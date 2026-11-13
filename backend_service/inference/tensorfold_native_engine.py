"""TensorFold 1.x native engine (``tensorfold-native serve``) for CUDA hosts.

Same shape as the Mac engine in ``tensorfold_engine.py`` (a local
OpenAI-compatible server we spawn and proxy), so request building, streaming,
tool calls and the acceptance metric are inherited unchanged. What differs:

* the binary comes from ``_tensorfold_native.find_binary`` (an archive unpacked
  into ``~/.chaosengine/tensorfold-native``, or a binary the user adopted), not
  from a Python venv;
* the capability gate runs before every load, from ``nvidia-smi`` and the
  binary's own ``capabilities --json``, and refuses with a readable message;
* serve flags are only passed when the binary lists them (``--vision``,
  ``--drafter``, ``--snapshot-dir``), so a newer or older build still starts;
* the model-memory check uses the measured VRAM figures in ``NATIVE_MODELS``
  against the card's free memory, not a fixed card size.

``load_model`` raises ``RuntimeError`` for every refusal or startup failure; the
controller does not fall back (there is no other engine for these checkpoints
on a CUDA host except llama.cpp, which the user picks explicitly).
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend_service.inference import _tensorfold_native as native
from backend_service.inference._tensorfold import match_model, native_context_window, resolve_model_dir, weights_complete
from backend_service.inference._utils import _append_runtime_note, _find_open_port, _isolated_child_env, _now_label
from backend_service.inference.base import BackendCapabilities, LoadedModelInfo
from backend_service.inference.tensorfold_engine import TensorFoldEngine, _emit_progress

_NOT_INSTALLED = "TensorFold (native) is not installed. Install it from the Setup tab."


def build_native_serve_command(
    binary: str,
    model_dir: Path,
    *,
    port: int,
    name: str,
    context: int | None,
    caps: native.NativeCapabilities | None,
    drafter_dir: Path | None = None,
    drafts: bool = True,
    vision: bool = False,
) -> list[str]:
    """The ``tensorfold-native serve`` argv. Optional flags appear only when ``caps`` lists them."""

    def has(flag: str) -> bool:
        return caps is None or caps.supports_flag(flag)

    command = [binary, "serve", str(model_dir), "--host", "127.0.0.1", "--port", str(port), "--name", name]
    if has("--no-update-check"):
        command.append("--no-update-check")
    if has("--snapshot-dir"):
        command += ["--snapshot-dir", "none"]
    if context and has("--context"):
        command += ["--context", str(context)]
    if has("--parallel"):
        command += ["--parallel", "1"]
    if vision and has("--vision"):
        command.append("--vision")
    if not drafts:
        if has("--no-drafts"):
            command.append("--no-drafts")
    elif drafter_dir is not None and has("--drafter"):
        command += ["--drafter", str(drafter_dir)]
    return command


def grammar_library_present(binary: Path) -> bool:
    """Structured output and required tool calls need ``tfgrammar`` beside the binary."""
    roots = (binary.parent, binary.parent / "lib", binary.parent.parent / "lib")
    return any(root.is_dir() and any(root.glob("*tfgrammar*")) for root in roots)


class TensorFoldNativeEngine(TensorFoldEngine):
    engine_name = "tensorfold-native"
    engine_label = "TensorFold 1.x (CUDA, exact speculative decoding)"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        super().__init__(capabilities)
        self._caps: native.NativeCapabilities | None = None

    def _tensorfold_bin(self) -> str:
        binary = native.find_binary()
        if binary is None:
            raise RuntimeError(_NOT_INSTALLED)
        return str(binary)

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str,
        canonical_repo: str | None,
        source: str,
        backend: str,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
        fit_model_in_memory: bool,
        context_tokens: int,
        speculative_decoding: bool = True,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        self.unload_model()
        binary = Path(self._tensorfold_bin())

        # Gate: this machine, this binary. The binary's own report wins.
        caps = native.probe_capabilities(binary)
        if caps is None:
            raise RuntimeError(f"{binary.name} did not answer 'capabilities --json'. Reinstall TensorFold from the Setup tab.")
        self._caps = caps
        gpu = native.read_gpu()
        gate = native.evaluate_gate(
            gpu, caps, key=native.platform_key(), asset_available=True, binary_present=True
        )
        if not gate.offered:
            raise RuntimeError(gate.message)

        model_dir = resolve_model_dir(path=path, runtime_target=runtime_target, repos=(canonical_repo, model_ref))
        if model_dir is None:
            raise RuntimeError(
                f"{model_name} is not downloaded. TensorFold serves local checkpoints only; download it from Discover first."
            )
        if not weights_complete(model_dir):
            raise RuntimeError(
                f"The download of {model_name} looks incomplete (weight files are missing from {model_dir}). "
                "Finish or repeat the download, then load it again."
            )

        model = native.match_native_model(canonical_repo, model_ref, runtime_target)
        # On the Metal build, the Mac family table names drafters and vision for
        # checkpoints the CUDA registry has no row for (MiniMax, Nanbeige, ...).
        # Its facts are Metal facts, and 1.x rows only (``native``).
        family = None
        if model is None and native.platform_key() == "macos-arm64":
            found = match_model(canonical_repo=canonical_repo, model_ref=model_ref, runtime_target=runtime_target, path=path)
            family = found.family if found is not None and found.family.native else None
        drafter_repo = model.drafter if model is not None else (family.drafter if family is not None else None)
        drafts = bool(speculative_decoding)
        drafter_dir: Path | None = None
        drafter_note: str | None = None
        if drafts and drafter_repo:
            drafter_dir, drafter_note = self._ensure_drafter(drafter_repo, progress_callback)
            if drafter_dir is None:
                drafts = False

        context = int(context_tokens or 0)
        window = native_context_window(model_dir)
        if window and context > window:
            context = window

        if model is not None and gpu is not None:
            need = native.estimate_vram_gb(model, context or model.ctx_ref, drafts=drafter_dir is not None)
            free = gpu.free_mb / 1000.0
            if need > free:
                raise RuntimeError(
                    f"{model.title} needs about {need:.1f} GB of VRAM at a {context or model.ctx_ref:,}-token window and "
                    f"{gpu.name} has {free:.1f} GB free. Close other GPU programs, lower the context, or pick a smaller model."
                )

        wants_vision = model.vision if model is not None else bool(family is not None and family.vision)
        vision = bool(wants_vision and caps.supports_flag("--vision"))
        image_note = None if vision else "TensorFold cannot read images for this model, so the attached image was ignored."

        self.port = _find_open_port()
        command = build_native_serve_command(
            str(binary),
            model_dir,
            port=self.port,
            name=model_ref,
            context=context or None,
            caps=caps,
            drafter_dir=drafter_dir,
            drafts=drafts,
            vision=vision,
        )

        temp_log = tempfile.NamedTemporaryFile(prefix="chaosengine-tensorfold-native-", suffix=".log", delete=False)
        temp_log.close()
        self.log_path = Path(temp_log.name)
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        home = Path.home() / ".chaosengine"
        home.mkdir(parents=True, exist_ok=True)
        _emit_progress(progress_callback, "starting", None, f"Starting TensorFold for {model_name}")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(home),
                env=_isolated_child_env({"HF_HUB_OFFLINE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1"}),
                stdout=self.log_handle,
                stderr=self.log_handle,
                text=True,
            )
        except OSError as exc:
            self._cleanup_process()
            raise RuntimeError(f"TensorFold could not be started: {exc}") from exc
        try:
            self._wait_for_server(model_name, progress_callback)
        except BaseException:
            self._cleanup_process()
            raise

        self._vision = vision
        self._grammar = grammar_library_present(binary)
        self._image_note = image_note

        if not drafts:
            note = "TensorFold serial decoding (speculative drafts off)."
        elif drafter_dir is not None:
            note = (
                f"TensorFold exact speculative decoding with draft model {drafter_repo}. "
                "Replies are identical to serial decoding."
            )
        else:
            note = "TensorFold exact speculative decoding (the model's own MTP head when it has one)."
        if drafter_note:
            note = _append_runtime_note(note, drafter_note)
        if gate.status == "unqualified" or gate.reasons:
            for reason in gate.reasons:
                note = _append_runtime_note(note, reason)
        if model is None and family is None:
            note = _append_runtime_note(note, "This checkpoint is not in the registry; replies stay exact, speed and VRAM are unmeasured.")
        if context and context != context_tokens:
            note = _append_runtime_note(note, f"Context window limited to {context:,} tokens, the most this checkpoint supports.")
        _emit_progress(progress_callback, "ready", 100.0, "TensorFold ready")

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=runtime_target or str(model_dir),
            runtimeNote=note,
            speculativeDecoding=drafts,
            dflashDraftModel=drafter_repo if drafter_dir is not None else None,
            visionEnabled=vision,
        )
        return self.loaded_model
