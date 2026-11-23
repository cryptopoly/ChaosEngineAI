"""vLLM-in-WSL inference engine (FU-056 Phase 8 follow-up).

vLLM ships no native Windows wheels. Windows users with the WSL2
bridge installed (Phase 8 foundation) get vLLM access through this
engine: it spawns vLLM's OpenAI-compatible HTTP server inside the
WSL Ubuntu venv, then proxies ``/v1/chat/completions`` from the
Windows-side backend.

Architecturally identical to ``MtplxEngine`` (subprocess + HTTP
proxy + ``_wait_for_server`` poll loop) but the command prefix is
``wsl -- ~/.chaosengine/vllm-venv/bin/python -m
vllm.entrypoints.openai.api_server`` instead of the host-native
mtplx binary.

WSL2 networking lets the Windows side reach the WSL listener on
``127.0.0.1:<port>`` transparently — the loopback adapter inside
WSL is mirrored to the Windows host's loopback by default. No
port forwarding needed.

Path translation: a model loaded from a Windows-side directory
(e.g. ``C:\\Users\\Dan\\AI_Models\\Qwen3-7B``) needs to be
addressed as ``/mnt/c/Users/Dan/AI_Models/Qwen3-7B`` when vLLM
runs inside WSL. HF repo ids (``Qwen/Qwen3.5-7B``) pass through
unchanged — vLLM downloads them into the WSL HF cache. The Windows
HF cache is reachable via ``/mnt/c/...`` but its NTFS-on-/mnt/c
performance is ~10× slower than the WSL ext4 home; we deliberately
let vLLM keep its own HF cache inside WSL.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from backend_service.inference._constants import (
    DEFAULT_LLAMA_TIMEOUT_SECONDS,
    WORKSPACE_ROOT,
)
from backend_service.inference._utils import (
    _find_open_port,
    _http_json,
    _normalize_message_content,
    _now_label,
    _read_log_with_root_cause,
    _read_text_tail,
)
from backend_service.reasoning_split import ThinkingTokenFilter, strip_thinking_tokens
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    StreamChunk,
)


# Same path the Phase 8 install endpoint writes into. Centralised
# here so the engine + the installer never drift out of sync.
_WSL_VLLM_VENV_PATH = "~/.chaosengine/vllm-venv"


def windows_path_to_wsl(path: str) -> str:
    """Translate a Windows-style path to its WSL ``/mnt/<drive>`` form.

    ``C:\\Users\\Dan\\AI_Models\\Qwen3-7B`` →
    ``/mnt/c/Users/Dan/AI_Models/Qwen3-7B``.

    Pass-throughs:
      - Already-WSL paths (start with ``/``): returned unchanged.
      - Forward-slash Windows paths (``C:/Users/...``): handled
        symmetrically to backslash form.
      - Non-path strings (HF repo ids, URLs, model names): returned
        unchanged. The detector is conservative — only translate
        strings that look like absolute Windows paths.
    """
    if not path:
        return path
    # Already a POSIX path; the user passed a WSL-native string.
    if path.startswith("/"):
        return path
    # Windows drive-letter pattern: ``X:\foo`` or ``X:/foo``. We don't
    # translate UNC paths (``\\server\share``) — those are rare for
    # local models and vLLM wouldn't load from them inside WSL anyway.
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if not match:
        return path
    drive = match.group(1).lower()
    tail = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{tail}"


# vLLM reserves ``--gpu-memory-utilization`` x total VRAM up front (weights plus the KV-cache pool) and
# defaults to 0.92, so a 1.1 GiB model took 23.6 of 24 GB and left nothing for the desktop or a second
# model. The bridge sizes the share to the model instead: its weights x 1.3 plus 4 GiB for the KV cache and
# activations (at least 6 GiB), never above the cap, and never more than what is free beside other apps.
# A big card gives a small model a small share; a model that needs more than the cap gets the cap, and
# ``CHAOSENGINE_VLLM_GPU_SHARE`` (0 to 1, the same switch as the native lane) overrides the lot.
_WSL_GPU_SHARE_CAP = 0.85
_WSL_GPU_KV_HEADROOM_BYTES = 4 * 1024**3
_WSL_GPU_MIN_BUDGET_BYTES = 6 * 1024**3
_WSL_GPU_FREE_MARGIN_BYTES = 1536 * 1024 * 1024
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def _gpu_free_total_bytes() -> tuple[int, int] | None:
    """Free and total memory of GPU 0 from the Windows driver's ``nvidia-smi`` (``None`` when unreadable)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode != 0:
            return None
        free_mib, total_mib = (int(part.strip()) for part in result.stdout.splitlines()[0].split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return free_mib * 1024**2, total_mib * 1024**2


def _local_weight_bytes(path: str | None) -> int:
    """Bytes of weight files in a local model folder; 0 for a repo id or a folder that is not there."""
    if not path:
        return 0
    try:
        folder = Path(path).expanduser()
        if not folder.is_dir():
            return 0
        return sum(
            entry.stat().st_size
            for entry in folder.iterdir()
            if entry.is_file() and entry.suffix.lower() in _WEIGHT_FILE_SUFFIXES
        )
    except OSError:
        return 0


def gpu_memory_utilization(weight_bytes: int = 0) -> float:
    """The ``--gpu-memory-utilization`` share for one load (see the note above the constants)."""
    override = os.environ.get("CHAOSENGINE_VLLM_GPU_SHARE", "").strip()
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        if 0.0 < value <= 1.0:
            return value
    sizes = _gpu_free_total_bytes()
    if sizes is None:
        return _WSL_GPU_SHARE_CAP
    free, total = sizes
    if total <= 0:
        return _WSL_GPU_SHARE_CAP
    share = _WSL_GPU_SHARE_CAP
    if weight_bytes > 0:
        budget = max(int(weight_bytes * 1.3) + _WSL_GPU_KV_HEADROOM_BYTES, _WSL_GPU_MIN_BUDGET_BYTES)
        share = min(share, budget / total)
    # vLLM refuses to start when its share is more than the free memory.
    share = min(share, (free - _WSL_GPU_FREE_MARGIN_BYTES) / total)
    return round(max(share, 0.05), 3)


def _dflash_speculative_config(
    *,
    wsl_vllm_version: str | None,
    canonical_repo: str | None,
    path: str | None,
    model_ref: str,
) -> tuple[dict[str, Any] | None, str]:
    """vLLM ``--speculative-config`` for a DFlash drafter, as the native lane builds it, plus a note.

    Needs vLLM >= ``VLLM_DFLASH_MIN_VERSION`` inside WSL and a drafter registered for the target.
    """
    from dflash import VLLM_DFLASH_MIN_VERSION, _version_tuple, get_draft_model

    parsed = _version_tuple(wsl_vllm_version or "")
    if parsed is None or parsed < VLLM_DFLASH_MIN_VERSION:
        need = ".".join(str(part) for part in VLLM_DFLASH_MIN_VERSION)
        return None, f"DFlash needs vLLM {need} or newer in WSL (found {wsl_vllm_version or 'unknown'})."
    from backend_service.model_resolution import resolve_dflash_target_ref
    from backend_service.vllm_engine import _dflash_num_speculative_tokens

    target = resolve_dflash_target_ref(canonical_repo=canonical_repo, path=path, model_ref=model_ref) or model_ref
    draft_model = get_draft_model(target, allow_dflash2=True)
    if not draft_model:
        return None, f"DFlash unavailable for '{target}': no compatible draft model is registered."
    config: dict[str, Any] = {"method": "dflash", "model": draft_model}
    draft_tokens = _dflash_num_speculative_tokens(draft_model)
    if draft_tokens:
        config["num_speculative_tokens"] = draft_tokens
    return config, f"DFlash speculative decoding active (draft: {draft_model})."


class VllmWslEngine(BaseInferenceEngine):
    """vLLM running inside the WSL isolated venv, proxied via HTTP.

    Spawn shape: ``wsl -- <venv>/bin/python -m
    vllm.entrypoints.openai.api_server --model X --host 127.0.0.1
    --port N``. We talk to it through the existing ``urllib`` HTTP
    helpers — same as MtplxEngine. The Windows process inherits all
    the stream / generate plumbing from ``BaseInferenceEngine`` so
    upstream callers don't need a "is this WSL?" branch.
    """

    engine_name = "vllm-wsl"
    engine_label = "vLLM (WSL bridge)"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.log_path: Path | None = None
        self.log_handle: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _server_url(self, path: str) -> str:
        if self.port is None:
            raise RuntimeError("vLLM WSL server is not running.")
        return f"http://127.0.0.1:{self.port}{path}"

    def _build_wsl_command(
        self,
        *,
        model_arg: str,
        port: int,
        max_model_len: int,
        served_model_name: str | None = None,
        gpu_memory_utilization: float | None = None,
        speculative_config: dict[str, Any] | None = None,
    ) -> list[str]:
        """Compose the ``wsl --exec bash -c "<inner cmd>"`` argv.

        We wrap the python invocation in ``bash -c`` so we can prepend
        the venv's ``bin/`` to ``PATH``. This matters because vLLM's
        runtime path (notably flashinfer) JIT-builds CUDA sampling
        kernels with ``ninja``, which lives at
        ``~/.chaosengine/vllm-venv/bin/ninja`` after the install.
        Without the venv bin on PATH the JIT build crashes with
        ``FileNotFoundError: 'ninja'`` mid-startup — caught live on the
        FU-056 Phase 8 follow-up live test against opt-125m.

        Arguments are ``shlex.quote``-escaped so a model path with
        spaces / quotes can't break the bash parse.
        """
        bin_path = f"{_WSL_VLLM_VENV_PATH}/bin"
        # Inside double quotes bash leaves ``~`` alone, so the PATH entry must name ``$HOME``: a literal ``~`` entry
        # is found by bash's own lookup but not by a child's ``execvp`` (vLLM's ninja call), FU-533.
        path_bin = "$HOME" + bin_path[1:]
        # PATH assignment is double-quoted — WSL2 interopts the Windows
        # PATH into bash, and that PATH contains paths with spaces
        # (e.g. ``/mnt/c/Program Files/NVIDIA...``). Without the quotes
        # bash word-splits the expanded ``$PATH`` and crashes with
        # ``export: 'Files/NVIDIA': not a valid identifier``.
        #
        # ``VLLM_USE_FLASHINFER_SAMPLER=0`` is the bypass for the vLLM 0.21+
        # flashinfer JIT path that otherwise crashes mid-startup with
        # ``FileNotFoundError: 'ninja'`` (the JIT subprocess can't see venv
        # bin on PATH through vLLM's multiprocessing fork chain); it is
        # still a live variable in vLLM 0.34. ``VLLM_ATTENTION_BACKEND=
        # TORCH_SDPA`` used to sit beside it, but vLLM 0.34 no longer has
        # that variable (it logs "Unknown vLLM environment variable" and
        # ignores it; ``TORCH_SDPA`` is only a ViT tag there now), so it was
        # dropped. The backend is picked with ``--attention-backend``.
        #
        # ``--host 127.0.0.1`` keeps vLLM bound to loopback — WSL2
        # mirrors loopback to the Windows host so the Windows backend
        # reaches the listener without any port-forward setup, and the
        # model never leaks to the LAN.
        #
        # ``--max-model-len`` is vLLM's context cap; the model card's
        # default is often too large for available VRAM.
        #
        # ``--trust-remote-code`` covers repos like Qwen3-VL that ship
        # custom modeling code; vLLM refuses to import those by default.
        #
        # ``--enforce-eager`` skips CUDA-graph compilation. Loses some
        # perf on long generations but avoids the second JIT path that
        # would otherwise need a system ``nvcc``.
        # ``--served-model-name`` keeps the name chat requests use (the model ref) valid when vLLM was started from a
        # local folder: vLLM otherwise serves the model under the folder path and answers 404 "model does not exist"
        # (RTX 4090, 2026-11-19).
        served = f"--served-model-name {shlex.quote(served_model_name)} " if served_model_name else ""
        # ``--gpu-memory-utilization``: the share of VRAM vLLM may reserve (see ``gpu_memory_utilization``).
        extra = ""
        if gpu_memory_utilization is not None:
            extra += f"--gpu-memory-utilization {gpu_memory_utilization:g} "
        if speculative_config:
            # DFlash targets are hybrid models (Gated DeltaNet) that reserve a state block per sequence:
            # vLLM's server default of 256 sequences left a 27B on a 24 GB card with room for about 30,
            # so the native lane runs 8 (``_DESKTOP_MAX_NUM_SEQS``) and the bridge does too.
            extra += (
                f"--speculative-config {shlex.quote(json.dumps(speculative_config, separators=(',', ':')))} "
                "--max-num-seqs 8 "
            )
        inner = (
            f'export PATH="{path_bin}:$PATH" && '
            'export VLLM_USE_FLASHINFER_SAMPLER=0 && '
            f"{bin_path}/python -m vllm.entrypoints.openai.api_server "
            f"--model {shlex.quote(model_arg)} "
            f"{served}"
            f"{extra}"
            f"--host 127.0.0.1 "
            f"--port {port} "
            f"--max-model-len {max_model_len} "
            f"--trust-remote-code "
            f"--enforce-eager"
        )
        # ``--exec`` runs bash directly. ``wsl -- bash -c`` hands the command line to the distro's default shell first,
        # which expands ``$(...)``, backticks and ``$VAR`` even inside the ``shlex.quote`` single quotes (they sit in
        # its double quotes), so a model ref or path naming ``$(...)`` ran as a command (security review 2026-11-20).
        return ["wsl", "--exec", "bash", "-c", inner]

    def _cleanup_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # vLLM loads big models — the graceful terminate can
                # take a few seconds while it tears down CUDA tensors.
                # If it's still alive after 10 s, SIGKILL.
                try:
                    self.process.kill()
                except (ProcessLookupError, OSError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self.process = None
        self.port = None
        if self.log_handle is not None:
            try:
                self.log_handle.close()
            except OSError:
                pass
        self.log_handle = None

    def process_pid(self) -> int | None:
        if self.process is None or self.process.poll() is not None:
            return None
        return int(self.process.pid)

    def _wait_for_server(self) -> None:
        """Poll ``/v1/models`` until vLLM accepts requests, or the subprocess dies.

        vLLM's startup is slow (~30-90 s for a 7B model on cold cache,
        ~10-30 s for tiny models) because it builds CUDA-graph caches
        + warms KV blocks. We give it the standard llama-timeout
        budget and surface the captured log if it dies before becoming
        ready.

        Probes ``/v1/models`` rather than ``/health`` — vLLM's
        ``/health`` returns 200 with an empty body, which trips
        ``_http_json``'s ``json.loads`` (caught live during the FU-056
        Phase 8 follow-up test). ``/v1/models`` returns the loaded
        model list as JSON so the JSON parse succeeds.
        """
        deadline = time.time() + DEFAULT_LLAMA_TIMEOUT_SECONDS
        last_error = "vLLM (WSL) did not become ready."
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                logs = _read_log_with_root_cause(self.log_path)
                raise RuntimeError(logs or "vLLM (WSL) exited during startup.")
            try:
                _http_json(self._server_url("/v1/models"), timeout=2.0)
                return
            except Exception as exc:  # noqa: BLE001 — best-effort poll
                last_error = str(exc)
            time.sleep(1.0)
        logs = _read_log_with_root_cause(self.log_path)
        raise RuntimeError(logs if logs else last_error)

    # ------------------------------------------------------------------
    # BaseInferenceEngine interface
    # ------------------------------------------------------------------

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
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> LoadedModelInfo:
        if sys.platform != "win32":
            raise RuntimeError(
                "vLLM WSL bridge is Windows-only. Use the native vLLM "
                "engine on Linux."
            )
        if not self.capabilities.wslVllmAvailable:
            raise RuntimeError(
                "vLLM isn't installed in WSL. Install it from the "
                "Diagnostics → WSL2 vLLM bridge panel."
            )

        self.unload_model()

        self.port = _find_open_port()

        # Pick the most precise model reference available:
        #   1. local path (translated to /mnt/c/... if Windows-style)
        #   2. runtime_target (catalog override)
        #   3. canonical HF repo (vLLM downloads to its WSL HF cache)
        #   4. model_ref (last resort — usually equal to #3)
        if path:
            model_arg = windows_path_to_wsl(path)
        elif runtime_target:
            model_arg = windows_path_to_wsl(runtime_target)
        else:
            model_arg = canonical_repo or model_ref

        gpu_share = gpu_memory_utilization(_local_weight_bytes(path or runtime_target))

        # DFlash over the bridge: the same ``speculative_config`` the native lane passes to ``LLM(...)``,
        # given to the OpenAI server as ``--speculative-config`` (vLLM >= 0.28).
        spec_config: dict[str, Any] | None = None
        spec_note = ""
        if speculative_decoding:
            try:
                spec_config, spec_note = _dflash_speculative_config(
                    wsl_vllm_version=self.capabilities.wslVllmVersion,
                    canonical_repo=canonical_repo,
                    path=path,
                    model_ref=model_ref,
                )
            except ImportError:
                spec_note = "DFlash is not available in this build."

        command = self._build_wsl_command(
            model_arg=model_arg,
            port=self.port,
            max_model_len=context_tokens,
            served_model_name=model_ref,
            gpu_memory_utilization=gpu_share,
            speculative_config=spec_config,
        )

        if progress_callback:
            progress_callback({
                "phase": "loading",
                "percent": 10.0,
                "message": f"Spawning vLLM in WSL for {model_name}...",
            })

        temp_log = tempfile.NamedTemporaryFile(
            prefix="chaosengine-vllm-wsl-", suffix=".log", delete=False
        )
        temp_log.close()
        self.log_path = Path(temp_log.name)
        self.log_handle = self.log_path.open("a", encoding="utf-8")

        self.process = subprocess.Popen(
            command,
            cwd=str(WORKSPACE_ROOT),
            stdout=self.log_handle,
            stderr=self.log_handle,
            text=True,
        )

        try:
            self._wait_for_server()
        except RuntimeError:
            self._cleanup_process()
            raise

        runtime_note = (
            f"vLLM running inside WSL ({self.capabilities.wslDistroName or 'Ubuntu'}) "
            f"venv at {_WSL_VLLM_VENV_PATH}."
        )
        if self.capabilities.wslVllmVersion:
            runtime_note = (
                f"vLLM {self.capabilities.wslVllmVersion} running inside WSL "
                f"({self.capabilities.wslDistroName or 'Ubuntu'})."
            )
        runtime_note += f" GPU memory share {gpu_share:.0%}."
        if spec_note:
            runtime_note += " " + spec_note

        if progress_callback:
            progress_callback({
                "phase": "ready",
                "percent": 100.0,
                "message": "vLLM (WSL) ready.",
            })

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=cache_strategy,
            cacheBits=cache_bits,
            fp16Layers=fp16_layers,
            fusedAttention=fused_attention,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=model_arg,
            runtimeNote=runtime_note,
            speculativeDecoding=spec_config is not None,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self._cleanup_process()
        self.loaded_model = None

    def generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The vLLM (WSL) server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({
                "role": role,
                "content": _normalize_message_content(message.get("text", "")),
            })
        messages.append({"role": "user", "content": prompt})

        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            response = _http_json(
                self._server_url("/v1/chat/completions"),
                payload=payload,
                timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = response.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        # vLLM's server leaves ``<think>...</think>`` inside ``content`` unless a reasoning parser is
        # configured; drop it from the reply the way the llama.cpp lane does.
        text = strip_thinking_tokens(str(message.get("content") or ""))

        return GenerationResult(
            text=text,
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=int(usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
            tokS=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=self.loaded_model.runtimeNote,
        )

    def stream_generate(
        self,
        *,
        prompt: str,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        images: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking_mode: str | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> Iterator[StreamChunk]:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The vLLM (WSL) server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({
                "role": role,
                "content": _normalize_message_content(message.get("text", "")),
            })
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        url = self._server_url("/v1/chat/completions")
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(request, timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(detail or str(exc)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(str(exc.reason)) from exc

        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0
        started_at = time.perf_counter()
        # Split ``<think>`` blocks into the reasoning channel, as the llama.cpp lane does.
        think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode or "off") != "off")

        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                # A server started with a reasoning parser sends the block as its own field.
                server_reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if server_reasoning:
                    yield StreamChunk(reasoning=str(server_reasoning))
                text_delta = delta.get("content")
                if text_delta:
                    split = think_filter.feed(str(text_delta))
                    if split.reasoning:
                        yield StreamChunk(reasoning=split.reasoning)
                    if split.reasoning_done:
                        yield StreamChunk(reasoning_done=True)
                    if split.text:
                        yield StreamChunk(text=split.text)
                # vLLM emits ``finish_reason`` on the last delta only.
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = str(fr)
                usage = chunk.get("usage") or {}
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)

        flushed = think_filter.flush()
        if flushed.reasoning:
            yield StreamChunk(reasoning=flushed.reasoning)
        if flushed.reasoning_done:
            yield StreamChunk(reasoning_done=True)
        if flushed.text:
            yield StreamChunk(text=flushed.text)

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            runtime_note=self.loaded_model.runtimeNote if self.loaded_model else None,
        )
