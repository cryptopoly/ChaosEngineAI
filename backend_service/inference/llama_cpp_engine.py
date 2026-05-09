"""llama.cpp / GGUF inference engine.

Wraps the ``llama-server`` binary (or the TurboQuant fork
``llama-server-turbo``) as a subprocess on a localhost port and proxies
chat completions through its ``/v1/chat/completions`` HTTP endpoint.

This module also owns the llama.cpp-specific helpers that previously
sat at the top of the monolithic ``inference.py``:

- ``_apply_llama_chat_template_fixes`` — Phase 3.8 follow-up:
  Gemma-family models reject the system role; fold it into the first
  user message client-side.
- ``_apply_sampler_kwargs`` — merge Phase 2.2 sampler overrides
  (top_p, top_k, mirostat, frequency_penalty, …) into the chat
  payload.
- ``_friendly_llama_error`` — translate startup log tails into
  actionable messages ("unknown architecture", "out of memory", …).
- ``_llama_server_help_text`` / ``_llama_server_supports`` /
  ``_llama_server_cache_types`` — query the binary's ``--help`` to
  feature-detect flags + supported cache types before spawning.
- ``_resolve_mmproj_path`` — locate the mmproj projector sibling for
  vision-capable GGUFs.
- ``_gguf_startup_fallback_note`` — runtime note when the requested
  cache strategy fails and we retry with native f16.

The two public test-imports — ``_llama_server_cache_types``,
``_STANDARD_CACHE_TYPES``, ``_CACHE_TYPE_CACHE`` — are re-exported from
``backend_service.inference`` so existing test code keeps working.

Extracted from ``inference.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import RLock
from typing import Any

from backend_service.inference._constants import (
    DEFAULT_LLAMA_TIMEOUT_SECONDS,
    WORKSPACE_ROOT,
)
from backend_service.inference._utils import (
    _append_runtime_note,
    _find_open_port,
    _http_json,
    _is_local_target,
    _normalize_message_content,
    _now_label,
    _read_text_tail,
    _resolve_gguf_path,
)
from backend_service.inference.base import (
    BackendCapabilities,
    BaseInferenceEngine,
    GenerationResult,
    LoadedModelInfo,
    RepeatedLineGuard,
    StreamChunk,
)
from backend_service.reasoning_split import (
    ThinkingTokenFilter,
    strip_thinking_tokens as _strip_thinking_tokens,
)


# Phase 2.2: keys forwarded as-is from `samplers` into the llama-server
# /v1/chat/completions payload. Anything not in this set is silently
# ignored so the frontend can blindly send the union of supported knobs
# without breaking older llama-server builds that don't recognise some.
_LLAMA_SAMPLER_KEYS: tuple[str, ...] = (
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "seed",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    # Phase 2.13: OpenAI-spec penalty fields. llama-server accepts these
    # natively under the same names. mlx-lm doesn't pass them through
    # but `_apply_sampler_kwargs` only adds them to the llama path
    # payload, so the worker subprocess is unaffected.
    "frequency_penalty",
    "presence_penalty",
    "stop",
    # Phase 3.3: per-token confidence info. llama-server returns
    # top-k alternatives with their logprobs in each delta when
    # `logprobs: true` + `top_logprobs: N` are set.
    "logprobs",
    "top_logprobs",
)


def _apply_llama_chat_template_fixes(
    messages: list[dict[str, Any]],
    loaded_model: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    """Phase 3.8 follow-up: apply known chat-template auto-fixes before
    sending the message list to llama-server.

    The llama.cpp server applies the chat template internally based on
    GGUF metadata, so we can't observe template Jinja directly. But we
    know certain families (Gemma) reject the system role entirely;
    folding the system message into the first user message client-side
    avoids the template error.

    Returns ``(new_messages, runtime_note)``. The note is None when no
    fix was applied; when set it's a single line suitable for the
    GenerationResult.runtimeNote channel so the substrate badge can
    show "auto-fixed: Gemma family — fold system into first user".
    """
    if not loaded_model or not messages:
        return messages, None

    from backend_service.helpers.chat_template import (
        fold_system_into_first_user,
        is_gemma_family,
    )

    model_ref = getattr(loaded_model, "ref", None)
    canonical = getattr(loaded_model, "canonicalRepo", None)
    target = canonical or model_ref

    if is_gemma_family(target):
        new_messages = fold_system_into_first_user(messages)
        if len(new_messages) != len(messages):
            return new_messages, "Chat template auto-fixed: Gemma family — fold system into first user message"
        return new_messages, None

    return messages, None


def _apply_sampler_kwargs(
    payload: dict[str, Any],
    *,
    samplers: dict[str, Any] | None,
    reasoning_effort: str | None,
    json_schema: dict[str, Any] | None,
) -> None:
    """Merge Phase 2.2 sampler overrides into a chat-completions payload.

    Mutates `payload` in place. Skips keys whose value is None so an
    explicit "use the default" from a UI that always sends every field
    doesn't override server-side defaults. Json-schema is wrapped in
    the OpenAI structured-outputs `response_format` envelope.
    """
    if samplers:
        for key in _LLAMA_SAMPLER_KEYS:
            value = samplers.get(key)
            if value is None:
                continue
            payload[key] = value
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": json_schema,
                "strict": True,
            },
        }


_LLAMA_HELP_CACHE: dict[str, str] = {}
_LLAMA_HELP_LOCK = RLock()


def _friendly_llama_error(logs: str | None) -> str:
    """Translate known llama.cpp startup failures into actionable messages.

    Falls back to the original log tail when nothing matches.
    """
    if not logs:
        return "llama.cpp server exited during startup."
    lower = logs.lower()
    if "unknown model architecture" in lower:
        match = re.search(r"unknown model architecture:\s*'([^']+)'", logs)
        arch = match.group(1) if match else "this model"
        return (
            f"llama.cpp does not recognise architecture '{arch}'. Your "
            f"llama.cpp build may be too old for this model. Update it "
            f"by installing a newer llama-server binary."
        )
    if "failed to allocate" in lower or "out of memory" in lower:
        return (
            "llama.cpp ran out of memory loading this model. Try a smaller "
            "quantisation, reduce the context window, or close other apps "
            "using the GPU."
        )
    info_only_lines = [
        line.strip()
        for line in logs.splitlines()
        if line.strip()
    ]
    if info_only_lines and all(
        re.match(r"^load_backend: loaded .* backend", line, re.IGNORECASE)
        or re.match(r"^ggml_metal_device_init: tensor api disabled .*", line, re.IGNORECASE)
        or re.match(r"^ggml_metal_library_init: using embedded metal library$", line, re.IGNORECASE)
        for line in info_only_lines
    ):
        return (
            "llama.cpp exited during startup before reporting a specific error. "
            "The visible ggml/Metal lines are informational startup messages, not the cause. "
            "Retry with Native f16 or inspect the full server log for the real failure."
        )
    return logs


def _llama_server_help_text(binary_path: str | None) -> str:
    if not binary_path:
        return ""
    with _LLAMA_HELP_LOCK:
        cached = _LLAMA_HELP_CACHE.get(binary_path)
        if cached is not None:
            return cached

    try:
        completed = subprocess.run(
            [binary_path, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8.0,
        )
        help_text = "\n".join(part for part in (completed.stdout, completed.stderr) if part).lower()
    except (OSError, subprocess.TimeoutExpired):
        help_text = ""

    with _LLAMA_HELP_LOCK:
        _LLAMA_HELP_CACHE[binary_path] = help_text
    return help_text


def _llama_server_supports(binary_path: str | None, option: str) -> bool:
    return option.lower() in _llama_server_help_text(binary_path)


# Baseline set assumed when help text cannot be parsed.
_STANDARD_CACHE_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
)

_CACHE_TYPE_CACHE: dict[str, frozenset[str]] = {}


def _llama_server_cache_types(binary_path: str | None) -> frozenset[str]:
    """Extract supported ``--cache-type-k`` values from the binary's help text."""
    if not binary_path:
        return _STANDARD_CACHE_TYPES
    cached = _CACHE_TYPE_CACHE.get(binary_path)
    if cached is not None:
        return cached

    help_text = _llama_server_help_text(binary_path)
    if not help_text:
        _CACHE_TYPE_CACHE[binary_path] = _STANDARD_CACHE_TYPES
        return _STANDARD_CACHE_TYPES

    # The help text contains lines like:
    #   allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1,
    #                   turbo2, turbo3, turbo4, planar3, iso3, planar4, iso4
    # Values may wrap across multiple lines.  We capture everything up to
    # the next ``(`` (which starts the default value parenthetical).
    match = re.search(
        r"cache-type-k.*?allowed\s+values:\s*([a-z0-9_, \n\r\t]+)",
        help_text,
        re.DOTALL,
    )
    if match:
        raw = match.group(1).replace("\n", " ").replace("\r", " ")
        types = frozenset(t.strip() for t in raw.split(",") if t.strip())
    else:
        types = _STANDARD_CACHE_TYPES
    _CACHE_TYPE_CACHE[binary_path] = types
    return types


def _resolve_mmproj_path(model_gguf_path: str | None) -> str | None:
    """Locate the mmproj projector sibling for a vision-capable GGUF.

    Vision support in llama.cpp is gated by the `--mmproj` flag; the
    projector lives as a separate `*mmproj*.gguf` file alongside the
    main weights. HF repos for vision-capable models usually ship both
    in the same snapshot (e.g. `gemma-3-27b-it-qat-4bit/` contains
    `model.gguf` and `mmproj.gguf`). This helper scans the same
    directory tree the main GGUF was found in and returns the largest
    matching projector file, or None when no projector is present (the
    model is text-only, or the user only downloaded the main weights).
    """
    if not model_gguf_path:
        return None
    main_path = Path(model_gguf_path)
    if not main_path.exists():
        return None

    # Search the parent directory + its immediate sibling directories
    # (covers the HF snapshot layout where projectors might live in a
    # `projectors/` peer to the `weights/` folder). We deliberately do
    # NOT recurse via `rglob` past one level — on macOS test rigs the
    # parent's parent is sometimes a system-cache root that raises
    # `OSError: Result too large` mid-scandir. Bounded depth keeps the
    # resolver predictable across hosts.
    candidates: list[Path] = []
    parent = main_path.parent
    if parent.is_dir():
        for entry in parent.iterdir():
            if entry.is_file() and entry.suffix.lower() == ".gguf" and "mmproj" in entry.name.lower():
                candidates.append(entry)
            elif entry.is_dir():
                try:
                    for child in entry.iterdir():
                        if (
                            child.is_file()
                            and child.suffix.lower() == ".gguf"
                            and "mmproj" in child.name.lower()
                        ):
                            candidates.append(child)
                except OSError:
                    continue
    grandparent = parent.parent
    if grandparent.is_dir() and grandparent != parent:
        try:
            for entry in grandparent.iterdir():
                if not entry.is_dir() or entry == parent:
                    continue
                try:
                    for child in entry.iterdir():
                        if (
                            child.is_file()
                            and child.suffix.lower() == ".gguf"
                            and "mmproj" in child.name.lower()
                            and child not in candidates
                        ):
                            candidates.append(child)
                except OSError:
                    continue
        except OSError:
            pass

    valid = [p for p in candidates if p.is_file() and p != main_path]
    if not valid:
        return None
    valid.sort(key=lambda f: f.stat().st_size, reverse=True)
    return str(valid[0])


def _gguf_startup_fallback_note(strategy_name: str) -> str:
    return (
        f"GGUF startup failed with {strategy_name} cache, so ChaosEngineAI retried "
        f"with the standard f16 KV cache."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LlamaCppEngine(BaseInferenceEngine):
    engine_name = "llama.cpp"
    engine_label = "llama.cpp + GGUF"

    def __init__(self, capabilities: BackendCapabilities) -> None:
        self.capabilities = capabilities
        self.loaded_model: LoadedModelInfo | None = None
        self.process: subprocess.Popen[str] | None = None
        self.port: int | None = None
        self.log_path: Path | None = None
        self.log_handle: Any = None

    def _server_url(self, path: str) -> str:
        if self.port is None:
            raise RuntimeError("llama.cpp server is not running.")
        return f"http://127.0.0.1:{self.port}{path}"

    def _cleanup_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            # llama-server now shares the Python backend's process group,
            # so we target just the single process. killpg would take the
            # entire backend down with it.
            try:
                self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
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

    def _select_llama_binary(self, strategy: Any) -> str:
        """Pick the correct llama-server binary for *strategy*.

        Routing rules:
        1. If the strategy requires ``"turbo"`` and the turbo binary is
           available, use it.
        2. Otherwise fall back to the standard binary.
        3. If neither binary is available, raise.
        """
        variant = strategy.required_llama_binary()
        if variant == "turbo" and self.capabilities.llamaServerTurboPath:
            return self.capabilities.llamaServerTurboPath
        if self.capabilities.llamaServerPath:
            return self.capabilities.llamaServerPath
        # If only the turbo binary exists (no standard), it is a superset
        # of standard and can serve all strategies.
        if self.capabilities.llamaServerTurboPath:
            return self.capabilities.llamaServerTurboPath
        raise RuntimeError("llama-server was not found on this machine.")

    def _build_command(
        self,
        *,
        path: str | None,
        runtime_target: str | None,
        cache_strategy: str,
        cache_bits: int,
        context_tokens: int,
        fit_enabled: bool,
        is_fallback: bool,
    ) -> tuple[list[str], str | None, bool, str | None]:
        """Build the llama-server command line.

        Returns ``(command, runtime_note, fell_back_to_native, mmproj_path)`` where
        *fell_back_to_native* is ``True`` when pre-validation detected
        unsupported cache types and silently switched to f16.
        """
        from cache_compression import registry as _strategy_registry
        strategy = _strategy_registry.get(cache_strategy) or _strategy_registry.default()

        binary = self._select_llama_binary(strategy)
        runtime_note = None
        fell_back_to_native = False
        self.port = _find_open_port()
        command = [
            binary,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--parallel",
            "1",
            "--ctx-size",
            str(max(256, context_tokens)),
            "--jinja",
        ]
        if _llama_server_supports(binary, "--reasoning-format"):
            command.extend(["--reasoning-format", "deepseek"])
        if _llama_server_supports(binary, "--reasoning"):
            command.extend(["--reasoning", "off"])
        if fit_enabled:
            command.extend(["--fit", "on"])
        else:
            command.extend(["--fit", "off"])

        try:
            cache_flags = strategy.llama_cpp_cache_flags(cache_bits)
        except NotImplementedError:
            cache_flags = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
            runtime_note = f"Cache strategy '{strategy.name}' does not support llama.cpp yet; using native f16 cache."

        if is_fallback:
            cache_flags = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
            runtime_note = (
                f"GGUF startup failed with {strategy.name} cache, so ChaosEngineAI retried with the standard f16 KV cache."
            )

        # Pre-validate cache types against the selected binary's
        # supported set.  If unsupported, fall back to f16 immediately
        # instead of waiting for a startup timeout.
        if not is_fallback and runtime_note is None:
            supported = _llama_server_cache_types(binary)
            for i, flag in enumerate(cache_flags):
                if flag.startswith("--cache-type-") and i + 1 < len(cache_flags):
                    cache_type = cache_flags[i + 1]
                    if cache_type not in supported:
                        variant = strategy.required_llama_binary()
                        if variant == "turbo" and not self.capabilities.llamaServerTurboPath:
                            runtime_note = (
                                f"{strategy.name} requires llama-server-turbo "
                                f"(the TurboQuant fork) which is not installed. "
                                f"Run scripts/build-llama-turbo.sh to install it. "
                                f"Using native f16 cache instead."
                            )
                        else:
                            runtime_note = (
                                f"Cache type '{cache_type}' is not supported by "
                                f"the installed llama-server; using f16 cache."
                            )
                        cache_flags = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
                        fell_back_to_native = True
                        break

        command.extend(cache_flags)

        target = runtime_target or path
        resolved_gguf = _resolve_gguf_path(path, target)
        if resolved_gguf:
            command.extend(["--model", resolved_gguf])
        elif path:
            command.extend(["--model", path])
        elif target:
            command.extend(["--hf-repo", target])
        else:
            raise RuntimeError("GGUF loading requires a local model path or a Hugging Face GGUF repository.")

        # Vision wiring: if a sibling mmproj file is present, pass it
        # via `--mmproj` so llama-server enables image input. Capture
        # the path so the caller can flip `LoadedModelInfo.visionEnabled`
        # to True; the capability resolver reads that flag to enable
        # the composer's image-attach button. Older llama-server builds
        # without `--mmproj` skip the flag silently — verify support
        # via the help-text gate to avoid startup failure on those.
        mmproj_path: str | None = None
        if resolved_gguf and _llama_server_supports(binary, "--mmproj"):
            mmproj_path = _resolve_mmproj_path(resolved_gguf)
            if mmproj_path:
                command.extend(["--mmproj", mmproj_path])

        return command, runtime_note, fell_back_to_native, mmproj_path

    def _wait_for_server(self) -> None:
        deadline = time.time() + DEFAULT_LLAMA_TIMEOUT_SECONDS
        last_error = "llama.cpp server did not become ready."

        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                logs = _read_text_tail(self.log_path)
                raise RuntimeError(_friendly_llama_error(logs))

            try:
                _http_json(self._server_url("/health"), timeout=2.0)
                models = _http_json(self._server_url("/v1/models"), timeout=2.0)
                if isinstance(models, dict):
                    return
            except Exception as exc:
                last_error = str(exc)
            time.sleep(1.0)

        logs = _read_text_tail(self.log_path)
        raise RuntimeError(_friendly_llama_error(logs) if logs else last_error)

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
        if not self.capabilities.ggufAvailable:
            raise RuntimeError("llama.cpp support is unavailable on this machine.")

        if _is_local_target(path) or _is_local_target(runtime_target):
            resolved_preflight = _resolve_gguf_path(path, runtime_target)
            if resolved_preflight is None:
                raise RuntimeError(
                    f"No .gguf weights found inside {path or runtime_target}. "
                    f"The download may be incomplete or corrupt. Re-download the model, "
                    f"or pick a specific .gguf file from the source directory."
                )

        self.unload_model()
        runtime_note = None
        actual_strategy = cache_strategy
        actual_fit = fit_model_in_memory
        from cache_compression import registry as _strategy_registry
        failed_strategy_name: str | None = None

        # Try the requested strategy first.  If it fails, try ChaosEngine
        # (which uses standard cache types on the standard llama-server),
        # then finally native f16.  This ensures the user gets the best
        # available compression even when the turbo binary can't load a
        # particular model architecture.
        attempts: list[tuple[str, bool, bool]] = [(cache_strategy, fit_model_in_memory, False)]
        if cache_strategy not in ("native", "chaosengine"):
            # Always include ChaosEngine as an intermediate fallback.  Its
            # llama.cpp path only emits standard cache-type flags (q4_0 etc.)
            # and runs on the standard binary — it does NOT require the
            # chaos_engine Python package to be installed.  Gating on
            # is_available() would skip this fallback on CI / dev machines
            # that don't have the package, breaking the 3-level chain.
            if _strategy_registry.get("chaosengine") is not None:
                attempts.append(("chaosengine", False, True))
        if cache_strategy != "native":
            attempts.append(("native", False, True))
        last_error: str | None = None

        attempt_mmproj_path: str | None = None
        for strategy_id, fit_enabled, is_fallback in attempts:
            strategy = _strategy_registry.get(strategy_id) or _strategy_registry.default()
            command, attempt_note, prevalidation_fallback, attempt_mmproj_path = self._build_command(
                path=path,
                runtime_target=runtime_target,
                cache_strategy=strategy_id,
                cache_bits=cache_bits,
                context_tokens=context_tokens,
                fit_enabled=fit_enabled,
                is_fallback=is_fallback,
            )

            temp_log = tempfile.NamedTemporaryFile(prefix="chaosengine-llama-", suffix=".log", delete=False)
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
                runtime_note = attempt_note
                actual_strategy = "native" if prevalidation_fallback else strategy_id
                actual_fit = fit_enabled
                if is_fallback and failed_strategy_name is not None:
                    fallback_strat = _strategy_registry.get(strategy_id) or _strategy_registry.default()
                    if strategy_id == "native":
                        runtime_note = _gguf_startup_fallback_note(failed_strategy_name)
                    else:
                        runtime_note = (
                            f"GGUF startup failed with {failed_strategy_name} cache "
                            f"(the turbo binary may not support this model architecture). "
                            f"Fell back to {fallback_strat.label(cache_bits, fp16_layers)} on the standard binary."
                        )
                break
            except RuntimeError as exc:
                last_error = str(exc)
                if not is_fallback:
                    failed_strategy_name = strategy.name
                self._cleanup_process()
        else:
            raise RuntimeError(last_error or "llama.cpp server failed to start.")

        strat = _strategy_registry.get(actual_strategy) or _strategy_registry.default()
        actual_cache_bits = cache_bits if actual_strategy != "native" else 0
        # The current llama.cpp / llama-server cache-type interface only exposes a
        # uniform KV dtype per cache (for example ``turbo3`` or ``q4_0``).  It does
        # not accept the mixed-precision ``fp16Layers`` split used by other backends.
        actual_fp16_layers = 0
        if runtime_note is None:
            runtime_note = (
                f"GGUF generation is running through the local llama.cpp server with "
                f"{strat.label(actual_cache_bits, actual_fp16_layers)} cache."
            )
        if actual_strategy != "native" and fp16_layers > 0:
            runtime_note = _append_runtime_note(
                runtime_note,
                "llama.cpp currently ignores the FP16 layers setting for compressed KV cache types.",
            )

        self.loaded_model = LoadedModelInfo(
            ref=model_ref,
            name=model_name,
            canonicalRepo=canonical_repo,
            backend=backend,
            source=source,
            engine=self.engine_name,
            cacheStrategy=actual_strategy,
            cacheBits=actual_cache_bits,
            fp16Layers=actual_fp16_layers,
            fusedAttention=fused_attention,
            fitModelInMemory=actual_fit,
            contextTokens=context_tokens,
            loadedAt=_now_label(),
            path=path,
            runtimeTarget=runtime_target or path,
            runtimeNote=runtime_note,
            visionEnabled=attempt_mmproj_path is not None,
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
            raise RuntimeError(logs or "The llama.cpp server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
        # Build user message with optional images
        if images:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        # Phase 3.8 follow-up: apply known chat-template auto-fixes
        # before the messages reach llama-server (e.g. Gemma family
        # rejects the system role outright).
        messages, template_fix_note = _apply_llama_chat_template_fixes(messages, self.loaded_model)

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
        _apply_sampler_kwargs(
            payload,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )
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
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        text = _strip_thinking_tokens(str(message.get("content") or ""))

        return GenerationResult(
            text=text,
            finishReason=str(choice.get("finish_reason") or "stop"),
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=total_tokens,
            tokS=round(completion_tokens / elapsed, 1) if completion_tokens else 0.0,
            responseSeconds=round(elapsed, 2),
            runtimeNote=(
                _append_runtime_note(self.loaded_model.runtimeNote, template_fix_note)
                if template_fix_note
                else self.loaded_model.runtimeNote
            ),
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
            raise RuntimeError(logs or "The llama.cpp server is not running.")

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for message in history:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
        if images:
            content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        # Phase 3.8 follow-up: chat-template auto-fix on the streaming
        # path matches the non-stream behaviour. The note is forwarded
        # via the final StreamChunk's runtime_note.
        messages, template_fix_note = _apply_llama_chat_template_fixes(messages, self.loaded_model)

        payload: dict[str, Any] = {
            "model": self.loaded_model.ref,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        _apply_sampler_kwargs(
            payload,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
        )
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
        stream_start = time.perf_counter()
        first_token_time: float | None = None
        runtime_note = self.loaded_model.runtimeNote
        if template_fix_note:
            runtime_note = _append_runtime_note(runtime_note, template_fix_note)
        think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode or "off") != "off")
        runaway_guard = RepeatedLineGuard()
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[len("data: "):]
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                # Phase 3.3: extract per-token logprobs when llama-server
                # returns them. The `logprobs.content` field is a list of
                # token entries with top_logprobs alternatives.
                logprob_entries: list[dict[str, Any]] | None = None
                logprobs_payload = choice.get("logprobs") or {}
                if isinstance(logprobs_payload, dict):
                    raw_entries = logprobs_payload.get("content")
                    if isinstance(raw_entries, list) and raw_entries:
                        logprob_entries = []
                        for entry in raw_entries:
                            if not isinstance(entry, dict):
                                continue
                            top = entry.get("top_logprobs") or []
                            logprob_entries.append({
                                "token": entry.get("token"),
                                "logprob": entry.get("logprob"),
                                "alternatives": [
                                    {"token": alt.get("token"), "logprob": alt.get("logprob")}
                                    for alt in top
                                    if isinstance(alt, dict)
                                ],
                            })
                if content:
                    split = think_filter.feed(str(content))
                    if split.reasoning:
                        yield StreamChunk(reasoning=split.reasoning)
                    if split.reasoning_done:
                        yield StreamChunk(reasoning_done=True)
                    if split.text:
                        runaway_guard.feed(split.text)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        completion_tokens += 1
                        yield StreamChunk(text=split.text, token_logprobs=logprob_entries)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            flushed = think_filter.flush()
            if flushed.reasoning:
                yield StreamChunk(reasoning=flushed.reasoning)
            if flushed.reasoning_done:
                yield StreamChunk(reasoning_done=True)
            if flushed.text:
                runaway_guard.feed(flushed.text)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                yield StreamChunk(text=flushed.text)
            runaway_guard.flush()
        except RuntimeError as exc:
            runtime_note = _append_runtime_note(runtime_note, str(exc))
            finish_reason = "stop"
        finally:
            resp.close()

        # Measure generation speed from first token to completion
        end_time = time.perf_counter()
        gen_elapsed = max(end_time - (first_token_time or stream_start), 1e-6)
        tok_s = round(completion_tokens / gen_elapsed, 1) if completion_tokens > 0 else 0.0

        yield StreamChunk(
            done=True,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=tok_s,
            runtime_note=runtime_note,
        )
