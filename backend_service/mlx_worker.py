from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from backend_service.reasoning_split import (
    RAW_REASONING_HEADING_RE,
    ThinkingTokenFilter,
    ThinkingStreamResult,
    strip_thinking_tokens as _strip_thinking_tokens,
)

_UNSUPPORTED_QUANT_ALGOS = {"NVFP4", "NVINT4"}


def _reject_unsupported_quant(model_path: str) -> None:
    """Raise early if the model uses a quantisation format MLX cannot handle."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        qcfg = cfg.get("quantization_config") or {}
        algo = qcfg.get("quant_algo", "")
        method = qcfg.get("quant_method", "")
        if algo in _UNSUPPORTED_QUANT_ALGOS:
            raise RuntimeError(
                f"This model uses {algo} quantisation (via {method}) which "
                f"is not supported by the MLX runtime. Try a GGUF or "
                f"standard MLX quantised version of this model instead."
            )
    except RuntimeError:
        raise
    except Exception:
        pass  # Don't block loading if config can't be parsed


def _normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content or "")


def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Ensure strict role alternation (user/assistant) after an optional system message.

    - Removes empty assistant messages.
    - Merges consecutive same-role messages with a newline separator.
    """
    sanitized: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "").strip()
        role = msg.get("role", "")
        # Drop empty assistant messages (from failed/mock responses)
        if role == "assistant" and not content:
            continue
        # Merge consecutive same-role messages
        if sanitized and sanitized[-1]["role"] == role and role != "system":
            sanitized[-1]["content"] += "\n" + content
        else:
            sanitized.append({"role": role, "content": content})
    return sanitized

_TRANSCRIPT_ROLE_LINE_RE = re.compile(r"^\s*(SYSTEM|USER|ASSISTANT):\s*(.*)$", re.IGNORECASE)

# Phase 2.0.5-F: RunawayGuard now lives in `backend_service.runaway_guard`
# so the llama.cpp stream loop in `state.py` can use the same detector. Re-
# export the symbol here so existing callers / tests keep working without
# import-path churn.
from backend_service.runaway_guard import RunawayGuard  # noqa: E402,F401


def _build_mlx_sampler(request: dict[str, Any]) -> Any:
    """Phase 2.2: build an mlx-lm sampler with whichever Phase 2.2 sampler
    overrides the installed `make_sampler` actually supports.

    `mlx_lm.sample_utils.make_sampler` has gained kwargs across versions
    (top_p, top_k, min_p, ...). Call sites used to pass `temp` only — we
    now collect the request's `samplers` block and forward whatever
    survives a signature filter, so newer mlx-lm builds get the full
    sampler chain while older builds fall back gracefully.
    """
    import inspect

    from mlx_lm.sample_utils import make_sampler

    kwargs: dict[str, Any] = {"temp": float(request.get("temperature") or 0.0)}
    samplers = request.get("samplers") or {}
    if isinstance(samplers, dict):
        for src in ("top_p", "top_k", "min_p"):
            value = samplers.get(src)
            if value is not None:
                kwargs[src] = value

    try:
        sig = inspect.signature(make_sampler)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        filtered = {"temp": kwargs["temp"]}
    return make_sampler(**filtered)


def _format_tools_for_prompt(tools: list[dict[str, Any]] | None) -> str | None:
    """Format tool schemas into a system prompt block for open-source models.

    Since MLX models don't have a native function-calling API, we inject
    tool descriptions into the system prompt so the model knows what tools
    are available and how to call them.
    """
    if not tools:
        return None

    lines = [
        "You have access to the following tools. To use a tool, respond with a JSON block wrapped in <tool_call> tags.",
        "Example: <tool_call>{\"name\": \"calculator\", \"arguments\": {\"expression\": \"2+2\"}}</tool_call>",
        "",
        "Available tools:",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_parts = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else ""
            param_parts.append(f"    - {pname}: {ptype}{req} — {pdesc}")

        lines.append(f"\n- {name}: {desc}")
        if param_parts:
            lines.append("  Parameters:")
            lines.extend(param_parts)

    lines.append("")
    lines.append("If you don't need a tool, just respond normally without <tool_call> tags.")
    return "\n".join(lines)


class TranscriptLoopFilter:
    """Suppress plain transcript continuations like USER:/ASSISTANT: loops.

    This is only used when we had to fall back to a raw text chat prompt because
    the tokenizer had no usable chat template. In that mode, some models keep
    continuing the transcript forever instead of answering once.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._stopped = False
        self._at_start = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def feed(self, text: str) -> str:
        if self._stopped:
            return ""
        self._buffer += text
        output_parts: list[str] = []
        while "\n" in self._buffer and not self._stopped:
            line, self._buffer = self._buffer.split("\n", 1)
            processed = self._process_line(line)
            if processed:
                output_parts.append(processed + "\n")
        return "".join(output_parts)

    def flush(self) -> str:
        if self._stopped:
            self._buffer = ""
            return ""
        if not self._buffer:
            return ""
        remaining = self._process_line(self._buffer)
        self._buffer = ""
        return remaining

    def _process_line(self, line: str) -> str:
        if self._stopped:
            return ""
        match = _TRANSCRIPT_ROLE_LINE_RE.match(line)
        if match:
            role = match.group(1).upper()
            content = match.group(2)
            if role == "ASSISTANT" and self._at_start:
                self._at_start = False
                return content
            self._stopped = True
            return ""
        if line.strip():
            self._at_start = False
        return line


def _plain_chat_fallback_active(runtime_note: str | None) -> bool:
    return bool(runtime_note and "plain chat fallback prompt" in runtime_note.lower())


def _trim_transcript_continuation(text: str) -> tuple[str, bool]:
    filter_ = TranscriptLoopFilter()
    emitted = filter_.feed(text)
    emitted += filter_.flush()
    return emitted.strip(), filter_.stopped


def _fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def _merge_runtime_notes(*notes: str | None) -> str | None:
    merged = " ".join(note.strip() for note in notes if note and note.strip())
    return merged or None


def _should_retry_cache_failure(exc: BaseException) -> bool:
    detail = str(exc).lower()
    return (
        "broadcast" in detail
        or "shape" in detail
        or "create_attention_mask" in detail
        or "swapaxes" in detail
    )


def _build_prompt_text(
    tokenizer: Any,
    history: list[dict[str, Any]],
    prompt: str,
    system_prompt: str | None,
    model_ref: str | None = None,
) -> tuple[str, str | None]:
    # Phase 3.8: detect chat-template quirks at render time and apply
    # the matching auto-fix. Today: Gemma family rejects the system role
    # entirely, so we fold the system prompt into the first user message
    # before handing off to apply_chat_template. The report's
    # `to_runtime_note()` surfaces the fix to the UI's substrate badge.
    from backend_service.helpers.chat_template import (
        fold_system_into_first_user,
        inspect_chat_template,
        is_gemma_family,
    )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in history:
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append({"role": role, "content": _normalize_message_content(message.get("text", ""))})
    messages.append({"role": "user", "content": prompt})
    messages = _sanitize_messages(messages)

    template_note: str | None = None
    if is_gemma_family(model_ref):
        messages = fold_system_into_first_user(messages)
        report = inspect_chat_template(getattr(tokenizer, "chat_template", None), model_ref)
        template_note = report.to_runtime_note()

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
            if isinstance(rendered, str):
                return rendered, template_note
        except TypeError:
            try:
                rendered = apply_template(messages, add_generation_prompt=True)
                if isinstance(rendered, str):
                    return rendered, template_note
                if isinstance(rendered, list):
                    return tokenizer.decode(rendered), template_note
            except Exception as exc:  # pragma: no cover - exercised via fallback path below
                reason = str(exc).strip() or exc.__class__.__name__
                return (
                    _fallback_chat_prompt(messages),
                    f"Tokenizer chat template was unavailable, so MLX used a plain chat fallback prompt. ({reason})",
                )
        except Exception as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            return (
                _fallback_chat_prompt(messages),
                f"Tokenizer chat template was unavailable, so MLX used a plain chat fallback prompt. ({reason})",
            )

    return (
        _fallback_chat_prompt(messages),
        "Tokenizer chat template was unavailable, so MLX used a plain chat fallback prompt.",
    )


# Dedicated JSON sink for the protocol channel. When the worker runs as a
# subprocess (the ``serve`` / ``probe`` / ``gguf-metadata`` entrypoints),
# ``_install_stdio_redirect`` replaces this with a private file object
# pointing at the original stdout FD, then redirects FD 1 to stderr at the
# OS level. That way ``mlx-lm``'s print-to-stdout warnings (e.g. the
# "Generating with a model that requires 48128 MB which is close to the
# maximum recommended size of 53084 MB" chatter on large models) land on
# stderr instead of corrupting the JSON stream the parent reads.
#
# Default value keeps in-process tests working: they patch ``_emit``
# directly and never go through ``main()``.
_JSON_OUT: io.TextIOBase = sys.stdout  # type: ignore[assignment]


def _install_stdio_redirect() -> None:
    """Split the JSON protocol channel from warning chatter.

    The JSON protocol uses stdout (file descriptor 1). ``mlx-lm`` and some
    diffusers/torch paths print warnings and progress to stdout as well —
    without isolation, a single ``[WARNING] Generating with a model that
    requires ...`` line crashes the caller's ``json.loads`` and the user
    sees "MLX worker returned invalid JSON".

    Duplicate the original stdout FD into a fresh file object reserved for
    protocol output, then point FD 1 at stderr so anything writing through
    the normal stdout path (Python ``print()``, C-extension writes, tqdm
    auto-detecting stdout) lands on stderr instead. Finally rebind
    ``sys.stdout`` to ``sys.stderr`` so libraries that cached a reference
    at import time follow along.
    """
    global _JSON_OUT
    json_fd = os.dup(1)
    os.dup2(2, 1)
    _JSON_OUT = os.fdopen(json_fd, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr


def _emit(payload: dict[str, Any]) -> None:
    _JSON_OUT.write(json.dumps(payload) + "\n")
    _JSON_OUT.flush()


def emit_progress(phase: str, percent: float | None, message: str | None = None) -> None:
    try:
        _emit(
            {
                "ok": True,
                "progress": {
                    "phase": phase,
                    "percent": percent,
                    "message": message,
                },
            }
        )
    except Exception:
        pass


def probe() -> int:
    mlx_available = importlib.util.find_spec("mlx") is not None
    mlx_lm_available = importlib.util.find_spec("mlx_lm") is not None
    payload: dict[str, Any] = {
        "mlxAvailable": mlx_available,
        "mlxLmAvailable": mlx_lm_available,
        "mlxUsable": False,
        "mlxVersion": None,
        "mlxLmVersion": None,
        "message": None,
    }

    if not (mlx_available and mlx_lm_available):
        _emit(payload)
        return 0

    try:
        import mlx.core as mx
        import mlx_lm

        payload["mlxUsable"] = True
        payload["mlxVersion"] = getattr(mx, "__version__", None)
        payload["mlxLmVersion"] = getattr(mlx_lm, "__version__", None)
        try:
            payload["deviceInfo"] = mx.device_info()
        except Exception:
            payload["deviceInfo"] = None
        _emit(payload)
        return 0
    except Exception as exc:
        payload["message"] = str(exc)
        _emit(payload)
        return 1


def gguf_metadata(path: str) -> int:
    try:
        from gguf import GGUFReader
    except Exception as exc:
        _emit({"error": str(exc)})
        return 1

    try:
        reader = GGUFReader(path, "r")
        base_model_repos: list[str] = []
        for key, field in reader.fields.items():
            if key.startswith("general.base_model.") and key.endswith(".repo_url"):
                value = field.contents()
                if isinstance(value, str):
                    base_model_repos.append(value)

        def normalize_repo(value: str | None) -> str | None:
            if not value:
                return None
            if value.startswith("https://huggingface.co/"):
                return value.removeprefix("https://huggingface.co/").strip("/")
            return value

        payload = {
            "path": str(Path(path).resolve()),
            "name": reader.get_field("general.name").contents() if reader.get_field("general.name") else Path(path).stem,
            "architecture": reader.get_field("general.architecture").contents() if reader.get_field("general.architecture") else None,
            "tokenizerModel": reader.get_field("tokenizer.ggml.model").contents() if reader.get_field("tokenizer.ggml.model") else None,
            "baseModelRepos": [normalize_repo(item) for item in base_model_repos if normalize_repo(item)],
            "baseModelRepo": normalize_repo(base_model_repos[0]) if base_model_repos else None,
        }
        _emit(payload)
        return 0
    except Exception as exc:
        _emit({"error": str(exc)})
        return 1


class WorkerState:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.config: dict[str, Any] | None = None
        self.cache_strategy = "native"
        self.cache_bits = 0
        self.fp16_layers = 0
        self.fused_attention = False
        self.context_tokens = 8192
        self.speculative_decoding = False
        self._dflash_generator = None  # Loaded DFlash draft model when active
        self._dflash_target = None     # Target model prepared by dflash_mlx.runtime
        self.tree_budget = 0
        self._ddtree_draft = None     # DFlashDraftModel for DDTree
        self._ddtree_target = None    # target model loaded via dflash_mlx for DDTree

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        op = request.get("op")
        if op == "load_model":
            return self.load_model(request)
        if op == "update_profile":
            return self.update_profile(request)
        if op == "unload_model":
            return self.unload_model()
        if op == "generate":
            return self.generate(request)
        if op == "stream_generate":
            self.stream_generate(request)
            return None
        if op == "eval_perplexity":
            return self.eval_perplexity(request)
        if op == "eval_task_accuracy":
            return self.eval_task_accuracy(request)
        raise ValueError(f"Unsupported worker operation: {op}")

    def load_model(self, request: dict[str, Any]) -> dict[str, Any]:
        from mlx_lm import load

        target = str(request["target"])
        requested_cache_strategy = str(request.get("cacheStrategy", "native"))
        requested_cache_bits = int(request.get("cacheBits", 0))
        requested_fp16_layers = int(request.get("fp16Layers", 0))
        requested_fused_attention = bool(request.get("fusedAttention", False))
        self.context_tokens = int(request.get("contextTokens", 8192))
        self.speculative_decoding = bool(request.get("speculativeDecoding", False))
        dflash_draft_model = request.get("dflashDraftModel")
        self._dflash_generator = None
        self._dflash_target = None
        self._ddtree_draft = None
        self._ddtree_target = None
        self.tree_budget = 0

        emit_progress("resolving", 5.0, f"Resolving model target: {target}")

        # Pre-resolve the snapshot so we can stream download progress. Skip if
        # `target` is already a local path, and fall back to letting mlx_lm.load
        # handle non-HF targets on any failure.
        local_path = target
        is_local = False
        try:
            candidate = Path(target).expanduser()
            if target.startswith("/") or target.startswith("~") or candidate.exists():
                is_local = True
                local_path = str(candidate)
        except Exception:
            is_local = False

        if not is_local:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
                from huggingface_hub.utils import (  # type: ignore
                    GatedRepoError,
                    RepositoryNotFoundError,
                    HfHubHTTPError,
                )
                from tqdm import tqdm  # type: ignore
            except ImportError:
                # huggingface_hub / tqdm not installed — let mlx_lm.load
                # handle resolution itself. Matches pre-progress behaviour.
                local_path = target
            else:
                class ProgressTqdm(tqdm):  # type: ignore[misc]
                    def update(self, n: int = 1):  # type: ignore[override]
                        result = super().update(n)
                        try:
                            total = float(self.total or 0)
                            done = float(self.n or 0)
                            if total > 0:
                                frac = max(0.0, min(1.0, done / total))
                                pct = 20.0 + frac * 40.0  # 20% -> 60%
                                done_mb = int(done // (1024 * 1024))
                                total_mb = int(total // (1024 * 1024))
                                emit_progress(
                                    "downloading",
                                    pct,
                                    f"{done_mb} / {total_mb} MB",
                                )
                            else:
                                emit_progress("downloading", 20.0, "Fetching weights")
                        except Exception:
                            pass
                        return result

                emit_progress("downloading", 20.0, "Fetching weights from Hugging Face")
                try:
                    # Use max_workers=1 to avoid multiprocessing semaphore
                    # leaks on macOS that crash the worker subprocess.
                    local_path = snapshot_download(
                        repo_id=target,
                        tqdm_class=ProgressTqdm,
                        max_workers=1,
                    )
                except GatedRepoError as exc:
                    raise RuntimeError(
                        f"This model is gated on Hugging Face. Accept the licence "
                        f"at https://huggingface.co/{target} and set HF_TOKEN in "
                        f"Settings, then retry."
                    ) from exc
                except RepositoryNotFoundError as exc:
                    raise RuntimeError(
                        f"Hugging Face repository not found: {target}"
                    ) from exc
                except HfHubHTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status in (401, 403):
                        raise RuntimeError(
                            f"Hugging Face refused access to {target} (HTTP {status}). "
                            f"Set HF_TOKEN in Settings and make sure you have accepted "
                            f"the licence at https://huggingface.co/{target}."
                        ) from exc
                    raise RuntimeError(
                        f"Hugging Face download failed for {target}: {exc}"
                    ) from exc
                except OSError as exc:
                    # Network / filesystem failures — bubble up the detail.
                    raise RuntimeError(
                        f"Could not download {target} from Hugging Face: {exc}"
                    ) from exc

        # Start a heartbeat that ticks the UI every 2s while mlx_lm.load
        # blocks. mlx_lm doesn't expose a progress callback, so large models
        # (20B+) would otherwise sit at a frozen 60% for 1-2 minutes.
        import threading
        load_done = threading.Event()
        load_start = time.monotonic()
        emit_progress("loading", 60.0, "Loading weights into MLX")

        def _heartbeat() -> None:
            tick = 0
            while not load_done.wait(2.0):
                tick += 1
                elapsed = int(time.monotonic() - load_start)
                # Creep the percent very slowly from 60 → 90 so the bar feels
                # alive without overstating progress we can't measure.
                pct = min(90.0, 60.0 + tick * 1.2)
                emit_progress(
                    "loading",
                    pct,
                    f"Loading weights into MLX... ({elapsed}s)",
                )

        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            # Reject quantisation formats that MLX cannot dequantize.
            _reject_unsupported_quant(local_path)
            self.model, self.tokenizer, self.config = load(local_path, return_config=True)
        finally:
            load_done.set()
            heartbeat_thread.join(timeout=0.5)
        emit_progress("ready", 95.0, "Finalising")

        # Initialise DFLASH speculative decoding if requested
        dflash_note = None
        self.tree_budget = int(request.get("treeBudget") or 0)
        if self.speculative_decoding and dflash_draft_model:
            try:
                from dflash_mlx.runtime import configure_full_attention_split, load_draft_bundle
                emit_progress("dflash", 96.0, f"Loading DFLASH draft model: {dflash_draft_model}")
                # Reuse the already loaded MLX target model. Loading a second
                # target bundle can duplicate the full model footprint and
                # trigger SIGKILL on large models during DFLASH startup.
                self._dflash_target = self.model
                configure_full_attention_split(self._dflash_target, enabled=True)
                self._dflash_generator, _ = load_draft_bundle(dflash_draft_model, lazy=True)
                dflash_note = f"DFLASH speculative decoding active (draft: {dflash_draft_model})."
            except ImportError as exc:
                dflash_note = f"dflash-mlx could not be imported ({exc}). Falling back to standard generation."
                self.speculative_decoding = False
            except Exception as exc:
                dflash_note = f"DFLASH initialisation failed: {exc}. Falling back to standard generation."
                self.speculative_decoding = False

            # Load DDTree components when tree budget is set
            if self.speculative_decoding and self.tree_budget > 0:
                try:
                    emit_progress("ddtree", 97.0, "Preparing DDTree runtime")
                    self._ddtree_target = self._dflash_target
                    self._ddtree_draft = self._dflash_generator
                    dflash_note = f"DDTree speculative decoding active (budget={self.tree_budget}, draft: {dflash_draft_model})."
                except Exception as exc:
                    dflash_note = f"DDTree init failed ({exc}). Using linear DFLASH."
                    self.tree_budget = 0
                    self._ddtree_draft = None
                    self._ddtree_target = None

        profile_note = self._apply_cache_profile(
            cache_strategy=requested_cache_strategy,
            cache_bits=requested_cache_bits,
            fp16_layers=requested_fp16_layers,
            fused_attention=requested_fused_attention,
        )

        return {
            "resolvedTarget": target,
            "layerCount": len(getattr(self.model, "layers", [])),
            "config": {
                "numHiddenLayers": (self.config or {}).get("num_hidden_layers"),
                "numAttentionHeads": (self.config or {}).get("num_attention_heads"),
                "hiddenSize": (self.config or {}).get("hidden_size"),
            },
            "cacheStrategy": self.cache_strategy,
            "cacheBits": self.cache_bits,
            "fp16Layers": self.fp16_layers,
            "fusedAttention": self.fused_attention,
            "speculativeDecoding": bool(self.speculative_decoding and self._dflash_generator is not None),
            "dflashDraftModel": (
                str(dflash_draft_model)
                if self.speculative_decoding and self._dflash_generator is not None and dflash_draft_model
                else None
            ),
            "treeBudget": self.tree_budget if self.speculative_decoding and self._dflash_generator is not None else 0,
            "note": _merge_runtime_notes(profile_note, dflash_note),
        }

    def unload_model(self) -> dict[str, Any]:
        self.model = None
        self.tokenizer = None
        self._dflash_generator = None
        self._dflash_target = None
        self._ddtree_draft = None
        self._ddtree_target = None
        self.speculative_decoding = False
        self.tree_budget = 0
        self.config = None
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        return {"unloaded": True}

    def update_profile(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")
        note = self._apply_cache_profile(
            cache_strategy=str(request.get("cacheStrategy", self.cache_strategy)),
            cache_bits=int(request.get("cacheBits", self.cache_bits)),
            fp16_layers=int(request.get("fp16Layers", self.fp16_layers)),
            fused_attention=bool(request.get("fusedAttention", self.fused_attention)),
        )
        return {
            "cacheStrategy": self.cache_strategy,
            "cacheBits": self.cache_bits,
            "fp16Layers": self.fp16_layers,
            "fusedAttention": self.fused_attention,
            "note": note,
        }

    def _apply_cache_profile(
        self,
        *,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> str | None:
        self.cache_strategy = cache_strategy
        self.cache_bits = cache_bits
        self.fp16_layers = fp16_layers
        self.fused_attention = fused_attention

        if self.cache_strategy == "native":
            self.cache_bits = 0
            self.fp16_layers = 0
            return None

        preview_cache, note = self._make_cache()
        if preview_cache is not None:
            preview_cache = None
            import gc
            gc.collect()

        if note:
            self.cache_strategy = "native"
            self.cache_bits = 0
            self.fp16_layers = 0

        return note

    def _runtime_fields(
        self,
        *,
        prompt_cache: Any | None,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
    ) -> dict[str, Any]:
        cache_strategy = self.cache_strategy
        cache_bits = self.cache_bits
        fp16_layers = self.fp16_layers
        if prompt_cache is None or cache_strategy == "native":
            cache_strategy = "native"
            cache_bits = 0
            fp16_layers = 0
        actual_speculative = bool(speculative_decoding)
        return {
            "cacheStrategy": cache_strategy,
            "cacheBits": int(cache_bits),
            "fp16Layers": int(fp16_layers),
            "speculativeDecoding": actual_speculative,
            "treeBudget": int(tree_budget or 0) if actual_speculative else 0,
        }

    def _make_cache(self) -> tuple[Any | None, str | None]:
        """Build the prompt cache for the active strategy. Returns (cache, note)."""
        from cache_compression import registry
        strategy = registry.get(self.cache_strategy)
        if strategy is None or self.cache_strategy == "native":
            return None, None
        try:
            cache = strategy.make_mlx_cache(
                len(getattr(self.model, "layers", [])),
                bits=self.cache_bits,
                fp16_layers=self.fp16_layers,
                fused=self.fused_attention,
                model=self.model,
            )
            return cache, None
        except (ValueError, NotImplementedError) as exc:
            return None, (
                f"Cache strategy '{strategy.name}' is unavailable for this MLX architecture, "
                f"so generation fell back to native f16 cache. ({exc})"
            )

    def _generate_dflash(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate using DFLASH speculative decoding."""
        from dflash_mlx.runtime import stream_dflash_generate

        # Build prompt text
        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )

        prompt_tokens = self.tokenizer.encode(prompt_text)
        eos_token_ids = list(getattr(self.tokenizer, "eos_token_ids", None) or [])
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and int(eos_token_id) not in eos_token_ids:
            eos_token_ids.append(int(eos_token_id))

        # ``stream_dflash_generate`` (upstream v0.1.4) yields per-token events
        # followed by a final ``{"event": "summary", ...}`` payload whose shape
        # matches what the old ``generate_dflash_once`` helper returned.
        summary: dict[str, Any] = {}
        # Phase 3.1: per-token accepted-from-draft tracking. Tokens that
        # share `cycles_completed` with the previous token are commits
        # from the same DDTree cycle — the first is verifier-decoded,
        # the rest are draft-accepted. Build a parallel list of
        # (token_text, accepted: bool) so the UI can tint accepted runs.
        per_token_accepted: list[bool] = []
        per_token_text: list[str] = []
        prev_cycle: int = -1
        prev_gen_count: int = 0
        for event in stream_dflash_generate(
            target_model=self._dflash_target or self.model,
            tokenizer=self.tokenizer,
            draft_model=self._dflash_generator,
            prompt=prompt_text,
            max_new_tokens=int(request.get("maxTokens") or 256),
            use_chat_template=False,
            stop_token_ids=eos_token_ids,
            prompt_tokens_override=prompt_tokens,
        ):
            if event.get("event") == "summary":
                summary = dict(event)
                continue
            if event.get("event") != "token":
                continue
            cycle = int(event.get("cycles_completed") or 0)
            gen_count = int(event.get("generated_tokens") or 0)
            token_id = event.get("token_id")
            if token_id is None:
                continue
            # First token of a new cycle (cycle increments) is
            # verifier-decoded; subsequent tokens within the same
            # cycle are draft-accepted. Cycle 0 (the initial seed
            # token) is also verifier-decoded.
            if gen_count <= prev_gen_count:
                # Defensive — skip duplicates / out-of-order events.
                continue
            accepted = cycle == prev_cycle and prev_cycle > 0
            per_token_accepted.append(accepted)
            try:
                per_token_text.append(self.tokenizer.decode([int(token_id)]))
            except Exception:
                per_token_text.append("")
            prev_cycle = cycle
            prev_gen_count = gen_count

        gen_tokens = [int(token_id) for token_id in summary.get("generated_token_ids", [])]
        text = self.tokenizer.decode(gen_tokens).strip() if gen_tokens else ""
        # Respect thinkingMode: only strip raw reasoning patterns when thinking
        # is enabled. XML <think> tags are always processed regardless.
        thinking_mode = request.get("thinkingMode") or "off"
        if text:
            think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode != "off"))
            result = think_filter.feed(text)
            flushed = think_filter.flush()
            text = f"{result.text}{flushed.text}".strip()
        if not text:
            text = "Generation completed without decoded text."

        output_tokens = int(summary.get("generation_tokens") or len(gen_tokens))
        prompt_token_count = int(summary.get("prompt_token_count") or len(prompt_tokens))
        elapsed = max(float(summary.get("elapsed_us") or 0.0) / 1e6, 1e-6)
        phase_timings = dict(summary.get("phase_timings_us") or {})
        prefill_elapsed = max(0.0, float(phase_timings.get("prefill") or 0.0) / 1e6)
        generation_elapsed = max(elapsed - prefill_elapsed, 1e-6)
        tok_s = round(output_tokens / generation_elapsed, 1) if output_tokens else 0.0
        cycles_completed = int(summary.get("cycles_completed") or 0)
        accepted_from_draft = int(summary.get("accepted_from_draft") or 0)
        acceptance_rate = (
            accepted_from_draft / cycles_completed
            if cycles_completed > 0
            else None
        )

        runtime_note = _merge_runtime_notes(
            prompt_note,
            (
                f"DFLASH speculative decoding. Acceptance rate: {acceptance_rate:.1f} avg tokens."
                if acceptance_rate is not None
                else "DFLASH speculative decoding."
            ),
        )

        # Phase 3.1: build run-length-encoded accepted spans from the
        # per-token accepted bools. Each span has start (char offset
        # into the rendered text), length (chars), and accepted (bool).
        accepted_spans: list[dict[str, Any]] = []
        if per_token_accepted and per_token_text:
            offset = 0
            run_start = 0
            run_kind = per_token_accepted[0]
            for idx, accepted in enumerate(per_token_accepted):
                tok_text = per_token_text[idx] if idx < len(per_token_text) else ""
                if accepted != run_kind:
                    accepted_spans.append({
                        "start": run_start,
                        "length": offset - run_start,
                        "accepted": run_kind,
                    })
                    run_start = offset
                    run_kind = accepted
                offset += len(tok_text)
            accepted_spans.append({
                "start": run_start,
                "length": offset - run_start,
                "accepted": run_kind,
            })

        return {
            "text": text,
            "finishReason": "stop",
            "promptTokens": prompt_token_count,
            "completionTokens": output_tokens,
            "totalTokens": prompt_token_count + output_tokens,
            "tokS": tok_s,
            "promptTokS": 0.0,
            "peakMemoryGb": round(float(summary.get("peak_memory_gb") or 0.0), 3),
            "runtimeNote": runtime_note,
            "dflashAcceptanceRate": round(float(acceptance_rate), 2) if acceptance_rate is not None else None,
            "acceptedSpans": accepted_spans,
            "acceptedTokenText": "".join(per_token_text) if per_token_text else None,
            **self._runtime_fields(prompt_cache=None, speculative_decoding=True, tree_budget=0),
        }

    def _generate_ddtree(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate using DDTree tree-based speculative decoding."""
        from backend_service.ddtree import generate_ddtree_mlx

        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )

        # Tokenize prompt
        prompt_tokens = self.tokenizer.encode(prompt_text)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        stop_ids = [eos] if eos is not None else []

        result = generate_ddtree_mlx(
            target_model=self._ddtree_target,
            tokenizer=self.tokenizer,
            draft_model=self._ddtree_draft,
            prompt_tokens=prompt_tokens,
            max_new_tokens=int(request.get("maxTokens") or 256),
            tree_budget=self.tree_budget,
            stop_token_ids=stop_ids,
        )

        # Decode output tokens
        gen_tokens = result["generated_tokens"]
        text = self.tokenizer.decode(gen_tokens).strip()
        # Respect thinkingMode: only strip raw reasoning patterns when thinking
        # is enabled. XML <think> tags are always processed regardless.
        thinking_mode = request.get("thinkingMode") or "off"
        if text:
            think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode != "off"))
            filter_result = think_filter.feed(text)
            flushed = think_filter.flush()
            text = f"{filter_result.text}{flushed.text}".strip()
        if not text:
            text = "Generation completed without decoded text."

        output_tokens = result["output_tokens"]
        elapsed = result["elapsed_seconds"]
        tok_s = round(output_tokens / max(elapsed, 1e-6), 1)
        acceptance_rate = result["avg_acceptance_length"]

        runtime_note = _merge_runtime_notes(
            prompt_note,
            f"DDTree speculative decoding (budget={result['tree_budget']}). Acceptance rate: {acceptance_rate:.1f} avg tokens."
            if acceptance_rate else f"DDTree speculative decoding (budget={result['tree_budget']}).",
        )

        return {
            "text": text,
            "finishReason": "stop",
            "promptTokens": len(prompt_tokens),
            "completionTokens": output_tokens,
            "totalTokens": len(prompt_tokens) + output_tokens,
            "tokS": tok_s,
            "promptTokS": 0.0,
            "peakMemoryGb": 0.0,
            "runtimeNote": runtime_note,
            "dflashAcceptanceRate": round(float(acceptance_rate), 2) if acceptance_rate else None,
            # Phase 3.1 follow-up: DDTree path now ships accepted-span
            # data alongside the linear DFLASH path so the frontend
            # AcceptedTokenOverlay tints draft-accepted ranges for
            # both speculative-decode strategies.
            "acceptedSpans": result.get("accepted_spans") or [],
            "acceptedTokenText": result.get("accepted_token_text"),
            **self._runtime_fields(
                prompt_cache=None,
                speculative_decoding=True,
                tree_budget=result["tree_budget"],
            ),
        }

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        # Use DDTree if tree budget is set and components are loaded
        if self.speculative_decoding and self.tree_budget > 0 and self._ddtree_draft is not None:
            try:
                return self._generate_ddtree(request)
            except Exception as exc:
                runtime_fallback_note = f"DDTree generation failed ({exc}). Falling back to linear DFLASH."
                # Fall through to linear DFLASH below

        # Use DFLASH if active
        if self.speculative_decoding and self._dflash_generator is not None:
            try:
                return self._generate_dflash(request)
            except Exception as exc:
                # Fall back to standard generation on DFLASH failure
                runtime_fallback_note = f"DFLASH generation failed ({exc}). Fell back to standard generation."
                result = self._generate_standard(request)
                result["runtimeNote"] = _merge_runtime_notes(result.get("runtimeNote"), runtime_fallback_note)
                return result

        return self._generate_standard(request)

    def _generate_standard(self, request: dict[str, Any]) -> dict[str, Any]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Inject tool schemas into system prompt for open-source models
        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )
        sampler = _build_mlx_sampler(request)
        prompt_cache, runtime_note = self._make_cache()
        runtime_note = _merge_runtime_notes(runtime_note, prompt_note)
        runtime_fields = self._runtime_fields(prompt_cache=prompt_cache)
        transcript_fallback = _plain_chat_fallback_active(prompt_note)

        runaway_guard = RunawayGuard()
        runaway_stopped = False
        try:
            text_parts: list[str] = []
            last_response = None
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt_text,
                    max_tokens=int(request.get("maxTokens") or 256),
                    sampler=sampler,
                    prompt_cache=prompt_cache,
            ):
                if response.text:
                    text_parts.append(response.text)
                    try:
                        runaway_guard.feed(response.text)
                    except RuntimeError:
                        runaway_stopped = True
                        break
                last_response = response
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            _should_retry = (
                prompt_cache is not None
                and _should_retry_cache_failure(exc)
            )
            if _should_retry:
                # Cache strategy produced incompatible shapes or mask errors.
                # Retry with the model's default (native) cache.
                runtime_note = (
                    _merge_runtime_notes(
                        prompt_note,
                        f"Cache strategy failed ({exc}). Fell back to native f16 cache.",
                    )
                )
                runtime_fields = self._runtime_fields(prompt_cache=None)
                runaway_guard = RunawayGuard()
                runaway_stopped = False
                text_parts = []
                last_response = None
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt_text,
                    max_tokens=int(request.get("maxTokens") or 256),
                    sampler=sampler,
                    prompt_cache=None,
                ):
                    if response.text:
                        text_parts.append(response.text)
                    last_response = response
            else:
                raise

        if last_response is None:
            raise RuntimeError("MLX generation did not return a response.")

        if runaway_stopped:
            runtime_note = _merge_runtime_notes(
                runtime_note,
                "Stopped runaway generation: model was repeating itself.",
            )

        raw_text = "".join(text_parts).strip()
        # Respect thinkingMode: only strip raw reasoning when thinking is on.
        thinking_mode = request.get("thinkingMode") or "off"
        think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode != "off"))
        filter_result = think_filter.feed(raw_text)
        flushed = think_filter.flush()
        text = f"{filter_result.text}{flushed.text}".strip()
        if transcript_fallback:
            text, transcript_trimmed = _trim_transcript_continuation(text)
            if transcript_trimmed:
                runtime_note = _merge_runtime_notes(
                    runtime_note,
                    "Suppressed a plain-chat transcript continuation to stop a runaway loop.",
                )
        if not text:
            text = "Generation completed without decoded text."

        return {
            "text": text,
            "finishReason": last_response.finish_reason or "stop",
            "promptTokens": int(last_response.prompt_tokens),
            "completionTokens": int(last_response.generation_tokens),
            "totalTokens": int(last_response.prompt_tokens + last_response.generation_tokens),
            "tokS": round(float(last_response.generation_tps), 1),
            "promptTokS": round(float(last_response.prompt_tps), 1),
            "peakMemoryGb": round(float(last_response.peak_memory), 3),
            "runtimeNote": runtime_note,
            **runtime_fields,
        }


    def stream_generate(self, request: dict[str, Any]) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        speculative_stream_fallback_note = None
        # DFLASH/DDTree don't support token-level streaming natively, so
        # emit the full result as a single chunk in the streaming protocol.
        # Prefer DDTree (tree-based) when tree_budget > 0, else linear DFlash.
        if self.speculative_decoding and self.tree_budget > 0 and self._ddtree_draft is not None:
            try:
                result = self._generate_ddtree(request)
                if result.get("text"):
                    _emit({"ok": True, "chunk": {"text": result["text"]}})
                _emit({
                    "ok": True,
                    "done": True,
                    "result": {
                        "finishReason": result.get("finishReason", "stop"),
                        "promptTokens": result.get("promptTokens", 0),
                        "completionTokens": result.get("completionTokens", 0),
                        "totalTokens": result.get("totalTokens", 0),
                        "tokS": result.get("tokS", 0.0),
                        "promptTokS": result.get("promptTokS", 0.0),
                        "peakMemoryGb": result.get("peakMemoryGb", 0.0),
                        "runtimeNote": result.get("runtimeNote"),
                        "dflashAcceptanceRate": result.get("dflashAcceptanceRate"),
                        "cacheStrategy": result.get("cacheStrategy"),
                        "cacheBits": result.get("cacheBits"),
                        "fp16Layers": result.get("fp16Layers"),
                        "speculativeDecoding": result.get("speculativeDecoding"),
                        "treeBudget": result.get("treeBudget"),
                    },
                })
                return
            except Exception as exc:
                speculative_stream_fallback_note = (
                    f"DDTree stream path failed ({exc}). "
                    "Falling back to linear DFLASH."
                )
                # Fall through to linear DFLASH below

        if self.speculative_decoding and self._dflash_generator is not None:
            try:
                result = self._generate_dflash(request)
                if result.get("text"):
                    _emit({"ok": True, "chunk": {"text": result["text"]}})
                _emit({
                    "ok": True,
                    "done": True,
                    "result": {
                        "finishReason": result.get("finishReason", "stop"),
                        "promptTokens": result.get("promptTokens", 0),
                        "completionTokens": result.get("completionTokens", 0),
                        "totalTokens": result.get("totalTokens", 0),
                        "tokS": result.get("tokS", 0.0),
                        "promptTokS": result.get("promptTokS", 0.0),
                        "peakMemoryGb": result.get("peakMemoryGb", 0.0),
                        "runtimeNote": result.get("runtimeNote"),
                        "dflashAcceptanceRate": result.get("dflashAcceptanceRate"),
                        "cacheStrategy": result.get("cacheStrategy"),
                        "cacheBits": result.get("cacheBits"),
                        "fp16Layers": result.get("fp16Layers"),
                        "speculativeDecoding": result.get("speculativeDecoding"),
                        "treeBudget": result.get("treeBudget"),
                    },
                })
                return
            except Exception as exc:
                speculative_stream_fallback_note = (
                    f"Speculative decoding stream path failed ({exc}). "
                    "Fell back to standard generation."
                )

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Inject tool schemas into system prompt for open-source models
        system_prompt = request.get("systemPrompt")
        tools_block = _format_tools_for_prompt(request.get("tools"))
        if tools_block:
            system_prompt = (tools_block + "\n\n" + (system_prompt or "")).strip()

        prompt_text, prompt_note = _build_prompt_text(
            self.tokenizer,
            history=list(request.get("history") or []),
            prompt=str(request.get("prompt") or ""),
            system_prompt=system_prompt,
        )
        sampler = _build_mlx_sampler(request)
        prompt_cache, runtime_note = self._make_cache()
        runtime_note = _merge_runtime_notes(runtime_note, prompt_note)
        runtime_note = _merge_runtime_notes(runtime_note, speculative_stream_fallback_note)
        runtime_fields = self._runtime_fields(prompt_cache=prompt_cache)
        transcript_fallback = _plain_chat_fallback_active(prompt_note)

        thinking_mode = request.get("thinkingMode") or "off"
        think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode != "off"))
        transcript_filter = TranscriptLoopFilter() if transcript_fallback else None
        transcript_trimmed = False
        runaway_guard = RunawayGuard()
        runaway_stopped = False

        try:
            last_response = None
            for response in mlx_stream_generate(
                self.model,
                self.tokenizer,
                prompt_text,
                max_tokens=int(request.get("maxTokens") or 256),
                sampler=sampler,
                prompt_cache=prompt_cache,
            ):
                if response.text:
                    # Check for runaway loops before emitting
                    try:
                        runaway_guard.feed(response.text)
                    except RuntimeError:
                        runaway_stopped = True
                        last_response = response
                        break
                    filtered = think_filter.feed(response.text)
                    if filtered.reasoning:
                        _emit({"ok": True, "chunk": {"reasoning": filtered.reasoning}})
                    if filtered.reasoning_done:
                        _emit({"ok": True, "chunk": {"reasoningDone": True}})
                    visible_text = filtered.text
                    if visible_text and transcript_filter is not None:
                        visible_text = transcript_filter.feed(visible_text)
                        if transcript_filter.stopped:
                            transcript_trimmed = True
                    if visible_text:
                        _emit({"ok": True, "chunk": {"text": visible_text}})
                    if transcript_filter is not None and transcript_filter.stopped:
                        last_response = response
                        break
                last_response = response
            # Flush any remaining buffered text
            flushed = think_filter.flush()
            if flushed.reasoning:
                _emit({"ok": True, "chunk": {"reasoning": flushed.reasoning}})
            if flushed.reasoning_done:
                _emit({"ok": True, "chunk": {"reasoningDone": True}})
            visible_text = flushed.text
            if visible_text and transcript_filter is not None:
                visible_text = transcript_filter.feed(visible_text) + transcript_filter.flush()
                transcript_trimmed = transcript_trimmed or transcript_filter.stopped
            if visible_text:
                _emit({"ok": True, "chunk": {"text": visible_text}})
        except (ValueError, RuntimeError, TypeError, AttributeError) as exc:
            _should_retry = (
                prompt_cache is not None
                and _should_retry_cache_failure(exc)
            )
            if _should_retry:
                runtime_note = (
                    _merge_runtime_notes(
                        prompt_note,
                        f"Cache strategy failed ({exc}). Fell back to native f16 cache.",
                    )
                )
                runtime_fields = self._runtime_fields(prompt_cache=None)
                think_filter = ThinkingTokenFilter(detect_raw_reasoning=(thinking_mode != "off"))
                transcript_filter = TranscriptLoopFilter() if transcript_fallback else None
                transcript_trimmed = False
                runaway_guard = RunawayGuard()
                runaway_stopped = False
                last_response = None
                for response in mlx_stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt_text,
                    max_tokens=int(request.get("maxTokens") or 256),
                    sampler=sampler,
                    prompt_cache=None,
                ):
                    if response.text:
                        try:
                            runaway_guard.feed(response.text)
                        except RuntimeError:
                            runaway_stopped = True
                            last_response = response
                            break
                        filtered = think_filter.feed(response.text)
                        if filtered.reasoning:
                            _emit({"ok": True, "chunk": {"reasoning": filtered.reasoning}})
                        if filtered.reasoning_done:
                            _emit({"ok": True, "chunk": {"reasoningDone": True}})
                        visible_text = filtered.text
                        if visible_text and transcript_filter is not None:
                            visible_text = transcript_filter.feed(visible_text)
                            if transcript_filter.stopped:
                                transcript_trimmed = True
                        if visible_text:
                            _emit({"ok": True, "chunk": {"text": visible_text}})
                        if transcript_filter is not None and transcript_filter.stopped:
                            last_response = response
                            break
                    last_response = response
                flushed = think_filter.flush()
                if flushed.reasoning:
                    _emit({"ok": True, "chunk": {"reasoning": flushed.reasoning}})
                if flushed.reasoning_done:
                    _emit({"ok": True, "chunk": {"reasoningDone": True}})
                visible_text = flushed.text
                if visible_text and transcript_filter is not None:
                    visible_text = transcript_filter.feed(visible_text) + transcript_filter.flush()
                    transcript_trimmed = transcript_trimmed or transcript_filter.stopped
                if visible_text:
                    _emit({"ok": True, "chunk": {"text": visible_text}})
            else:
                raise

        if last_response is None:
            raise RuntimeError("MLX generation did not return a response.")

        if transcript_trimmed:
            runtime_note = _merge_runtime_notes(
                runtime_note,
                "Suppressed a plain-chat transcript continuation to stop a runaway loop.",
            )
        if runaway_stopped:
            runtime_note = _merge_runtime_notes(
                runtime_note,
                "Stopped runaway generation: model was repeating itself.",
            )

        _emit({
            "ok": True,
            "done": True,
            "result": {
                "finishReason": last_response.finish_reason or "stop",
                "promptTokens": int(last_response.prompt_tokens),
                "completionTokens": int(last_response.generation_tokens),
                "totalTokens": int(last_response.prompt_tokens + last_response.generation_tokens),
                "tokS": round(float(last_response.generation_tps), 1),
                "promptTokS": round(float(last_response.prompt_tps), 1),
                "peakMemoryGb": round(float(last_response.peak_memory), 3),
                "runtimeNote": runtime_note,
                **runtime_fields,
            },
        })


    def eval_perplexity(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        import math
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np

        dataset = request.get("dataset", "wikitext-2")
        num_samples = int(request.get("numSamples", 64))
        seq_length = int(request.get("seqLength", 512))
        batch_size = int(request.get("batchSize", 4))

        dataset_map = {
            "wikitext-2": "wikitext/wikitext-2-raw-v1",
        }
        data_path = dataset_map.get(dataset, dataset)

        emit_progress("loading_data", 10.0, "Loading evaluation dataset...")
        from mlx_lm.perplexity import load_data
        np.random.seed(123)
        data = load_data(self.tokenizer, data_path, num_samples, seq_length)

        emit_progress("evaluating", 20.0, f"Evaluating perplexity on {len(data)} samples...")
        start = time.monotonic()

        all_losses: list[mx.array] = []
        num_batches = (len(data) + batch_size - 1) // batch_size
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            logits = self.model(batch[:, :-1]).astype(mx.float32)
            losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
            mx.eval(losses)
            all_losses.append(losses.flatten())

            pct = 20.0 + (i / len(data)) * 70.0
            emit_progress("evaluating", pct, f"Batch {i // batch_size + 1}/{num_batches}")

        all_losses_cat = mx.concatenate(all_losses)
        mean_loss = all_losses_cat.mean().item()
        ppl = math.exp(mean_loss)
        std_dev = mx.sqrt(mx.var(all_losses_cat, ddof=1)).item()
        se_ppl = ppl * (std_dev / math.sqrt(all_losses_cat.size))

        elapsed = time.monotonic() - start
        tokens_eval = data.shape[0] * (data.shape[1] - 1)

        emit_progress("done", 100.0, f"Perplexity: {ppl:.2f}")
        return {
            "perplexity": round(ppl, 3),
            "standardError": round(se_ppl, 3),
            "evalSeconds": round(elapsed, 2),
            "evalTokensPerSecond": round(tokens_eval / elapsed, 1),
            "numSamples": len(data),
            "seqLength": seq_length,
            "dataset": dataset,
        }

    def eval_task_accuracy(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No MLX model is loaded.")

        from mlx_lm import stream_generate as mlx_stream_generate
        from mlx_lm.sample_utils import make_sampler
        from backend_service.task_datasets import load_task_data, score_answer

        task_name = request.get("taskName", "mmlu")
        limit = int(request.get("limit", 100))
        num_shots = int(request.get("numShots", 5))

        emit_progress("loading_tasks", 10.0, f"Loading {task_name} task data...")
        tasks = load_task_data(task_name, limit, num_shots)

        sampler = make_sampler(temp=0.0)  # greedy for accuracy
        correct = 0
        total = len(tasks)
        start = time.monotonic()

        for idx, task in enumerate(tasks):
            text_parts: list[str] = []
            for resp in mlx_stream_generate(
                self.model,
                self.tokenizer,
                task["prompt"],
                max_tokens=task.get("max_tokens", 3),
                sampler=sampler,
            ):
                if resp.text:
                    text_parts.append(resp.text)

            answer = "".join(text_parts).strip()
            if score_answer(task_name, answer, task["correct_answer"], task.get("choices")):
                correct += 1

            pct = 10.0 + ((idx + 1) / total) * 85.0
            emit_progress(
                "evaluating", pct,
                f"Question {idx + 1}/{total} — {correct}/{idx + 1} correct",
            )

        elapsed = time.monotonic() - start
        accuracy = round(correct / total, 4) if total > 0 else 0.0
        emit_progress("done", 100.0, f"Accuracy: {accuracy:.1%} ({correct}/{total})")
        return {
            "taskName": task_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "numShots": num_shots,
            "evalSeconds": round(elapsed, 2),
        }


def serve() -> int:
    state = WorkerState()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            result = state.handle(request)
            if result is not None:
                _emit({"ok": True, "result": result})
        except Exception as exc:
            _emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=4),
                }
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    # Install the stdout split before any subcommand runs — probe() and
    # gguf_metadata() call _emit too, and both import mlx/gguf machinery
    # that can print to stdout on their own.
    _install_stdio_redirect()

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m backend_service.mlx_worker [probe|gguf-metadata|serve]", file=sys.stderr)
        return 1

    command = argv[0]
    if command == "probe":
        return probe()
    if command == "gguf-metadata":
        if len(argv) < 2:
            print("gguf-metadata requires a path argument", file=sys.stderr)
            return 1
        return gguf_metadata(argv[1])
    if command == "serve":
        return serve()

    print(f"unknown command: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
