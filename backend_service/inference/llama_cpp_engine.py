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
import os
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
from backend_service.inference._tensorfold_wire import _TOOL_CALL_BLOCK, _structured_calls
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
from backend_service.helpers.gpu import nvidia_shared_memory
from cache_compression.native import native_llama_cache_type
from backend_service.helpers.request_guards import friendly_llama_error, llama_error
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
    # Modern anti-repetition / quality samplers llama-server supports
    # natively. Forward-only: builds that don't recognise them ignore the
    # field, so old binaries are unaffected. DRY beats plain repeat_penalty
    # at killing verbatim loops; XTC adds creative variety; top-n-sigma is
    # a temperature-stable truncator.
    "dry_multiplier",
    "dry_base",
    "dry_allowed_length",
    "xtc_probability",
    "xtc_threshold",
    "top_n_sigma",
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
    thinking_mode: str | None = None,
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
    if thinking_mode is not None:
        # Templates that gate reasoning on ``enable_thinking`` (Qwen3 family,
        # LFM2.5, ...) reason by default; the server-wide ``--reasoning off``
        # does not reach every one, so say it per request.
        payload["chat_template_kwargs"] = {"enable_thinking": thinking_mode != "off"}
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

    # Cache only a real answer: a first --help that timed out (a cold binary
    # on a busy machine) used to pin an empty result for the life of the
    # backend, so every later TurboQuant load fell back to f16.
    if help_text:
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


# Cache types llama-server lists in --help but only runs on CUDA (FU-178):
# b12387 aborts on Metal when it allocates an f8_e4m3 cache (SET_ROWS is not
# supported on MTL0, with or without flash attention), so pre-validation
# against --help alone would let it through.
_CUDA_ONLY_CACHE_TYPES: frozenset[str] = frozenset({"f8_e4m3"})


def _refused_cache_type(cache_type: str) -> str | None:
    """Why ``cache_type`` cannot run on this host, or None when it can."""
    if cache_type not in _CUDA_ONLY_CACHE_TYPES:
        return None
    try:
        from backend_service.helpers.hardware_class import host_hardware_class  # noqa: PLC0415

        if host_hardware_class().kind == "cuda":
            return None
    except Exception:  # noqa: BLE001
        pass
    return (
        f"The {cache_type} (FP8) KV cache needs an NVIDIA GPU: llama-server stops at load "
        "with it on Apple Silicon and the CPU. Using f16 cache."
    )


# llama-server prints one line per finished request that drafted anything
# (``server_slot::print_timings``), e.g.
# ``slot print_timing: id  0 | task 0 | draft acceptance = 0.63158 (  120 accepted /   190 generated), mean len =  2.71``
_DRAFT_ACCEPTANCE_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)


def _draft_acceptance(timings: Any, log_text: str | None) -> float | None:
    """Average draft tokens the target accepted per verify round, or None.

    That is the unit ``dflashAcceptanceRate`` carries on every lane ("N avg
    accepted"). The reply's ``timings`` give the drafted and accepted counts
    (``draft_n`` / ``draft_n_accepted``) but not the number of verify rounds;
    the server log's ``draft acceptance`` line for the same request does, as
    ``mean len`` = 1 + accepted / rounds. The line is matched on both counts,
    so a line left by an earlier request is never used.
    """
    if not isinstance(timings, dict):
        return None
    drafted = timings.get("draft_n")
    accepted = timings.get("draft_n_accepted")
    if not isinstance(drafted, int) or not isinstance(accepted, int) or drafted <= 0:
        return None
    for match in reversed(list(_DRAFT_ACCEPTANCE_RE.finditer(log_text or ""))):
        if int(match.group(2)) == accepted and int(match.group(3)) == drafted:
            return round(max(float(match.group(4)) - 1.0, 0.0), 2)
    return None


def _resolve_mmproj_path(model_gguf_path: str | None) -> str | None:
    """Locate the mmproj projector sibling for a vision-capable GGUF.

    Vision support in llama.cpp is gated by the `--mmproj` flag; the
    projector lives as a separate `*mmproj*.gguf` file alongside the
    main weights in the same repo directory. We deliberately scan ONLY
    the main GGUF's own directory — earlier revisions walked sibling
    subdirectories and the grandparent, which under a flat layout like
    `~/AI_Models/<org>/<repo>/` picked up the mmproj of an unrelated
    neighbouring model and crashed llama-server with an n_embd
    mismatch. Returns None for text-only models (no mmproj present).
    """
    if not model_gguf_path:
        return None
    main_path = Path(model_gguf_path)
    if not main_path.exists():
        return None

    parent = main_path.parent
    if not parent.is_dir():
        return None

    # A model in a subfolder of a Hugging Face snapshot (Inkling 2 Mini's ``Mainline_Support/``, FU-540) may keep
    # its projector at the snapshot root. The snapshot is one repo, so its root is still never a neighbour's.
    folders = [parent]
    if parent.parent.parent.name == "snapshots" and parent.parent.parent.parent.name.startswith("models--"):
        folders.append(parent.parent)
    candidates: list[Path] = []
    for folder in folders:
        try:
            for entry in folder.iterdir():
                if (
                    entry.is_file()
                    and entry != main_path
                    and entry.suffix.lower() == ".gguf"
                    and "mmproj" in entry.name.lower()
                ):
                    candidates.append(entry)
        except OSError:
            return None
        if candidates:
            break

    if not candidates:
        return None
    candidates.sort(key=lambda f: f.stat().st_size, reverse=True)
    return str(candidates[0])


def _speculative_args(
    binary: str | None,
    *,
    resolved_gguf: str | None,
    path: str | None,
    repo: str,
) -> tuple[list[str], str | None]:
    """llama-server speculative-decoding flags for a GGUF load.

    Preference order (FU-047 / FU-089 / FU-178):

    1. A block drafter (DFlash / DFlash 2 / DSpark): one the author publishes
       in a repo of its own (pinned by repo id), else a ``dspark-`` /
       ``dflash-`` sidecar beside the GGUF → ``-md <file>`` + its type.
       Measured ahead of MTP on Qwen3.8-27B (M4 Max: 55-59 tok/s against
       32-34 for the baked-in MTP head and 15 without drafting). DFlash 2 drafters
       run as ``draft-dflash``: llama.cpp reads the variant from the file.
    2. Baked-in MTP heads in the model itself → ``draft-mtp``.
    3. Any other sidecar (``mtp-``, ``eagle3-``), in llama.cpp's order.
    4. Nothing to draft with → ``ngram-mod``: draft-free n-gram lookup,
       lossless, ~16 MB, pays off on repetitive text (code edits,
       summaries, reasoning echoed in the answer).

    Each type is gated on the binary listing it in ``--help``; an older
    llama-server gets standard decode plus a note, never a startup error.
    """
    from backend_service.helpers.text_gguf import (  # noqa: PLC0415
        BLOCK_DRAFT_TYPES,
        resolve_pinned_drafter,
        resolve_spec_sidecar,
    )
    from backend_service.inference._mtp import (  # noqa: PLC0415
        get_mtp_draft_n,
        is_mtp_gguf_repo,
        model_has_mtp_tensors,
    )

    if not _llama_server_supports(binary, "--spec-type"):
        return [], (
            "Speculative decoding needs a newer llama-server (no --spec-type in "
            "the installed binary; run scripts/update-llama-cpp.sh). Using "
            "standard decode."
        )

    def supports(spec_type: str) -> bool:
        return _llama_server_supports(binary, spec_type)

    def drafter_args(spec_type: str, sidecar_path: str) -> tuple[list[str], str]:
        args = ["--model-draft", sidecar_path, "--spec-type", spec_type]
        if spec_type == "draft-mtp":
            args += ["--spec-draft-n-max", str(get_mtp_draft_n(repo) or 2)]
        elif spec_type in BLOCK_DRAFT_TYPES:
            # Block drafters: ask for a full block; llama.cpp clamps to
            # the drafter's trained block size (15 for a b16 DFlash).
            args += ["--spec-draft-n-max", "15"]
        return args, (
            f"Speculative decoding active ({spec_type} with drafter "
            f"{Path(sidecar_path).name})."
        )

    sidecar = resolve_pinned_drafter(repo, resolved_gguf) or resolve_spec_sidecar(resolved_gguf)
    sidecar_note: str | None = None
    if sidecar is not None and not supports(sidecar[0]):
        sidecar_note = (
            f"Found drafter {Path(sidecar[1]).name} but the installed "
            f"llama-server has no {sidecar[0]}; run scripts/update-llama-cpp.sh."
        )
        sidecar = None

    # 1. A block drafter in hand.
    if sidecar is not None and sidecar[0] in BLOCK_DRAFT_TYPES:
        return drafter_args(*sidecar)

    # 2. Baked-in MTP heads. Authoritative check peeks the GGUF tensors
    # when a local path is supplied; repo-name aliases otherwise.
    tensor_probe = model_has_mtp_tensors(path)
    has_mtp_tensors = tensor_probe if tensor_probe is not None else is_mtp_gguf_repo(repo)
    if has_mtp_tensors and supports("draft-mtp"):
        n_max = get_mtp_draft_n(repo) or 2
        return ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n_max)], (
            f"MTP speculative decoding active (draft-mtp, spec-draft-n-max={n_max}). "
            "Prompt-processing speed may dip slightly vs. standard decode due to "
            "D2H embedding transfers (per upstream PR #22673)."
        )

    # 3. Any other drafter (an mtp- or eagle3- sidecar).
    if sidecar is not None:
        return drafter_args(*sidecar)

    # 4. Draft-free n-gram lookup.
    if supports("ngram-mod"):
        note = (
            "Speculative decoding active (ngram-mod: no draft model; speeds up "
            "repetitive output such as code edits and summaries)."
        )
        return ["--spec-type", "ngram-mod"], f"{sidecar_note} {note}" if sidecar_note else note

    fallback = "No speculative-decoding mode available for this model; using standard decode."
    return [], f"{sidecar_note} {fallback}" if sidecar_note else fallback


def _discrete_cuda_host() -> bool:
    try:
        from backend_service.helpers.hardware_class import host_hardware_class

        hardware = host_hardware_class()
    except Exception:  # noqa: BLE001 - an unreadable host is "not a discrete card"
        return False
    return hardware.kind == "cuda" and not hardware.unified


def _auto_cpu_moe_layers(gguf: str, context_tokens: int, cache_flags: list[str]) -> int | None:
    """Blocks whose experts must stay in system RAM for this GGUF to fit the card's free memory now, or None when
    nothing has to move (a dense model, it fits, or the card cannot be read)."""
    from backend_service.helpers.chat_memory import LLAMA_CACHE_BITS
    from backend_service.helpers.gguf_layout import gguf_layout
    from backend_service.helpers.memory_gate import cuda_free_vram_gb
    from backend_service.helpers.model_arch import read_kv_architecture
    from backend_service.helpers.moe_offload import GPU_MARGIN_GB, auto_cpu_layers, is_moe

    layout = gguf_layout(gguf)
    if not is_moe(layout):
        return None
    free_gb = cuda_free_vram_gb()
    if free_gb is None:
        return None
    arch = read_kv_architecture(gguf)
    cache_type = cache_flags[cache_flags.index("--cache-type-k") + 1] if "--cache-type-k" in cache_flags[:-1] else "f16"
    cache_bytes = arch.cache_bytes(context_tokens) * LLAMA_CACHE_BITS.get(cache_type, 16.0) / 16.0 if arch else 0.0
    layers = auto_cpu_layers(layout, cache_bytes, free_gb - GPU_MARGIN_GB)
    if layers is None:
        # Even every expert in RAM leaves too much on the card: move them all and let llama.cpp report the rest.
        return len(layout.expert_bytes_by_layer) if layout is not None else None
    return layers or None


def _expert_offload_args(
    binary: str,
    *,
    requested: int | None,
    fit_enabled: bool,
    gguf: str | None,
    context_tokens: int,
    cache_flags: list[str],
) -> tuple[list[str], int | None, str | None]:
    """Mixture-of-experts expert offload (FU-550): ``(arguments, blocks moved, note)``.

    ``requested`` is the number of leading blocks whose routed experts stay in system RAM: 0 keeps them all on the
    GPU, None is automatic. Automatic with fit-in-memory on leaves the choice to llama.cpp's own ``--fit``, which
    moves experts to system memory one block at a time when the model does not fit (read from b12649's log on the
    RTX 4090). With fit-in-memory off nothing else adjusts the load, so the app picks the count from the card's
    free memory and the file's tensor table. Gated on the binary listing ``--n-cpu-moe`` in ``--help``, like the
    cache types.
    """
    if requested == 0:
        return [], None, None
    if requested is None:
        if fit_enabled or gguf is None or not _discrete_cuda_host():
            return [], None, None
        requested = _auto_cpu_moe_layers(gguf, context_tokens, cache_flags)
        if not requested:
            return [], None, None
    if not _llama_server_supports(binary, "--n-cpu-moe"):
        return [], None, (
            "This llama-server cannot keep mixture-of-experts weights in system RAM (no --n-cpu-moe); "
            "the expert offload setting was ignored."
        )
    blocks: int | None = None
    if gguf is not None:
        from backend_service.helpers.gguf_layout import gguf_layout
        from backend_service.helpers.moe_offload import block_count, is_moe

        layout = gguf_layout(gguf)
        if layout is not None and not is_moe(layout):
            return [], None, "This is not a mixture-of-experts model, so the expert offload setting was ignored."
        if layout is not None:
            blocks = block_count(layout)
    if blocks is not None and requested >= blocks and _llama_server_supports(binary, "--cpu-moe"):
        args, moved = ["--cpu-moe"], blocks
    else:
        args, moved = ["--n-cpu-moe", str(requested)], (min(requested, blocks) if blocks is not None else requested)
    of_blocks = f" of {blocks}" if blocks is not None else ""
    note = (
        f"The experts of {moved}{of_blocks} layers stay in system RAM and run on the CPU, so the model fits the GPU; "
        "replies are slower than with every expert on the GPU."
    )
    return args, moved, note


def _gguf_startup_fallback_note(strategy_name: str) -> str:
    return (
        f"GGUF startup failed with {strategy_name} cache, so ChaosEngineAI retried "
        f"with the standard f16 KV cache."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _chat_messages(
    system_prompt: str | None,
    history: list[dict[str, Any]],
    prompt: str,
    images: list[str] | None,
) -> list[dict[str, Any]]:
    """Chat state -> llama-server messages, keeping tool calls structured.

    The agent loop records an assistant tool call with ``tool_calls`` and its
    result as a ``tool`` turn with ``tool_call_id``; both are passed on so the
    model's chat template renders them natively. Its follow-up requests send
    an empty ``prompt`` (the tool result is the last turn), which adds no
    user turn.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in history:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        entry: dict[str, Any] = {"role": role, "content": _normalize_message_content(message.get("text", ""))}
        if role == "assistant":
            calls = _structured_calls(message.get("tool_calls"))
            if calls:
                entry["tool_calls"] = calls
                entry["content"] = _TOOL_CALL_BLOCK.sub("", entry["content"]).strip()
        elif role == "tool" and message.get("tool_call_id"):
            entry["tool_call_id"] = str(message["tool_call_id"])
        messages.append(entry)
    if images:
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        messages.append({"role": "user", "content": content_parts})
    elif prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


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
        self.applied_moe_cpu_layers: int | None = None
        self.moe_offload_note: str | None = None

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
        speculative_decoding: bool = False,
        fused_attention: bool = False,
        canonical_repo: str | None = None,
        model_ref: str = "",
        moe_cpu_layers: int | None = None,
    ) -> tuple[list[str], str | None, bool, str | None]:
        """Build the llama-server command line.

        Returns ``(command, runtime_note, fell_back_to_native, mmproj_path)`` where
        *fell_back_to_native* is ``True`` when pre-validation detected
        unsupported cache types and silently switched to f16. The number of
        blocks whose experts stay in system RAM is left on
        ``self.applied_moe_cpu_layers`` (None when none do).
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
        # Reuse the single slot's KV cache across chat turns: a growing
        # conversation re-prefills only the new suffix instead of the whole
        # history (turn-2+ TTFT drops sharply on long chats). Forward-gated
        # on binary support so older llama-server builds are unaffected.
        if _llama_server_supports(binary, "--cache-reuse"):
            command.extend(["--cache-reuse", "256"])
        # On a GPU that shares system memory (DGX Spark, GH200) loading through
        # mmap is slow; read the file in instead. CHAOSENGINE_LLAMA_MMAP=1 keeps mmap.
        # llama.cpp 1.0 replaced --no-mmap (now an invalid argument) with
        # --load-mode none; older builds and the turbo fork still take --no-mmap.
        if os.environ.get("CHAOSENGINE_LLAMA_MMAP") != "1" and nvidia_shared_memory():
            if _llama_server_supports(binary, "--load-mode"):
                command.extend(["--load-mode", "none"])
            elif _llama_server_supports(binary, "--no-mmap"):
                command.append("--no-mmap")
        # Honour the user's fused-attention toggle. It was plumbed into
        # load_model + stored on LoadedModelInfo but never emitted as a
        # flag. Flash attention is a large decode + KV-memory win on Metal
        # and is required by the quantized KV cache types. Opt-in via the
        # existing flag so a model/quant combo that dislikes it can disable.
        if fused_attention and _llama_server_supports(binary, "--flash-attn"):
            command.extend(["--flash-attn", "on"])
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
                    refusal = _refused_cache_type(cache_type)
                    if refusal is not None:
                        runtime_note = refusal
                        cache_flags = ["--cache-type-k", "f16", "--cache-type-v", "f16"]
                        fell_back_to_native = True
                        break
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

        offload_args, self.applied_moe_cpu_layers, self.moe_offload_note = _expert_offload_args(
            binary,
            requested=moe_cpu_layers,
            fit_enabled=fit_enabled,
            gguf=resolved_gguf,
            context_tokens=context_tokens,
            cache_flags=cache_flags,
        )
        command.extend(offload_args)
        # Automatic with Fit in memory on passes no expert flags, but llama.cpp's own --fit then moves experts to
        # system RAM by tensor override when the file is bigger than the card (FU-550), so it needs the same read-in.
        if offload_args or (
            moe_cpu_layers is None
            and fit_enabled
            and resolved_gguf is not None
            and _discrete_cuda_host()
            and _auto_cpu_moe_layers(resolved_gguf, context_tokens, cache_flags)
        ):
            # llama.cpp warns that CPU tensor overrides under mmap are slow: reading the file in loaded in 6.6 s
            # instead of 12.6 s and decoded at 84 tok/s instead of 72 to 83 (Qwen3.6-35B-A3B, 20 blocks, 4090).
            if (
                "--load-mode" not in command
                and os.environ.get("CHAOSENGINE_LLAMA_MMAP") != "1"
                and _llama_server_supports(binary, "--load-mode")
            ):
                command.extend(["--load-mode", "none"])

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

        if speculative_decoding:
            spec_args, spec_note = _speculative_args(
                binary,
                resolved_gguf=resolved_gguf,
                path=path,
                repo=canonical_repo or runtime_target or model_ref or path or "",
            )
            command.extend(spec_args)
            if spec_note:
                runtime_note = f"{runtime_note} {spec_note}".strip() if runtime_note else spec_note

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
        moe_cpu_layers: int | None = None,
    ) -> LoadedModelInfo:
        if not self.capabilities.ggufAvailable:
            raise RuntimeError("llama.cpp support is unavailable on this machine.")

        if _is_local_target(path) or _is_local_target(runtime_target):
            resolved_preflight = _resolve_gguf_path(path, runtime_target)
            if resolved_preflight is None:
                from backend_service.helpers.weights_format import describe_contents, weights_format

                source = path or runtime_target
                found_format = weights_format(source) if source else None
                if found_format in {"safetensors", "pytorch"}:
                    hint = "It holds a safetensors checkpoint: load it with the MLX backend instead."
                else:
                    hint = (
                        "The download may be incomplete or corrupt. Re-download the model, "
                        "or pick a specific .gguf file from the source directory."
                    )
                raise RuntimeError(
                    f"No .gguf weights found inside {source} (found: {describe_contents(source)}). {hint}"
                )
            from backend_service.helpers.weights_format import has_gguf_magic

            if has_gguf_magic(Path(resolved_preflight)) is False:
                raise RuntimeError(
                    f"{resolved_preflight} is not a GGUF file: it does not start with the GGUF header. "
                    "It may be an unfinished download or a Git LFS pointer. Re-download the model."
                )

        self.unload_model()
        runtime_note = None
        actual_strategy = cache_strategy
        actual_fit = fit_model_in_memory
        from cache_compression import registry as _strategy_registry
        failed_strategy_name: str | None = None

        # Try the requested strategy first. If it fails, fall back to
        # native f16 directly. The previous chain inserted a ChaosEngine
        # intermediate step; ChaosEngine was dropped in FU-030 because
        # its llama.cpp path only emitted standard q-type flags (q4_0 etc.)
        # which are already a subset of what native + the ggml cache types
        # cover. The two-level chain is shorter to reason about and the
        # behaviour at the boundary is unchanged for users.
        cache_strategy = _strategy_registry.resolve_legacy_id(cache_strategy)
        attempts: list[tuple[str, bool, bool]] = [(cache_strategy, fit_model_in_memory, False)]
        native_quantized = cache_strategy == "native" and native_llama_cache_type(cache_bits) != "f16"
        if cache_strategy != "native" or native_quantized:
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
                speculative_decoding=speculative_decoding,
                fused_attention=fused_attention,
                canonical_repo=canonical_repo,
                model_ref=model_ref,
                moe_cpu_layers=moe_cpu_layers,
            )

            self.last_command = list(command)  # read by the Activity view's detail row (D-88)
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
                # Native at f16 after a refused / unsupported / failed quantized cache.
                actual_native_f16 = prevalidation_fallback or is_fallback
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
                    failed_strategy_name = (
                        strategy.label(cache_bits, 0) if strategy_id == "native" else strategy.name
                    )
                self._cleanup_process()
        else:
            raise RuntimeError(last_error or "llama.cpp server failed to start.")

        strat = _strategy_registry.get(actual_strategy) or _strategy_registry.default()
        if actual_strategy != "native":
            actual_cache_bits = cache_bits
        else:
            # Native keeps 8 / 4 bits when llama-server runs q8_0 / q4_0 (FU-178).
            actual_cache_bits = (
                cache_bits
                if not actual_native_f16 and native_llama_cache_type(cache_bits) != "f16"
                else 0
            )
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
        if self.moe_offload_note:
            runtime_note = _append_runtime_note(runtime_note, self.moe_offload_note)

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
            # Drafter sidecar, baked-in MTP or ngram-mod: whatever
            # _speculative_args emitted is what the server is running.
            speculativeDecoding="--spec-type" in command,
            moeCpuLayers=self.applied_moe_cpu_layers,
            requestedMoeCpuLayers=moe_cpu_layers,
        )
        return self.loaded_model

    def unload_model(self) -> None:
        self._cleanup_process()
        self.loaded_model = None

    def _acceptance_rate(self, timings: Any) -> float | None:
        """``dflashAcceptanceRate`` for a finished reply (``_draft_acceptance``).

        llama-server writes the log line just before it sends the final reply;
        its logger runs on a thread of its own, so give it a moment to land.
        """
        if not isinstance(timings, dict) or not timings.get("draft_n"):
            return None
        for _ in range(5):
            rate = _draft_acceptance(timings, _read_text_tail(self.log_path, 120))
            if rate is not None:
                return rate
            time.sleep(0.05)
        return None

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
        thinking_mode: str | None = None,
        samplers: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> GenerationResult:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        if self.process is None or self.process.poll() is not None:
            logs = _read_text_tail(self.log_path)
            raise RuntimeError(logs or "The llama.cpp server is not running.")

        messages = _chat_messages(system_prompt, history, prompt, images)

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
            # Reuse the slot's cached prompt prefix across turns (pairs with
            # the server's --cache-reuse) so unchanged history isn't reprocessed.
            "cache_prompt": True,
        }
        if tools:
            payload["tools"] = tools
        _apply_sampler_kwargs(
            payload,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
            thinking_mode=thinking_mode,
        )
        try:
            response = _http_json(
                self._server_url("/v1/chat/completions"),
                payload=payload,
                # A non-streaming reply sends nothing until it is whole: wait for the longest reply the request
                # allows (2 tok/s worst case), not a fixed 120 s (the same cut-off the MLX path had, FU-186).
                timeout=min(max(DEFAULT_LLAMA_TIMEOUT_SECONDS, 60.0 + max(int(max_tokens or 0), 0) * 0.5), 1800.0),
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise llama_error(detail, str(exc)) from exc
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
        # llama-server parses tool calls with the model's own template and
        # returns them structured, leaving ``content`` empty.
        tool_calls = _structured_calls(message.get("tool_calls"))

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
            tool_calls=tool_calls or None,
            dflashAcceptanceRate=self._acceptance_rate(response.get("timings")),
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

        messages = _chat_messages(system_prompt, history, prompt, images)

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
            # Reuse the slot's cached prompt prefix across turns (pairs with
            # the server's --cache-reuse) so unchanged history isn't reprocessed.
            "cache_prompt": True,
        }
        if tools:
            payload["tools"] = tools
        _apply_sampler_kwargs(
            payload,
            samplers=samplers,
            reasoning_effort=reasoning_effort,
            json_schema=json_schema,
            thinking_mode=thinking_mode,
        )
        url = self._server_url("/v1/chat/completions")
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(request, timeout=DEFAULT_LLAMA_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise llama_error(detail, str(exc)) from exc
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
        # llama-server parses <think> blocks into ``reasoning_content`` itself
        # (--reasoning-format deepseek), and some models reason even with
        # --reasoning off. Track whether that channel still needs closing.
        server_reasoning_open = False
        timings: Any = None
        # llama-server streams a call in pieces (``delta.tool_calls``: index, id, name, argument text): join them.
        call_parts: dict[int, dict[str, Any]] = {}
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
                reasoning_content = delta.get("reasoning_content")
                for piece in delta.get("tool_calls") or []:
                    if not isinstance(piece, dict):
                        continue
                    part = call_parts.setdefault(int(piece.get("index") or 0), {"function": {"name": "", "arguments": ""}})
                    if piece.get("id"):
                        part["id"] = piece["id"]
                    function = piece.get("function") or {}
                    if function.get("name"):
                        part["function"]["name"] += str(function["name"])
                    if function.get("arguments"):
                        part["function"]["arguments"] += str(function["arguments"])
                if reasoning_content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    completion_tokens += 1
                    server_reasoning_open = True
                    yield StreamChunk(reasoning=str(reasoning_content))
                if content and server_reasoning_open:
                    server_reasoning_open = False
                    yield StreamChunk(reasoning_done=True)
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
                if chunk.get("timings"):
                    timings = chunk["timings"]
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
            if server_reasoning_open:
                yield StreamChunk(reasoning_done=True)
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
            dflash_acceptance_rate=self._acceptance_rate(timings),
            tool_calls=_structured_calls([call_parts[k] for k in sorted(call_parts)]) or None,
        )
