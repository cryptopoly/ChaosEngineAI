"""DFLASH speculative decoding integration for ChaosEngineAI.

Maps target models to their DFLASH draft model checkpoints and detects
whether the MLX or vLLM DFLASH backends are installed.  This module is
safe to import on any platform — unavailable backends simply report as
not installed.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any


# ======================================================================
# Draft model registry
# ======================================================================

# target repo → draft checkpoint.  Keys are canonical HuggingFace repo
# IDs (case-sensitive).  The lookup helpers below apply fuzzy matching
# to handle quantised/community variants.

DRAFT_MODEL_MAP: dict[str, str] = {
    # ----- Qwen3 family -----
    "Qwen/Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
    "Qwen/Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
    # ----- Qwen3-Coder family -----
    # 2026-09-28: dropped Qwen3-Coder-4B / -8B and Qwen3.5-7B / -14B —
    # none of those drafter repos exist on the Hub (the API 401s for an
    # unauthenticated caller, the signature of a missing repo; gated
    # repos answer 200), so a match only produced a failed download.
    "Qwen/Qwen3-Coder-30B-A3B": "z-lab/Qwen3-Coder-30B-A3B-DFlash",
    "Qwen/Qwen3-Coder-Next": "z-lab/Qwen3-Coder-Next-DFlash",
    # ----- Qwen3.5 family -----
    "Qwen/Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen/Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen/Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen/Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    # 2026-05-10: z-lab published a 122B-A10B drafter for the largest
    # Qwen3.5 MoE checkpoint. Same naming pattern as the smaller A3B.
    "Qwen/Qwen3.5-122B-A10B": "z-lab/Qwen3.5-122B-A10B-DFlash",
    # ----- Qwen3.6 family -----
    # Dense 27B drafter (2026-04-23, MIT). Same DFlashDraftModel config
    # shape as the Qwen3-4B drafter the matrix validates; upstream
    # dflash-mlx benchmarks it at 2.78-3.06x on Qwen3.6-27B-4bit.
    "Qwen/Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash",
    "Qwen/Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    # ----- Gemma 4 family (added 2026-05-10) -----
    # dflash-mlx 0.1.5 commit 05cc456 added the Gemma4 backend; z-lab
    # ships matched draft checkpoints for both flagship variants.
    "google/gemma-4-31B-it": "z-lab/gemma-4-31B-it-DFlash",
    "google/gemma-4-26B-A4B-it": "z-lab/gemma-4-26B-A4B-it-DFlash",
    # 12B drafter (2026-06); z-lab names it ``gemma4-`` not ``gemma-4-``.
    "google/gemma-4-12B-it": "z-lab/gemma4-12B-it-DFlash",
    # ----- LLaMA family -----
    # ``Llama-3.1-8B-Instruct-DFlash`` never existed publicly; the
    # published drafter is the UltraChat-trained one.
    "meta-llama/Llama-3.1-8B-Instruct": "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
    # ----- gpt-oss family -----
    # Real target org is ``openai`` and the drafter repos are lowercase
    # (the old ``-20B`` / ``-120B`` ids only resolved via a 307 redirect).
    "openai/gpt-oss-20b": "z-lab/gpt-oss-20b-DFlash",
    "openai/gpt-oss-120b": "z-lab/gpt-oss-120b-DFlash",
    # ----- MiniMax family (preview drafts, added 2026-05-10) -----
    "MiniMaxAI/MiniMax-M2.5": "z-lab/MiniMax-M2.5-DFlash",
    "MiniMaxAI/MiniMax-M2.7": "z-lab/MiniMax-M2.7-DFlash",
    # ----- Kimi -----
    "moonshotai/Kimi-K2.5": "z-lab/Kimi-K2.5-DFlash",
    "moonshotai/Kimi-K2.6": "z-lab/Kimi-K2.6-DFlash",
}

# Additional aliases that map community / MLX repos to the same drafts.
_ALIASES: dict[str, str] = {
    "mlx-community/Qwen3-4B-bf16": "Qwen/Qwen3-4B",
    "mlx-community/Qwen3-4B-4bit": "Qwen/Qwen3-4B",
    "mlx-community/Qwen3-4B-8bit": "Qwen/Qwen3-4B",
    "mlx-community/Qwen3-8B-bf16": "Qwen/Qwen3-8B",
    "mlx-community/Qwen3-8B-4bit": "Qwen/Qwen3-8B",
    "mlx-community/Qwen3-8B-8bit": "Qwen/Qwen3-8B",
    "lmstudio-community/Qwen3-Coder-Next-MLX-4bit": "Qwen/Qwen3-Coder-Next",
    # FU-041 (2026-05-10): canonicalRepo for the lmstudio-community
    # Coder-Next MLX 4-bit ships as the same repo path (no rename), so
    # alias both the exact community ref AND the bf16 / 8bit siblings
    # for completeness. The earlier FU-038 mappings that pointed
    # ``mlx-community/Qwen3.6-27B-4bit`` at Coder-Next were wrong — that
    # repo is the dense Qwen3.6-27B (text-only Coder, ``qwen3``
    # architecture, hidden_size=5120), not the Qwen3-Next MoE coder
    # (``qwen3_next`` architecture, 512 experts, hidden_size=2048).
    # Inspecting the local config.json under ~/AI_Models/
    # lmstudio-community/Qwen3-Coder-Next-MLX-4bit confirms the latter.
    # Coder-Next uses ``z-lab/Qwen3-Coder-Next-DFlash``; the dense
    # 27B-4bit gets its own ``z-lab/Qwen3.6-27B-DFlash`` (aliased below).
    "lmstudio-community/Qwen3-Coder-Next-MLX-4bit-Instruct": "Qwen/Qwen3-Coder-Next",
    "mlx-community/Qwen3.5-4B-bf16": "Qwen/Qwen3.5-4B",
    "mlx-community/Qwen3.5-27B-bf16": "Qwen/Qwen3.5-27B",
    "mlx-community/Qwen3.6-27B-4bit": "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3.6-27B-8bit": "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3.6-27B-bf16": "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3.6-35B-A3B-bf16": "Qwen/Qwen3.6-35B-A3B",
    "mlx-community/Qwen3.6-35B-A3B-4bit": "Qwen/Qwen3.6-35B-A3B",
    "mlx-community/Qwen3.6-35B-A3B-8bit": "Qwen/Qwen3.6-35B-A3B",
    # ----- Qwen3.5-122B-A10B (added 2026-05-10) -----
    "mlx-community/Qwen3.5-122B-A10B-bf16": "Qwen/Qwen3.5-122B-A10B",
    "mlx-community/Qwen3.5-122B-A10B-4bit": "Qwen/Qwen3.5-122B-A10B",
    "mlx-community/Qwen3.5-122B-A10B-8bit": "Qwen/Qwen3.5-122B-A10B",
    # ----- Gemma 4 (added 2026-05-10) -----
    "mlx-community/gemma-4-31B-it-bf16": "google/gemma-4-31B-it",
    "mlx-community/gemma-4-31B-it-4bit": "google/gemma-4-31B-it",
    "mlx-community/gemma-4-31B-it-8bit": "google/gemma-4-31B-it",
    "mlx-community/gemma-4-26B-A4B-it-bf16": "google/gemma-4-26B-A4B-it",
    "mlx-community/gemma-4-26B-A4B-it-4bit": "google/gemma-4-26B-A4B-it",
    "mlx-community/gemma-4-26B-A4B-it-8bit": "google/gemma-4-26B-A4B-it",
    # ----- MiniMax (added 2026-05-10) -----
    "mlx-community/MiniMax-M2.5-bf16": "MiniMaxAI/MiniMax-M2.5",
    "mlx-community/MiniMax-M2.5-4bit": "MiniMaxAI/MiniMax-M2.5",
    "mlx-community/MiniMax-M2.5-8bit": "MiniMaxAI/MiniMax-M2.5",
    "mlx-community/MiniMax-M2.7-bf16": "MiniMaxAI/MiniMax-M2.7",
    "mlx-community/MiniMax-M2.7-4bit": "MiniMaxAI/MiniMax-M2.7",
    "mlx-community/MiniMax-M2.7-8bit": "MiniMaxAI/MiniMax-M2.7",
    # ----- LLaMA 3.1 -----
    "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16": "meta-llama/Llama-3.1-8B-Instruct",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "meta-llama/Llama-3.1-8B-Instruct",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit": "meta-llama/Llama-3.1-8B-Instruct",
    # ----- gpt-oss -----
    "mlx-community/gpt-oss-20B-bf16": "openai/gpt-oss-20b",
    "mlx-community/gpt-oss-20B-4bit": "openai/gpt-oss-20b",
    "mlx-community/gpt-oss-20B-8bit": "openai/gpt-oss-20b",
    "mlx-community/gpt-oss-120B-bf16": "openai/gpt-oss-120b",
    "mlx-community/gpt-oss-120B-4bit": "openai/gpt-oss-120b",
    "mlx-community/gpt-oss-120B-8bit": "openai/gpt-oss-120b",
    # ----- Kimi -----
    "mlx-community/Kimi-K2.5-bf16": "moonshotai/Kimi-K2.5",
    "mlx-community/Kimi-K2.5-4bit": "moonshotai/Kimi-K2.5",
    "mlx-community/Kimi-K2.5-8bit": "moonshotai/Kimi-K2.5",
    "mlx-community/Kimi-K2.6-bf16": "moonshotai/Kimi-K2.6",
    "mlx-community/Kimi-K2.6-4bit": "moonshotai/Kimi-K2.6",
    "mlx-community/Kimi-K2.6-8bit": "moonshotai/Kimi-K2.6",
}

# Suffixes stripped during fuzzy matching (order matters — longest first).
_QUANT_SUFFIXES = re.compile(
    r"[-_](?:bf16|fp16|f16|4bit|8bit|3bit|q4_k_m|q5_k_m|q8_0|GGUF|gguf|instruct|Instruct)$",
    re.IGNORECASE,
)

# Community repo prefixes that should be stripped for fuzzy matching.
_COMMUNITY_PREFIXES = ("mlx-community/", "lmstudio-community/", "TheBloke/", "bartowski/")


def _normalize_ref(model_ref: str) -> str:
    """Strip quantisation/format suffixes and community prefixes for fuzzy matching."""
    ref = model_ref.strip()
    # Repeatedly strip known suffixes (handles stacked ones like ``-8bit-Instruct``)
    for _ in range(3):
        ref = _QUANT_SUFFIXES.sub("", ref)
    return ref


def get_draft_model(target_ref: str) -> str | None:
    """Return the DFLASH draft model checkpoint for *target_ref*, or ``None``."""
    # 1. Exact match
    if target_ref in DRAFT_MODEL_MAP:
        return DRAFT_MODEL_MAP[target_ref]

    # 2. Explicit alias
    canonical = _ALIASES.get(target_ref)
    if canonical and canonical in DRAFT_MODEL_MAP:
        return DRAFT_MODEL_MAP[canonical]

    # 3. Fuzzy: strip quant suffixes and retry
    normalised = _normalize_ref(target_ref)
    if normalised in DRAFT_MODEL_MAP:
        return DRAFT_MODEL_MAP[normalised]

    # 4. Fuzzy: strip community prefix, then normalise
    for prefix in _COMMUNITY_PREFIXES:
        if target_ref.startswith(prefix):
            base = target_ref[len(prefix):]
            normalised_base = _normalize_ref(base)
            # Try matching against the model name portion of each key
            for key, draft in DRAFT_MODEL_MAP.items():
                key_model = key.split("/", 1)[-1] if "/" in key else key
                if normalised_base == key_model or _normalize_ref(key_model) == normalised_base:
                    return draft
            break

    # 5. Substring: community finetunes often embed the base model name
    #    (e.g. "MLX-Qwen3.5-9B-Claude-Opus-Distilled-8bit" contains "Qwen3.5-9B").
    #    Try each draft model's base name as a case-insensitive substring.
    ref_lower = _normalize_ref(target_ref).lower()
    # Also strip the model name portion after a community prefix
    for prefix in _COMMUNITY_PREFIXES:
        if target_ref.lower().startswith(prefix):
            ref_lower = _normalize_ref(target_ref[len(prefix):]).lower()
            break

    # Sort by longest key first so "Qwen3.5-35B-A3B" matches before "Qwen3.5-3"
    for key, draft in sorted(DRAFT_MODEL_MAP.items(), key=lambda kv: -len(kv[0])):
        key_model = (key.split("/", 1)[-1] if "/" in key else key).lower()
        if key_model in ref_lower:
            return draft

    return None


# ======================================================================
# Availability detection
# ======================================================================

def _spec_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def is_mlx_available() -> bool:
    """True when ``dflash_mlx`` is importable."""
    return _spec_exists("dflash_mlx")


# vLLM serves DFlash natively via ``speculative_config={"method":
# "dflash"}``; 0.28 is the release that also carries DFlash 2's
# candidate-path selector (vllm-project/vllm#52816). Below that we don't
# claim the lane.
VLLM_DFLASH_MIN_VERSION: tuple[int, int] = (0, 28)


def vllm_version() -> str | None:
    """Installed vLLM version from package metadata (never imports vllm)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - stdlib on 3.8+
        return None
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def _version_tuple(raw: str) -> tuple[int, int] | None:
    match = re.match(r"(\d+)\.(\d+)", raw.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_vllm_available() -> bool:
    """True when the installed vLLM can serve DFlash drafts.

    FU-091: this used to probe ``dflash.model`` — the PyPI ``dflash``
    package, whose top-level module has the same name as this one. This
    module always wins on ``sys.path``, so the probe could never pass,
    and vLLM doesn't use that package anyway. Gate on vLLM's own version
    instead (metadata only, so no torch import on the startup path).
    """
    raw = vllm_version()
    if raw is None:
        return False
    parsed = _version_tuple(raw)
    return parsed is not None and parsed >= VLLM_DFLASH_MIN_VERSION


def is_available() -> bool:
    """True when at least one DFLASH backend is usable."""
    return is_mlx_available() or is_vllm_available()


def supported_models() -> list[str]:
    """Return target model refs (canonical + community aliases) with known DFlash drafts."""
    return sorted(set(DRAFT_MODEL_MAP.keys()) | set(_ALIASES.keys()))


def is_ddtree_available() -> bool:
    """True when DDTree (tree-based speculative decoding) can run.

    DDTree requires the same dflash_mlx runtime as linear DFlash, plus
    access to ``dflash_mlx.runtime`` primitives for tree verification.

    The required-symbol set mirrors what our code actually imports from
    ``dflash_mlx.runtime`` (see FU-006): ``resolve_target_ops`` is the
    per-family adapter entry point ``backend_service/ddtree.py`` calls to
    reach ``forward_with_hidden_capture`` / ``extract_context_feature`` /
    ``make_cache`` (these moved off the runtime top level onto a
    ``target_ops`` object in dflash-mlx 0.1.5); ``load_draft_bundle`` is
    used by the worker lifecycle; ``stream_dflash_generate`` drives the
    linear path. The pre-0.1.5 symbol ``target_forward_with_hidden_states``
    was renamed to ``target_ops.forward_with_hidden_capture`` and must NOT
    be required here, or the probe wrongly reports DDTree unavailable on
    every modern dflash-mlx build.
    """
    try:
        runtime_spec = importlib.util.find_spec("dflash_mlx.runtime")
    except ModuleNotFoundError:
        return False
    if runtime_spec is None:
        return False
    runtime_path = getattr(runtime_spec, "origin", None)
    if not runtime_path:
        return True
    try:
        source = Path(runtime_path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True
    required_symbols = (
        "resolve_target_ops",
        "load_draft_bundle",
        "stream_dflash_generate",
    )
    return all(symbol in source for symbol in required_symbols)


def availability_info() -> dict[str, Any]:
    """Return a JSON-friendly dict for the frontend system stats."""
    return {
        "available": is_available(),
        "mlxAvailable": is_mlx_available(),
        "vllmAvailable": is_vllm_available(),
        "ddtreeAvailable": is_ddtree_available(),
        "supportedModels": supported_models(),
    }
