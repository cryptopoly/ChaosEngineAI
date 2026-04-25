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
    "Qwen/Qwen3-Coder-4B": "z-lab/Qwen3-Coder-4B-DFlash",
    "Qwen/Qwen3-Coder-8B": "z-lab/Qwen3-Coder-8B-DFlash",
    "Qwen/Qwen3-Coder-30B-A3B": "z-lab/Qwen3-Coder-30B-A3B-DFlash",
    "Qwen/Qwen3-Coder-Next": "z-lab/Qwen3-Coder-Next-DFlash",
    # ----- Qwen3.5 family -----
    "Qwen/Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen/Qwen3.5-7B": "z-lab/Qwen3.5-7B-DFlash",
    "Qwen/Qwen3.5-9B": "z-lab/Qwen3.5-9B-DFlash",
    "Qwen/Qwen3.5-14B": "z-lab/Qwen3.5-14B-DFlash",
    "Qwen/Qwen3.5-27B": "z-lab/Qwen3.5-27B-DFlash",
    "Qwen/Qwen3.5-35B-A3B": "z-lab/Qwen3.5-35B-A3B-DFlash",
    # ----- Qwen3.6 family -----
    "Qwen/Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    # ----- LLaMA family -----
    "meta-llama/Llama-3.1-8B-Instruct": "z-lab/Llama-3.1-8B-Instruct-DFlash",
    # ----- gpt-oss family -----
    "gpt-oss/gpt-oss-20B": "z-lab/gpt-oss-20B-DFlash",
    "gpt-oss/gpt-oss-120B": "z-lab/gpt-oss-120B-DFlash",
    # ----- Kimi -----
    "moonshotai/Kimi-K2.5": "z-lab/Kimi-K2.5-DFlash",
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
    "mlx-community/Qwen3.5-4B-bf16": "Qwen/Qwen3.5-4B",
    "mlx-community/Qwen3.5-7B-bf16": "Qwen/Qwen3.5-7B",
    "mlx-community/Qwen3.5-14B-bf16": "Qwen/Qwen3.5-14B",
    "mlx-community/Qwen3.5-27B-bf16": "Qwen/Qwen3.5-27B",
    "mlx-community/Qwen3.6-35B-A3B-bf16": "Qwen/Qwen3.6-35B-A3B",
    "mlx-community/Qwen3.6-35B-A3B-4bit": "Qwen/Qwen3.6-35B-A3B",
    "mlx-community/Qwen3.6-35B-A3B-8bit": "Qwen/Qwen3.6-35B-A3B",
    # ----- LLaMA 3.1 -----
    "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16": "meta-llama/Llama-3.1-8B-Instruct",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit": "meta-llama/Llama-3.1-8B-Instruct",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit": "meta-llama/Llama-3.1-8B-Instruct",
    # ----- gpt-oss -----
    "mlx-community/gpt-oss-20B-bf16": "gpt-oss/gpt-oss-20B",
    "mlx-community/gpt-oss-20B-4bit": "gpt-oss/gpt-oss-20B",
    "mlx-community/gpt-oss-20B-8bit": "gpt-oss/gpt-oss-20B",
    "mlx-community/gpt-oss-120B-bf16": "gpt-oss/gpt-oss-120B",
    "mlx-community/gpt-oss-120B-4bit": "gpt-oss/gpt-oss-120B",
    "mlx-community/gpt-oss-120B-8bit": "gpt-oss/gpt-oss-120B",
    # ----- Kimi -----
    "mlx-community/Kimi-K2.5-bf16": "moonshotai/Kimi-K2.5",
    "mlx-community/Kimi-K2.5-4bit": "moonshotai/Kimi-K2.5",
    "mlx-community/Kimi-K2.5-8bit": "moonshotai/Kimi-K2.5",
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


def is_vllm_available() -> bool:
    """True when ``dflash`` (the PyTorch/CUDA package) is importable."""
    # The PyTorch package installs as ``dflash`` but we need to check
    # for the *model* submodule, not this ChaosEngineAI integration module.
    # We check for ``dflash.model`` which is the core PyTorch implementation.
    try:
        spec = importlib.util.find_spec("dflash.model")
        return spec is not None
    except (ModuleNotFoundError, ValueError):
        return False


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
        "target_forward_with_hidden_states",
        "load_draft_bundle",
        "load_target_bundle",
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
