"""MTP (Multi-Token Prediction) model registry.

Maps models that carry baked-in MTP heads to their recommended
``spec-draft-n-max`` count.  Used by two separate inference paths:

- MLX path  → ``MtplxEngine`` (this module, ``has_mtp_heads``)
- GGUF path → ``LlamaCppEngine._build_command`` (``--spec-type mtp``,
              Phase 2 of feature/mtplx)

Only models trained with MTP objectives belong here — standard MLX quants
of base/instruct checkpoints that strip MTP heads at conversion time should
NOT be listed.  MTPLX auto-detects MTP heads at load time; this map is used
by ChaosEngineAI to decide whether to offer the MTPLX toggle and, for the
GGUF path, how many draft tokens to request.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MLX / transformers repos with baked-in MTP heads
# ---------------------------------------------------------------------------
#
# Key   → canonical HuggingFace repo id (case-sensitive)
# Value → recommended spec-draft-n-max (1–3); start conservatively at 1
#         and bump when acceptance rate benchmarks justify it.

MTP_MODEL_MAP: dict[str, int] = {
    # ----- Youssofal MTPLX-Optimized (upstream-verified for MTPLX v0.3.5) -----
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed": 1,
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16": 1,
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality": 1,
    # ----- Qwen3.5 family -----
    "Qwen/Qwen3.5-4B": 1,
    "Qwen/Qwen3.5-7B": 1,
    "Qwen/Qwen3.5-9B": 1,
    "Qwen/Qwen3.5-14B": 1,
    "Qwen/Qwen3.5-27B": 1,
    "Qwen/Qwen3.5-35B-A3B": 1,
    "Qwen/Qwen3.5-122B-A10B": 1,
    # ----- Qwen3.6 family -----
    "Qwen/Qwen3.6-27B": 1,
    "Qwen/Qwen3.6-35B-A3B": 1,
    # ----- Qwen3-Coder-Next -----
    "Qwen/Qwen3-Coder-Next": 1,
    # ----- DeepSeek V3 / R1 -----
    "deepseek-ai/DeepSeek-V3": 1,
    "deepseek-ai/DeepSeek-V3-0324": 1,
    "deepseek-ai/DeepSeek-R1": 1,
}

# Community MLX conversions that preserve MTP heads.
# Maps community repo → canonical repo (for draft-n lookup).
_MTP_ALIASES: dict[str, str] = {
    # Qwen3.5
    "mlx-community/Qwen3.5-4B-4bit": "Qwen/Qwen3.5-4B",
    "mlx-community/Qwen3.5-4B-8bit": "Qwen/Qwen3.5-4B",
    "mlx-community/Qwen3.5-7B-4bit": "Qwen/Qwen3.5-7B",
    "mlx-community/Qwen3.5-7B-8bit": "Qwen/Qwen3.5-7B",
    "mlx-community/Qwen3.5-9B-4bit": "Qwen/Qwen3.5-9B",
    "mlx-community/Qwen3.5-9B-8bit": "Qwen/Qwen3.5-9B",
    "mlx-community/Qwen3.5-14B-4bit": "Qwen/Qwen3.5-14B",
    "mlx-community/Qwen3.5-14B-8bit": "Qwen/Qwen3.5-14B",
    "mlx-community/Qwen3.5-27B-4bit": "Qwen/Qwen3.5-27B",
    "mlx-community/Qwen3.5-27B-8bit": "Qwen/Qwen3.5-27B",
    # Qwen3.6
    "mlx-community/Qwen3.6-27B-4bit": "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3.6-27B-8bit": "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3.6-27B-bf16": "Qwen/Qwen3.6-27B",
    "mlx-community/Qwen3.6-35B-A3B-4bit": "Qwen/Qwen3.6-35B-A3B",
    "lmstudio-community/Qwen3.6-27B-GGUF": "Qwen/Qwen3.6-27B",
    # Qwen3-Coder-Next
    "lmstudio-community/Qwen3-Coder-Next-MLX-4bit": "Qwen/Qwen3-Coder-Next",
    # ----- llama.cpp MTP GGUF mirrors (FU-047, PR #22673 merged 2026-05-16) -----
    # The ggml-org/* repos are the canonical mirrors am17an published with
    # the PR; am17an/* are the author's pre-merge drafts. Both ship the
    # same baked-in MTP heads. Aliasing them to the canonical safetensors
    # repo means ``has_mtp_heads`` returns True and ``get_mtp_draft_n``
    # picks up the same N as the MLX path.
    "ggml-org/Qwen3.6-27B-MTP-GGUF": "Qwen/Qwen3.6-27B",
    "ggml-org/Qwen3.6-35B-A3B-MTP-GGUF": "Qwen/Qwen3.6-35B-A3B",
    "am17an/Qwen3.6-27B-MTP-GGUF": "Qwen/Qwen3.6-27B",
    "am17an/Qwen3.6-35BA3B-MTP-GGUF": "Qwen/Qwen3.6-35B-A3B",
}


# ---------------------------------------------------------------------------
# GGUF MTP detection (FU-047)
# ---------------------------------------------------------------------------

_MTP_GGUF_REPOS: frozenset[str] = frozenset({
    "ggml-org/Qwen3.6-27B-MTP-GGUF",
    "ggml-org/Qwen3.6-35B-A3B-MTP-GGUF",
    "am17an/Qwen3.6-27B-MTP-GGUF",
    "am17an/Qwen3.6-35BA3B-MTP-GGUF",
})


def is_mtp_gguf_repo(repo: str | None) -> bool:
    """True when *repo* names a GGUF mirror carrying baked-in MTP heads.

    Used by the llama.cpp engine to decide whether to emit
    ``--spec-type draft-mtp --spec-draft-n-max N`` flags.

    Two checks: an exact-match set for the canonical mirrors am17an
    published with PR #22673, plus a defensive ``-MTP-GGUF`` substring
    heuristic so future mirrors get picked up as long as their canonical
    is registered.
    """
    if not repo:
        return False
    if repo in _MTP_GGUF_REPOS:
        return True
    if "-MTP-GGUF" not in repo:
        return False
    aliased = _MTP_ALIASES.get(repo)
    return aliased is not None and aliased in MTP_MODEL_MAP


def get_mtp_draft_n(repo: str) -> int | None:
    """Return the recommended spec-draft-n-max for *repo*, or None.

    Returns None when the repo is not known to carry MTP heads — callers
    should not enable MTP speculative decoding for that model.
    """
    if repo in MTP_MODEL_MAP:
        return MTP_MODEL_MAP[repo]
    canonical = _MTP_ALIASES.get(repo)
    if canonical:
        return MTP_MODEL_MAP.get(canonical)
    return None


def has_mtp_heads(repo: str) -> bool:
    """True when *repo* (or a community alias of it) carries baked-in MTP heads."""
    return get_mtp_draft_n(repo) is not None
