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
    # Depth 3 matches MTPLX's own UI default for these models; benchmarks
    # showed depth=1 hurt rather than helped because the HTTP-proxy
    # overhead per token wasn't amortised across enough draft tokens.
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed": 3,
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed-FP16": 3,
    "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality": 3,
    # ----- Qwen3.5 family -----
    "Qwen/Qwen3.5-4B": 1,
    "Qwen/Qwen3.5-7B": 1,
    "Qwen/Qwen3.5-9B": 1,
    "Qwen/Qwen3.5-14B": 1,
    "Qwen/Qwen3.5-27B": 1,
    "Qwen/Qwen3.5-35B-A3B": 1,
    "Qwen/Qwen3.5-122B-A10B": 1,
    # ----- Qwen3.6 family -----
    # N=3 verified post-merge: 1.46x speedup @ N=1 on Q8_0 GGUF (M5);
    # upstream PR #22673 reports ~72% acceptance @ N=3 on Qwen3.6-27B.
    "Qwen/Qwen3.6-27B": 3,
    "Qwen/Qwen3.6-35B-A3B": 3,
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
    """True when *repo* (or a community alias of it) carries baked-in MTP heads.

    Name-only check. Use ``model_has_mtp_tensors(path)`` for the
    authoritative tensor-level probe when a local path is available.
    """
    return get_mtp_draft_n(repo) is not None


# ---------------------------------------------------------------------------
# Tensor-level MTP head detection (colleague feedback 2026-05-16):
# repo-name aliasing has false-positives (FU-041 mismatched Coder-Next
# variants) and false-negatives (any new MTP-bearing repo we haven't
# enumerated yet). The authoritative signal is the safetensors weight
# index — models with MTP heads ship ``mtp_*`` keys (``mtp_heads.*``,
# ``model.mtp.*``, ``mtp.safetensors`` shard). Probe these directly
# whenever a local path is known; fall back to name match otherwise.
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

# Tensor-name fragments that uniquely identify baked-in MTP heads.
# Verified against: Youssofal MTPLX-Optimized-Speed (mtp.safetensors
# sibling + model.safetensors.index.json entries), Qwen3.6-MTP-GGUF
# (mtp_decoder / mtp_emb tensor names).
_MTP_TENSOR_HINTS: tuple[str, ...] = (
    "mtp_heads.",
    "mtp_decoder.",
    "mtp_emb.",
    "model.mtp.",
    ".mtp.",
)


def _read_safetensors_index(model_dir: _Path) -> dict[str, str] | None:
    """Return the weight_map from a sharded ``model.safetensors.index.json``."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        payload = _json.loads(index_path.read_text())
    except (OSError, _json.JSONDecodeError):
        return None
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    return weight_map if isinstance(weight_map, dict) else None


def model_has_mtp_tensors(path: str | None) -> bool | None:
    """Authoritative MTP-head probe via local model files.

    Returns:
      True  — model has MTP tensor keys (either a ``mtp.safetensors``
              shard, or ``mtp_*`` keys in the safetensors index, or
              ``mtp_decoder`` / ``mtp_emb`` tensors in a GGUF header).
      False — local files probed, no MTP keys found.
      None  — could not probe (path missing, no recognizable index).
              Callers should fall back to ``has_mtp_heads(repo)``
              for the name-only check.
    """
    if not path:
        return None
    p = _Path(path)
    if not p.exists():
        return None
    # GGUF case: peek the model file for the MTP / Next-N markers.
    # PR #22673 emits the metadata key ``<arch>.nextn_predict_layers``
    # in the GGUF header (near the top, well under 2 MB), and ships
    # tensor weights under ``blk.{N}.nextn.*``. Older drafts used
    # ``mtp_decoder`` / ``mtp_emb`` / ``mtp_heads``. The metadata key
    # is the cheapest reliable signal — it's emitted by PR #22673 only
    # for MTP-bearing models and lives in the first few KB.
    if p.is_file() and p.suffix.lower() == ".gguf":
        try:
            with p.open("rb") as fh:
                head = fh.read(2 * 1024 * 1024)
        except OSError:
            return None
        for needle in (
            b"nextn_predict",       # PR #22673 metadata key
            b"mtp_decoder",         # legacy / pre-merge naming
            b"mtp_emb",
            b"mtp_heads",
        ):
            if needle in head:
                return True
        return False

    # MLX / safetensors case: look for the dedicated shard or any
    # ``mtp_*`` key in the index.
    model_dir = p if p.is_dir() else p.parent
    if (model_dir / "mtp.safetensors").exists():
        return True
    weight_map = _read_safetensors_index(model_dir)
    if weight_map is None:
        return None
    for tensor_name in weight_map.keys():
        # FU-076: Qwen3.5 / Qwen3.6 ship the MTP head as *top-level*
        # ``mtp.layers.*`` / ``mtp.fc.weight`` keys (no leading prefix),
        # which the nested ``.mtp.`` / ``model.mtp.`` hints miss — that
        # made ``has_mtp_heads_strict`` return False and silently routed
        # these models to the DFlash path instead of MtplxEngine. Match a
        # bare ``mtp.`` prefix as well.
        if tensor_name.startswith("mtp.") or any(hint in tensor_name for hint in _MTP_TENSOR_HINTS):
            return True
    return False


def has_mtp_heads_strict(repo: str, path: str | None = None) -> bool:
    """Combined check: tensor probe (when path available) ELSE name alias.

    Use this in routing logic when ``path`` is known — picks up new
    MTP-bearing repos we haven't enumerated, rejects name-collisions
    that don't actually have the tensors.
    """
    if path:
        probe = model_has_mtp_tensors(path)
        if probe is not None:
            return probe
    return has_mtp_heads(repo)
