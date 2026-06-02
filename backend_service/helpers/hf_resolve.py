"""Resolve an arbitrary Hugging Face repo into a loadable descriptor (#5).

Lets a user paste any GGUF / MLX repo and run it without a curated
catalog row. The previous behaviour (FU-041) fuzzy-matched off-catalog
repos against the nearest catalog entry, picking up the wrong context
window, capabilities, and DFlash drafter. This module instead reads the
repo's own file list + ``config.json`` and synthesises a descriptor, and
the caller passes ``canonicalRepo=<repo>`` to ``load_model`` so
``_resolve_canonical_repo`` returns it verbatim — no fuzzy match.

``resolve_hf_model`` is pure (no network): it takes the already-fetched
file list and optional parsed ``config.json``. The route layer fetches
those via ``_hub_repo_files`` + a best-effort ``config.json`` read.
"""

from __future__ import annotations

from typing import Any

# GGUF quantization preference when a repo ships several. Quality/size
# sweet spots first; everything unlisted sorts last but is still runnable.
_GGUF_QUANT_PRIORITY = (
    "q4_k_m",
    "q5_k_m",
    "q4_k_s",
    "q5_k_s",
    "q8_0",
    "q6_k",
    "q4_0",
    "q3_k_m",
    "iq4_nl",
)

_DEFAULT_CONTEXT = 8192
_MIN_CONTEXT = 2048
_MAX_CONTEXT = 131072

# config.json keys that carry the trained context length, most-specific
# first.
_CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "n_ctx")


def _is_gguf(path: str) -> bool:
    return path.lower().endswith(".gguf")


def _gguf_score(path: str) -> tuple[int, int]:
    """Lower is better. (quant_rank, shard_penalty)."""
    lowered = path.lower()
    quant_rank = len(_GGUF_QUANT_PRIORITY)
    for idx, tag in enumerate(_GGUF_QUANT_PRIORITY):
        if tag in lowered:
            quant_rank = idx
            break
    # Prefer a non-sharded file; if sharded, only the first shard is a
    # valid entry point for llama.cpp.
    is_shard = "-of-" in lowered
    is_first_shard = "00001-of-" in lowered
    shard_penalty = 0 if not is_shard else (1 if is_first_shard else 2)
    return (quant_rank, shard_penalty)


def _pick_gguf(gguf_paths: list[str], requested_file: str | None) -> str | None:
    if not gguf_paths:
        return None
    if requested_file and requested_file in gguf_paths:
        return requested_file
    # Drop non-first shards from contention; if every candidate is a
    # non-first shard (unusual), fall back to the full list.
    primary = [p for p in gguf_paths if "-of-" not in p.lower() or "00001-of-" in p.lower()]
    pool = primary or gguf_paths
    return sorted(pool, key=_gguf_score)[0]


def _context_from_config(config: dict[str, Any] | None) -> int | None:
    if not isinstance(config, dict):
        return None
    # Some multimodal configs nest the LM config under text_config.
    sources = [config]
    text_cfg = config.get("text_config")
    if isinstance(text_cfg, dict):
        sources.append(text_cfg)
    for src in sources:
        for key in _CONTEXT_KEYS:
            value = src.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
    return None


def _infer_capabilities(config: dict[str, Any] | None, has_mmproj: bool) -> dict[str, bool]:
    vision = has_mmproj
    if isinstance(config, dict):
        if config.get("vision_config") or config.get("image_token_id") is not None:
            vision = True
    return {"text": True, "vision": bool(vision)}


def resolve_hf_model(
    repo: str,
    *,
    files: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
    requested_file: str | None = None,
) -> dict[str, Any]:
    """Synthesise a loadable descriptor for an arbitrary HF repo.

    ``files`` are records as produced by ``_hub_repo_files`` siblings:
    ``{"path", "sizeBytes", "kind"}``. ``config`` is the parsed
    ``config.json`` when available. Never raises for a well-formed file
    list; surfaces uncertainty via ``warnings``.
    """
    paths = [str(f.get("path") or "") for f in files if f.get("path")]
    size_by_path = {str(f.get("path") or ""): int(f.get("sizeBytes") or 0) for f in files}

    gguf_paths = [p for p in paths if _is_gguf(p)]
    safetensors_paths = [p for p in paths if p.lower().endswith(".safetensors")]
    has_mmproj = any("mmproj" in p.lower() for p in paths)

    warnings: list[str] = []
    gguf_file: str | None = None

    if gguf_paths:
        backend = "llama.cpp"
        gguf_file = _pick_gguf(gguf_paths, requested_file)
        size_bytes = size_by_path.get(gguf_file or "", 0)
        if not size_bytes:
            size_bytes = sum(size_by_path.get(p, 0) for p in gguf_paths)
    elif repo.startswith("mlx-community/") or _looks_like_mlx(config):
        backend = "mlx"
        size_bytes = sum(size_by_path.get(p, 0) for p in safetensors_paths)
    elif safetensors_paths:
        # Raw (non-MLX) safetensors: runnable only via a CUDA backend or
        # after conversion. Surface it honestly rather than guessing.
        backend = "vllm"
        size_bytes = sum(size_by_path.get(p, 0) for p in safetensors_paths)
        warnings.append(
            "This repo ships raw safetensors weights (no GGUF, not an MLX conversion). "
            "On Apple Silicon, convert it to MLX or pick a GGUF mirror; the vLLM backend "
            "is CUDA-only."
        )
    else:
        backend = "unknown"
        size_bytes = sum(size_by_path.values())
        warnings.append("No GGUF or safetensors weights found in this repo.")

    ctx_from_config = _context_from_config(config)
    if ctx_from_config is not None:
        context_tokens = max(_MIN_CONTEXT, min(_MAX_CONTEXT, ctx_from_config))
    else:
        context_tokens = _DEFAULT_CONTEXT
        if backend == "llama.cpp":
            warnings.append(
                f"Context length not read from metadata; defaulting to {_DEFAULT_CONTEXT}. "
                "Adjust in launch settings if the model supports more."
            )

    return {
        "repo": repo,
        "ref": repo,
        "label": repo.split("/")[-1],
        "backend": backend,
        "ggufFile": gguf_file,
        "contextTokens": context_tokens,
        "capabilities": _infer_capabilities(config, has_mmproj),
        "sizeBytes": size_bytes,
        "family": "custom",
        "custom": True,
        "warnings": warnings,
    }


def _looks_like_mlx(config: dict[str, Any] | None) -> bool:
    """Heuristic: an MLX-converted repo carries an MLX quantization stanza."""
    if not isinstance(config, dict):
        return False
    if "quantization" in config and isinstance(config["quantization"], dict):
        # mlx-lm writes {"group_size": N, "bits": M} under "quantization".
        q = config["quantization"]
        if "group_size" in q or "bits" in q:
            return True
    return False
