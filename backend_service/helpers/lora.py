"""Fuse a catalog LoRA into a diffusers pipeline."""

from __future__ import annotations

from typing import Any


def fuse_catalog_lora(pipeline: Any, repo_id: str, filename: str, scale: float | None) -> str:
    """Fetch ``filename`` from ``repo_id``, fuse it into ``pipeline`` and return a load note.

    The load is cache-only and nothing downloads a catalog LoRA ahead of time, so
    the first run fetches it here (a cache hit costs no network). Raises when the
    file cannot be fetched or fused; callers keep generating without the LoRA and
    report the error as a note.
    """
    from backend_service.helpers.hf_local import fetch_file

    fetch_file(repo_id, filename)
    pipeline.load_lora_weights(repo_id, weight_name=filename, local_files_only=True)
    effective_scale = float(scale) if scale is not None else 1.0
    pipeline.fuse_lora(lora_scale=effective_scale)
    try:
        pipeline.unload_lora_weights()
    except Exception:
        # Older diffusers do not always manage this after a fuse; the fused
        # weights are correct either way.
        pass
    return f"LoRA: {repo_id}/{filename} @ scale {effective_scale:.3f}"
