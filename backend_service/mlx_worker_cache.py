"""Cache profile + runtime-fields helpers for the MLX worker.

Two pure helpers lifted out of ``WorkerState``:

* ``runtime_fields`` — assemble the ``cacheStrategy`` / ``cacheBits`` /
  ``fp16Layers`` / ``speculativeDecoding`` / ``treeBudget`` block that
  the parent process expects in every generation response. When the
  prompt cache is missing or the strategy is ``native`` the helper
  zeros the related fields so the UI doesn't show stale numbers.
* ``make_mlx_cache`` — instantiate the prompt-cache object the active
  ``CacheStrategy`` exposes via ``make_mlx_cache``. Returns
  ``(cache, note)`` — ``cache=None`` + ``note=<reason>`` signals the
  caller to fall back to native f16 cache.

Extracted from ``backend_service/mlx_worker.py`` as part of the v0.8.0
refactor. ``WorkerState._runtime_fields`` / ``_make_cache`` are now
thin wrappers.
"""

from __future__ import annotations

from typing import Any


def runtime_fields(
    *,
    cache_strategy: str,
    cache_bits: int,
    fp16_layers: int,
    prompt_cache: Any | None,
    speculative_decoding: bool = False,
    tree_budget: int = 0,
) -> dict[str, Any]:
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


def make_mlx_cache(
    *,
    model: Any,
    cache_strategy: str,
    cache_bits: int,
    fp16_layers: int,
    fused_attention: bool,
) -> tuple[Any | None, str | None]:
    """Build the prompt cache for the active strategy. Returns (cache, note)."""
    from cache_compression import registry
    strategy = registry.get(cache_strategy)
    if strategy is None or cache_strategy == "native":
        return None, None
    try:
        cache = strategy.make_mlx_cache(
            len(getattr(model, "layers", [])),
            bits=cache_bits,
            fp16_layers=fp16_layers,
            fused=fused_attention,
            model=model,
        )
        return cache, None
    except (ValueError, NotImplementedError) as exc:
        return None, (
            f"Cache strategy '{strategy.name}' is unavailable for this MLX architecture, "
            f"so generation fell back to native f16 cache. ({exc})"
        )
