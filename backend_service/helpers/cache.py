"""Cache estimation and preview computations."""
from __future__ import annotations

from typing import Any

import psutil

from backend_service.helpers.formatting import _bytes_to_gb


def _estimate_baseline_tok_s(system_stats: dict[str, Any]) -> float:
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count() or 8
    baseline = 15.0 + cpu_count * 1.1 + system_stats["totalMemoryGb"] * 0.5
    return round(baseline, 1)


def _strategy_speed_map(strategy: str) -> dict[int, float]:
    """Speed ratio maps by strategy and bit count (fraction of baseline FP16 speed)."""
    maps: dict[str, dict[int, float]] = {
        "rotorquant":   {1: 0.42, 2: 0.50, 3: 0.57, 4: 0.65},
        "triattention": {1: 0.48, 2: 0.56, 3: 0.63, 4: 0.70},
        "turboquant":   {1: 0.44, 2: 0.52, 3: 0.60, 4: 0.67},
    }
    return maps.get(strategy, {1: 0.45, 2: 0.53, 3: 0.59, 4: 0.68})


def _strategy_quality_base(strategy: str) -> dict[int, float]:
    """Base quality percentage by strategy and bit count (before fp16_layers bonus)."""
    maps: dict[str, dict[int, float]] = {
        "rotorquant":   {1: 88.0, 2: 91.0, 3: 93.5, 4: 96.0},
        "triattention": {1: 89.5, 2: 92.0, 3: 94.5, 4: 97.0},
        "turboquant":   {1: 87.5, 2: 90.5, 3: 93.0, 4: 95.5},
    }
    return maps.get(strategy, {1: 87.0, 2: 90.0, 3: 92.0, 4: 95.6})


def compute_cache_preview(
    *,
    bits: int = 3,
    fp16_layers: int = 4,
    num_layers: int = 32,
    num_heads: int = 32,
    num_kv_heads: int | None = None,
    hidden_size: int = 4096,
    context_tokens: int = 8192,
    params_b: float = 7.0,
    system_stats: dict[str, Any] | None = None,
    strategy: str = "native",
    build_system_snapshot=None,
) -> dict[str, Any]:
    from cache_compression import registry as _cache_registry

    num_layers = max(1, num_layers)
    num_heads = max(1, num_heads)
    if num_kv_heads is not None:
        num_kv_heads = max(1, min(num_kv_heads, num_heads))
    hidden_size = max(num_heads, hidden_size)
    context_tokens = max(256, context_tokens)

    strat = _cache_registry.get(strategy) or _cache_registry.default()

    if strategy == "native" or bits <= 0:
        # Native FP16: no compression
        baseline_bytes, optimized_bytes = strat.estimate_cache_bytes(
            num_layers, num_heads, hidden_size, context_tokens, 0, 0, num_kv_heads,
        )
        compression_ratio = 1.0
        speed_ratio = 1.0
        quality_percent = 100.0
        bits = 0
    else:
        bits = max(1, min(bits, 8))
        baseline_bytes, optimized_bytes = strat.estimate_cache_bytes(
            num_layers, num_heads, hidden_size, context_tokens, bits, fp16_layers, num_kv_heads,
        )
        compression_ratio = baseline_bytes / optimized_bytes if optimized_bytes else 1.0

        speed_map = _strategy_speed_map(strategy)
        clamped_bits = max(min(bits, max(speed_map.keys())), min(speed_map.keys()))
        speed_ratio = speed_map.get(clamped_bits, 0.6) + min(fp16_layers, 8) * 0.012
        speed_ratio = min(speed_ratio, 0.92)

        quality_map = _strategy_quality_base(strategy)
        quality_percent = min(99.5, quality_map.get(clamped_bits, 94.0) + min(fp16_layers, 8) * 0.35)

    if system_stats is None and build_system_snapshot is not None:
        preview_system = build_system_snapshot()
    else:
        preview_system = system_stats or {}
    baseline_tok_s = _estimate_baseline_tok_s(preview_system)
    model_scale = min(1.55, max(0.2, (7.0 / max(params_b, 1.0)) ** 0.38))
    baseline_tok_s *= model_scale
    estimated_tok_s = round(baseline_tok_s * speed_ratio, 1)

    # Estimate on-disk size: BF16 models are ~2 bytes/param, 4-bit quant ~0.5 bytes/param.
    disk_size_gb = round(params_b * 1.2, 1)

    strat_label = strat.label(bits, fp16_layers) if bits > 0 else "Native f16"
    if strategy == "native" or bits <= 0:
        summary = (
            f"Native f16 cache uses {_bytes_to_gb(baseline_bytes):.1f} GB "
            f"with an estimated {estimated_tok_s:.1f} tok/s on this machine."
        )
    else:
        summary = (
            f"{strat_label} lowers cache use to "
            f"{_bytes_to_gb(optimized_bytes):.1f} GB from {_bytes_to_gb(baseline_bytes):.1f} GB, "
            f"about {compression_ratio:.1f}x smaller, with an estimated {estimated_tok_s:.1f} tok/s on this machine."
        )

    return {
        "bits": bits,
        "fp16Layers": fp16_layers,
        "numLayers": num_layers,
        "numHeads": num_heads,
        "numKvHeads": num_kv_heads or num_heads,
        "hiddenSize": hidden_size,
        "contextTokens": context_tokens,
        "paramsB": params_b,
        "baselineCacheGb": _bytes_to_gb(baseline_bytes),
        "optimizedCacheGb": _bytes_to_gb(optimized_bytes),
        "compressionRatio": round(compression_ratio, 1),
        "estimatedTokS": estimated_tok_s,
        "speedRatio": round(speed_ratio, 2),
        "qualityPercent": round(quality_percent, 1),
        "diskSizeGb": disk_size_gb,
        "summary": summary,
    }
