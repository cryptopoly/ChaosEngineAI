"""Cache estimation and preview computations."""
from __future__ import annotations

from typing import Any

from backend_service.helpers.formatting import _bytes_to_gb


# Decode reads the active weights (and the KV cache) once per token, so its
# speed is bandwidth / bytes per token, plus a fixed per-token cost (kernel
# launches, sampling; larger for mixture-of-experts routing). Fitted on an M4
# Max (546 GB/s) against this machine's median chat speeds, 2026-10-20
# (measured / estimate, tok/s): Qwen3-4B bf16 52 / 51, Qwen3-0.6B 4-bit
# 346 / 350, Qwen3-0.6B GGUF 311 / 325, Qwen2.5-0.5B 4-bit 467 / 382,
# Bonsai 1.7B Q1_0 328 / 375, Ternary Bonsai 27B 42 / 49, Qwen3.6-35B-A3B
# 4-bit 94 / 109, Gemma 4 26B-A4B bf16 52 / 43. Within about 20% except the
# smallest models, where per-token overhead dominates.
_DECODE_BANDWIDTH_EFFICIENCY = 0.84
_TOKEN_OVERHEAD_SECONDS_DENSE = 0.002
_TOKEN_OVERHEAD_SECONDS_MOE = 0.006
# KV cache read per token, sized at a typical chat turn rather than the
# full window. 2K keeps small models honest: Ternary Bonsai 4B measured 213
# tok/s (catalog smoke run) against 234 estimated here, but 144 at 8K.
_TYPICAL_KV_TOKENS = 2048


def estimate_decode_tok_s(
    *,
    weights_gb: float,
    active_fraction: float,
    kv_bytes_per_token_read: float,
    bandwidth_gbps: float,
) -> float:
    """Tokens per second for single-stream decode with the native cache."""
    bytes_per_token = weights_gb * 1e9 * max(0.01, min(1.0, active_fraction)) + kv_bytes_per_token_read
    overhead = _TOKEN_OVERHEAD_SECONDS_MOE if active_fraction < 0.9 else _TOKEN_OVERHEAD_SECONDS_DENSE
    seconds = bytes_per_token / (_DECODE_BANDWIDTH_EFFICIENCY * bandwidth_gbps * 1e9) + overhead
    return 1.0 / seconds


def _strategy_speed_map(strategy: str) -> dict[int, float]:
    """Speed ratio maps by strategy and bit count (fraction of baseline FP16 speed).

    TurboQuant is a memory saver, not a speed option. Measured with
    ``scripts/calibrate-cache-speed.py`` on Qwen3-4B-bf16, M4 Max
    (2026-10-01): Native 54.3 / 50.1 tok/s at 512 / 4096 prompt tokens against
    9.8-10.7 / 6.4-6.5 for 2-4 bits, so 0.18-0.20 at short context and 0.13 at
    4K. The map holds the median of the two; 1 bit is unmeasured and kept at
    the 2-bit level. Decode is flat across bit widths, so the ratios are too.
    The TriAttention row is an unmeasured estimate.
    """
    maps: dict[str, dict[int, float]] = {
        "triattention": {1: 0.48, 2: 0.56, 3: 0.63, 4: 0.70},
        "turboquant":   {1: 0.16, 2: 0.16, 3: 0.15, 4: 0.16},
    }
    return maps.get(strategy, {1: 0.45, 2: 0.53, 3: 0.59, 4: 0.68})


def _strategy_quality_base(strategy: str) -> dict[int, float]:
    """Base quality percentage by strategy and bit count (before fp16_layers bonus)."""
    maps: dict[str, dict[int, float]] = {
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
    arch=None,
    weights_gb: float | None = None,
    measured_tok_s: float | None = None,
) -> dict[str, Any]:
    """Cache size, speed and quality preview for one launch configuration.

    ``arch`` is the model's real attention layout (``helpers.model_arch``).
    When given, the f16 cache is sized from it and a compressed strategy keeps
    its own ratio; otherwise the layer / head arguments are an estimate.

    ``system_stats`` / ``build_system_snapshot`` are accepted for older
    callers and no longer used.

    Speed: ``measured_tok_s`` (this machine's own history for the model) wins;
    otherwise it is estimated from memory bandwidth and the bytes read per
    token (``weights_gb`` on disk, scaled by the active-parameter share).
    """
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

    if arch is not None:
        # Real layout: size the f16 cache from it and keep the strategy's own
        # compression ratio from the nominal estimate above.
        ratio = (optimized_bytes / baseline_bytes) if baseline_bytes else 1.0
        baseline_bytes = arch.cache_bytes(context_tokens)
        optimized_bytes = int(baseline_bytes * ratio)
        num_layers = arch.num_layers
        if arch.kv_heads:
            num_kv_heads = arch.kv_heads

    from backend_service.helpers.bandwidth import memory_bandwidth

    bandwidth_gbps, bandwidth_label, bandwidth_known = memory_bandwidth()
    kv_tokens = min(context_tokens, _TYPICAL_KV_TOKENS)
    kv_read = (
        arch.cache_bytes(kv_tokens)
        if arch is not None
        else baseline_bytes * kv_tokens / max(context_tokens, 1)
    )
    if measured_tok_s and measured_tok_s > 0:
        baseline_tok_s = measured_tok_s
        speed_source = "measured"
    else:
        baseline_tok_s = estimate_decode_tok_s(
            # Without the file size, assume a 4-bit build (~0.56 GB per billion).
            weights_gb=weights_gb if weights_gb and weights_gb > 0 else params_b * 0.56,
            active_fraction=arch.active_fraction if arch is not None else 1.0,
            kv_bytes_per_token_read=kv_read,
            bandwidth_gbps=bandwidth_gbps,
        )
        speed_source = "estimate"
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
        # "config" / "gguf" when the cache was sized from the model's own
        # files, "estimate" when it was guessed from the parameter count.
        "archSource": arch.source if arch is not None else "estimate",
        # "measured" when the speed comes from this machine's own runs of the
        # model, "estimate" when it comes from memory bandwidth.
        "speedSource": speed_source,
        "bandwidthGbps": bandwidth_gbps,
        "bandwidthLabel": bandwidth_label,
        "bandwidthKnown": bandwidth_known,
    }
