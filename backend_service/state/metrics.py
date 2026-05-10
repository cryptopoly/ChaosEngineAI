"""Metrics + cache-profile helpers extracted from state.py.

Two responsibilities:

1. **Cache labels** — turn ``(strategy_id, bits, fp16_layers)`` into the
   human-readable string the UI shows ("Native f16 cache", "TurboQ
   3-bit 4+4", etc.). Cache strategies registered in
   ``cache_compression.registry`` get their label from the strategy
   itself; everything else falls back to the native naming.

2. **Profile-change detection** — given a freshly-loaded model and a
   set of requested runtime params, return the list of differences
   ("cache bits 8 -> 4", "fp16 layers 0 -> 4", ...). The chat path
   uses this to decide whether a request needs a full reload or can
   reuse the in-memory model.

3. **Metrics payloads** — assemble the per-turn metrics dict that the
   frontend's reasoning panel + perf strip depend on. Pure assembly;
   callers pass in the runtime + result objects.

All functions are pure (modulo the optional ``cache_compression``
registry lookup for label resolution). They were lifted off
``ChaosEngineState`` because nothing here needs the lock, the session
list, or any other instance state.
"""

from __future__ import annotations

import time
from typing import Any


def _time_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Cache labels
# ---------------------------------------------------------------------------


def native_cache_strategy_label(bits: int, fp16_layers: int) -> str:
    if bits and bits < 16:
        return f"Native {bits}-bit {fp16_layers}+{fp16_layers}"
    return "Native f16 cache"


def cache_label(*, cache_strategy: str, bits: int, fp16_layers: int) -> str:
    """Resolve a cache strategy id to its UI label."""
    from cache_compression import registry as cache_registry

    strategy = cache_registry.get(cache_strategy)
    if strategy is not None:
        return strategy.label(bits, fp16_layers)
    return native_cache_strategy_label(bits, fp16_layers)


# ---------------------------------------------------------------------------
# Profile-change detection
# ---------------------------------------------------------------------------


def cache_profile_change_reasons(
    loaded_model: Any | None,
    *,
    cache_bits: int,
    fp16_layers: int,
    fused_attention: bool,
    cache_strategy: str,
) -> list[str]:
    if loaded_model is None:
        return []

    changes: list[str] = []
    if loaded_model.cacheStrategy != cache_strategy:
        changes.append(f"cache strategy {loaded_model.cacheStrategy} -> {cache_strategy}")
    if loaded_model.cacheBits != cache_bits:
        changes.append(f"cache bits {loaded_model.cacheBits} -> {cache_bits}")
    if loaded_model.fp16Layers != fp16_layers:
        changes.append(f"fp16 layers {loaded_model.fp16Layers} -> {fp16_layers}")
    if loaded_model.fusedAttention != fused_attention:
        changes.append(
            f"fused attention {'on' if loaded_model.fusedAttention else 'off'} -> {'on' if fused_attention else 'off'}"
        )
    return changes


def runtime_profile_change_reasons(
    loaded_model: Any | None,
    *,
    cache_bits: int,
    fp16_layers: int,
    fused_attention: bool,
    cache_strategy: str,
    fit_model_in_memory: bool,
    context_tokens: int,
    speculative_decoding: bool,
    tree_budget: int,
) -> list[str]:
    if loaded_model is None:
        return []

    changes: list[str] = []
    if loaded_model.contextTokens != context_tokens:
        changes.append(f"context {loaded_model.contextTokens} -> {context_tokens}")
    changes.extend(
        cache_profile_change_reasons(
            loaded_model,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
            cache_strategy=cache_strategy,
        )
    )
    if loaded_model.fitModelInMemory != fit_model_in_memory:
        changes.append(
            f"fit-in-memory {'on' if loaded_model.fitModelInMemory else 'off'} -> {'on' if fit_model_in_memory else 'off'}"
        )
    if bool(loaded_model.speculativeDecoding) != speculative_decoding:
        changes.append(
            f"speculative decoding {'on' if loaded_model.speculativeDecoding else 'off'} -> {'on' if speculative_decoding else 'off'}"
        )
    if int(loaded_model.treeBudget or 0) != int(tree_budget):
        changes.append(f"tree budget {loaded_model.treeBudget or 0} -> {tree_budget}")
    return changes


# ---------------------------------------------------------------------------
# Metrics payloads
# ---------------------------------------------------------------------------


def loaded_model_metrics_fields(runtime: Any) -> dict[str, Any]:
    loaded = runtime.loaded_model
    return {
        "model": loaded.name if loaded else None,
        "modelRef": loaded.ref if loaded else None,
        "canonicalRepo": loaded.canonicalRepo if loaded else None,
        "backend": loaded.backend if loaded else None,
        "engineLabel": runtime.engine.engine_label,
        "cacheLabel": cache_label(
            cache_strategy=str(loaded.cacheStrategy) if loaded else "native",
            bits=int(loaded.cacheBits) if loaded else 0,
            fp16_layers=int(loaded.fp16Layers) if loaded else 0,
        ),
        "cacheStrategy": loaded.cacheStrategy if loaded else None,
        "cacheBits": loaded.cacheBits if loaded else None,
        "fp16Layers": loaded.fp16Layers if loaded else None,
        "fusedAttention": loaded.fusedAttention if loaded else None,
        "fitModelInMemory": loaded.fitModelInMemory if loaded else None,
        "speculativeDecoding": loaded.speculativeDecoding if loaded else None,
        "dflashDraftModel": loaded.dflashDraftModel if loaded else None,
        "modelSource": loaded.source if loaded else None,
        "modelPath": loaded.path if loaded else None,
        "contextTokens": loaded.contextTokens if loaded else None,
        "treeBudget": loaded.treeBudget if loaded else 0,
        "generatedAt": _time_label(),
    }


def requested_runtime_metrics_fields(
    *,
    cache_strategy: str,
    cache_bits: int,
    fp16_layers: int,
    fit_model_in_memory: bool,
    speculative_decoding: bool,
    tree_budget: int,
) -> dict[str, Any]:
    return {
        "requestedCacheLabel": cache_label(
            cache_strategy=cache_strategy,
            bits=cache_bits,
            fp16_layers=fp16_layers,
        ),
        "requestedCacheStrategy": cache_strategy,
        "requestedCacheBits": cache_bits,
        "requestedFp16Layers": fp16_layers,
        "requestedFitModelInMemory": fit_model_in_memory,
        "requestedSpeculativeDecoding": speculative_decoding,
        "requestedTreeBudget": tree_budget,
    }


def result_runtime_metrics_fields(result: Any | None) -> dict[str, Any]:
    if result is None:
        return {}
    metrics: dict[str, Any] = {}
    cache_strategy = getattr(result, "cache_strategy", None)
    if cache_strategy is not None:
        cache_bits = int(getattr(result, "cache_bits", 0) or 0)
        fp16_layers = int(getattr(result, "fp16_layers", 0) or 0)
        metrics.update({
            "cacheLabel": cache_label(
                cache_strategy=str(cache_strategy),
                bits=cache_bits,
                fp16_layers=fp16_layers,
            ),
            "cacheStrategy": str(cache_strategy),
            "cacheBits": cache_bits,
            "fp16Layers": fp16_layers,
        })
    speculative_decoding = getattr(result, "speculative_decoding", None)
    if speculative_decoding is not None:
        metrics["speculativeDecoding"] = bool(speculative_decoding)
        metrics["treeBudget"] = int(getattr(result, "tree_budget", 0) or 0)
        if not speculative_decoding:
            metrics["dflashDraftModel"] = None
    return metrics


def assistant_metrics_payload(
    runtime: Any,
    result: Any,
    *,
    requested_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        **loaded_model_metrics_fields(runtime),
        **result_runtime_metrics_fields(result),
        **result.to_metrics(),
        **(requested_runtime or {}),
    }


def stream_assistant_metrics_payload(
    runtime: Any,
    *,
    final_chunk: Any,
    tok_s: float,
    response_seconds: float,
    requested_runtime: dict[str, Any] | None = None,
    ttft_seconds: float | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "finishReason": final_chunk.finish_reason if final_chunk else "stop",
        "promptTokens": final_chunk.prompt_tokens if final_chunk else 0,
        "completionTokens": final_chunk.completion_tokens if final_chunk else 0,
        "totalTokens": (final_chunk.prompt_tokens + final_chunk.completion_tokens) if final_chunk else 0,
        "tokS": tok_s,
        "responseSeconds": response_seconds,
        "runtimeNote": final_chunk.runtime_note if final_chunk else None,
    }
    if final_chunk and getattr(final_chunk, "dflash_acceptance_rate", None) is not None:
        metrics["dflashAcceptanceRate"] = final_chunk.dflash_acceptance_rate
    if ttft_seconds is not None:
        metrics["ttftSeconds"] = ttft_seconds
    accepted_spans = getattr(final_chunk, "accepted_spans", None) if final_chunk else None
    if accepted_spans:
        metrics["acceptedSpans"] = accepted_spans
    accepted_token_text = getattr(final_chunk, "accepted_token_text", None) if final_chunk else None
    if accepted_token_text:
        metrics["acceptedTokenText"] = accepted_token_text

    # Best-effort perf telemetry capture; never block finalisation.
    try:
        from backend_service.helpers.perf import snapshot_perf_telemetry
        telemetry = snapshot_perf_telemetry()
        if not telemetry.is_empty:
            metrics["perfTelemetry"] = telemetry.to_dict()
    except Exception:
        pass

    return {
        **loaded_model_metrics_fields(runtime),
        **result_runtime_metrics_fields(final_chunk),
        **metrics,
        **(requested_runtime or {}),
    }


def should_reload_for_profile(
    runtime: Any,
    *,
    model_ref: str | None,
    cache_bits: int,
    fp16_layers: int,
    fused_attention: bool,
    cache_strategy: str,
    fit_model_in_memory: bool,
    context_tokens: int,
    speculative_decoding: bool,
    tree_budget: int,
) -> bool:
    loaded = runtime.loaded_model
    if model_ref and (
        loaded is None
        or model_ref not in {loaded.ref, loaded.runtimeTarget}
    ):
        return True

    if loaded is None:
        return True

    return bool(
        runtime_profile_change_reasons(
            loaded,
            cache_bits=cache_bits,
            fp16_layers=fp16_layers,
            fused_attention=fused_attention,
            cache_strategy=cache_strategy,
            fit_model_in_memory=fit_model_in_memory,
            context_tokens=context_tokens,
            speculative_decoding=speculative_decoding,
            tree_budget=tree_budget,
        )
    )
