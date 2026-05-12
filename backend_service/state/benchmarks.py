"""Benchmark orchestration for ``ChaosEngineState``.

Two helpers lifted out of ``state/__init__.py``:

* ``append_benchmark_run`` — push a fresh run onto the rolling window
  (``MAX_BENCHMARK_RUNS`` cap) + persist to disk.
* ``run_benchmark`` — orchestrate a single benchmark across the
  perplexity / task-accuracy / throughput modes. Resolves catalog +
  library context for the requested model, decides whether the
  loaded model needs swapping, runs the right eval entrypoint on the
  active engine, and returns the run + benchmarks-list + runtime
  status payload the routes layer hands back to the UI.

Both take the ``ChaosEngineState`` instance as their first argument.
The class methods become thin wrappers.

Extracted as part of the v0.8.0 Phase 1a-5 refactor.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from backend_service.helpers.formatting import _benchmark_label
from backend_service.helpers.persistence import (
    MAX_BENCHMARK_RUNS,
    _default_chat_variant,
    _save_benchmark_runs,
)
from backend_service.models import BenchmarkRunRequest, LoadModelRequest


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def append_benchmark_run(state: ChaosEngineState, run: dict[str, Any]) -> None:
    state.benchmark_runs = [
        run,
        *[item for item in state.benchmark_runs if item["id"] != run["id"]],
    ][:MAX_BENCHMARK_RUNS]
    _save_benchmark_runs(state.benchmark_runs, state._benchmarks_path)


def run_benchmark(state: ChaosEngineState, request: BenchmarkRunRequest) -> dict[str, Any]:
    from backend_service.app import compute_cache_preview

    with state._lock:
        default_variant = _default_chat_variant()
        effective_model_ref = (
            request.modelRef
            or (state.runtime.loaded_model.ref if state.runtime.loaded_model is not None else None)
            or default_variant["id"]
        )
        catalog_entry = state._find_catalog_entry(effective_model_ref)
        library_entry = state._find_library_entry(request.path, effective_model_ref)
        model_name = request.modelName
        if model_name is None and library_entry is not None:
            model_name = str(library_entry.get("name") or "")
        if model_name is None and catalog_entry is not None:
            model_name = str(catalog_entry.get("name") or "")
        if model_name is None:
            model_name = str(effective_model_ref or default_variant["name"])

        if library_entry is not None and library_entry.get("broken"):
            reason = library_entry.get("brokenReason") or "incomplete or corrupt"
            raise RuntimeError(
                f"Cannot benchmark '{library_entry.get('name') or effective_model_ref}': {reason}."
            )
        effective_source = request.source or ("library" if library_entry is not None else "catalog")
        effective_path = (
            request.path
            if request.path is not None
            else (library_entry.get("path") if library_entry is not None else None)
        )
        effective_backend = request.backend or (
            "llama.cpp"
            if (library_entry and library_entry.get("format") == "GGUF")
            or (catalog_entry and catalog_entry.get("format") == "GGUF")
            else "mlx"
        )

    load_seconds = 0.0
    effective_cache_strategy = "native" if request.speculativeDecoding else request.cacheStrategy
    effective_cache_bits = 0 if request.speculativeDecoding else request.cacheBits
    effective_fp16_layers = 0 if request.speculativeDecoding else request.fp16Layers

    if state._should_reload_for_profile(
        model_ref=effective_model_ref,
        cache_bits=effective_cache_bits,
        fp16_layers=effective_fp16_layers,
        fused_attention=request.fusedAttention,
        cache_strategy=effective_cache_strategy,
        fit_model_in_memory=request.fitModelInMemory,
        context_tokens=request.contextTokens,
        speculative_decoding=request.speculativeDecoding,
        tree_budget=0,
    ):
        load_started = time.perf_counter()
        state.load_model(
            LoadModelRequest(
                modelRef=str(effective_model_ref),
                modelName=model_name,
                canonicalRepo=state._resolve_canonical_repo(
                    model_ref=str(effective_model_ref),
                    path=effective_path,
                    canonical_repo=None,
                ),
                source=effective_source,
                backend=effective_backend,
                path=effective_path,
                cacheStrategy=request.cacheStrategy,
                cacheBits=request.cacheBits,
                fp16Layers=request.fp16Layers,
                fusedAttention=request.fusedAttention,
                fitModelInMemory=request.fitModelInMemory,
                contextTokens=request.contextTokens,
                speculativeDecoding=request.speculativeDecoding,
            )
        )
        load_seconds = round(time.perf_counter() - load_started, 2)

    with state._lock:
        params_b = (
            float(catalog_entry.get("paramsB"))
            if catalog_entry and catalog_entry.get("paramsB") is not None
            else 7.0
        )
        preview = compute_cache_preview(
            bits=request.cacheBits if request.cacheBits else 4,
            fp16_layers=request.fp16Layers,
            context_tokens=request.contextTokens,
            params_b=params_b,
            system_stats=state._system_snapshot(),
        )
        use_compressed = request.cacheBits > 0
        cache_gb = preview["optimizedCacheGb"] if use_compressed else preview["baselineCacheGb"]
        baseline_cache_gb = preview["baselineCacheGb"]
        compression = (
            round(baseline_cache_gb / cache_gb, 1) if use_compressed and cache_gb else 1.0
        )
        quality = int(round(preview["qualityPercent"])) if use_compressed else 100
        cache_label = state._cache_label(
            cache_strategy=request.cacheStrategy,
            bits=request.cacheBits,
            fp16_layers=request.fp16Layers,
        )

    base_run: dict[str, Any] = {
        "id": f"bench-{uuid.uuid4().hex[:8]}",
        "mode": request.mode,
        "model": model_name,
        "modelRef": effective_model_ref,
        "backend": state.runtime.loaded_model.backend if state.runtime.loaded_model else effective_backend,
        "engineLabel": state.runtime.engine.engine_label,
        "source": effective_source,
        "measuredAt": state._time_label(),
        "bits": request.cacheBits if request.cacheBits > 0 else 16,
        "fp16Layers": request.fp16Layers,
        "cacheStrategy": request.cacheStrategy,
        "cacheLabel": cache_label,
        "cacheGb": cache_gb,
        "baselineCacheGb": baseline_cache_gb,
        "compression": compression,
        "contextTokens": request.contextTokens,
        "maxTokens": request.maxTokens,
        "loadSeconds": load_seconds,
    }

    if request.mode == "perplexity":
        eval_result = state.runtime.engine.eval_perplexity(
            dataset=request.perplexityDataset,
            num_samples=request.perplexityNumSamples,
            seq_length=request.perplexitySeqLength,
            batch_size=request.perplexityBatchSize,
        )
        run = {
            **base_run,
            "label": request.label or f"{model_name} / Perplexity / {request.perplexityDataset}",
            "perplexity": eval_result["perplexity"],
            "perplexityStdError": eval_result["standardError"],
            "perplexityDataset": eval_result["dataset"],
            "perplexityNumSamples": eval_result["numSamples"],
            "evalTokensPerSecond": eval_result["evalTokensPerSecond"],
            "evalSeconds": eval_result["evalSeconds"],
            "quality": quality,
            "tokS": eval_result["evalTokensPerSecond"],
            "responseSeconds": eval_result["evalSeconds"],
            "totalSeconds": round(load_seconds + eval_result["evalSeconds"], 2),
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "notes": (
                f"Perplexity: {eval_result['perplexity']:.2f} ± "
                f"{eval_result['standardError']:.2f} on {eval_result['dataset']} "
                f"({eval_result['numSamples']} samples)"
            ),
        }
    elif request.mode == "task_accuracy":
        eval_result = state.runtime.engine.eval_task_accuracy(
            task_name=request.taskName,
            limit=request.taskLimit,
            num_shots=request.taskNumShots,
        )
        accuracy_pct = round(eval_result["accuracy"] * 100, 1)
        run = {
            **base_run,
            "label": (
                request.label
                or f"{model_name} / {request.taskName.upper()} / "
                f"{eval_result['correct']}/{eval_result['total']}"
            ),
            "taskName": eval_result["taskName"],
            "taskAccuracy": eval_result["accuracy"],
            "taskCorrect": eval_result["correct"],
            "taskTotal": eval_result["total"],
            "taskNumShots": eval_result["numShots"],
            "evalSeconds": eval_result["evalSeconds"],
            "quality": quality,
            "tokS": 0,
            "responseSeconds": eval_result["evalSeconds"],
            "totalSeconds": round(load_seconds + eval_result["evalSeconds"], 2),
            "promptTokens": 0,
            "completionTokens": 0,
            "totalTokens": 0,
            "notes": (
                f"{request.taskName.upper()}: {accuracy_pct}% "
                f"({eval_result['correct']}/{eval_result['total']}) "
                f"{eval_result['numShots']}-shot"
            ),
        }
    else:
        prompt = request.prompt or (
            "Summarize the practical trade-offs of this runtime profile for a local desktop user in six short bullets."
        )
        result = state.runtime.generate(
            prompt=prompt,
            history=[],
            system_prompt="Return a concise but complete answer so ChaosEngineAI can benchmark response speed consistently.",
            max_tokens=request.maxTokens,
            temperature=request.temperature,
        )
        run = {
            **base_run,
            "label": request.label
            or _benchmark_label(
                model_name,
                cache_strategy=request.cacheStrategy,
                bits=request.cacheBits,
                fp16_layers=request.fp16Layers,
                context_tokens=request.contextTokens,
            ),
            "tokS": round(result.tokS, 1),
            "quality": quality,
            "responseSeconds": round(result.responseSeconds, 2),
            "totalSeconds": round(load_seconds + result.responseSeconds, 2),
            "promptTokens": result.promptTokens,
            "completionTokens": result.completionTokens,
            "totalTokens": result.totalTokens,
            "notes": result.runtimeNote,
        }

    with state._lock:
        append_benchmark_run(state, run)
        mode_label = {"perplexity": "Perplexity", "task_accuracy": "Task accuracy"}.get(
            request.mode, "Throughput"
        )
        state.add_log(
            "benchmark",
            "info",
            f"{mode_label} benchmark completed for {model_name}: {run.get('notes', '')}",
        )
        state.add_activity("Benchmark completed", run["label"])
        return {
            "result": run,
            "benchmarks": state.benchmark_runs,
            "runtime": state.runtime.status(
                active_requests=state.active_requests,
                requests_served=state.requests_served,
            ),
        }
