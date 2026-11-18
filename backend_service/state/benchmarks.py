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

import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend_service.helpers.formatting import _benchmark_label
from backend_service.helpers.persistence import (
    MAX_BENCHMARK_RUNS,
    _default_chat_variant,
    _save_benchmark_runs,
)
from backend_service.models import BenchmarkRunRequest, LoadModelRequest
from backend_service.progress import BENCHMARK_PROGRESS


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def append_benchmark_run(state: ChaosEngineState, run: dict[str, Any]) -> None:
    state.benchmark_runs = [
        run,
        *[item for item in state.benchmark_runs if item["id"] != run["id"]],
    ][:MAX_BENCHMARK_RUNS]
    _save_benchmark_runs(state.benchmark_runs, state._benchmarks_path)


def _local_kv_architecture(path: str | None) -> Any:
    """Attention layout of the benchmarked model, read from its own files.

    ``None`` (no path, or a layout that cannot be read) leaves the cache size
    to the parameter-count guess, as before.
    """
    if not path:
        return None
    try:
        from backend_service.helpers.model_arch import read_kv_architecture
        from backend_service.inference.conversion import _resolve_local_model_dir

        # A Hugging Face cache folder keeps config.json under snapshots/<revision>/.
        return read_kv_architecture(path) or read_kv_architecture(_resolve_local_model_dir(path))
    except Exception:  # noqa: BLE001 - sizing is best effort, the run itself must not fail on it
        return None


def delete_benchmark_run(state: ChaosEngineState, run_id: str) -> bool:
    """Drop one run from the saved history. False when no run has that id."""
    with state._lock:
        kept = [item for item in state.benchmark_runs if item.get("id") != run_id]
        if len(kept) == len(state.benchmark_runs):
            return False
        state.benchmark_runs = kept
        _save_benchmark_runs(state.benchmark_runs, state._benchmarks_path)
        return True


class BenchmarkCancelled(RuntimeError):
    """The user stopped the run. Nothing is saved for it."""


class BenchmarkBusy(RuntimeError):
    """A benchmark is already running; the app runs one at a time."""


@dataclass
class _ActiveRun:
    """The one benchmark in flight: what ``cancel_benchmark`` needs to stop it."""

    cancel_file: Path
    # True once this run started loading a model: that model is unloaded on cancel.
    # A model that was already loaded with the same profile is left alone when the
    # run can stop between items.
    loaded_by_benchmark: bool = False
    # True when cancel ended the model process to stop the run (see ``cancel_benchmark``).
    hard_stopped: bool = False
    done: threading.Event = field(default_factory=threading.Event)


_RUN_LOCK = threading.Lock()
_ACTIVE_RUN: _ActiveRun | None = None

# How long a cancel waits for the worker to stop between two items before it ends the model process.
_COOPERATIVE_CANCEL_SECONDS = 8.0
# Phases in which the MLX worker checks the cancel file between items.
_COOPERATIVE_PHASES = {"dataset", "evaluating"}

CANCELLED_MESSAGE = "Benchmark cancelled. Nothing was saved."
CANCELLED_UNLOADED_NOTE = "The model was unloaded."
CANCELLED_KEPT_NOTE = "The model that was loaded before the run stays loaded."


def cancel_benchmark(state: ChaosEngineState) -> bool:
    """Stop the benchmark in flight. True when one was told to stop.

    Perplexity and task accuracy stop between two items (the worker reads a
    cancel file), so a model that was loaded before the run survives. A model
    load or a throughput generation cannot be interrupted from outside, so
    there the model process is ended: that is also what unloads a model the
    run loaded itself. Takes no state lock: the run thread may hold it.
    """
    run = _ACTIVE_RUN
    if run is None or not BENCHMARK_PROGRESS.request_cancel():
        return False
    try:
        run.cancel_file.touch()
    except OSError:
        pass
    cooperative = not run.loaded_by_benchmark and BENCHMARK_PROGRESS.phase() in _COOPERATIVE_PHASES
    if cooperative and run.done.wait(_COOPERATIVE_CANCEL_SECONDS):
        return True
    if not run.done.is_set():
        run.hard_stopped = True
        try:
            state.runtime.unload_model()
        except Exception:  # noqa: BLE001 - the run thread unloads again and reports
            pass
    return True


def _raise_if_cancelled() -> None:
    if BENCHMARK_PROGRESS.is_cancelled():
        raise BenchmarkCancelled(CANCELLED_MESSAGE)


def _worker_progress(payload: dict[str, Any]) -> None:
    """Progress lines from the MLX worker's eval loops, copied onto the tracker."""
    phase = str(payload.get("phase") or "")
    if phase in {"dataset", "evaluating"} and BENCHMARK_PROGRESS.phase() != phase:
        BENCHMARK_PROGRESS.set_phase(phase, str(payload.get("message") or ""))
    step, total = payload.get("step"), payload.get("total")
    if isinstance(step, int) and isinstance(total, int):
        BENCHMARK_PROGRESS.set_step(step, total)
        if payload.get("message"):
            BENCHMARK_PROGRESS.set_message(str(payload["message"]))
    correct = payload.get("correct")
    if isinstance(correct, int):
        BENCHMARK_PROGRESS.set_detail(correct=correct)


def run_benchmark(state: ChaosEngineState, request: BenchmarkRunRequest) -> dict[str, Any]:
    """Run one benchmark; a second request while one runs is refused.

    A cancelled run answers ``{"cancelled": True, "result": None, ...}`` and
    leaves no entry in the history.
    """
    global _ACTIVE_RUN
    if not _RUN_LOCK.acquire(blocking=False):
        raise BenchmarkBusy("A benchmark is already running. Wait for it to finish or cancel it first.")
    run = _ActiveRun(cancel_file=Path(tempfile.gettempdir()) / f"chaosengine-benchmark-cancel-{uuid.uuid4().hex}")
    try:
        BENCHMARK_PROGRESS.begin(run_label=request.mode, phase="preparing")
        _ACTIVE_RUN = run
        try:
            return _run_benchmark(state, request, run)
        except BenchmarkCancelled:
            return _cancelled_response(state, run)
        except Exception:
            if BENCHMARK_PROGRESS.is_cancelled():
                # Ending the model process makes the load or the generation fail: that is the cancel, not an error.
                return _cancelled_response(state, run)
            raise
    finally:
        run.done.set()
        _ACTIVE_RUN = None
        BENCHMARK_PROGRESS.finish()
        try:
            run.cancel_file.unlink(missing_ok=True)
        except OSError:
            pass
        _RUN_LOCK.release()


def _cancelled_response(state: ChaosEngineState, run: _ActiveRun) -> dict[str, Any]:
    model_unloaded = run.loaded_by_benchmark or run.hard_stopped
    if model_unloaded:
        try:
            state.runtime.unload_model()
        except Exception as exc:  # noqa: BLE001 - report the cancel either way
            state.add_log("benchmark", "warning", f"Unload after a cancelled benchmark failed: {exc}")
    note = CANCELLED_UNLOADED_NOTE if model_unloaded else CANCELLED_KEPT_NOTE
    with state._lock:
        state.add_log("benchmark", "info", f"{CANCELLED_MESSAGE} {note}")
        return {
            "cancelled": True,
            "modelUnloaded": model_unloaded,
            "message": f"{CANCELLED_MESSAGE} {note}",
            "result": None,
            "benchmarks": state.benchmark_runs,
            "runtime": state.runtime.status(
                active_requests=state.active_requests,
                requests_served=state.requests_served,
            ),
        }


def _run_benchmark(state: ChaosEngineState, request: BenchmarkRunRequest, run: _ActiveRun) -> dict[str, Any]:
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
        _raise_if_cancelled()
        run.loaded_by_benchmark = True
        BENCHMARK_PROGRESS.set_phase("loading", f"Loading {model_name}")
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
    _raise_if_cancelled()

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
            # Sized from the model's own config, like the preview next to the Run button. Without it a
            # model outside the catalog was sized as a 7B one (4 GB of cache reported for a 0.6B model).
            arch=_local_kv_architecture(effective_path),
        )
        use_compressed = request.cacheBits > 0
        cache_gb = preview["optimizedCacheGb"] if use_compressed else preview["baselineCacheGb"]
        baseline_cache_gb = preview["baselineCacheGb"]
        compression = (
            round(baseline_cache_gb / cache_gb, 1) if use_compressed and cache_gb else 1.0
        )
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
        BENCHMARK_PROGRESS.set_phase("dataset", "Loading the evaluation text")
        eval_result = state.runtime.engine.eval_perplexity(
            dataset=request.perplexityDataset,
            num_samples=request.perplexityNumSamples,
            seq_length=request.perplexitySeqLength,
            batch_size=request.perplexityBatchSize,
            progress_callback=_worker_progress,
            cancel_file=str(run.cancel_file),
        )
        run_record = {
            **base_run,
            "label": request.label or f"{model_name} / Perplexity / {request.perplexityDataset}",
            "perplexity": eval_result["perplexity"],
            "perplexityStdError": eval_result["standardError"],
            "perplexityDataset": eval_result["dataset"],
            "perplexityNumSamples": eval_result["numSamples"],
            "evalTokensPerSecond": eval_result["evalTokensPerSecond"],
            "evalSeconds": eval_result["evalSeconds"],
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
        BENCHMARK_PROGRESS.set_phase("dataset", "Loading the questions")
        eval_result = state.runtime.engine.eval_task_accuracy(
            task_name=request.taskName,
            limit=request.taskLimit,
            num_shots=request.taskNumShots,
            progress_callback=_worker_progress,
            cancel_file=str(run.cancel_file),
        )
        accuracy_pct = round(eval_result["accuracy"] * 100, 1)
        run_record = {
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
            # How the answer was picked. Runs saved before this field used "generated"
            # (three generated tokens searched for a letter); the UI says so when two runs differ.
            "taskScoringMethod": eval_result.get("scoringMethod") or "generated",
            "evalSeconds": eval_result["evalSeconds"],
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
        # One blocking call: the engine reports no token count until it returns.
        BENCHMARK_PROGRESS.set_phase("generating", f"Generating up to {request.maxTokens} tokens")
        result = state.runtime.generate(
            prompt=prompt,
            history=[],
            system_prompt="Return a concise but complete answer so ChaosEngineAI can benchmark response speed consistently.",
            max_tokens=request.maxTokens,
            temperature=request.temperature,
        )
        run_record = {
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
            "responseSeconds": round(result.responseSeconds, 2),
            "totalSeconds": round(load_seconds + result.responseSeconds, 2),
            "promptTokens": result.promptTokens,
            "completionTokens": result.completionTokens,
            "totalTokens": result.totalTokens,
            "notes": result.runtimeNote,
        }

    _raise_if_cancelled()
    BENCHMARK_PROGRESS.set_phase("saving", "Saving the result")
    with state._lock:
        append_benchmark_run(state, run_record)
        mode_label = {"perplexity": "Perplexity", "task_accuracy": "Task accuracy"}.get(
            request.mode, "Throughput"
        )
        state.add_log(
            "benchmark",
            "info",
            f"{mode_label} benchmark completed for {model_name}: {run_record.get('notes', '')}",
        )
        state.add_activity("Benchmark completed", run_record["label"])
        return {
            "result": run_record,
            "benchmarks": state.benchmark_runs,
            "runtime": state.runtime.status(
                active_requests=state.active_requests,
                requests_served=state.requests_served,
            ),
        }
