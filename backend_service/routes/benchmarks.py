from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.i18n import localized_detail
from backend_service.models import BenchmarkRunRequest
from backend_service.state.benchmarks import BenchmarkBusy, cancel_benchmark as _cancel_benchmark

router = APIRouter()


@router.post("/api/benchmarks/run")
def run_benchmark(request: Request, body: BenchmarkRunRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    try:
        return state.run_benchmark(body)
    except BenchmarkBusy as exc:
        raise HTTPException(status_code=409, detail=localized_detail(request, str(exc))) from exc
    except RuntimeError as exc:
        detail = str(exc)
        state.add_log("benchmark", "error", f"Benchmark failed: {detail}")
        raise HTTPException(
            status_code=400,
            detail=localized_detail(request, detail),
        ) from exc


@router.get("/api/benchmarks/progress")
def benchmark_progress(request: Request) -> dict[str, Any]:
    """What the running benchmark is doing: phase, items done of total, and the model load's own message.

    Polled by the progress dialog. ``totalSteps`` is 0 for a phase the engine cannot count
    (a model load, a throughput generation); the dialog then shows the phase with no percentage.
    """
    from backend_service.progress import BENCHMARK_PROGRESS

    snapshot = BENCHMARK_PROGRESS.snapshot()
    if snapshot["active"] and snapshot["phase"] == "loading":
        # The loader reports through the runtime; no state lock is taken (the run may hold it).
        loading, _tail = request.app.state.chaosengine.runtime.loading_progress()
        if loading and loading.get("message"):
            snapshot["message"] = str(loading["message"])
    return {"progress": snapshot}


@router.post("/api/benchmarks/cancel")
def cancel_benchmark(request: Request) -> dict[str, Any]:
    """Stop the benchmark that is running. The waiting run request then answers ``{"cancelled": true}``."""
    # No state lock here: the run request may hold it while a model loads.
    return {"cancelled": bool(_cancel_benchmark(request.app.state.chaosengine))}


@router.delete("/api/benchmarks/{run_id}")
def delete_benchmark(request: Request, run_id: str) -> dict[str, Any]:
    """Remove one saved run from the history."""
    state = request.app.state.chaosengine
    if not state.delete_benchmark_run(run_id):
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return {"deleted": run_id, "benchmarks": state.benchmark_runs}
