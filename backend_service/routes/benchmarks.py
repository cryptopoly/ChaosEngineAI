from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.i18n import localized_detail
from backend_service.models import BenchmarkRunRequest

router = APIRouter()


@router.post("/api/benchmarks/run")
def run_benchmark(request: Request, body: BenchmarkRunRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    try:
        return state.run_benchmark(body)
    except RuntimeError as exc:
        detail = str(exc)
        state.add_log("benchmark", "error", f"Benchmark failed: {detail}")
        raise HTTPException(
            status_code=400,
            detail=localized_detail(request, detail),
        ) from exc


@router.delete("/api/benchmarks/{run_id}")
def delete_benchmark(request: Request, run_id: str) -> dict[str, Any]:
    """Remove one saved run from the history."""
    state = request.app.state.chaosengine
    if not state.delete_benchmark_run(run_id):
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return {"deleted": run_id, "benchmarks": state.benchmark_runs}
