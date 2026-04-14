from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

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
        raise HTTPException(status_code=400, detail=detail) from exc
