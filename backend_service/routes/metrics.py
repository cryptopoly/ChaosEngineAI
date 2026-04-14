"""GPU and system metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend_service.helpers.gpu import get_gpu_metrics

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/gpu")
async def gpu_snapshot():
    """Return current GPU / accelerator metrics."""
    return get_gpu_metrics()
