"""GPU and system metrics endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from backend_service.helpers.gpu import get_gpu_metrics

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/gpu")
def gpu_snapshot():
    """Return current GPU / accelerator metrics.

    A plain ``def`` so FastAPI runs it in its thread pool: the snapshot shells out
    (nvidia-smi, sysctl, or a torch probe that takes seconds on Windows), and as an
    ``async def`` it blocked the event loop, stalling every other request while the
    UI polled it.
    """
    return get_gpu_metrics()
