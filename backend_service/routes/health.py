from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from backend_service.helpers.gpu import gpu_status_snapshot
from backend_service.helpers.system import _runtime_label

router = APIRouter()


@router.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    from backend_service.app import WORKSPACE_ROOT, app_version

    runtime_status = state.runtime.status(
        active_requests=state.active_requests,
        requests_served=state.requests_served,
    )
    return {
        "status": "ok",
        "workspaceRoot": str(WORKSPACE_ROOT),
        "runtime": _runtime_label(),
        "appVersion": app_version,
        "engine": runtime_status["engine"],
        "loadedModel": runtime_status["loadedModel"],
        "nativeBackends": runtime_status["nativeBackends"],
    }


@router.get("/api/workspace")
def workspace(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.workspace()


@router.get("/api/runtime")
def runtime_status(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.runtime.status(
        active_requests=state.active_requests,
        requests_served=state.requests_served,
    )


@router.get("/api/system/gpu-status")
def system_gpu_status() -> dict[str, Any]:
    """Unified GPU availability summary for the frontend warning banner.

    Returns whether torch sees CUDA / MPS on the current host, whether an
    NVIDIA driver is visible on ``PATH``, and a human-readable recommendation
    string when torch fell back to CPU on a box that clearly has an NVIDIA
    GPU. Safe to call before any model is loaded.
    """
    return gpu_status_snapshot()
