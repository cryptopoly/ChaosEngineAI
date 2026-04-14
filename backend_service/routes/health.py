from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

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
