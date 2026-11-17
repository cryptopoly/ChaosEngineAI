"""Keep going routes (FU-191): turn a chat's run on, read its state, stop it, resume it.

Under /api/, so only the app's own token reaches them. Starting is idempotent (one active run per chat); the
render grant is chosen on every start and Resume and never survives an app restart.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend_service.i18n import localized_detail

router = APIRouter()


class KeepGoingStart(BaseModel):
    cap: int = Field(20, ge=1, le=100)
    grant: Literal["auto", "manual"] = "manual"
    hours: float = Field(6, ge=0.25, le=12)
    gpuMode: Literal["auto", "always", "never"] = "auto"


class KeepGoingResume(BaseModel):
    grant: Literal["auto", "manual"] = "manual"
    hours: float = Field(6, ge=0.25, le=12)
    cap: int | None = Field(None, ge=1, le=100)


def _controller(request: Request) -> Any:
    from backend_service import keep_going

    return keep_going.for_state(request.app.state.chaosengine)


_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _ensure_session(request: Request, session_id: str) -> bool:
    """The chat exists, or is a new chat the app has not sent yet (a draft id): it is created empty, as the
    first message would create it, so Keep going can be armed before the brief is sent."""
    if not _SESSION_ID.match(session_id):
        return False
    state = request.app.state.chaosengine
    with state._lock:
        state._ensure_session(session_id)
    return True


@router.get("/api/chat/keepgoing/{session_id}")
def keep_going_status(request: Request, session_id: str) -> dict[str, Any]:
    return {"run": _controller(request).status(session_id)}


@router.post("/api/chat/keepgoing/{session_id}/start")
def keep_going_start(request: Request, session_id: str, body: KeepGoingStart) -> dict[str, Any]:
    if not _ensure_session(request, session_id):
        raise HTTPException(status_code=404, detail=localized_detail(request, "Session not found."))
    controller = _controller(request)
    controller.start(session_id, cap=body.cap, grant_mode=body.grant, hours=body.hours, app_request=request,
                     gpu_mode=body.gpuMode)
    return {"run": controller.status(session_id)}


@router.post("/api/chat/keepgoing/{session_id}/stop")
def keep_going_stop(request: Request, session_id: str) -> dict[str, Any]:
    controller = _controller(request)
    controller.stop(session_id, "user_stopped")
    return {"run": controller.status(session_id)}


@router.post("/api/chat/keepgoing/{session_id}/studio-off")
def keep_going_studio_off(request: Request, session_id: str) -> dict[str, Any]:
    """Studio tools were switched off in the composer: the run ends (it needs them)."""
    controller = _controller(request)
    controller.stop(session_id, "studio_off")
    return {"run": controller.status(session_id)}


@router.post("/api/chat/keepgoing/{session_id}/resume")
def keep_going_resume(request: Request, session_id: str, body: KeepGoingResume) -> dict[str, Any]:
    controller = _controller(request)
    try:
        controller.resume(session_id, grant_mode=body.grant, hours=body.hours, cap=body.cap, app_request=request)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=localized_detail(request, str(exc))) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=localized_detail(request, str(exc))) from exc
    return {"run": controller.status(session_id)}
