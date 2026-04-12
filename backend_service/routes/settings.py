from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from backend_service.models import UpdateSettingsRequest

router = APIRouter()


@router.get("/api/settings")
def settings(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    library = state._library()
    return {"settings": state._settings_payload(library)}


@router.patch("/api/settings")
def update_settings(request: Request, body: UpdateSettingsRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.update_settings(body)
