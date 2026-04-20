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
    result = state.update_settings(body)
    # Hot-apply the API-auth toggle — the middleware reads this flag per
    # request, so flipping it should take effect immediately without
    # forcing the user to restart the server. Env var, if set, still wins.
    from backend_service.app import _resolve_require_api_auth
    request.app.state.chaosengine_require_api_auth = _resolve_require_api_auth(state.settings)
    return result
