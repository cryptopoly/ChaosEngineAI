from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.app import _is_loopback_host

router = APIRouter()


@router.get("/api/auth/session")
def auth_session(request: Request) -> dict[str, Any]:
    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        raise HTTPException(status_code=403, detail="Session bootstrap is only available from localhost.")

    origin = request.headers.get("origin", "").strip()
    allowed_origins = set(getattr(request.app.state, "chaosengine_allowed_origins", ()))
    if origin and origin not in allowed_origins:
        raise HTTPException(status_code=403, detail="Origin is not allowed to bootstrap an API session.")

    state = request.app.state.chaosengine
    return {
        "apiToken": request.app.state.chaosengine_api_token,
        "apiBase": f"http://127.0.0.1:{state.server_port}",
        "tokenType": "Bearer",
    }
