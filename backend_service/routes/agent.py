"""``/mcp`` (the agent bridge) and ``/api/agent/access`` (the Settings card's switch and token)."""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from backend_service.mcp_bridge import access
from backend_service.mcp_bridge.protocol import handle_message

router = APIRouter()


class AgentAccessUpdate(BaseModel):
    enabled: bool | None = None
    rotateToken: bool = False


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "").strip()
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-chaosengine-token", "").strip()


def _authorized(request: Request) -> bool:
    provided = _bearer(request)
    if not provided:
        return False
    app_token = str(getattr(request.app.state, "chaosengine_api_token", "") or "")
    return access.token_ok(provided) or (bool(app_token) and secrets.compare_digest(provided, app_token))


@router.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    # /mcp sits outside the /api/ auth middleware, so check here. While access is off, every call is
    # answered (tools say how to turn it on) so an agent can tell the user instead of seeing a failure.
    if access.is_enabled() and not _authorized(request):
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized. Send the Agent access token as a Bearer token."},
        )
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error."}},
        )
    import anyio

    if isinstance(payload, list):
        responses = [r for r in [await anyio.to_thread.run_sync(handle_message, m, request) for m in payload] if r]
        return JSONResponse(responses) if responses else Response(status_code=202)
    response = await anyio.to_thread.run_sync(handle_message, payload, request)
    return JSONResponse(response) if response is not None else Response(status_code=202)


@router.get("/mcp")
def mcp_get() -> Response:
    # No server-initiated stream: the spec lets a server answer 405.
    return Response(status_code=405, headers={"Allow": "POST"})


@router.get("/api/agent/access")
def get_agent_access(request: Request) -> dict[str, Any]:
    port = request.app.state.chaosengine.server_port
    return {
        "enabled": access.is_enabled(),
        "token": access.get_token(),
        "mcpUrl": f"http://127.0.0.1:{port}/mcp",
    }


@router.post("/api/agent/access")
def update_agent_access(request: Request, body: AgentAccessUpdate) -> dict[str, Any]:
    if body.enabled is not None:
        access.set_enabled(body.enabled)
    if body.rotateToken:
        access.rotate_token()
    return get_agent_access(request)
