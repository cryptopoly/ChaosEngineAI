"""``/mcp`` (the agent bridge) and ``/api/agent/access`` (the Settings card's switch and token)."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from backend_service.mcp_bridge import access
from backend_service.mcp_bridge.protocol import handle_message
from backend_service.mcp_bridge.scopes import scope_error

router = APIRouter()


class AgentAccessUpdate(BaseModel):
    enabled: bool | None = None
    rotateToken: bool = False
    scopes: list[str] | None = None  # scopes of the main token ("read" is always kept)
    remoteClone: bool | None = None  # let callers on other computers use "clone" when the token has it
    readToken: bool | None = None  # True makes (or replaces) the read-only token, False removes it


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "").strip()
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-chaosengine-token", "").strip()


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("::ffff:127.")


def _principal(request: Request) -> access.Principal | None:
    """Who is calling: the app's own token (everything), the agent token, or the read-only token."""
    provided = _bearer(request)
    if not provided:
        return None
    app_token = str(getattr(request.app.state, "chaosengine_api_token", "") or "")
    if app_token and secrets.compare_digest(provided, app_token):
        return access.Principal(access.ALL_SCOPES, "app")
    return access.principal_for(provided, _is_loopback(request))


def _authorized(request: Request) -> bool:
    principal = _principal(request)
    request.state.agent_principal = principal
    return principal is not None


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

    from backend_service.mcp_bridge import progress

    token = progress.token_of(payload)
    if token is not None:  # the client asked for progress: a short event stream, then the answer
        from fastapi.responses import StreamingResponse

        return StreamingResponse(progress.stream(handle_message, payload, request, token), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})
    if isinstance(payload, list):
        responses = [r for r in [await anyio.to_thread.run_sync(handle_message, m, request) for m in payload] if r]
        return JSONResponse(responses) if responses else Response(status_code=202)
    response = await anyio.to_thread.run_sync(handle_message, payload, request)
    return JSONResponse(response) if response is not None else Response(status_code=202)


@router.get("/mcp")
def mcp_get() -> Response:
    # No server-initiated stream: the spec lets a server answer 405.
    return Response(status_code=405, headers={"Allow": "POST"})


def _require_agent(request: Request, scope: str = "files") -> None:
    if not access.is_enabled():
        raise HTTPException(status_code=403, detail=access.OFF_MESSAGE)
    if not _authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized. Send the Agent access token as a Bearer token.")
    if scope not in request.state.agent_principal.scopes:
        raise HTTPException(status_code=403, detail=scope_error("this route", scope))


@router.post("/api/agent/uploads")
async def upload_input_image(request: Request) -> dict[str, Any]:
    """Raw image, sound or video bytes in the body; returns a handle tools accept in place of a path.

    The body is streamed into the inbox; the type is read from its first bytes and the size cap for that
    type is enforced while it arrives.
    """
    _require_agent(request)
    from backend_service.helpers.media_input import InputImageError
    from backend_service.mcp_bridge import files

    part = files.begin_upload()
    try:
        head, cap, size = b"", None, 0
        with part.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if cap is None:
                    head += chunk[:64]
                    if len(head) >= 16:
                        cap = files.max_upload_bytes(files.sniff(head))
                if cap is not None and size > cap:
                    raise HTTPException(status_code=413, detail=f"The upload is larger than {cap // (1024 * 1024)} MB.")
                handle.write(chunk)
        return files.finish_upload(part)
    except InputImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        part.unlink(missing_ok=True)


@router.get("/api/agent/files")
def download_output_file(request: Request, path: str) -> FileResponse:
    """A finished job's output. Only files inside the app's own output folders."""
    _require_agent(request)
    from backend_service.mcp_bridge.files import fetchable

    real = fetchable(path)
    if real is None:
        raise HTTPException(status_code=404, detail="Not an output file of this app.")
    return FileResponse(real, filename=real.name)


def _stdio_command(port: int) -> list[str] | None:
    """Command for MCP clients that only start local programs: this Python running the CLI relay."""
    cli = Path(__file__).resolve().parents[2] / "scripts" / "chaosengine-cli"
    if not cli.is_file():
        return None
    return [sys.executable, str(cli), "--port", str(port), "mcp"]


@router.get("/api/agent/access")
def get_agent_access(request: Request) -> dict[str, Any]:
    port = request.app.state.chaosengine.server_port
    return {
        "enabled": access.is_enabled(),
        "token": access.get_token(),
        "mcpUrl": f"http://127.0.0.1:{port}/mcp",
        "stdioCommand": _stdio_command(port),
        "scopes": access.get_scopes(),
        "remoteClone": access.remote_clone_allowed(),
        "readToken": access.get_read_token(),
    }


@router.post("/api/agent/access")
def update_agent_access(request: Request, body: AgentAccessUpdate) -> dict[str, Any]:
    if body.enabled is not None:
        access.set_enabled(body.enabled)
    if body.rotateToken:
        access.rotate_token()
    if body.scopes is not None:
        access.set_scopes(body.scopes)
    if body.remoteClone is not None:
        access.set_remote_clone(body.remoteClone)
    if body.readToken is True:
        access.make_read_token()
    elif body.readToken is False:
        access.revoke_read_token()
    return get_agent_access(request)
