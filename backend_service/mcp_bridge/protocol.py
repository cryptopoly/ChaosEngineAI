"""JSON-RPC 2.0 for MCP over streamable HTTP (single JSON responses, no streaming)."""

from __future__ import annotations

import json
from typing import Any

from backend_service.mcp_bridge import access
from backend_service.mcp_bridge.jobs import CURRENT_OWNER, STORE, JobLimitError
from backend_service.mcp_bridge.scopes import ALIASES, lowest_scope, required_scope, scope_error, starts_job
from backend_service.mcp_bridge.tools import HANDLERS, TOOLS, ToolError, make_context, validate

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "chaosengine", "version": "1"}
INSTRUCTIONS = (
    "Make images, video clips, speech, music, transcripts and short films (storyboards) with the local "
    "ChaosEngineAI app. Call system_status first: it shows free memory and the whole job queue with estimated "
    "times. Jobs run one at a time and each token may have 20 unfinished: start a job, call wait_for_job until "
    "it is done, then use the file in its output (output.download for another machine). list_jobs finds a job "
    "id you lost; jobs survive an app restart as 'interrupted'. If list_models is empty, call recommend_models "
    "and tell the user what to download; you cannot download. Music: make_music. Films: storyboard_project, "
    "storyboard_shot, storyboard_audio, storyboard_captions, storyboard_generate, storyboard_render, "
    "storyboard_export_otio (pass the revision from the last answer). music_library manages tracks. Cloned voices: "
    "list_cloned_voices and speak_cloned only, never create one. Resources and prompts are listed for reading "
    "films and starting common jobs. Send files in with upload_file. A token may lack a scope; the error says what to ask the user."
)


def _error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


def _tool_result(payload: Any, is_error: bool = False) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle_message(message: Any, request: Any) -> dict[str, Any] | None:
    """One JSON-RPC message in, one response out (None for notifications)."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Invalid JSON-RPC request.")
    rpc_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if rpc_id is None:  # notification: nothing to answer
        return None
    if method == "initialize":
        asked = str(params.get("protocolVersion") or "")
        version = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": False}, "resources": {"subscribe": False, "listChanged": False},
                                 "prompts": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {}}
    if method == "tools/list":
        principal = _principal(request)
        tools = TOOLS if principal is None else [t for t in TOOLS if lowest_scope(t["name"]) in principal.scopes]
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": tools}}
    if method == "tools/call":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": _call_tool(params, request)}
    if method in _ROOT_METHODS:
        return _resources_call(rpc_id, method, params, request)
    return _error(rpc_id, -32601, f"Method not found: {method}")


_ROOT_METHODS = ("resources/list", "resources/templates/list", "resources/read", "prompts/list", "prompts/get")


def _resources_call(rpc_id: Any, method: str, params: Any, request: Any) -> dict[str, Any]:
    """Resources and prompts: read-only, so every token may use them while access is on."""
    from backend_service.mcp_bridge import resources

    if not access.is_enabled():
        return _error(rpc_id, -32001, access.OFF_MESSAGE)
    if not isinstance(params, dict):
        return _error(rpc_id, -32602, "params must be an object.")
    try:
        if method == "resources/list":
            result = resources.list_resources(params)
        elif method == "resources/templates/list":
            result = resources.list_templates(params)
        elif method == "resources/read":
            result = resources.read_resource(params, make_context(request))
        elif method == "prompts/list":
            result = resources.list_prompts(params)
        else:
            result = resources.get_prompt(params)
    except resources.ResourceError as exc:
        return _error(rpc_id, exc.code, str(exc))
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint
        return _error(rpc_id, -32603, f"{method} failed: {type(exc).__name__}: {exc}")
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _principal(request: Any) -> access.Principal | None:
    return getattr(getattr(request, "state", None), "agent_principal", None)


def _call_tool(params: dict[str, Any], request: Any) -> dict[str, Any]:
    name = str(params.get("name") or "")
    if not access.is_enabled():
        return _tool_result(access.OFF_MESSAGE, True)
    name = ALIASES.get(name, name)
    handler = HANDLERS.get(name)
    tool = next((t for t in TOOLS if t["name"] == name), None)
    if handler is None or tool is None:
        return _tool_result(f"Unknown tool '{name}'. Call tools/list.", True)
    args = params.get("arguments") or {}
    problem = validate(tool["inputSchema"], args)
    if problem:
        return _tool_result(f"Invalid input for {name}: {problem}.", True)
    principal = _principal(request)
    if principal is not None:
        scope = required_scope(name, args)
        if scope not in principal.scopes:
            return _tool_result(scope_error(name, scope), True)
    token = CURRENT_OWNER.set(principal.owner if principal is not None else "app")
    try:
        if starts_job(name, args):
            STORE.check_capacity()
        return _tool_result(handler(make_context(request), args))
    except (ToolError, JobLimitError) as exc:
        return _tool_result(str(exc), True)
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint on a tool bug
        return _tool_result(f"{name} failed: {type(exc).__name__}: {exc}", True)
    finally:
        CURRENT_OWNER.reset(token)
