"""JSON-RPC 2.0 for MCP over streamable HTTP (single JSON responses, no streaming)."""

from __future__ import annotations

import json
from typing import Any

from backend_service.mcp_bridge import access
from backend_service.mcp_bridge.tools import HANDLERS, TOOLS, ToolError, make_context, validate

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "chaosengine", "version": "1"}
INSTRUCTIONS = (
    "Make images, video clips, speech and transcripts with the local ChaosEngineAI app. Call system_status "
    "first. Jobs run one at a time: start one, call wait_for_job until it is done, then use the file path "
    "in its output. If list_models is empty, call recommend_models and tell the user what to download."
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
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": INSTRUCTIONS,
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        return {"jsonrpc": "2.0", "id": rpc_id, "result": _call_tool(params, request)}
    return _error(rpc_id, -32601, f"Method not found: {method}")


def _call_tool(params: dict[str, Any], request: Any) -> dict[str, Any]:
    name = str(params.get("name") or "")
    if not access.is_enabled():
        return _tool_result(access.OFF_MESSAGE, True)
    handler = HANDLERS.get(name)
    tool = next((t for t in TOOLS if t["name"] == name), None)
    if handler is None or tool is None:
        return _tool_result(f"Unknown tool '{name}'. Call tools/list.", True)
    args = params.get("arguments") or {}
    problem = validate(tool["inputSchema"], args)
    if problem:
        return _tool_result(f"Invalid input for {name}: {problem}.", True)
    try:
        return _tool_result(handler(make_context(request), args))
    except ToolError as exc:
        return _tool_result(str(exc), True)
    except Exception as exc:  # noqa: BLE001 - never crash the endpoint on a tool bug
        return _tool_result(f"{name} failed: {type(exc).__name__}: {exc}", True)
