"""Adapter that exposes an MCP server tool as a `BaseTool`.

Phase 2.10: lets the existing agent loop dispatch MCP tools using the
same interface it already uses for built-ins. The adapter holds a
reference to the live `McpClient` and routes each `execute(...)` call
through `client.call_tool`. Errors from the remote tool are converted
to a string return so the agent loop's existing tool-call result path
handles them — no exception surface change.

Provenance
----------
Each adapter exposes a `provenance` property tagged
``"mcp:<server-id>"``. The /api/tools route reads this so the UI can
render a source badge next to each tool ("Built-in" vs "MCP: filesystem").
"""

from __future__ import annotations

import re
from typing import Any

from backend_service.mcp.client import McpClient, McpClientError, McpToolDescriptor
from backend_service.tools import BaseTool


# MCP tool names can include slashes / colons that aren't legal in
# OpenAI function-calling identifiers. Sanitise to a safe identifier
# while keeping a deterministic mapping back to the original.
_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_name(server_id: str, tool_name: str) -> str:
    """Build a registry-safe name. Format: `mcp__<server>__<tool>`."""
    safe_server = _NAME_SAFE_RE.sub("_", server_id).strip("_") or "server"
    safe_tool = _NAME_SAFE_RE.sub("_", tool_name).strip("_") or "tool"
    return f"mcp__{safe_server}__{safe_tool}"


class McpTool(BaseTool):
    """One MCP tool wrapped as a backend-native `BaseTool`."""

    def __init__(self, client: McpClient, descriptor: McpToolDescriptor) -> None:
        self._client = client
        self._descriptor = descriptor
        self._safe_name = _safe_name(descriptor.server_id, descriptor.name)

    @property
    def name(self) -> str:
        return self._safe_name

    @property
    def description(self) -> str:
        # Prefix the description with the server id so the UI can
        # surface provenance even when the schema list is rendered
        # without per-tool styling.
        base = self._descriptor.description.strip()
        suffix = f" (via MCP: {self._descriptor.server_id})"
        if base:
            return base + suffix
        return f"Tool from MCP server '{self._descriptor.server_id}'"

    @property
    def provenance(self) -> str:
        """Phase 2.10: tag for the API surface + UI badging."""
        return f"mcp:{self._descriptor.server_id}"

    @property
    def remote_name(self) -> str:
        """The tool name on the remote server (before _safe_name munging)."""
        return self._descriptor.name

    def parameters_schema(self) -> dict[str, Any]:
        # MCP exposes JSON Schema directly under `inputSchema`. Pass
        # through verbatim so the model sees the upstream-published
        # shape. Default to a permissive object schema if the server
        # left it empty.
        return self._descriptor.input_schema or {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> str:
        try:
            return self._client.call_tool(self._descriptor.name, kwargs)
        except McpClientError as exc:
            # Surface the failure as text so the agent loop still has
            # something to feed back to the model. Raising would
            # require a more invasive change to the loop's error path.
            return f"[MCP server '{self._descriptor.server_id}' error] {exc}"
