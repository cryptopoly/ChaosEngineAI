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
from backend_service.tools import BaseTool, StructuredToolOutput


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

    def execute_structured(self, **kwargs: Any) -> StructuredToolOutput | None:
        """Phase 2.8: surface MCP content parts as structured output.

        MCP servers return a list of content parts under
        ``result.content`` (text, image, embedded resources). When the
        first part is an image we render it inline; when there's a
        single text part we leave it for the legacy fallback so the UI
        can still pick markdown / table renderers added later by tool
        introspection. Multiple-part results render as markdown with
        each part stringified.
        """
        try:
            raw = self._client.call_tool_raw(self._descriptor.name, kwargs)
        except AttributeError:
            # Older clients without the raw helper — just fall through
            # to the plain text path.
            return None
        except McpClientError as exc:
            return StructuredToolOutput(
                text=f"[MCP server '{self._descriptor.server_id}' error] {exc}",
                render_as="markdown",
            )
        if not isinstance(raw, dict):
            return None
        content = raw.get("content")
        if not isinstance(content, list) or not content:
            return None

        # Single image part: render inline.
        if len(content) == 1 and isinstance(content[0], dict) and content[0].get("type") == "image":
            img = content[0]
            data_uri = _image_part_to_data_uri(img)
            if data_uri:
                return StructuredToolOutput(
                    text=f"[image: {img.get('mimeType', 'image/png')}]",
                    render_as="image",
                    data={"src": data_uri, "alt": img.get("alt", "")},
                )

        # Multiple parts or non-image: stringify into markdown so the
        # UI shows each part with its own framing.
        from backend_service.mcp.client import _flatten_tool_result

        text = _flatten_tool_result(raw)
        return StructuredToolOutput(
            text=text,
            render_as="markdown",
            data={"markdown": text},
        )


def _image_part_to_data_uri(part: dict[str, Any]) -> str | None:
    """Convert an MCP image content part to a `data:` URI for inline render."""
    data = part.get("data")
    if not isinstance(data, str) or not data:
        return None
    mime = part.get("mimeType") or "image/png"
    return f"data:{mime};base64,{data}"
