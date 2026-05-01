"""MCP (Model Context Protocol) client — Phase 2.10.

ChaosEngineAI's chat agent loop dispatches built-in tools (web search,
calculator, file reader, code executor) through `backend_service.tools`.
This package extends that surface with externally-provided MCP tools:
the user configures one or more MCP servers in settings, and at startup
each server's exported tools are discovered and registered alongside
the built-ins. From the agent loop's perspective the new tools look
identical — same `BaseTool` interface, same OpenAI-shaped function
schema, same `execute(...)` calling convention.

Transport
---------
First ship supports stdio only. The user gives us a command line; we
spawn the process, talk JSON-RPC 2.0 over its stdin/stdout, and tear
the subprocess down at app shutdown. SSE / WebSocket transports are
future work.

Provenance
----------
Every adapted MCP tool tags its `provenance` so the API surface and
the eventual UI can show which server a tool came from. Built-in
tools tag as `"builtin"`; MCP tools tag as `"mcp:<server-id>"`.
"""

from backend_service.mcp.client import (
    McpClient,
    McpClientError,
    McpServerConfig,
    McpToolDescriptor,
)
from backend_service.mcp.tool_adapter import McpTool

__all__ = [
    "McpClient",
    "McpClientError",
    "McpServerConfig",
    "McpToolDescriptor",
    "McpTool",
]
