"""High-level MCP loader — spawn servers, discover tools, build adapters.

The single entry point `load_mcp_tools` is what the app should call
at startup (and after the user updates `mcpServers` in settings). It
takes a list of server configs and returns:

  * a flat list of `McpTool` adapters ready to feed into
    `ToolRegistry.replace_mcp_tools`;
  * a list of live `McpClient` instances the caller must close on
    shutdown (or when reloading).

A misbehaving server (bad command, init timeout, malformed
`tools/list` response) is isolated: its client is closed and skipped,
the loader logs via the supplied callback, and other servers proceed
normally. The chat path always sees the union of healthy servers'
tools — never an all-or-nothing failure.
"""

from __future__ import annotations

from typing import Callable, Iterable

from backend_service.mcp.client import (
    McpClient,
    McpClientError,
    McpServerConfig,
)
from backend_service.mcp.tool_adapter import McpTool


LogFn = Callable[[str, str], None]


def load_mcp_tools(
    configs: Iterable[McpServerConfig],
    *,
    log: LogFn | None = None,
) -> tuple[list[McpTool], list[McpClient]]:
    """Spawn each enabled server and collect its tools.

    `log(level, message)` is the optional logging callback. When
    omitted, failures are silent (callers like tests can pass
    ``log=None``); production callers should plumb in `state.add_log`
    so users see a settings → log entry per misbehaving server.
    """
    tools: list[McpTool] = []
    clients: list[McpClient] = []

    for config in configs:
        if not config.enabled:
            continue
        client = McpClient(config)
        try:
            client.initialize()
            descriptors = client.list_tools()
        except McpClientError as exc:
            if log is not None:
                log("warning", f"MCP server '{config.id}' failed to start: {exc}")
            client.close()
            continue
        except Exception as exc:  # noqa: BLE001 — protect chat path from any subprocess weirdness
            if log is not None:
                log("warning", f"MCP server '{config.id}' raised unexpected error: {exc}")
            client.close()
            continue

        if not descriptors:
            if log is not None:
                log("info", f"MCP server '{config.id}' is up but exports zero tools.")
            # Keep the client around — the server may export tools
            # later, and the user might still rely on resources/prompts
            # in a future release.
            clients.append(client)
            continue

        clients.append(client)
        for descriptor in descriptors:
            tools.append(McpTool(client, descriptor))
        if log is not None:
            log("info", f"MCP server '{config.id}' loaded ({len(descriptors)} tool(s)).")

    return tools, clients


def close_all(clients: Iterable[McpClient]) -> None:
    """Tear down every client — call on app shutdown / reload.

    Errors during close are swallowed: a hung subprocess shouldn't
    block the parent app from exiting. Each client's `close()` method
    sends terminate + falls back to kill after 5 s.
    """
    for client in clients:
        try:
            client.close()
        except Exception:
            continue
