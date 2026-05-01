"""Tool registry and base interface for ChaosEngineAI agent tool-use.

Each tool implements BaseTool and registers itself via the ToolRegistry.
The agent loop inspects model responses for tool_calls, dispatches them
through the registry, and injects results back into the conversation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# Phase 2.8: rich tool output payload.
#
# `text` is what the language model sees on the next turn (preserves
# the existing contract — the agent loop feeds tool results back as
# message content). `render_as` + `data` are an optional UI hint the
# frontend's `ToolCallCard` reads to render a table / code block /
# markdown / image / chart instead of dumping raw JSON. Tools that
# don't override `execute_structured` continue to return plain text
# and the UI falls back to the existing collapsible-JSON view.
RenderAsLiteral = str  # "table" | "code" | "markdown" | "image" | "chart" | "json"


@dataclass
class StructuredToolOutput:
    text: str
    render_as: RenderAsLiteral = "json"
    data: dict[str, Any] | None = None


class BaseTool(ABC):
    """Interface every tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in function-calling schemas (e.g. ``"web_search"``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description shown to the model in the tool list."""

    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's input parameters."""

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Run the tool with the given arguments and return a text result."""

    def execute_structured(self, **kwargs: Any) -> StructuredToolOutput | None:
        """Phase 2.8: optional rich-output entry point.

        Tools that want the UI to render a table / code block / markdown
        instead of a JSON dump override this to return a
        `StructuredToolOutput`. The agent loop calls this first; when
        it returns None (the default), the loop falls back to
        `execute(...)` and treats the result as plain text. Built-in
        tools that haven't been migrated yet keep working unchanged.
        """
        return None

    @property
    def provenance(self) -> str:
        """Phase 2.10: where this tool came from. Built-ins return
        ``"builtin"``; MCP-adapted tools override to ``"mcp:<server>"``.
        Surfaced via /api/tools so the UI can render a source badge.
        """
        return "builtin"

    def openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling representation of this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


class ToolRegistry:
    """Discover-register-get registry for agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        # Phase 2.10: keep MCP-sourced tools in a parallel set so we
        # can refresh them (re-spawn server, swap configs) without
        # disturbing the built-in registrations.
        self._mcp_tool_names: set[str] = set()

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._mcp_tool_names.discard(name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in OpenAI function-calling format."""
        return [tool.openai_schema() for tool in self._tools.values()]

    def available_names(self) -> list[str]:
        return list(self._tools.keys())

    def discover(self) -> None:
        """Import and register all built-in tools."""
        from backend_service.tools.web_search import WebSearchTool
        from backend_service.tools.calculator import CalculatorTool
        from backend_service.tools.code_executor import CodeExecutorTool, code_executor_enabled
        from backend_service.tools.file_reader import FileReaderTool

        tool_classes = [WebSearchTool, CalculatorTool, FileReaderTool]
        if code_executor_enabled():
            tool_classes.append(CodeExecutorTool)

        for cls in tool_classes:
            instance = cls()
            self.register(instance)

    def replace_mcp_tools(self, tools: list[BaseTool]) -> None:
        """Phase 2.10: swap the registry's MCP-sourced tools.

        Drops every previously-registered MCP tool and registers the
        provided list. Built-in tools are untouched. Called whenever
        the user updates `mcpServers` in settings or the app starts up.
        """
        for stale in list(self._mcp_tool_names):
            self._tools.pop(stale, None)
        self._mcp_tool_names.clear()
        for tool in tools:
            self.register(tool)
            self._mcp_tool_names.add(tool.name)


# Module-level singleton
registry = ToolRegistry()
registry.discover()
