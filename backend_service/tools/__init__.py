"""Tool registry and base interface for ChaosEngineAI agent tool-use.

Each tool implements BaseTool and registers itself via the ToolRegistry.
The agent loop inspects model responses for tool_calls, dispatches them
through the registry, and injects results back into the conversation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

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


# Module-level singleton
registry = ToolRegistry()
registry.discover()
