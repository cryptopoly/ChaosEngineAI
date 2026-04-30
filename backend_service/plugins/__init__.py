"""Universal plugin system for ChaosEngineAI.

Generalizes the compression CacheStrategyRegistry pattern into a
multi-type plugin registry supporting inference engines, tools,
model sources, and post-processing pipelines.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any
import importlib
import json

class PluginType(str, Enum):
    CACHE_STRATEGY = "cache_strategy"
    INFERENCE_ENGINE = "inference_engine"
    TOOL = "tool"
    MODEL_SOURCE = "model_source"
    POST_PROCESSOR = "post_processor"

@dataclass
class PluginManifest:
    id: str
    name: str
    plugin_type: PluginType
    version: str = "0.1.0"
    author: str = ""
    description: str = ""
    entry_point: str = ""
    builtin: bool = False
    enabled: bool = True

class BasePlugin(ABC):
    @property
    @abstractmethod
    def manifest(self) -> PluginManifest: ...

class PluginRegistry:
    def __init__(self, *, auto_register_builtins: bool = False):
        self._plugins: dict[str, tuple[PluginManifest, Any]] = {}
        self._auto_register_builtins = auto_register_builtins
        self._builtins_registered = False
        self._lock = RLock()

    def register(self, manifest: PluginManifest, instance: Any = None):
        self._plugins[manifest.id] = (manifest, instance)

    def ensure_builtins(self) -> None:
        if not self._auto_register_builtins or self._builtins_registered:
            return
        with self._lock:
            if not self._builtins_registered:
                self.register_builtins()

    def get(self, plugin_id: str) -> tuple[PluginManifest, Any] | None:
        self.ensure_builtins()
        return self._plugins.get(plugin_id)

    def list_all(self) -> list[PluginManifest]:
        self.ensure_builtins()
        return [m for m, _ in self._plugins.values()]

    def list_by_type(self, plugin_type: PluginType) -> list[tuple[PluginManifest, Any]]:
        self.ensure_builtins()
        return [(m, i) for m, i in self._plugins.values() if m.plugin_type == plugin_type]

    def enable(self, plugin_id: str) -> bool:
        self.ensure_builtins()
        if plugin_id in self._plugins:
            self._plugins[plugin_id][0].enabled = True
            return True
        return False

    def disable(self, plugin_id: str) -> bool:
        self.ensure_builtins()
        if plugin_id in self._plugins:
            self._plugins[plugin_id][0].enabled = False
            return True
        return False

    def discover_from_directory(self, plugins_dir: Path):
        """Scan a directory for plugin manifest.json files and load them."""
        if not plugins_dir.exists():
            return
        for manifest_path in plugins_dir.glob("*/manifest.json"):
            try:
                data = json.loads(manifest_path.read_text())
                manifest = PluginManifest(
                    id=data["id"],
                    name=data["name"],
                    plugin_type=PluginType(data["type"]),
                    version=data.get("version", "0.1.0"),
                    author=data.get("author", ""),
                    description=data.get("description", ""),
                    entry_point=data.get("entry_point", ""),
                )
                self.register(manifest)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    def register_builtins(self):
        """Register all built-in components as plugins."""
        # Cache strategies
        from cache_compression import registry as cache_registry
        for strategy in cache_registry.strategies():
            manifest = PluginManifest(
                id=f"cache.{strategy.strategy_id}",
                name=strategy.name,
                plugin_type=PluginType.CACHE_STRATEGY,
                builtin=True,
                description=f"Cache compression: {strategy.name}",
            )
            self.register(manifest, strategy)

        # Tools
        from backend_service.tools import registry as tool_registry
        for tool in tool_registry.list_tools():
            manifest = PluginManifest(
                id=f"tool.{tool.name}",
                name=tool.name,
                plugin_type=PluginType.TOOL,
                builtin=True,
                description=tool.description,
            )
            self.register(manifest, tool)
        self._builtins_registered = True

# Module singleton
plugin_registry = PluginRegistry(auto_register_builtins=True)
