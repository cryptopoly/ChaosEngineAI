"""Plugin management endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend_service.plugins import plugin_registry, PluginType

router = APIRouter(prefix="/api/plugins", tags=["plugins"])


@router.get("")
async def list_plugins():
    """List all plugins grouped by type."""
    grouped: dict[str, list[dict]] = {}
    for manifest in plugin_registry.list_all():
        entry = {
            "id": manifest.id,
            "name": manifest.name,
            "type": manifest.plugin_type.value,
            "version": manifest.version,
            "author": manifest.author,
            "description": manifest.description,
            "builtin": manifest.builtin,
            "enabled": manifest.enabled,
        }
        grouped.setdefault(manifest.plugin_type.value, []).append(entry)
    return grouped


@router.post("/{plugin_id}/enable")
async def enable_plugin(plugin_id: str):
    """Enable a plugin by ID."""
    if not plugin_registry.enable(plugin_id):
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")
    return {"ok": True, "plugin_id": plugin_id, "enabled": True}


@router.post("/{plugin_id}/disable")
async def disable_plugin(plugin_id: str):
    """Disable a plugin by ID."""
    if not plugin_registry.disable(plugin_id):
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_id}' not found")
    return {"ok": True, "plugin_id": plugin_id, "enabled": False}
