"""The ``music_library`` agent tool: list, get, rename (and tag) and delete tracks, like the Music tab's buttons.

One tool grouped by verb (D-30). Answers carry ids, titles and a ``download`` link, never a bare local path.
Delete is permanent and says so in its annotations (destructive); list and get need only the read scope.
"""

from __future__ import annotations

from typing import Any, Callable

from backend_service.mcp_bridge.tools import ToolError, _obj

LIBRARY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "music_library",
        "description": "Edit: the Music Library. list(query, tag, limit, offset), get(track_id; adds a download link), "
        "rename(track_id, title and/or tags), delete(track_id; permanent).",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["list", "get", "rename", "delete"]},
            "track_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "query": {"type": "string", "maxLength": 200, "description": "list: matches title, caption and tags."},
            "tag": {"type": "string", "maxLength": 64},
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20, "description": "rename: replaces the tags."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            "offset": {"type": "integer", "minimum": 0},
        }, ["action"]),
    },
]

_KEYS = ("id", "title", "caption", "tags", "durationS", "seed", "createdAt")


def _row(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: meta[k] for k in _KEYS if k in meta}


def music_library(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service import music_library as lib
    from backend_service.mcp_bridge.files import download_info

    action = args["action"]
    if action == "list":
        items = lib.list_items(str(args.get("query") or ""), str(args.get("tag") or ""))
        offset, limit = int(args.get("offset", 0)), int(args.get("limit", 25))
        out: dict[str, Any] = {"total": len(items), "tracks": [_row(m) for m in items[offset:offset + limit]]}
        if offset + limit < len(items):
            out["nextOffset"] = offset + limit
        return out
    if "track_id" not in args:
        raise ToolError("This action needs track_id (from list).")
    track_id = str(args["track_id"])
    if action == "get":
        meta = lib.get(track_id)
        if meta is None:
            raise ToolError(f"Unknown track '{track_id}'. Call music_library list.")
        path = lib.audio_path(track_id)
        return {**meta, "download": download_info(path) if path else None}
    if action == "rename":
        if "title" not in args and "tags" not in args:
            raise ToolError("rename needs title and/or tags.")
        meta = lib.update(track_id, title=args.get("title"), tags=args.get("tags"))
        if meta is None:
            raise ToolError(f"Unknown track '{track_id}'. Call music_library list.")
        return _row(meta)
    if not lib.delete(track_id):
        raise ToolError(f"Unknown track '{track_id}'. Call music_library list.")
    return {"deleted": track_id}


LIBRARY_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {"music_library": music_library}
