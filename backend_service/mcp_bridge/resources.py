"""MCP resources and prompts (A8). Read-only views of the app, plus three starter prompts.

Resources (all need only the ``read`` scope, answered as JSON text, never a bare local path):

* ``chaosengine://system/status``       free memory and the job queue
* ``chaosengine://music/library``       tracks (paged); ``chaosengine://music/{id}`` one track
* ``chaosengine://gallery/recent``      the newest image and video outputs
* ``chaosengine://storyboard/{id}``     one film project (shots, takes, timeline, revision)

``resources/list`` pages with an opaque cursor (an offset); the storyboard projects come after the fixed
resources. ``prompts/list`` and ``prompts/get`` serve ``short_film_from_brief``, ``poster_set`` and
``soundtrack_for_project``: each returns a user message that tells the agent which tools to call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PAGE = 50
GALLERY_LIMIT = 30
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_FIXED = [
    ("chaosengine://system/status", "System status", "Free memory and the job queue with estimated times."),
    ("chaosengine://music/library", "Music library", "Tracks in the Music Library, newest first (first 50; use music_library to page)."),
    ("chaosengine://gallery/recent", "Recent outputs", "The newest images and videos the app made."),
]
TEMPLATES = [
    {"uriTemplate": "chaosengine://storyboard/{id}", "name": "Storyboard project", "mimeType": "application/json",
     "description": "One film project: shots, takes, audio, timeline and revision."},
    {"uriTemplate": "chaosengine://music/{id}", "name": "Music track", "mimeType": "application/json",
     "description": "One Music Library track with a download link."},
]


class ResourceError(Exception):
    """Unknown or malformed resource URI (JSON-RPC -32002 / -32602)."""

    def __init__(self, message: str, code: int = -32002) -> None:
        super().__init__(message)
        self.code = code


def _cursor(params: dict[str, Any]) -> int:
    raw = params.get("cursor")
    if raw in (None, ""):
        return 0
    try:
        value = int(str(raw))
    except ValueError as exc:
        raise ResourceError("Invalid cursor.", -32602) from exc
    if value < 0:
        raise ResourceError("Invalid cursor.", -32602)
    return value


def list_resources(params: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes.storyboard import get_store

    rows = [{"uri": u, "name": n, "description": d, "mimeType": "application/json"} for u, n, d in _FIXED]
    try:
        projects = [p for p in get_store().list() if "name" in p]
    except Exception:  # noqa: BLE001 - no storyboard folder is an empty list, not an error
        projects = []
    rows += [{"uri": f"chaosengine://storyboard/{p['id']}", "name": f"Film: {p['name']}", "mimeType": "application/json",
              "description": f"Storyboard project, {p['shots']} shots, revision {p['revision']}."} for p in projects]
    start = _cursor(params)
    page = rows[start:start + PAGE]
    result: dict[str, Any] = {"resources": page}
    if start + PAGE < len(rows):
        result["nextCursor"] = str(start + PAGE)
    return result


def list_templates(params: dict[str, Any]) -> dict[str, Any]:
    return {"resourceTemplates": TEMPLATES}


def _json(uri: str, payload: Any) -> dict[str, Any]:
    return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(payload, indent=2, default=str)}]}


def _gallery() -> dict[str, Any]:
    from backend_service.app import _current_image_outputs_dir, _current_video_outputs_dir
    import urllib.parse

    from backend_service.mcp_bridge.files import fetchable

    files: list[Path] = []
    for folder, exts in ((_current_image_outputs_dir(), (".png", ".jpg", ".jpeg", ".webp")), (_current_video_outputs_dir(), (".mp4", ".webm", ".mov"))):
        try:
            files += [p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in exts]
        except OSError:
            pass
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for path in files[:GALLERY_LIMIT]:
        real = fetchable(path)  # no hashing here: a listing must stay fast
        items.append({"file": path.name, "kind": "video" if path.suffix.lower() in (".mp4", ".webm", ".mov") else "image",
                      "modifiedAt": int(path.stat().st_mtime), "bytes": path.stat().st_size,
                      "download": {"url": "/api/agent/files?path=" + urllib.parse.quote(str(real), safe="")} if real else None})
    return {"outputs": items, "note": "Newest first. Use the download url (relative to the app address) to fetch a file."}


def read_resource(params: dict[str, Any], ctx: Any) -> dict[str, Any]:
    from backend_service.mcp_bridge import library_tools, tools

    uri = str(params.get("uri") or "")
    if uri == "chaosengine://system/status":
        return _json(uri, tools.system_status(ctx, {}))
    if uri == "chaosengine://music/library":
        return _json(uri, library_tools.music_library(ctx, {"action": "list"}))
    if uri == "chaosengine://gallery/recent":
        return _json(uri, _gallery())
    match = re.fullmatch(r"chaosengine://(storyboard|music)/([^/?#]+)", uri)
    if match is None or not _ID.match(match.group(2)):
        raise ResourceError(f"Unknown resource '{uri[:120]}'. Call resources/list.")
    kind, item_id = match.groups()
    try:
        if kind == "music":
            return _json(uri, library_tools.music_library(ctx, {"action": "get", "track_id": item_id}))
        from backend_service.storyboard.models import dump
        from backend_service.storyboard.timeline import compute_timeline
        from backend_service.routes.storyboard import get_store

        project = get_store().load(item_id)
        return _json(uri, {"project": dump(project), "timeline": compute_timeline(project).as_dict()})
    except tools.ToolError as exc:
        raise ResourceError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - unknown project
        raise ResourceError(str(exc) or f"Unknown resource '{uri}'.") from exc


# ---- prompts ----
_ARG = lambda name, description, required=True: {"name": name, "description": description, "required": required}  # noqa: E731

PROMPTS: list[dict[str, Any]] = [
    {"name": "short_film_from_brief", "title": "Short film from a brief",
     "description": "Plan, generate and render a short film (storyboard) from a written brief.",
     "arguments": [_ARG("brief", "What the film is about."), _ARG("seconds", "Target length in seconds.", False)]},
    {"name": "poster_set", "title": "Poster set",
     "description": "Make a matching set of poster images from one idea.",
     "arguments": [_ARG("subject", "What the posters show."), _ARG("count", "How many posters (default 3).", False),
                   _ARG("style", "Visual style.", False)]},
    {"name": "soundtrack_for_project", "title": "Soundtrack for a film project",
     "description": "Make music that fits a storyboard project and add it to the film's audio lane.",
     "arguments": [_ARG("project_id", "Storyboard project id (see the chaosengine://storyboard resources)."),
                   _ARG("mood", "Mood or genre.", False)]},
]


def list_prompts(params: dict[str, Any]) -> dict[str, Any]:
    start = _cursor(params)
    result: dict[str, Any] = {"prompts": PROMPTS[start:start + PAGE]}
    if start + PAGE < len(PROMPTS):
        result["nextCursor"] = str(start + PAGE)
    return result


def _clip(text: Any, limit: int = 2000) -> str:
    return " ".join(str(text).split())[:limit]


def get_prompt(params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name") or "")
    prompt = next((p for p in PROMPTS if p["name"] == name), None)
    if prompt is None:
        raise ResourceError(f"Unknown prompt '{name[:80]}'. Call prompts/list.", -32602)
    args = params.get("arguments") or {}
    if not isinstance(args, dict):
        raise ResourceError("arguments must be an object.", -32602)
    missing = [a["name"] for a in prompt["arguments"] if a["required"] and not str(args.get(a["name"], "")).strip()]
    if missing:
        raise ResourceError(f"Missing argument: {', '.join(missing)}.", -32602)
    if name == "short_film_from_brief":
        secs = _clip(args.get("seconds") or "30", 8)
        text = (f"Make a short film of about {secs} seconds from this brief: {_clip(args['brief'])}\n\n"
                "1. Call system_status, then list_models for image and video; if one is missing, call recommend_models and tell me.\n"
                "2. storyboard_project create, then storyboard_shot add for each shot (title, stillPrompt, motionPrompt, seconds). "
                "Keep every call's revision current.\n"
                "3. storyboard_generate, then wait_for_job until done. Fix weak shots with storyboard_shot redo and generate again.\n"
                "4. Optionally add music with make_music and storyboard_audio add.\n"
                "5. storyboard_render, wait_for_job, and give me the output file and download link.")
    elif name == "poster_set":
        count = _clip(args.get("count") or "3", 3)
        style = f" in this style: {_clip(args['style'], 300)}" if str(args.get("style", "")).strip() else ""
        text = (f"Make {count} matching posters of: {_clip(args['subject'])}{style}.\n\n"
                "Call list_models for image, pick the best installed fit, and keep one seed family so they match. "
                "Run generate_image once per poster, one job at a time (wait_for_job each), vary only the composition, "
                "and list the finished files with their download links.")
    else:
        mood = f" Mood: {_clip(args['mood'], 300)}." if str(args.get("mood", "")).strip() else ""
        text = (f"Make a soundtrack for storyboard project {_clip(args['project_id'], 64)}.{mood}\n\n"
                "Read the project (resource chaosengine://storyboard/<id> or storyboard_project get) for its length and shot "
                "prompts. Call make_music (instrumental=true, seconds close to the film length), wait_for_job, then "
                "storyboard_audio add with kind=music, the track path, and fields duck=true. Report the new revision.")
    return {"description": prompt["description"], "messages": [{"role": "user", "content": {"type": "text", "text": text}}]}
