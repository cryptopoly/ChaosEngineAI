"""The ``make_music`` agent tool: a song or instrumental into the Music Library, as a job.

It runs the same request model and generation code as ``/api/music/generate`` on the shared heavy-job
queue (kind ``music``, source ``agent``); ``wait_for_job`` returns the FLAC paths with a ``download``
link, and ``cancel_job`` stops a running job through the worker. The description is short on purpose.
"""

from __future__ import annotations

from typing import Any, Callable

from backend_service.mcp_bridge.jobs import STORE
from backend_service.mcp_bridge.tools import ToolError, _PROJECT_PROPS, _obj, _submitted, project_scope

MUSIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "make_music",
        "description": "Make: start a music job (returns a job id): a song with lyrics, or instrumental=true. 10 to 600 s, "
        "FLAC into the Music Library. Cancellable while running.",
        "inputSchema": _obj(
            {
                "caption": {"type": "string", "minLength": 1, "maxLength": 512, "description": "Style: genre, instruments, mood, voice."},
                "lyrics": {"type": "string", "maxLength": 4096, "description": "Lyrics with [verse] / [chorus] tags. Needed unless instrumental."},
                "instrumental": {"type": "boolean"},
                "seconds": {"type": "number", "minimum": 10, "maximum": 600},
                "bpm": {"type": "integer", "minimum": 30, "maximum": 300},
                "key": {"type": "string", "maxLength": 16, "description": "For example 'A minor' or 'F# major'."},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "variations": {"type": "integer", "minimum": 1, "maximum": 4, "description": "1, 2 or 4 takes with seeds seed, seed+1, ..."},
                "model": {"type": "string", "description": "Model id from list_models kind music. Default: ACE-Step 1.5 turbo."},
                **_PROJECT_PROPS,
            },
            ["caption"],
        ),
    },
]


def make_music(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from backend_service import music_runtime
    from backend_service.mcp_bridge import models as bridge_models
    from backend_service.routes.music import MusicGenerateRequest, generate_music_locked

    if int(args.get("variations", 1)) not in (1, 2, 4):
        raise ToolError("variations must be 1, 2 or 4.")
    installed = {str(r["id"]) for r in bridge_models.installed_models(ctx.app.state.chaosengine, "music")}
    model = args.get("model")
    if model and model not in installed:
        raise ToolError(f"'{model}' is not an installed music model. Installed: {', '.join(sorted(installed)) or 'none'}.")
    try:
        variant = music_runtime.resolve_variant(model)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    not_ready = music_runtime.readiness_error(variant)
    if not_ready:
        raise ToolError(not_ready)
    fields: dict[str, Any] = {"modelId": variant["id"], "caption": args["caption"]}
    for name, key in (("lyrics", "lyrics"), ("instrumental", "instrumental"), ("seconds", "seconds"),
                      ("bpm", "bpm"), ("key", "key"), ("seed", "seed"), ("variations", "variations")):
        if name in args:
            fields[key] = args[name]
    try:
        body = MusicGenerateRequest(**fields)
    except ValidationError as exc:
        raise ToolError(f"Invalid music request: {exc.errors()[0]['msg']}") from exc
    request = ctx.request
    scope = project_scope(args)

    def run() -> tuple[dict[str, Any], list[str]]:
        from backend_service import music_library

        with scope:
            result = generate_music_locked(request, body, source="agent")
        tracks = []
        for item in result["items"]:
            path = music_library.audio_path(item["id"])
            tracks.append({"id": item["id"], "path": str(path) if path else None, "seed": item.get("seed"),
                           "durationSeconds": item.get("durationS"), "title": item.get("title")})
        output = {
            "path": tracks[0]["path"] if tracks else None,
            "tracks": tracks,
            "seed": result.get("seed"),
            "model": variant["name"],
            "seconds": result.get("seconds"),
        }
        notes = ["More takes than the first are in tracks[]; each has its own path."] if len(tracks) > 1 else []
        return output, notes

    label = f"{int(body.seconds)} s {'instrumental' if body.instrumental else 'song'}"
    per_10s = (variant.get("estimates") or {}).get("secondsPer10sAudio")
    estimate = round(body.seconds / 10 * per_10s) if per_10s else None
    return _submitted(STORE.submit("music", label, run, estimate_seconds=estimate))


MUSIC_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {"make_music": make_music}
