"""The ``make_sound`` agent tool: a short sound effect or loop into the Music Library, as a job.

It runs the same request model and generation code as ``/api/music/sound/generate`` on the shared heavy-job
queue (kind ``music``, source ``agent``); ``wait_for_job`` returns the FLAC path with a ``download`` link and
``cancel_job`` stops a running job through the sounds worker. NOT registered here: the bridge stream adds
``SOUND_TOOLS`` to the tool list and ``SOUND_HANDLERS`` to the dispatch table (like ``music_tools``). The model
is gated, so an install that has not accepted the Stability AI licence on Hugging Face gets a plain error.
"""

from __future__ import annotations

from typing import Any, Callable

from backend_service.mcp_bridge.jobs import STORE
from backend_service.mcp_bridge.tools import ToolError, _obj, _submitted

SOUND_TOOLS: list[dict[str, Any]] = [
    {
        "name": "make_sound",
        "description": "Make: start a sound-effect job (returns a job id): footsteps, doors, ambience, a loop. 1 to 11 s, "
        "English prompt, FLAC in the Music Library. Not songs or speech. Needs the gated Stable Audio Open model the user installed.",
        "inputSchema": _obj(
            {
                "caption": {"type": "string", "minLength": 1, "maxLength": 300, "description": "The sound: 'heavy wooden door creaking open, close mic'."},
                "seconds": {"type": "number", "minimum": 1, "maximum": 11},
                "loop": {"type": "boolean", "description": "Make the clip a seamless loop (ambience, beats)."},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "variations": {"type": "integer", "minimum": 1, "maximum": 4, "description": "1, 2 or 4 takes with seeds seed, seed+1, ..."},
            },
            ["caption"],
        ),
    },
]


def make_sound(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from backend_service import music_runtime
    from backend_service.catalog.music_models import music_variants
    from backend_service.routes.music import SoundGenerateRequest, generate_sound_locked

    if int(args.get("variations", 1)) not in (1, 2, 4):
        raise ToolError("variations must be 1, 2 or 4.")
    variant = next((v for v in music_variants() if music_runtime.is_sound(v)), None)
    if variant is None:
        raise ToolError("No sound model is available in this build.")
    not_ready = music_runtime.readiness_error(variant)
    if not_ready:
        raise ToolError(not_ready)
    fields: dict[str, Any] = {"modelId": variant["id"], "caption": args["caption"]}
    for key in ("seconds", "loop", "seed", "variations"):
        if key in args:
            fields[key] = args[key]
    try:
        body = SoundGenerateRequest(**fields)
    except ValidationError as exc:
        raise ToolError(f"Invalid sound request: {exc.errors()[0]['msg']}") from exc
    request = ctx.request

    def run() -> tuple[dict[str, Any], list[str]]:
        from backend_service import music_library

        result = generate_sound_locked(request, body, source="agent")
        tracks = []
        for item in result["items"]:
            path = music_library.audio_path(item["id"])
            tracks.append({"id": item["id"], "path": str(path) if path else None, "seed": item.get("seed"),
                           "durationSeconds": item.get("durationS"), "loop": bool(item.get("loop")), "title": item.get("title")})
        output = {
            "path": tracks[0]["path"] if tracks else None,
            "tracks": tracks,
            "seed": result.get("seed"),
            "model": variant["name"],
            "seconds": result.get("seconds"),
            "licence": music_runtime.licence_note(variant),
        }
        notes = ["More takes than the first are in tracks[]; each has its own path."] if len(tracks) > 1 else []
        notes.append("Made with Stable Audio Open: keep 'Powered by Stability AI' where you publish with it.")
        return output, notes

    label = f"{body.seconds:g} s {'loop' if body.loop else 'sound'}"
    per_10s = (variant.get("estimates") or {}).get("secondsPer10sAudio")
    estimate = round(body.seconds / 10 * per_10s) + 5 if per_10s else None
    return _submitted(STORE.submit("music", label, run, estimate_seconds=estimate))


SOUND_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {"make_sound": make_sound}
