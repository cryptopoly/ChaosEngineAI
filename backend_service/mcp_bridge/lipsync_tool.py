"""The ``make_lipsync_shot`` agent tool: a start still plus a vocal clip become a lip-synced clip, as a job (D-103).

It builds the same ``VideoGenerationRequest`` (with ``audioPath``) as POST /api/video/generate and runs it through
``_generate_video_locked`` on the shared heavy-job queue (kind ``video``), like ``generate_video``; ``wait_for_job``
returns the mp4 path and ``cancel_job`` stops it. Registered from ``tools.py`` like the music and sound tools.
"""

from __future__ import annotations

from typing import Any, Callable

from backend_service.mcp_bridge.jobs import STORE
from backend_service.mcp_bridge.tools import (
    ToolError,
    _PROJECT_PROPS,
    _artifact_notes,
    _obj,
    _pick_variant,
    _refuse_placeholder,
    _shape_like,
    _state,
    _submitted,
    project_scope,
)

LIPSYNC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "make_lipsync_shot",
        "description": "Make: start a lip-sync video job (returns a job id): a start still and a vocal clip (WAV or FLAC, "
        "speech or singing) become a 3 to 8 s clip whose mouth follows the audio, with that audio as its sound. Needs an "
        "LTX-2.3 model with lipSync in list_models (4-bit on a 24 GB NVIDIA card with about 37 GB free RAM; the Mac path "
        "is untested). About 1-2 minutes per 5 s shot on an RTX 4090. For singing (measured): a vocal stem, not the mix; set "
        "singing; start the shot about 0.4 s before a phrase and keep it to 3-5 s; a frontal head-and-shoulders still; "
        "make three takes and keep the best (about 1 take in 4 sings with a shut mouth or turns away).",
        "inputSchema": _obj(
            {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "The shot: who sings or speaks, framing, light. Keep the camera still."},
                "start_image": {"type": "string", "description": "Path or upload: handle of the still the clip starts from (a face, frontal works best)."},
                "audio": {"type": "string", "description": "upload: handle, or a file in the app's output folders: the vocal clip (WAV or FLAC; mono is fine). A vocal stem follows best; the full mix works less well."},
                "singing": {"type": "boolean", "description": "The vocal is sung: adds singing wording to the prompt (measured to stop takes that listen with a shut mouth). Leave off for speech."},
                "audio_start": {"type": "number", "minimum": 0, "maximum": 3600, "description": "Where in the audio the shot starts, in seconds (default 0)."},
                "seconds": {"type": "number", "minimum": 3, "maximum": 8, "description": "Shot length; default: the audio from audio_start, held to 3 to 8 s. Frames follow 8k+1 at 24 fps."},
                "model": {"type": "string", "description": "Model id from list_models with lipSync. Default: the installed one."},
                "width": {"type": "integer", "minimum": 256, "maximum": 1280},
                "height": {"type": "integer", "minimum": 256, "maximum": 1280},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                **_PROJECT_PROPS,
            },
            ["prompt", "start_image", "audio"],
        ),
    },
]

FPS = 24


def _lipsync_variant(ctx: Any, model: str | None) -> dict[str, Any]:
    from backend_service import lipsync_runtime
    from backend_service.mcp_bridge import models as bridge_models

    if model:
        variant = _pick_variant(ctx, "video", model)
        if not lipsync_runtime.supports(variant):
            raise ToolError(f"{variant.get('name')} cannot lip-sync. Use a video model with lipSync in list_models.")
        return variant
    for row in bridge_models.installed_models(_state(ctx), "video"):
        if row.get("lipSync"):
            return _pick_variant(ctx, "video", str(row["id"]))
    raise ToolError("No lip-sync model is installed. Tell the user to download LTX-2.3 distilled 4-bit (NVIDIA, 24 GB "
                    "card) from ChaosEngineAI > Video > Discover.")


def make_lipsync_shot(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from backend_service import lipsync_runtime
    from backend_service.helpers.generation_limits import parse_resolution, shot_clip_frames
    from backend_service.helpers.media_input import InputImageError, image_size, resolve_input_image
    from backend_service.mcp_bridge.storyboard_tools import _agent_source
    from backend_service.models import VideoGenerationRequest
    from backend_service.routes.video import _generate_video_locked
    from backend_service.storyboard import media

    variant = _lipsync_variant(ctx, args.get("model"))
    try:
        start = resolve_input_image(args["start_image"])
        # Like a Storyboard vocal: an upload handle or an output file only. The clip carries this audio as its sound,
        # so any other path would let an agent copy any WAV or FLAC on the machine (security review 2026-11-20).
        audio = lipsync_runtime.resolve_audio(_agent_source(args["audio"]))
    except (InputImageError, lipsync_runtime.LipSyncError) as exc:
        raise ToolError(str(exc)) from exc
    audio_start = float(args.get("audio_start", 0.0))
    notes: list[str] = []
    if "seconds" in args:
        seconds = float(args["seconds"])
    else:
        length = media.probe_seconds(audio)
        if length is None:
            raise ToolError("The audio length could not be read; give seconds.")
        seconds = min(lipsync_runtime.MAX_SECONDS, max(lipsync_runtime.MIN_SECONDS, length - audio_start))
        if length - audio_start > lipsync_runtime.MAX_SECONDS:
            notes.append(f"The audio runs {length - audio_start:.1f} s from audio_start; this shot takes the first "
                         f"{lipsync_runtime.MAX_SECONDS:g} s. Make more shots with a later audio_start for the rest.")
    frames = shot_clip_frames(str(variant.get("repo") or ""), seconds, FPS)
    size = parse_resolution(variant.get("recommendedResolution")) or (768, 512)
    found = image_size(start)
    if found and "width" not in args and "height" not in args:
        size = _shape_like(found, size)
    fields: dict[str, Any] = {
        "modelId": variant["id"], "prompt": args["prompt"], "numFrames": frames, "fps": FPS,
        "width": args.get("width", size[0]), "height": args.get("height", size[1]),
        "startImagePath": str(start), "audioPath": str(audio), "audioStartSeconds": audio_start,
    }
    if "seed" in args:
        fields["seed"] = args["seed"]
    if args.get("singing"):
        fields["lipSyncSinging"] = True
    try:
        body = VideoGenerationRequest(**fields)
        lipsync_runtime.shot_frames(body.numFrames, body.fps)
    except ValidationError as exc:
        raise ToolError(f"Invalid lip-sync request: {exc.errors()[0]['msg']}") from exc
    except lipsync_runtime.LipSyncError as exc:
        raise ToolError(str(exc)) from exc
    request = ctx.request
    scope = project_scope(args)

    def run() -> tuple[dict[str, Any], list[str]]:
        with scope:
            result = _generate_video_locked(request, body)
        _refuse_placeholder(result.get("runtime"))
        art = result["artifact"]
        output = {
            "path": art.get("videoPath"), "width": art.get("width"), "height": art.get("height"),
            "durationSeconds": art.get("clipDurationSeconds"), "frames": art.get("numFrames"), "fps": art.get("fps"),
            "hasAudio": art.get("hasAudio"), "seed": art.get("seed"), "model": art.get("modelName"),
            "timings": art.get("timings"),
        }
        return output, notes + _artifact_notes(art)

    estimate = lipsync_runtime.estimate_seconds(variant, int(body.width), int(body.height), body.numFrames)
    return _submitted(STORE.submit("video", f"lip sync · {variant.get('name') or variant['id']}", run,
                                   estimate_seconds=estimate))


LIPSYNC_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {"make_lipsync_shot": make_lipsync_shot}
