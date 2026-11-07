"""The fixed v1 tool set. Handlers take (ctx, args) and return a JSON-able dict, or raise ToolError."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from backend_service.mcp_bridge import models as bridge_models
from backend_service.mcp_bridge.jobs import STORE


class ToolError(Exception):
    """Shown to the agent as a readable tool error."""


def _obj(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


_KIND = {"type": "string", "enum": ["image", "video", "voice", "stt", "music"]}
_JOB_ID = {"type": "string", "description": "Job id from a tool that starts a job, or list_jobs."}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "system_status",
        "description": "Read: free memory, platform and the whole job queue with positions and estimated times. Call first.",
        "inputSchema": _obj({}),
    },
    {
        "name": "recommend_models",
        "description": "Read: best models not installed yet that fit this machine. You cannot download them; tell the user which to get.",
        "inputSchema": _obj({"kind": _KIND, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ["kind"]),
    },
    {
        "name": "list_models",
        "description": "Read: installed models of a kind with id, size, fit, limits and frame rules.",
        "inputSchema": _obj({"kind": _KIND}, ["kind"]),
    },
    {"name": "list_voices", "description": "Read: ready text-to-speech voices with language.", "inputSchema": _obj({})},
    {
        "name": "generate_image",
        "description": "Make: start an image job (returns a job id). Sizes snap to what the model supports; see the job notes.",
        "inputSchema": _obj(
            {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 4000},
                "model": {"type": "string", "description": "Model id from list_models. Default: best installed fit."},
                "width": {"type": "integer", "minimum": 256, "maximum": 4096},
                "height": {"type": "integer", "minimum": 256, "maximum": 4096},
                "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                "reference_images": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 4,
                    "description": "Paths or upload: handles. Only for models with acceptsImageInput (list_models); some take one, FLUX 3 Klein up to four.",
                },
                "image_strength": {"type": "number", "minimum": 0.05, "maximum": 1.0, "description": "Single start image only: higher stays closer to it (mflux default 0.4)."},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "include_preview": {"type": "boolean", "description": "Also return the PNG as base64."},
            },
            ["prompt"],
        ),
    },
    {
        "name": "generate_video",
        "description": "Make: start a video job (returns a job id). Give seconds; frames follow the model's rule. Takes minutes.",
        "inputSchema": _obj(
            {
                "prompt": {"type": "string", "minLength": 1, "maxLength": 4000},
                "model": {"type": "string", "description": "Model id from list_models. Default: best installed fit."},
                "width": {"type": "integer", "minimum": 256, "maximum": 4096},
                "height": {"type": "integer", "minimum": 256, "maximum": 4096},
                "seconds": {"type": "number", "minimum": 0.5, "maximum": 20},
                "frames": {"type": "integer", "minimum": 8, "maximum": 257},
                "start_image": {"type": "string", "description": "Path or upload: handle of the first frame. Only for models with acceptsImageInput (list_models)."},
                "end_image": {"type": "string", "description": "Path or upload: handle of the last frame. May be used alone."},
                "image_strength": {"type": "number", "minimum": 0.1, "maximum": 1.0, "description": "1.0 follows the image closely (default); lower lets the clip drift."},
                "fps": {"type": "integer", "minimum": 1, "maximum": 60},
                "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
            },
            ["prompt"],
        ),
    },
    {
        "name": "speak",
        "description": "Make: start a text-to-speech job (returns a job id). Cancellable while running.",
        "inputSchema": _obj(
            {
                "text": {"type": "string", "minLength": 1, "maxLength": 20000},
                "voice": {"type": "string", "description": "Voice id from list_voices."},
                "speed": {"type": "number", "minimum": 0.5, "maximum": 2.0},
            },
            ["text"],
        ),
    },
    {
        "name": "upload_file",
        "description": "Send: a small image, sound or video (about 8 MB) as base64. Returns an upload: handle to use in place "
        "of a path; lasts 24 hours. Larger files: POST the bytes to /api/agent/uploads.",
        "inputSchema": _obj(
            {"data_base64": {"type": "string", "minLength": 1, "maxLength": 12_000_000, "description": "The file, base64."}},
            ["data_base64"],
        ),
    },
    {
        "name": "transcribe",
        "description": "Make: start a speech-to-text job for a file on the machine running the app. Cancellable while running.",
        "inputSchema": _obj(
            {"path": {"type": "string", "description": "Absolute path of the file."}, "language": {"type": "string"}},
            ["path"],
        ),
    },
    {"name": "job_status", "description": "Read: state, progress, queue position, notes and output of a job.", "inputSchema": _obj({"job_id": _JOB_ID}, ["job_id"])},
    {
        "name": "wait_for_job",
        "description": "Read: like job_status, but waits up to timeout_seconds (max 60). Call again if not finished.",
        "inputSchema": _obj({"job_id": _JOB_ID, "timeout_seconds": {"type": "number", "minimum": 1, "maximum": 60}}, ["job_id"]),
    },
    {
        "name": "list_jobs",
        "description": "Read: recent jobs (newest first), kept 7 days and across restarts. Use to find a lost job id.",
        "inputSchema": _obj({"state": {"type": "string", "enum": ["queued", "running", "done", "failed", "cancelled", "interrupted"]},
                             "limit": {"type": "integer", "minimum": 1, "maximum": 100}}),
    },
    {"name": "cancel_job", "description": "Stop: cancel a queued or running job (image, video, music, speech, transcript, storyboard).", "inputSchema": _obj({"job_id": _JOB_ID}, ["job_id"])},
]

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(schema: dict[str, Any], args: Any) -> str | None:
    """A message for the first problem in ``args``, or None. Covers the keywords the tool schemas use."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    props = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in args:
            return f"missing required argument '{name}'"
    for name, value in args.items():
        spec = props.get(name)
        if spec is None:
            return f"unknown argument '{name}'"
        expected = _TYPES[spec["type"]]
        if isinstance(value, bool) and spec["type"] != "boolean":
            return f"'{name}' must be {spec['type']}"
        if not isinstance(value, expected):
            return f"'{name}' must be {spec['type']}"
        if isinstance(value, list):
            if len(value) > spec.get("maxItems", 10**9):
                return f"'{name}' takes at most {spec['maxItems']} items"
            item_type = spec.get("items", {}).get("type")
            if item_type and not all(isinstance(v, _TYPES[item_type]) for v in value):
                return f"'{name}' items must be {item_type}"
        if "enum" in spec and value not in spec["enum"]:
            return f"'{name}' must be one of {', '.join(spec['enum'])}"
        if isinstance(value, str):
            if len(value.strip()) < spec.get("minLength", 0):
                return f"'{name}' is too short"
            if len(value) > spec.get("maxLength", 10**9):
                return f"'{name}' is too long (max {spec['maxLength']})"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < spec.get("minimum", float("-inf")) or value > spec.get("maximum", float("inf")):
                return f"'{name}' must be between {spec.get('minimum')} and {spec.get('maximum')}"
    return None


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def _state(ctx: Any) -> Any:
    return ctx.app.state.chaosengine


def system_status(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    import platform

    from backend_service.helpers.memory_gate import snapshot_memory_signals

    free_gb, pressure = snapshot_memory_signals()
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "freeMemoryGb": round(free_gb, 1),
        "memoryPressurePercent": round(pressure),
        "agentAccess": True,
        "queue": STORE.queue_view(),
        "advice": "One heavy job runs at a time. Start a job, then wait_for_job; do not start several at once.",
    }


def recommend_models(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    return bridge_models.recommend(_state(ctx), args["kind"], int(args.get("limit", 3)))


def list_models(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    rows = bridge_models.installed_models(_state(ctx), args["kind"])
    out: dict[str, Any] = {"kind": args["kind"], "models": rows}
    if not rows:
        out["hint"] = "Nothing installed for this kind. Call recommend_models and tell the user what to download."
    return out


def list_voices(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    return {"voices": bridge_models.list_voices()}


def _pick_variant(ctx: Any, kind: str, model: str | None) -> dict[str, Any]:
    from backend_service.helpers.images import _find_image_variant
    from backend_service.helpers.video import _find_video_variant

    if not model:
        picked = bridge_models.default_model(_state(ctx), kind)
        if picked is None:
            raise ToolError(
                f"No {kind} model is installed. Call recommend_models with kind '{kind}' and tell the user "
                "which to download from ChaosEngineAI > Models."
            )
        model = str(picked["id"])
    installed = {str(r["id"]) for r in bridge_models.installed_models(_state(ctx), kind)}
    if model not in installed:
        raise ToolError(
            f"'{model}' is not an installed {kind} model. Installed: {', '.join(sorted(installed)) or 'none'}."
        )
    variant = (_find_image_variant if kind == "image" else _find_video_variant)(model)
    if variant is None:
        raise ToolError(f"Unknown {kind} model '{model}'.")
    return variant


def _recommended(variant: dict[str, Any]) -> dict[str, Any]:
    from backend_service.helpers.fit import annotate_variants

    rows = annotate_variants([dict(variant)], "video" if variant.get("defaultDurationSeconds") else "image")
    return dict(((rows[0].get("recommendedSettings") or {}).get("settings")) or {})


def _artifact_notes(artifact: dict[str, Any]) -> list[str]:
    note = artifact.get("runtimeNote")
    return [str(n).strip() for n in str(note).split("\n") if n.strip()] if note else []


def _shape_like(found: tuple[int, int], size: tuple[int, int]) -> tuple[int, int]:
    """Keep ``size``'s pixel count, take the image's shape (multiples of 64 as the engines want)."""
    pixels = size[0] * size[1]
    ratio = found[0] / found[1]
    width = max(256, round((pixels * ratio) ** 0.5 / 64) * 64)
    height = max(256, round(width / ratio / 64) * 64)
    return width, height


def generate_image(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from backend_service.helpers.generation_limits import parse_resolution
    from backend_service.models import ImageGenerationRequest
    from backend_service.routes.images import _generate_image_locked

    variant = _pick_variant(ctx, "image", args.get("model"))
    settings = _recommended(variant)
    fields: dict[str, Any] = {"modelId": variant["id"], "prompt": args["prompt"]}
    for key in ("width", "height", "seed"):
        if key in args:
            fields[key] = args[key]
    steps = args.get("steps", settings.get("steps"))
    if steps is not None:
        fields["steps"] = steps
    if settings.get("guidance") is not None:
        fields["guidance"] = settings["guidance"]
    notes: list[str] = []
    if args.get("reference_images"):
        from backend_service.helpers.media_input import InputImageError, checked_reference_images, image_size

        try:
            refs = checked_reference_images(list(args["reference_images"]), variant)
        except InputImageError as exc:
            raise ToolError(str(exc)) from exc
        fields["referenceImagePaths"] = refs
        if "image_strength" in args:
            fields["imageStrength"] = args["image_strength"]
        found = image_size(Path(refs[0]))
        if found and "width" not in args and "height" not in args:
            size = _shape_like(found, parse_resolution(variant.get("recommendedResolution")) or (1024, 1024))
            fields["width"], fields["height"] = size
            notes.append(f"Image shape {size[0]}x{size[1]} follows the first reference ({found[0]}x{found[1]}).")
    try:
        body = ImageGenerationRequest(**fields)
    except ValidationError as exc:
        raise ToolError(f"Invalid image request: {exc.errors()[0]['msg']}") from exc
    preview = bool(args.get("include_preview"))
    request = ctx.request

    def run() -> tuple[dict[str, Any], list[str]]:
        result = _generate_image_locked(request, body)
        art = result["artifacts"][0]
        output: dict[str, Any] = {
            "path": art.get("imagePath"),
            "width": art.get("width"),
            "height": art.get("height"),
            "seed": art.get("seed"),
            "model": art.get("modelName"),
            "durationSeconds": art.get("durationSeconds"),
        }
        if preview and art.get("imagePath"):
            output["previewBase64"] = base64.b64encode(Path(art["imagePath"]).read_bytes()).decode("ascii")
        return output, notes + _artifact_notes(art)

    return _submitted(STORE.submit("image", str(variant.get("name") or variant["id"]), run,
                                   estimate_seconds=variant.get("estimatedGenerationSeconds")))


def generate_video(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from pydantic import ValidationError

    from backend_service.helpers.generation_limits import align_video_frames, parse_resolution
    from backend_service.models import VideoGenerationRequest
    from backend_service.routes.video import _generate_video_locked

    variant = _pick_variant(ctx, "video", args.get("model"))
    settings = _recommended(variant)
    fps = int(args.get("fps", 24))
    if "frames" in args:
        frames = int(args["frames"])
    else:
        seconds = float(args.get("seconds") or variant.get("defaultDurationSeconds") or 4)
        frames = round(seconds * fps)
    frames = max(8, min(257, frames))
    notes: list[str] = []
    frames, frame_note = align_video_frames(str(variant.get("repo") or ""), frames)
    if frame_note:
        notes.append(frame_note)
    size = parse_resolution(variant.get("recommendedResolution")) or (768, 512)
    image_fields: dict[str, Any] = {}
    if args.get("start_image") or args.get("end_image"):
        from backend_service.helpers.media_input import InputImageError, image_size, resolve_input_image
        from backend_service.ltx2_mlx_runtime import _is_ltx2_mlx_repo

        if not _is_ltx2_mlx_repo(str(variant.get("repo") or "")):
            raise ToolError(f"{variant.get('name')} cannot start from an image. Use a model with acceptsImageInput in list_models.")
        try:
            start = resolve_input_image(args["start_image"]) if args.get("start_image") else None
            end = resolve_input_image(args["end_image"]) if args.get("end_image") else None
            if start:
                image_fields["startImagePath"] = str(start)
            if end:
                image_fields["endImagePath"] = str(end)
        except InputImageError as exc:
            raise ToolError(str(exc)) from exc
        if "image_strength" in args:
            image_fields["imageStrength"] = args["image_strength"]
        found = image_size(start or end)
        if found and "width" not in args and "height" not in args:
            # Keep the recommended pixel count, take the image's shape (multiples of 64 as the engine wants).
            size = _shape_like(found, size)
            notes.append(f"Clip shape {size[0]}x{size[1]} follows the {'start' if start else 'end'} image ({found[0]}x{found[1]}).")
    fields: dict[str, Any] = {
        **image_fields,
        "modelId": variant["id"],
        "prompt": args["prompt"],
        "numFrames": frames,
        "fps": fps,
        "width": args.get("width", size[0]),
        "height": args.get("height", size[1]),
    }
    if "seed" in args:
        fields["seed"] = args["seed"]
    for key in ("steps", "guidance"):
        value = args.get(key, settings.get(key))
        if value is not None:
            fields[key] = value
    try:
        body = VideoGenerationRequest(**fields)
    except ValidationError as exc:
        raise ToolError(f"Invalid video request: {exc.errors()[0]['msg']}") from exc
    request = ctx.request

    def run() -> tuple[dict[str, Any], list[str]]:
        result = _generate_video_locked(request, body)
        art = result["artifact"]
        output = {
            "path": art.get("videoPath"),
            "width": art.get("width"),
            "height": art.get("height"),
            "durationSeconds": art.get("clipDurationSeconds"),
            "frames": art.get("numFrames"),
            "fps": art.get("fps"),
            "hasAudio": art.get("hasAudio"),
            "seed": art.get("seed"),
            "model": art.get("modelName"),
        }
        return output, notes + _artifact_notes(art)

    return _submitted(STORE.submit("video", str(variant.get("name") or variant["id"]), run,
                                   estimate_seconds=variant.get("estimatedGenerationSeconds")))


def speak(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service import voice_library, voice_runtime

    text = args["text"]
    settings = voice_runtime.load_settings()
    voice = args.get("voice") or settings.get("ttsVoice")
    speed = float(args.get("speed", settings.get("speed", 1.0)))
    ready = {v["id"] for v in bridge_models.list_voices()}
    if args.get("voice") and voice not in ready:
        raise ToolError(f"Voice '{voice}' is not available. Call list_voices.")

    def run() -> tuple[dict[str, Any], list[str]]:
        wav, duration = voice_runtime.synthesize(text, voice, speed)
        with tempfile.NamedTemporaryFile(prefix="chaosengine-agent-", suffix=".wav", delete=False) as tmp:
            tmp.write(wav)
        item = voice_library.create(
            "speech",
            {"text": text, "voice": voice, "speed": speed, "durationS": duration},
            audio_file=Path(tmp.name),
            audio_ext=".wav",
        )
        path = voice_library.audio_path(item["id"])
        return {"path": str(path) if path else None, "durationSeconds": round(duration, 2), "voice": voice}, []

    return _submitted(STORE.submit("voice", "speech", run))


def upload_file(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.helpers.media_input import InputImageError
    from backend_service.mcp_bridge.files import save_upload

    try:
        data = base64.b64decode(str(args["data_base64"]), validate=True)
    except ValueError as exc:
        raise ToolError("data_base64 is not valid base64.") from exc
    try:
        return save_upload(data)
    except InputImageError as exc:
        raise ToolError(str(exc)) from exc


def transcribe(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service import voice_runtime

    path = Path(args["path"]).expanduser()
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ToolError(f"File not found: {args['path']}") from exc
    if not path.is_file():
        raise ToolError(f"Not a file: {path}")
    language = str(args.get("language") or "").strip() or None

    def run() -> tuple[dict[str, Any], list[str]]:
        result = voice_runtime.transcribe_file(path, "", language)
        return {k: result.get(k) for k in ("text", "language", "durationS", "model") if k in result}, []

    return _submitted(STORE.submit("voice", f"transcribe {path.name}", run))


def _submitted(job: Any) -> dict[str, Any]:
    return {**job.describe(), "next": "Call wait_for_job with this jobId."}


def _job(args: dict[str, Any]) -> Any:
    job = STORE.get(str(args["job_id"]))
    if job is None:
        raise ToolError(f"Unknown job '{args['job_id']}'. Call list_jobs to see recent jobs (kept 7 days).")
    return job


def job_status(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    return _job(args).describe()


def wait_for_job(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    import time

    from backend_service.mcp_bridge import progress

    job = _job(args)
    deadline = time.monotonic() + min(60.0, float(args.get("timeout_seconds", 30)))
    started = time.monotonic()
    # With a progressToken the wait is cut into 1 s steps that each send a notification; without one it is one wait.
    while not job.done.is_set() and (left := deadline - time.monotonic()) > 0:
        if progress.PROGRESS.get() is None:
            job.done.wait(timeout=left)
            break
        job.done.wait(timeout=min(1.0, left))
        snap = job.describe().get("progress") or {}
        step, total = snap.get("step"), snap.get("totalSteps")
        text = f"{job.state}" + (f", {snap['phase']}" if snap.get("phase") else "")
        if job.state == "queued":
            text += f", position {job.describe().get('queuePosition', 0)}"
        if isinstance(step, (int, float)) and isinstance(total, (int, float)) and total:
            progress.report(float(step), float(total), text)
        else:
            progress.report(round(time.monotonic() - started, 1), None, text)
    info = job.describe()
    if not job.done.is_set():
        info["next"] = "Not finished. Call wait_for_job again."
    return info


def list_jobs(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    jobs = STORE.list(args.get("state"), int(args.get("limit", 20)))
    return {"jobs": [j.describe() for j in jobs]}


def cancel_job(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    job = _job(args)
    return {"cancelRequested": STORE.cancel(job), **job.describe()}


HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "system_status": system_status,
    "recommend_models": recommend_models,
    "list_models": list_models,
    "list_voices": list_voices,
    "generate_image": generate_image,
    "generate_video": generate_video,
    "speak": speak,
    "upload_file": upload_file,
    "upload_image": upload_file,  # old name, kept so existing setups keep working
    "list_jobs": list_jobs,
    "transcribe": transcribe,
    "job_status": job_status,
    "wait_for_job": wait_for_job,
    "cancel_job": cancel_job,
}


def make_context(request: Any) -> Any:
    return SimpleNamespace(app=request.app, request=request)


# Storyboard tools live in their own module; registered here so the tool list stays in one place.
from backend_service.mcp_bridge.storyboard_tools import STORYBOARD_HANDLERS, STORYBOARD_TOOLS  # noqa: E402

TOOLS.extend(STORYBOARD_TOOLS)
HANDLERS.update(STORYBOARD_HANDLERS)

from backend_service.mcp_bridge.music_tools import MUSIC_HANDLERS, MUSIC_TOOLS  # noqa: E402

TOOLS.extend(MUSIC_TOOLS)
HANDLERS.update(MUSIC_HANDLERS)

from backend_service.mcp_bridge.voice_clone_tool import CLONE_HANDLERS, CLONE_TOOLS  # noqa: E402

TOOLS.extend(CLONE_TOOLS)
HANDLERS.update(CLONE_HANDLERS)

from backend_service.mcp_bridge.sound_tool import SOUND_HANDLERS, SOUND_TOOLS  # noqa: E402

TOOLS.extend(SOUND_TOOLS)
HANDLERS.update(SOUND_HANDLERS)

from backend_service.mcp_bridge.library_tools import LIBRARY_HANDLERS, LIBRARY_TOOLS  # noqa: E402

TOOLS.extend(LIBRARY_TOOLS)
HANDLERS.update(LIBRARY_HANDLERS)


from backend_service.mcp_bridge.scopes import apply_annotations  # noqa: E402

apply_annotations(TOOLS)
