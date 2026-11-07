"""The real still and clip generators a storyboard batch uses, wired to the Studio code paths.

They build requests the way the agent tools do (``mcp_bridge.tools`` model pick and recommended settings,
the model's frame rule) and call the shared ``_generate_image_locked`` / ``_generate_video_locked`` inside
a heavy-queue slot with source ``storyboard`` and the batch id, so a Studio Cancel never touches a batch
and the batch's own Cancel reaches its running and waiting tickets. Heavy imports stay inside the calls.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from backend_service.storyboard import media
from backend_service.storyboard.models import Project, Shot


def _variant(request: Any, kind: str, model: str | None) -> dict[str, Any]:
    from backend_service.mcp_bridge.tools import ToolError, _pick_variant, make_context

    try:
        return _pick_variant(make_context(request), kind, model)
    except ToolError as exc:
        raise RuntimeError(str(exc)) from exc


def _readable(exc: Exception) -> RuntimeError:
    return RuntimeError(str(getattr(exc, "detail", None) or exc))


def studio_generators(request: Any, batch_id: str, folder: Path) -> dict[str, Callable[..., Any]]:
    from backend_service.helpers.job_queue import HEAVY_JOBS
    from backend_service.progress import IMAGE_PROGRESS, VIDEO_PROGRESS

    def make_still(shot: Shot, project: Project) -> tuple[str, dict[str, Any]]:
        from fastapi import HTTPException

        from backend_service.mcp_bridge.tools import _recommended
        from backend_service.models import ImageGenerationRequest
        from backend_service.routes.images import _generate_image_locked

        prompt = shot.stillPrompt or shot.title
        if not prompt:
            raise RuntimeError("the shot has no still prompt")
        variant = _variant(request, "image", project.defaults.imageModel)
        settings = _recommended(variant)
        fields: dict[str, Any] = {"modelId": variant["id"], "prompt": prompt,
                                  "width": project.defaults.width, "height": project.defaults.height}
        if shot.seed is not None:
            fields["seed"] = shot.seed
        for key in ("steps", "guidance"):
            if settings.get(key) is not None:
                fields[key] = settings[key]
        refs = [project.reference(r) for r in shot.referenceImages]
        paths = [str(media.inside(folder, r.image)) for r in refs if r is not None and r.image]
        if paths:
            fields["referenceImagePaths"] = paths
        body = ImageGenerationRequest(**fields)
        try:
            with HEAVY_JOBS.slot("image", "storyboard", f"{project.name}: {shot.id} still", batch=batch_id):
                result = _generate_image_locked(request, body)
        except HTTPException as exc:
            raise _readable(exc) from exc
        art = result["artifacts"][0]
        info = {"model": art.get("modelName") or variant.get("name"), "seed": art.get("seed"),
                "steps": body.steps, "width": art.get("width"), "height": art.get("height")}
        return str(art["imagePath"]), {"settings": info}

    def make_clip(shot: Shot, project: Project, still: Path | None) -> tuple[str, dict[str, Any]]:
        from fastapi import HTTPException

        from backend_service.helpers.generation_limits import align_video_frames
        from backend_service.ltx2_mlx_runtime import _is_ltx2_mlx_repo
        from backend_service.mcp_bridge.tools import _recommended
        from backend_service.models import VideoGenerationRequest
        from backend_service.routes.video import _generate_video_locked

        prompt = shot.motionPrompt or shot.stillPrompt or shot.title
        if not prompt:
            raise RuntimeError("the shot has no motion prompt")
        variant = _variant(request, "video", project.defaults.videoModel)
        settings = _recommended(variant)
        fps = project.defaults.fps
        frames, _ = align_video_frames(str(variant.get("repo") or ""), max(8, min(257, round(shot.seconds * fps))))
        fields: dict[str, Any] = {"modelId": variant["id"], "prompt": prompt, "numFrames": frames, "fps": fps,
                                  "width": project.defaults.width, "height": project.defaults.height}
        if _is_ltx2_mlx_repo(str(variant.get("repo") or "")):
            if still is not None:
                fields["startImagePath"] = str(still)
            if shot.endImage:
                fields["endImagePath"] = str(media.inside(folder, shot.endImage))
        elif still is not None or shot.startFrom in ("generate", "previous-shot"):
            raise RuntimeError(_no_image_input(variant))  # never a silent text-to-video for a shot that asked for an image
        if shot.seed is not None:
            fields["seed"] = shot.seed
        for key in ("steps", "guidance"):
            if settings.get(key) is not None:
                fields[key] = settings[key]
        body = VideoGenerationRequest(**fields)
        try:
            with HEAVY_JOBS.slot("video", "storyboard", f"{project.name}: {shot.id} clip", batch=batch_id):
                result = _generate_video_locked(request, body)
        except HTTPException as exc:
            raise _readable(exc) from exc
        art = result["artifact"]
        n, rate = art.get("numFrames") or frames, art.get("fps") or fps
        info = {"model": art.get("modelName") or variant.get("name"), "seed": art.get("seed"),
                "frames": n, "fps": rate, "width": art.get("width"), "height": art.get("height")}
        return str(art["videoPath"]), {"settings": info, "measuredSeconds": round(n / rate, 4) if rate else None}

    def clip_image_refusal(project: Project) -> str | None:
        from backend_service.ltx2_mlx_runtime import _is_ltx2_mlx_repo

        try:
            variant = _variant(request, "video", project.defaults.videoModel)
        except RuntimeError:
            return None  # no model yet: the clip step fails with the readable pick error
        return None if _is_ltx2_mlx_repo(str(variant.get("repo") or "")) else _no_image_input(variant)

    def unload(kind: str) -> None:
        state = request.app.state.chaosengine
        try:
            if kind == "image":
                state.image_runtime.unload()
            elif kind == "video":
                state.video_runtime.unload()
        except Exception as exc:  # noqa: BLE001 - freeing memory is best effort
            state.add_log("storyboard", "warning", f"Could not unload the {kind} model: {type(exc).__name__}")

    def stop_running(kind: str) -> bool:
        tracker = {"image": IMAGE_PROGRESS, "video": VIDEO_PROGRESS}.get(kind)
        return bool(tracker and tracker.request_cancel())

    return {"make_still": make_still, "make_clip": make_clip, "unload": unload,
            "last_frame": last_frame, "stop_running": stop_running, "clip_image_refusal": clip_image_refusal}


def _no_image_input(variant: dict[str, Any]) -> str:
    return (f"{variant.get('name') or variant.get('id') or 'This video model'} cannot start a clip from an image, so it "
            "would ignore this shot's still or the previous shot. Pick LTX-2.3 or LTX-2.5 (the models with "
            "acceptsImageInput) as the project's video model, or import a clip.")


def last_frame(clip: Path) -> Path:
    """The clip's final frame as a PNG (bundled ffmpeg), for ``startFrom: previous-shot``."""
    exe = media.ffmpeg_exe()
    if not exe:
        raise RuntimeError("the bundled ffmpeg is not available")
    out = Path(tempfile.mkdtemp(prefix="chaosengine-frame-")) / f"{clip.parent.name}-last.png"
    result = subprocess.run([exe, "-hide_banner", "-nostdin", "-loglevel", "error", "-sseof", "-1", "-i", str(clip),
                             "-update", "1", "-frames:v", "1000", "-y", str(out)], capture_output=True, timeout=120)
    if result.returncode != 0 or not out.is_file():
        raise RuntimeError("could not read the previous clip's last frame")
    return out
