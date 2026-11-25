"""Video upscale jobs, fast tier (``video_upscale``): models, start, status, cancel, resume.

* ``GET  /api/video/upscale/models``: the catalog rows with ``fit`` for this machine and whether each is installed.
* ``POST /api/video/upscale``: start a job (``inputPath``, ``modelId``, ``scale`` or ``width`` + ``height``,
  optional ``outputPath`` and ``deinterlace``). The same request again returns the same job (its id is a hash of
  the request), so a client that lost the id finds it.
* ``GET  /api/video/upscale/jobs`` (the 20 newest), ``GET .../jobs/{id}``, ``POST .../cancel``, ``POST .../resume``.
* ``GET  /api/video/upscale/jobs/{id}/file`` streams a finished film (``Range`` works), ``.../frame?which=input|output``
  is the first frame as a PNG (cached beside the work folders).
* ``POST /api/video/upscale/probe``: size, frame rate, length, sound and field order of a file, checked like a start.

Paths are hostile input: the input must be a real local video file (resolved, regular, a video extension, under the
size cap); the output must resolve inside one of the app's output folders (``mcp_bridge.files.output_roots``) with
an ``.mp4``, ``.mov`` or ``.mkv`` name, and is never the input. Without ``outputPath`` the film goes to
``<video outputs>/upscaled/``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from backend_service.catalog.video_upscale_models import DEFAULT_MODEL_ID, catalog_with_fit, find_model
from backend_service.video_upscale import addon, jobs, pipeline
from backend_service.video_upscale.engine import UpscaleError, check_target, target_size

router = APIRouter()

INPUT_EXTENSIONS = frozenset({".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".ts", ".m2ts"})
OUTPUT_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv"})
MAX_INPUT_BYTES = 256 * 1024**3  # an hour of DV tape is about 13 GB
MAX_LISTED_JOBS = 20
_VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}


class UpscaleRequest(BaseModel):
    inputPath: str = Field(min_length=1, max_length=4096)
    modelId: str = Field(DEFAULT_MODEL_ID, max_length=64)
    scale: float | None = Field(None, ge=1.0, le=4.0)
    width: int | None = Field(None, ge=16, le=8192)
    height: int | None = Field(None, ge=16, le=8192)
    outputPath: str | None = Field(None, max_length=4096)
    deinterlace: bool = False

    @model_validator(mode="after")
    def _one_size(self) -> "UpscaleRequest":
        if (self.width is None) != (self.height is None):
            raise ValueError("give both width and height, or neither")
        if self.width is not None and self.scale is not None:
            raise ValueError("give a scale or a width and height, not both")
        return self


class ProbeRequest(BaseModel):
    inputPath: str = Field(min_length=1, max_length=4096)


def resolve_input(raw: str) -> Path:
    """The real path of a local video file, or ``UpscaleError``."""
    try:
        path = Path(raw.strip()).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UpscaleError(f"File not found: {Path(raw).name}") from exc
    if not path.is_file():
        raise UpscaleError(f"Not a file: {path.name}")
    if path.suffix.lower() not in INPUT_EXTENSIONS:
        raise UpscaleError(f"'{path.suffix or 'no extension'}' is not a video file this tool reads: {path.name}")
    size = path.stat().st_size
    if size == 0 or size > MAX_INPUT_BYTES:
        raise UpscaleError(f"File is empty or larger than {MAX_INPUT_BYTES // 1024**3} GB: {path.name}")
    return path


def output_roots() -> list[Path]:
    from backend_service.mcp_bridge.files import output_roots as roots  # noqa: PLC0415

    return roots()


def default_output_dir() -> Path:
    from backend_service.app import _current_video_outputs_dir  # noqa: PLC0415

    return Path(_current_video_outputs_dir()) / "upscaled"


def resolve_output(raw: str | None, src: Path, model_id: str, size: tuple[int, int]) -> Path:
    """Where the film goes: ``raw`` when it is inside an output folder, else a name in the default folder."""
    if raw:
        candidate = Path(raw.strip()).expanduser()
        if not candidate.is_absolute():
            raise UpscaleError("outputPath must be an absolute path inside one of the app's output folders.")
    else:
        candidate = default_output_dir() / f"{src.stem}-{model_id}-{size[0]}x{size[1]}.mp4"
    if candidate.suffix.lower() not in OUTPUT_EXTENSIONS:
        raise UpscaleError(f"The output must end in {', '.join(sorted(OUTPUT_EXTENSIONS))}.")
    try:
        if not raw:
            candidate.parent.mkdir(parents=True, exist_ok=True)
        real = candidate.parent.resolve(strict=True) / candidate.name
    except (OSError, RuntimeError) as exc:
        raise UpscaleError(f"The output folder does not exist: {candidate.parent}") from exc
    if real.is_symlink() or not any(real.is_relative_to(root) for root in output_roots()):
        raise UpscaleError("The output must be inside one of the app's output folders (images, videos, music, "
                           "voice or a Storyboard project's exports).")
    if real == src:
        raise UpscaleError("The output must not be the input file.")
    return real


def _http(exc: Exception) -> HTTPException:
    from backend_service.storyboard.ffmpeg_bin import FfmpegMissing  # noqa: PLC0415

    if isinstance(exc, FfmpegMissing):
        return HTTPException(status_code=409, detail={"code": exc.action, "message": str(exc)})
    if isinstance(exc, UpscaleError):
        return HTTPException(status_code=exc.status, detail=str(exc))
    raise exc


def _fit_refusal(model_id: str) -> str | None:
    row = next((r for r in catalog_with_fit() if r["id"] == model_id), None)
    fit = (row or {}).get("fit") or {}
    if fit.get("status") == "wont-fit":
        return f"{row['name']} will not fit on this machine: {fit.get('reason')}"
    return None


def _deinterlace_available() -> bool:
    """Whether the ffmpeg a job would use has the deinterlace filter (``False`` when there is no ffmpeg yet)."""
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    try:
        return pipeline.DEINTERLACE_FILTER in ffmpeg_bin.ffmpeg_filters()
    except (ffmpeg_bin.FfmpegMissing, OSError, subprocess.SubprocessError):
        return False


@router.get("/api/video/upscale/models")
def upscale_models() -> dict[str, Any]:
    rows = catalog_with_fit()
    for row in rows:
        row["installed"] = addon.site_installed() and addon.model_installed(row)
    return {"models": rows, "addonInstalled": addon.site_installed(), "deinterlaceAvailable": _deinterlace_available()}


@router.post("/api/video/upscale")
def start_upscale(body: UpscaleRequest) -> dict[str, Any]:
    try:
        if find_model(body.modelId) is None:
            raise UpscaleError(f"Unknown video upscale model '{body.modelId}'.")
        jobs.check_ready(body.modelId)
        refusal = _fit_refusal(body.modelId)
        if refusal:
            raise UpscaleError(refusal, 409)
        src = resolve_input(body.inputPath)
        from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

        info = pipeline.probe_source(ffmpeg_bin.ffmpeg_exe(), src)
        size = (body.width, body.height) if body.width else target_size(info.width, info.height, body.scale or 2.0)
        check_target(*size)  # type: ignore[arg-type]
        dst = resolve_output(body.outputPath, src, body.modelId, size)  # type: ignore[arg-type]
        spec = jobs.build_spec(src, dst, body.modelId, size, deinterlace=body.deinterlace)  # type: ignore[arg-type]
        return jobs.JOBS.start(spec).to_dict()
    except Exception as exc:  # noqa: BLE001 - mapped to a status, or re-raised
        raise _http(exc) from exc


@router.get("/api/video/upscale/jobs/{job_id}")
def upscale_status(job_id: str) -> dict[str, Any]:
    job = jobs.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No upscale job '{job_id}'.")
    return job.to_dict()


@router.post("/api/video/upscale/jobs/{job_id}/cancel")
def cancel_upscale(job_id: str) -> dict[str, Any]:
    return {"cancelled": jobs.JOBS.cancel(job_id)}


@router.post("/api/video/upscale/jobs/{job_id}/resume")
def resume_upscale(job_id: str) -> dict[str, Any]:
    try:
        return jobs.JOBS.resume(job_id).to_dict()
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


@router.post("/api/video/upscale/probe")
def probe_upscale_input(body: ProbeRequest) -> dict[str, Any]:
    try:
        from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

        src = resolve_input(body.inputPath)
        info, interlaced = pipeline.probe_details(ffmpeg_bin.ffmpeg_exe(), src)
        return {"inputPath": str(src), "width": info.width, "height": info.height, "fps": round(pipeline.fps_value(info.fps), 3),
                "durationSeconds": info.duration, "frames": info.frames, "hasAudio": info.has_audio,
                "interlaced": interlaced}
    except Exception as exc:  # noqa: BLE001
        raise _http(exc) from exc


@router.get("/api/video/upscale/jobs")
def list_upscale_jobs() -> dict[str, Any]:
    return {"jobs": [job.to_dict() for job in jobs.JOBS.recent(MAX_LISTED_JOBS)]}


def _finished_output(job_id: str) -> Path:
    """The finished film of a job: 404 for an unknown or unfinished job, or one whose file is gone or outside the output folders."""
    job = jobs.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No upscale job '{job_id}'.")
    if job.state != "done":
        raise HTTPException(status_code=404, detail="The film is not finished yet.")
    raw = Path(job.spec.output)
    try:
        real = raw.parent.resolve(strict=True) / raw.name
        ok = real.is_file() and not real.is_symlink() and real.suffix.lower() in OUTPUT_EXTENSIONS and any(
            real.is_relative_to(root) for root in output_roots())
    except (OSError, RuntimeError):
        ok = False
    if not ok:
        raise HTTPException(status_code=404, detail="The finished film is missing or outside the app's output folders.")
    return real


@router.get("/api/video/upscale/jobs/{job_id}/file")
def upscale_file(job_id: str) -> FileResponse:
    """Stream the finished film; Starlette's ``FileResponse`` answers ``Range`` requests, so the player can seek."""
    real = _finished_output(job_id)
    return FileResponse(real, media_type=_VIDEO_TYPES.get(real.suffix.lower(), "application/octet-stream"))


@router.get("/api/video/upscale/jobs/{job_id}/frame")
def upscale_frame(job_id: str, which: Literal["input", "output"] = "output") -> FileResponse:
    """The first frame of the job's input or finished output as a PNG, made once with ffmpeg and cached."""
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    try:
        if which == "output":
            src = _finished_output(job_id)
        else:
            job = jobs.JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"No upscale job '{job_id}'.")
            src = resolve_input(job.spec.input)
        cache = jobs.JOBS.root() / "frames" / f"{job_id}-{which}.png"
        if not (cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime):
            cache.parent.mkdir(parents=True, exist_ok=True)
            part = cache.with_name(f"{job_id}-{which}.part.png")
            run = subprocess.run([ffmpeg_bin.ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-protocol_whitelist",
                                  ffmpeg_bin.PROTOCOLS, "-i", str(src), "-an", "-frames:v", "1", "-y", str(part)],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            if run.returncode != 0 or not part.is_file():
                part.unlink(missing_ok=True)
                raise UpscaleError(f"Could not read a frame from {src.name}.", 422)
            os.replace(part, cache)
        return FileResponse(cache, media_type="image/png")
    except Exception as exc:  # noqa: BLE001 - mapped to a status, or re-raised
        if isinstance(exc, HTTPException):
            raise
        raise _http(exc) from exc
