"""``/api/storyboard``: projects, edit ops, undo / redo, batches, renders, bundles and project files.

Every route that changes a project takes the ``revision`` the caller last saw; a stale one answers 409
with the current revision and project, so the UI and an agent can edit at once. Generation and renders
run in background threads and are polled. Files are served only from inside a project (``media/`` and
``exports/``). The agent tools call the same helpers (``start_batch``, ``start_render``).
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from backend_service.storyboard import store as store_module
from backend_service.storyboard.models import Line, dump

router = APIRouter()
_EXPORT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateProject(_Body):
    name: Line = Field(min_length=1)
    defaults: dict[str, Any] | None = None


class OpRequest(_Body):
    revision: int
    op: str = Field(max_length=40)
    args: dict[str, Any] = Field(default_factory=dict)


class RevisionOnly(_Body):
    revision: int


class BatchRequest(_Body):
    revision: int
    what: Literal["still", "clip", "both", "final", "redo-clip", "redo-still"] = "both"
    shotIds: list[str] | None = Field(None, max_length=500)
    force: bool = False
    draft: bool = False  # Make drafts: the fast still model at a smaller size (not with final)


class RenderRequest(_Body):
    revision: int
    preview: bool = True
    notify: bool = True  # a system banner when a long export ends


class CaptionsRequest(_Body):
    revision: int
    clipIds: list[str] | None = Field(None, max_length=200)
    language: str = Field("", max_length=16)


class EstimateRequest(_Body):
    what: Literal["still", "clip", "both", "final", "redo-clip", "redo-still"] = "both"
    shotIds: list[str] | None = Field(None, max_length=500)
    draft: bool = False


class ImportRequest(_Body):
    path: str = Field(min_length=1, max_length=4096)


def get_store() -> store_module.ProjectStore:
    return store_module.default_store()


def _payload(project: Any) -> dict[str, Any]:
    from backend_service.storyboard.timeline import compute_timeline

    return {"project": dump(project), "timeline": compute_timeline(project).as_dict()}


def _http(exc: Exception) -> HTTPException:
    from backend_service.storyboard.media import MediaError
    from backend_service.storyboard.ffmpeg_bin import FfmpegMissing
    from backend_service.storyboard.render_adapter import RenderBusy

    if isinstance(exc, FfmpegMissing):
        return HTTPException(status_code=409, detail={"code": exc.action, "message": str(exc)})
    if isinstance(exc, store_module.RevisionConflict):
        return HTTPException(status_code=409, detail={"message": str(exc), "revision": exc.current.revision,
                                                      "project": dump(exc.current)})
    if isinstance(exc, store_module.ProjectNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (RenderBusy, store_module.ProjectBusy)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (store_module.StoreError, MediaError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    raise exc


def _guard(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - mapped to a readable status, others re-raised
        raise _http(exc) from exc


# ---- shared with the agent tools ----
def start_batch(request: Request, project_id: str, revision: int | None, what: str,
                shot_ids: list[str] | None, force: bool, draft: bool = False) -> Any:
    from backend_service.storyboard.batch import Batch
    from backend_service.storyboard.generators import studio_generators

    store = get_store()
    folder = store.folder(project_id)
    project = store.load(project_id)
    image, _ = _project_variant(request, "image", project.defaults.imageModel)
    video, _ = _project_variant(request, "video", project.defaults.videoModel)
    final_image = None
    if what == "final":
        final_image, why = _project_variant(request, "image", _final_model(request, project))
        if final_image is None:
            raise store_module.StoreError(why or "No image model is installed for the final pass.")
    batch = Batch(store, project_id, what, make_still=lambda s, p: None, make_clip=lambda s, p, i: None,
                  shot_ids=shot_ids, force=force, draft=draft,
                  # A redo keeps the pass of the take it redoes, so a draft take is redone at the draft size.
                  draft_sizes=_draft_sizes(project, image, video) if draft or what.startswith("redo-") else None,
                  final_image_model=str(final_image["id"]) if final_image else None)
    wiring = studio_generators(request, batch.id, folder)
    batch.make_still, batch.make_clip = wiring["make_still"], wiring["make_clip"]
    batch.unload, batch.last_frame, batch.stop_running = wiring["unload"], wiring["last_frame"], wiring["stop_running"]
    batch.clip_image_refusal = wiring.get("clip_image_refusal")
    # Name the project's models so the first step of each kind shows a time left before any step has finished.
    batch.use_models(final_image or image, video)
    batch.start(revision)
    batch.start_thread()
    return batch


def start_render(project_id: str, revision: int | None, preview: bool, notify: bool = False) -> Any:
    from backend_service.storyboard.render_adapter import RENDERS
    from backend_service.storyboard.timeline import build_render_plan

    store = get_store()
    project = store.load(project_id)
    if revision is not None and project.revision != revision:
        raise store_module.RevisionConflict(project)
    plan = build_render_plan(project, preview, store.folder(project_id))
    from backend_service.storyboard import ffmpeg_bin

    ffmpeg_bin.ffmpeg_exe()  # FfmpegMissing now, so the app can offer the install before a job exists
    return RENDERS.start(project_id, plan, notify)


def _batch_info(project_id: str, batch_id: str) -> dict[str, Any]:
    project = get_store().load(project_id)
    record = next((b for b in project.batches if b.id == batch_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown batch '{batch_id}'.")
    info: dict[str, Any] = {"batch": record.model_dump(mode="json"), "revision": project.revision}
    from backend_service.storyboard.batch import get_batch

    live = get_batch(batch_id)
    if live is not None and not live.done.is_set():
        info["progress"] = live.progress()  # "clip 2 of 4", elapsed, and time left from the clips made so far
    return info


# ---- projects ----
@router.get("/api/storyboard/projects")
def list_projects() -> dict[str, Any]:
    return {"projects": get_store().list()}


@router.post("/api/storyboard/projects")
def create_project(body: CreateProject) -> dict[str, Any]:
    return _payload(_guard(get_store().create, body.name, body.defaults))


@router.get("/api/storyboard/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return _payload(_guard(get_store().load, project_id))


@router.delete("/api/storyboard/projects/{project_id}")
def delete_project(project_id: str, revision: int) -> dict[str, Any]:
    return _guard(get_store().delete, project_id, revision)


@router.post("/api/storyboard/projects/{project_id}/ops")
def run_op(project_id: str, body: OpRequest) -> dict[str, Any]:
    from backend_service.storyboard.ops import run_op as apply

    return _payload(_guard(apply, get_store(), project_id, body.revision, body.op, body.args))


@router.post("/api/storyboard/projects/{project_id}/undo")
def undo(project_id: str, body: RevisionOnly) -> dict[str, Any]:
    return _payload(_guard(get_store().undo, project_id, body.revision))


@router.post("/api/storyboard/projects/{project_id}/redo")
def redo(project_id: str, body: RevisionOnly) -> dict[str, Any]:
    return _payload(_guard(get_store().redo, project_id, body.revision))


# ---- batches ----
@router.post("/api/storyboard/projects/{project_id}/batches")
def create_batch(request: Request, project_id: str, body: BatchRequest) -> dict[str, Any]:
    batch = _guard(start_batch, request, project_id, body.revision, body.what, body.shotIds, body.force, body.draft)
    return {"batchId": batch.id, **_batch_info(project_id, batch.id)}


def _project_variant(request: Request, kind: str, model: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """The model a batch would use for ``kind`` (the project's own, else the default), or a plain sentence why none."""
    from backend_service.mcp_bridge import models as bridge_models
    from backend_service.mcp_bridge.tools import ToolError, _pick_variant, make_context

    area = "Images" if kind == "image" else "Video"
    state = request.app.state.chaosengine
    if model is None and bridge_models.default_model(state, kind) is None:
        return None, f"No {kind} model is installed. Install one under {area} > Models to see an estimate."
    try:
        return _pick_variant(make_context(request), kind, model), None
    except ToolError:
        return None, f"The project's {kind} model ({model}) is not installed. Install it under {area} > Models, or pick another in Project settings."


def _final_model(request: Request, project: Any) -> str | None:
    """The still model of the final pass: the project's own choice, else the best-quality installed one."""
    if project.defaults.finalImageModel:
        return project.defaults.finalImageModel
    from backend_service.mcp_bridge import models as bridge_models

    picked = bridge_models.quality_model(request.app.state.chaosengine, "image")
    return str(picked["id"]) if picked else None


def _min_side(variant: dict[str, Any] | None) -> int | None:
    value = (variant or {}).get("minSide")
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _draft_sizes(project: Any, image: dict[str, Any] | None, video: dict[str, Any] | None) -> dict[str, tuple[int, int]]:
    """The draft size of stills and clips, each kept at or above its model's smallest usable side."""
    from backend_service.storyboard.batch import draft_size

    w, h = project.defaults.width, project.defaults.height
    return {"still": draft_size(w, h, _min_side(image)), "clip": draft_size(w, h, _min_side(video))}


@router.get("/api/storyboard/still-models")
def still_models(request: Request, width: int = Query(1280, ge=64, le=4096),
                 height: int = Query(720, ge=64, le=4096)) -> dict[str, Any]:
    """The installed still models with the catalogue time per still at this size, and which one Automatic picks
    for Make and drafts (``fast``, the fastest) and for the final pass (``quality``). One rule with the agent."""
    from backend_service.helpers.images import _find_image_variant
    from backend_service.helpers.time_estimate import variant_seconds
    from backend_service.mcp_bridge import models as bridge_models

    state = request.app.state.chaosengine
    rows = []
    for row in bridge_models.installed_models(state, "image"):
        variant = _find_image_variant(str(row["id"])) or {}
        rows.append({"id": row["id"], "name": row["name"], "fit": row.get("fit"),
                     "territoryRestricted": bool(row.get("territoryRestricted")),
                     "secondsPerStill": variant_seconds(variant, "image", width, height)})
    fast, quality = bridge_models.default_model(state, "image"), bridge_models.quality_model(state, "image")
    return {"models": rows, "fast": fast["id"] if fast else None, "quality": quality["id"] if quality else None,
            "width": width, "height": height}


@router.post("/api/storyboard/projects/{project_id}/estimate")
def estimate_batch(request: Request, project_id: str, body: EstimateRequest) -> dict[str, Any]:
    """Time, peak memory and disk for a batch of shots (what Make would run), from this project's measured takes
    or the catalog. Never fails for a missing model: ``missing`` says what to install.

    ``passes`` (S6) adds both passes for the same shots: ``draft`` (fast model, draft size; ``seconds`` is the
    stills, ``clipSeconds`` the optional clips) and ``final`` (the kept shots with the quality model, plus clips)."""
    from backend_service.storyboard.estimate import estimate, passes, plan_steps

    store = get_store()
    project = _guard(store.load, project_id)
    steps = plan_steps(project, body.what, body.shotIds)
    missing: list[str] = []
    image, why_image = _project_variant(request, "image", project.defaults.imageModel)
    video, why_video = _project_variant(request, "video", project.defaults.videoModel)
    final_image = _project_variant(request, "image", _final_model(request, project))[0]
    if any(kind == "still" for _, kind in steps):
        missing += [why_image] if why_image else []
    if any(kind == "clip" for _, kind in steps):
        missing += [why_video] if why_video else []
    folder = store.folder(project_id)
    sizes = _draft_sizes(project, image, video)
    # A redo is made in the pass of the takes it redoes: priced as a draft when every one of them is a draft.
    redone = [shot.active() for shot, _ in steps] if body.what.startswith("redo-") else []
    draft = body.draft or (bool(redone) and all(t is not None and t.renderPass == "draft" for t in redone))
    out = estimate(project, folder, what=body.what, shot_ids=body.shotIds,
                   image_variant=final_image if body.what == "final" else image, video_variant=video, missing=missing,
                   render_pass="draft" if draft else "final", sizes=sizes if draft else None)
    out["passes"] = passes(project, folder, shot_ids=body.shotIds if body.what in ("still", "clip", "both") else None,
                           draft_variant=image, final_variant=final_image, video_variant=video, draft_sizes=sizes)
    return out


@router.get("/api/storyboard/projects/{project_id}/batches/{batch_id}")
def batch_status(project_id: str, batch_id: str) -> dict[str, Any]:
    return _guard(_batch_info, project_id, batch_id)


@router.post("/api/storyboard/projects/{project_id}/batches/{batch_id}/cancel")
def cancel_batch(project_id: str, batch_id: str) -> dict[str, Any]:
    from backend_service.storyboard.batch import get_batch

    batch = get_batch(batch_id)
    if batch is None or batch.project_id != project_id:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' is not running.")
    return {"cancelled": batch.cancel()}


# ---- renders ----
@router.post("/api/storyboard/projects/{project_id}/render")
def render(project_id: str, body: RenderRequest) -> dict[str, Any]:
    return {"render": _guard(start_render, project_id, body.revision, body.preview, body.notify).describe()}


@router.get("/api/storyboard/projects/{project_id}/render")
def render_status(project_id: str) -> dict[str, Any]:
    from backend_service.storyboard.render_adapter import RENDERS

    folder = _guard(get_store().folder, project_id)
    job = RENDERS.status(folder.name, folder)
    return {"render": job.describe() if job else None}


@router.post("/api/storyboard/projects/{project_id}/render/cancel")
def cancel_render(project_id: str) -> dict[str, Any]:
    from backend_service.storyboard.render_adapter import RENDERS

    return {"cancelled": RENDERS.cancel(project_id)}


# ---- captions, timeline export ----
@router.post("/api/storyboard/projects/{project_id}/captions/generate")
def generate_captions(project_id: str, body: CaptionsRequest) -> dict[str, Any]:
    """Transcribe the voice clips (or the named ones) and store the result as captions."""
    from backend_service.storyboard import captions

    project, notes = _guard(captions.generate, get_store(), project_id, body.revision, body.clipIds, body.language)
    return {**_payload(project), "notes": notes}


@router.post("/api/storyboard/projects/{project_id}/captions/export")
def export_captions(project_id: str) -> dict[str, Any]:
    """Write the captions as an .srt in the project's ``exports/`` folder."""
    from backend_service.storyboard import captions
    from backend_service.storyboard.timeline import compute_timeline

    store = get_store()
    project = _guard(store.load, project_id)
    items = captions.cues(project, compute_timeline(project, ready_only=True))
    if not items:
        raise HTTPException(status_code=400, detail="There are no captions to export.")
    path = captions.write_srt(project, store.folder(project_id), items)
    return {"file": f"exports/{path.name}", "cues": len(items)}


@router.post("/api/storyboard/projects/{project_id}/otio")
def export_otio(project_id: str) -> dict[str, Any]:
    """Write the cut as an OpenTimelineIO (.otio) file in ``exports/``, for Resolve, Premiere and other editors."""
    from backend_service.storyboard import otio_export

    store = get_store()
    project = _guard(store.load, project_id)
    try:
        path, warnings = otio_export.write_otio(project, store.folder(project_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"file": f"exports/{path.name}", "warnings": warnings}


# ---- bundles ----
@router.post("/api/storyboard/projects/{project_id}/bundle")
def export_bundle(project_id: str) -> dict[str, Any]:
    from backend_service.storyboard.bundle import export_bundle as export

    path = _guard(export, get_store(), project_id)
    return {"file": f"exports/{path.name}", "bytes": path.stat().st_size}


@router.post("/api/storyboard/import")
async def import_bundle(request: Request) -> dict[str, Any]:
    """A ``.chaosfilm`` bundle: JSON ``{"path": ...}`` for a local file, or the raw zip bytes in the body."""
    import anyio

    from backend_service.storyboard import bundle

    store = get_store()
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = ImportRequest.model_validate(await request.json())
        except ValueError as exc:  # bad JSON or the wrong shape (a pydantic ValidationError is a ValueError)
            raise HTTPException(status_code=400, detail='Send {"path": "<a .chaosfilm file>"} or the bundle bytes.') from exc
        try:
            src = Path(body.path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(status_code=400, detail="Bundle not found.") from None
        if not src.is_file() or src.suffix.lower() not in (bundle.BUNDLE_SUFFIX, ".zip"):
            raise HTTPException(status_code=400, detail="Give a .chaosfilm file.")
        project = await anyio.to_thread.run_sync(lambda: _guard(bundle.import_bundle, store, src))
        return _payload(project)
    store.root.mkdir(parents=True, exist_ok=True)
    tmp = store.root / f".incoming-{secrets.token_hex(6)}.chaosfilm"
    try:
        size = 0
        with tmp.open("wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > bundle.MAX_TOTAL_BYTES:
                    raise HTTPException(status_code=413, detail="The bundle is too large.")
                out.write(chunk)
        project = await anyio.to_thread.run_sync(lambda: _guard(bundle.import_bundle, store, tmp))
        return _payload(project)
    finally:
        tmp.unlink(missing_ok=True)


@router.post("/api/storyboard/uploads")
async def upload_picture(request: Request) -> dict[str, Any]:
    """Raw picture bytes (a file the user chose, dropped or pasted on a Cast card); returns an ``upload:`` handle
    that ``add_reference`` / ``edit_reference`` take in place of a path. The type comes from the bytes (PNG,
    JPEG or WebP only) and the size cap is enforced while the body arrives; the file waits in the inbox, which
    is pruned after a day, until an op copies it into the project.
    """
    from backend_service.helpers.media_input import IMAGE_EXTENSIONS, InputImageError
    from backend_service.mcp_bridge import files

    part = files.begin_upload()
    try:
        head, cap, size = b"", None, 0
        with part.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if cap is None:
                    head += chunk[:64]
                    if len(head) >= 16:
                        kind = files.sniff(head)
                        if kind not in IMAGE_EXTENSIONS:
                            raise HTTPException(status_code=400, detail="Give a PNG, JPEG or WebP picture.")
                        cap = files.max_upload_bytes(kind)
                if cap is not None and size > cap:
                    raise HTTPException(status_code=413, detail=f"The picture is larger than {cap // (1024 * 1024)} MB.")
                handle.write(chunk)
        if cap is None:
            raise HTTPException(status_code=400, detail="Give a PNG, JPEG or WebP picture.")
        return files.finish_upload(part)
    except InputImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        part.unlink(missing_ok=True)


# ---- files ----
def project_file(project_id: str, rel: str) -> Path:
    """A file inside one project (``media/...`` or ``exports/<name>``), or 404."""
    from backend_service.storyboard.media import MediaError, inside

    folder = _guard(get_store().folder, project_id)
    try:
        if rel.startswith("exports/"):
            name = rel[len("exports/"):]
            if not _EXPORT_NAME_RE.match(name):
                raise MediaError("bad export name")
            real = (folder / "exports" / name).resolve(strict=True)
            if not real.is_relative_to(folder.resolve()) or not real.is_file():
                raise MediaError("outside")
            return real
        return inside(folder, rel)
    except (MediaError, OSError):
        raise HTTPException(status_code=404, detail="Not a file of this project.") from None


@router.get("/api/storyboard/projects/{project_id}/files")
def get_file(project_id: str, path: str, thumb: int | None = None) -> FileResponse:
    """A project file; with ``thumb`` (128, 256 or 512) a picture comes back as a small JPEG for grids."""
    real = project_file(project_id, path)
    if thumb is not None:
        from backend_service.storyboard import thumbs
        from backend_service.storyboard.media import kind_of

        if thumb not in thumbs.THUMB_SIZES:
            raise HTTPException(status_code=400, detail=f"thumb is one of {', '.join(map(str, thumbs.THUMB_SIZES))}.")
        if kind_of(real) == "image":
            try:
                small = thumbs.thumbnail(get_store().folder(project_id).resolve(), real, thumb)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return FileResponse(small, media_type="image/jpeg")
    return FileResponse(real, filename=real.name)


@router.get("/api/storyboard/projects/{project_id}/peaks")
def get_peaks(project_id: str, path: str, buckets: int = Query(400, ge=32, le=4000)) -> dict[str, Any]:
    """Waveform peaks (0 to 1) and the length of one project file's audio, cached under ``cache/peaks/``."""
    from backend_service.storyboard import peaks

    real = project_file(project_id, path)
    folder = _guard(get_store().folder, project_id)
    try:
        return peaks.peaks_for(folder, real, buckets)
    except peaks.PeaksError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:  # ffmpeg is missing (409 with the install code) or failed to run (503)
        from backend_service.storyboard.ffmpeg_bin import FfmpegMissing

        if isinstance(exc, FfmpegMissing):
            raise _http(exc) from exc
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---- the bundled sample film (S12) ----
class SampleRequest(_Body):
    name: Line | None = None  # the project's name in the user's language; the bundled name when absent


@router.get("/api/storyboard/sample")
def sample_info() -> dict[str, Any]:
    """Whether this install carries the sample film, and what the "Try the sample project" card says about it."""
    from backend_service.storyboard import sample

    return sample.info()


@router.get("/api/storyboard/sample/poster")
def sample_poster() -> FileResponse:
    """The sample's first still, for its card."""
    from backend_service.storyboard import sample

    try:
        return FileResponse(sample.poster_path())
    except store_module.StoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/storyboard/sample")
def install_sample(body: SampleRequest) -> dict[str, Any]:
    """Copy the sample film into the storyboards folder as an ordinary new project and return it."""
    from backend_service.storyboard import sample

    return _payload(_guard(sample.install, get_store(), body.name or None))
