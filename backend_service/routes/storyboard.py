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
    what: Literal["still", "clip", "both"] = "both"
    shotIds: list[str] | None = Field(None, max_length=500)
    force: bool = False


class RenderRequest(_Body):
    revision: int
    preview: bool = True


class ImportRequest(_Body):
    path: str = Field(min_length=1, max_length=4096)


def get_store() -> store_module.ProjectStore:
    return store_module.default_store()


def _payload(project: Any) -> dict[str, Any]:
    from backend_service.storyboard.timeline import compute_timeline

    return {"project": dump(project), "timeline": compute_timeline(project).as_dict()}


def _http(exc: Exception) -> HTTPException:
    from backend_service.storyboard.media import MediaError
    from backend_service.storyboard.render_adapter import RenderBusy

    if isinstance(exc, store_module.RevisionConflict):
        return HTTPException(status_code=409, detail={"message": str(exc), "revision": exc.current.revision,
                                                      "project": dump(exc.current)})
    if isinstance(exc, store_module.ProjectNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RenderBusy):
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
                shot_ids: list[str] | None, force: bool) -> Any:
    from backend_service.storyboard.batch import Batch
    from backend_service.storyboard.generators import studio_generators

    store = get_store()
    folder = store.folder(project_id)
    batch = Batch(store, project_id, what, make_still=lambda s, p: None, make_clip=lambda s, p, i: None,
                  shot_ids=shot_ids, force=force)
    wiring = studio_generators(request, batch.id, folder)
    batch.make_still, batch.make_clip = wiring["make_still"], wiring["make_clip"]
    batch.unload, batch.last_frame, batch.stop_running = wiring["unload"], wiring["last_frame"], wiring["stop_running"]
    batch.start(revision)
    batch.start_thread()
    return batch


def start_render(project_id: str, revision: int | None, preview: bool) -> Any:
    from backend_service.storyboard.render_adapter import RENDERS
    from backend_service.storyboard.timeline import build_render_plan

    store = get_store()
    project = store.load(project_id)
    if revision is not None and project.revision != revision:
        raise store_module.RevisionConflict(project)
    plan = build_render_plan(project, preview, store.folder(project_id))
    return RENDERS.start(project_id, plan)


def _batch_info(project_id: str, batch_id: str) -> dict[str, Any]:
    project = get_store().load(project_id)
    record = next((b for b in project.batches if b.id == batch_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown batch '{batch_id}'.")
    return {"batch": record.model_dump(mode="json"), "revision": project.revision}


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
    _guard(get_store().delete, project_id, revision)
    return {"deleted": project_id}


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
    batch = _guard(start_batch, request, project_id, body.revision, body.what, body.shotIds, body.force)
    return {"batchId": batch.id, **_batch_info(project_id, batch.id)}


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
    return {"render": _guard(start_render, project_id, body.revision, body.preview).describe()}


@router.get("/api/storyboard/projects/{project_id}/render")
def render_status(project_id: str) -> dict[str, Any]:
    from backend_service.storyboard.render_adapter import RENDERS

    job = RENDERS.status(_guard(get_store().folder, project_id).name)
    return {"render": job.describe() if job else None}


@router.post("/api/storyboard/projects/{project_id}/render/cancel")
def cancel_render(project_id: str) -> dict[str, Any]:
    from backend_service.storyboard.render_adapter import RENDERS

    return {"cancelled": RENDERS.cancel(project_id)}


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
        body = ImportRequest.model_validate(await request.json())
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
def get_file(project_id: str, path: str) -> FileResponse:
    real = project_file(project_id, path)
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
    except RuntimeError as exc:  # the bundled ffmpeg is missing
        raise HTTPException(status_code=503, detail=str(exc)) from exc
