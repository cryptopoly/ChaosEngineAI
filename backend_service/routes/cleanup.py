"""``/api/cleanup``: list and delete what the user chooses (storyboard projects, chat threads, gallery pictures and clips).

``GET /api/cleanup/items`` lists what the backend itself finds in its own data folders. ``POST /api/cleanup/delete``
takes ids from that list and nothing else: a request never carries a path. Every id is looked up in a fresh
listing (exact string match), so an id that is not in the listing (a traversal string, a glob, an old id) is
reported ``not found`` and touches nothing. Files are removed only when their resolved path sits under the
data root they were listed from, and links are never followed. A project with a live batch or render is refused.
Nothing here guesses what is "test data": the user ticks what to delete.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend_service.storyboard import store as store_module
from backend_service.storyboard.models import PROJECT_ID_RE

router = APIRouter()

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".svg")
_VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")
_MAX_IDS = 5000

_Id = Annotated[str, StringConstraints(min_length=1, max_length=400)]


class DeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projects: list[_Id] = Field(default_factory=list, max_length=_MAX_IDS)
    threads: list[_Id] = Field(default_factory=list, max_length=_MAX_IDS)
    outputs: list[_Id] = Field(default_factory=list, max_length=_MAX_IDS)


# ---- shared helpers ----
def _inside(path: Path, root: Path) -> bool:
    """True when ``path`` is a real (not linked) entry whose resolved location is under ``root``."""
    try:
        return not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _folder_bytes(folder: Path) -> int:
    """Bytes of the files under ``folder``; links are counted as themselves and never followed."""
    total = 0
    stack = [folder]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def _dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(ids))


# ---- storyboard projects ----
def _project_rows(store: store_module.ProjectStore) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    root = store.root
    if not root.is_dir():
        return rows
    for entry in sorted(root.iterdir()):
        project_file = entry / store_module.PROJECT_FILE
        if not PROJECT_ID_RE.match(entry.name) or entry.is_symlink() or not project_file.is_file() or project_file.is_symlink():
            continue
        try:
            project = store_module.parse_project(project_file.read_bytes())
            modified = project_file.stat().st_mtime
        except (store_module.StoreError, OSError):
            continue  # unreadable projects are not offered: opening one shows the reason
        if project.id != entry.name:
            continue
        rows[entry.name] = {
            "id": entry.name,
            "name": project.name,
            "shots": len(project.shots),
            "bytes": _folder_bytes(entry),
            "lastEdited": modified,
        }
    return rows


def _delete_project(store: store_module.ProjectStore, project_id: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = rows.get(project_id)
    if row is None:
        return _result("project", project_id, None, False, 0, "Not found. It may already be deleted.")
    name = row["name"]
    try:
        project = store.load(project_id)
        if any(batch.id in store_module.LIVE_BATCHES for batch in project.batches):
            return _result("project", project_id, name, False, 0,
                           "A batch is running for this project. Cancel it or wait, then delete.")
        outcome = store.delete(project_id, project.revision)
    except store_module.RevisionConflict:
        return _result("project", project_id, name, False, 0, "The project changed while deleting. Try again.")
    except store_module.StoreError as exc:  # includes ProjectBusy (a render is running) and ProjectNotFound
        return _result("project", project_id, name, False, 0, str(exc))
    except OSError as exc:
        return _result("project", project_id, name, False, 0, f"Could not delete the project: {exc.strerror or exc}")
    return _result("project", project_id, name, True, row["bytes"], outcome.get("message"))


# ---- chat threads ----
def _session_last_used(session: dict[str, Any]) -> float | None:
    stamps = [m.get("createdAt") for m in session.get("messages") or [] if isinstance(m, dict)]
    millis = [s for s in stamps if isinstance(s, (int, float)) and not isinstance(s, bool) and s > 0]
    if millis:
        return max(millis) / 1000.0
    try:
        return time.mktime(time.strptime(str(session.get("updatedAt")), "%Y-%m-%d %H:%M:%S"))
    except (ValueError, OverflowError):
        return None


def _thread_rows(state: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with state._lock:
        sessions = list(state.chat_sessions)
    for session in sessions:
        if not isinstance(session, dict) or not session.get("id"):
            continue
        session_id = str(session["id"])
        try:
            size = len(json.dumps(session, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            size = 0
        rows[session_id] = {
            "id": session_id,
            "title": str(session.get("title") or session_id),
            "messages": len(session.get("messages") or []),
            "bytes": size,
            "lastUsed": _session_last_used(session),
        }
    return rows


def _delete_thread(state: Any, thread_id: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = rows.get(thread_id)
    if row is None:
        return _result("thread", thread_id, None, False, 0, "Not found. It may already be deleted.")
    try:
        state.delete_session(thread_id)
    except HTTPException:
        return _result("thread", thread_id, row["title"], False, 0, "Not found. It may already be deleted.")
    return _result("thread", thread_id, row["title"], True, row["bytes"], None)


# ---- gallery pictures and clips ----
@dataclass
class _Output:
    row: dict[str, Any]
    metadata: Path
    media: list[Path] = field(default_factory=list)


def _output_entries(kind: str, root: Path) -> dict[str, _Output]:
    """Every saved picture or clip under ``root``, by the id the list shows (``<kind>:<path under root>``).

    Only the metadata file and the media files that sit next to it with the same name are ever listed, and
    only when they resolve inside ``root``; the ``imagePath`` / ``videoPath`` inside the metadata is ignored."""
    entries: dict[str, _Output] = {}
    if not root.is_dir():
        return entries
    extensions = _IMAGE_EXTENSIONS if kind == "image" else _VIDEO_EXTENSIONS
    for metadata in sorted(root.rglob("*.json")):
        if not _inside(metadata, root) or not metadata.is_file():
            continue
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or not payload.get("artifactId"):
            continue
        if ("videoPath" if kind == "image" else "imagePath") in payload:
            continue  # the other gallery's metadata (both folders may be set to one place)
        media = [metadata.with_suffix(ext) for ext in extensions]
        media = [path for path in media if path.is_file() and _inside(path, root)]
        try:
            size = metadata.stat().st_size + sum(path.stat().st_size for path in media)
        except OSError:
            continue
        item_id = f"{kind}:{metadata.relative_to(root).as_posix()}"
        prompt = " ".join(str(payload.get("prompt") or "").split())
        name = prompt[:80] + ("..." if len(prompt) > 80 else "") or str(payload.get("artifactId"))
        entries[item_id] = _Output(
            row={"id": item_id, "kind": kind, "name": name, "bytes": size,
                 "createdAt": str(payload.get("createdAt") or "")},
            metadata=metadata,
            media=media,
        )
    return entries


def _output_roots() -> dict[str, Path]:
    from backend_service.app import _current_image_outputs_dir, _current_video_outputs_dir

    return {"image": _current_image_outputs_dir(), "video": _current_video_outputs_dir()}


def _delete_output(item_id: str, entries: dict[str, _Output]) -> dict[str, Any]:
    entry = entries.get(item_id)
    if entry is None:
        return _result("output", item_id, None, False, 0, "Not found. It may already be deleted.")
    name = entry.row["name"]
    freed = 0
    try:
        for path in [*entry.media, entry.metadata]:  # the metadata goes last: a failure leaves a retryable entry
            size = path.stat().st_size
            path.unlink()
            freed += size
    except OSError as exc:
        return _result("output", item_id, name, False, freed, f"Could not delete the file: {exc.strerror or exc}")
    return _result("output", item_id, name, True, freed, None)


def _result(kind: str, item_id: str, name: str | None, ok: bool, freed: int, reason: str | None) -> dict[str, Any]:
    return {"kind": kind, "id": item_id, "name": name, "ok": ok, "bytes": freed if ok else 0, "reason": reason}


# ---- routes ----
@router.get("/api/cleanup/items")
def list_cleanup_items(request: Request) -> dict[str, Any]:
    roots = _output_roots()
    outputs = [e.row for kind in ("image", "video") for e in _output_entries(kind, roots[kind]).values()]
    outputs.sort(key=lambda row: row["createdAt"], reverse=True)
    projects = sorted(_project_rows(store_module.default_store()).values(), key=lambda r: r["lastEdited"], reverse=True)
    threads = sorted(_thread_rows(request.app.state.chaosengine).values(), key=lambda r: r["lastUsed"] or 0.0, reverse=True)
    return {"projects": projects, "threads": threads, "outputs": outputs}


@router.post("/api/cleanup/delete")
def delete_cleanup_items(request: Request, body: DeleteRequest) -> dict[str, Any]:
    projects, threads, outputs = _dedupe(body.projects), _dedupe(body.threads), _dedupe(body.outputs)
    if not (projects or threads or outputs):
        raise HTTPException(status_code=400, detail="Nothing was chosen to delete.")
    results: list[dict[str, Any]] = []
    if projects:
        store = store_module.default_store()
        rows = _project_rows(store)
        results += [_delete_project(store, pid, rows) for pid in projects]
    if threads:
        state = request.app.state.chaosengine
        rows = _thread_rows(state)
        results += [_delete_thread(state, tid, rows) for tid in threads]
    if outputs:
        roots = _output_roots()
        entries: dict[str, _Output] = {}
        for kind in ("image", "video"):
            if any(item.startswith(f"{kind}:") for item in outputs):
                entries.update(_output_entries(kind, roots[kind]))
        results += [_delete_output(oid, entries) for oid in outputs]
    done = [r for r in results if r["ok"]]
    return {
        "results": results,
        "deleted": len(done),
        "refused": len(results) - len(done),
        "freedBytes": sum(r["bytes"] for r in done),
    }
