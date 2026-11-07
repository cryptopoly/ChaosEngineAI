"""The project store: one folder per project, one writer at a time, every change revisioned and undoable.

Projects live in ``<app data>/storyboards/<project-id>/`` next to settings.json. A mutation takes the
project's process-wide lock, checks ``expected_revision`` (a stale caller gets ``RevisionConflict`` with
the current project), applies the change, bumps ``revision``, writes ``project.json`` atomically (temp
file in the same folder, then ``os.replace``) and appends one small diff file to ``history/``. Undo and
redo replay those diffs for the reversible ops, field by field: work done since by non-undoable ops
(batch picks and takes, relinked files) is kept, and a shot an undo removes keeps its takes for redo. On the first load in a
process, shots left ``queued`` / ``running`` go back to ``idle`` and unfinished batches are marked
``interrupted``; on every load, a shot or audio clip whose file is gone shows ``missing``.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from backend_service.storyboard import deps
from backend_service.storyboard.media import MediaError, inside, replace_file
from backend_service.storyboard.models import PROJECT_ID_RE, SCHEMA_VERSION, Defaults, Project, dump, new_project_id

PROJECT_FILE = "project.json"
MAX_PROJECT_BYTES = 16 * 1024 * 1024
HISTORY_KEEP = 500
_TOP_KEYS = ("name", "defaults", "references", "export")
_LIST_KEYS = ("shots", "audio")

# schemaVersion n -> function turning an n dict into an n+1 dict. Add one entry per format change.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_RECOVERED: set[str] = set()
LIVE_BATCHES: set[str] = set()  # batch ids running in this process (batch.py adds and removes)


class StoreError(ValueError):
    """A readable refusal (bad id, invalid or hostile project.json, newer schema)."""


class ProjectNotFound(StoreError):
    pass


class ProjectBusy(StoreError):
    """The project has a batch or render running; the caller should wait or cancel it (HTTP 409)."""


class RevisionConflict(Exception):
    def __init__(self, current: Project) -> None:
        super().__init__(f"The project changed (now revision {current.revision}). Reload and try again.")
        self.current = current


def storyboards_dir() -> Path:
    override = os.environ.get("CHAOSENGINE_STORYBOARDS_DIR")
    if override:
        return Path(override)
    from backend_service.app import SETTINGS_DIR

    return Path(SETTINGS_DIR) / "storyboards"


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    version = data.get("schemaVersion")
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise StoreError("project.json has no valid schemaVersion.")
    if version > SCHEMA_VERSION:
        raise StoreError(f"This project was saved by a newer ChaosEngineAI (format {version}, this app reads up to {SCHEMA_VERSION}). Update the app.")
    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise StoreError(f"No migration from project format {version}.")
        data = step(dict(data))
        version += 1
        data["schemaVersion"] = version
    return data


def parse_project(raw: bytes) -> Project:
    """Validate untrusted project.json bytes (size, JSON, migrations, schema)."""
    if len(raw) > MAX_PROJECT_BYTES:
        raise StoreError("project.json is too large.")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise StoreError("project.json is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise StoreError("project.json must hold an object.")
    try:
        return Project.model_validate(migrate(data))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise StoreError(f"project.json refused at {where or 'top level'}: {first.get('msg')}") from exc


def atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        replace_file(tmp, path)  # retried while a reader holds the file (Windows)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _encode(project: Project) -> bytes:
    return json.dumps(dump(project), indent=1, ensure_ascii=False).encode("utf-8")


def check_media(project: Project, folder: Path) -> None:
    """Mark shots and clips whose active files are gone ``missing``; bring them back when the files return."""
    for shot in project.shots:
        take = shot.active()
        rels = [r for r in ((take.still, take.clip) if take else ()) if r]
        gone = False
        for rel in rels:
            try:
                inside(folder, rel)
            except MediaError:
                gone = True
        if gone:
            shot.status = "missing"
        elif shot.status == "missing":
            shot.status = "done" if take else "idle"
    for clip in project.audio:
        try:
            inside(folder, clip.source.path)
            clip.status = "ok"
        except MediaError:
            clip.status = "missing"


def _diff(before: dict[str, Any], after: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    old: dict[str, Any] = {}
    new: dict[str, Any] = {}
    for key in _TOP_KEYS:
        if before.get(key) != after.get(key):
            old[key], new[key] = before.get(key), after.get(key)
    for key in _LIST_KEYS:
        b = {item["id"]: item for item in before.get(key, [])}
        a = {item["id"]: item for item in after.get(key, [])}
        changed = [i for i in dict.fromkeys([*b, *a]) if b.get(i) != a.get(i)]
        order_b, order_a = list(b), list(a)
        if changed or order_b != order_a:
            old[key] = {"items": {i: b.get(i) for i in changed}, "order": order_b}
            new[key] = {"items": {i: a.get(i) for i in changed}, "order": order_a}
    return old, new


def _merge_item(key: str, current: dict[str, Any], target: dict[str, Any], other: dict[str, Any] | None) -> dict[str, Any]:
    """Replay only the fields the history entry changed, and only where nothing else changed them since.

    ``target`` is the side being restored, ``other`` the side the entry left. A field is set back when the
    entry changed it (target != other) and it still holds what the entry left (current == other); work
    that is not undoable (a batch's pick, ``redo=False``, a relinked file) stays. Takes are never dropped:
    the current takes, with their current contents, plus any the target names that are gone.
    """
    merged = json.loads(json.dumps(current))
    for field, value in target.items():
        if field in ("takes", "status"):
            continue
        left = other.get(field) if other is not None else None
        if other is not None and value == left:
            continue  # the entry did not change this field
        if other is not None and current.get(field) != left:
            continue  # changed since by work that is not undoable; keep it
        merged[field] = json.loads(json.dumps(value))
    if key == "shots":
        known = {t["id"] for t in merged.get("takes", [])}
        merged["takes"] = merged.get("takes", []) + [json.loads(json.dumps(t)) for t in target.get("takes", []) if t["id"] not in known]
        ids = {t["id"] for t in merged["takes"]}
        if merged.get("activeTake") is not None and merged["activeTake"] not in ids:
            merged["activeTake"] = current.get("activeTake")
    return merged


def _apply_side(data: dict[str, Any], side: dict[str, Any], other: dict[str, Any] | None = None,
                stash: dict[str, Any] | None = None) -> None:
    """Replay one side of a history entry onto ``data``. ``stash`` holds shots an undo removed, as they were."""
    other = other or {}
    for key in _TOP_KEYS:
        if key in side:
            data[key] = side[key]
    for key in _LIST_KEYS:
        if key not in side:
            continue
        current = {item["id"]: item for item in data.get(key, [])}
        others = (other.get(key) or {}).get("items", {})
        for item_id, item in side[key]["items"].items():
            if item is None:
                current.pop(item_id, None)
                continue
            if item_id in current:
                current[item_id] = _merge_item(key, current[item_id], item, others.get(item_id))
                continue
            item = json.loads(json.dumps(item))
            kept = (stash or {}).get(item_id) if key == "shots" else None
            if kept is not None:  # a redo of a shot an undo removed: bring back the takes it had then
                item = _merge_item(key, kept, item, None)
                item["activeTake"] = kept.get("activeTake")
                item["status"] = kept.get("status", item.get("status"))
            current[item_id] = item
        order = [i for i in side[key]["order"] if i in current]
        order += [i for i in current if i not in order]
        data[key] = [current[i] for i in order]
        if key == "shots":
            for index, shot in enumerate(data[key]):
                shot["order"] = index


class ProjectStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else storyboards_dir()

    # ---- paths and locks ----
    def folder(self, project_id: str) -> Path:
        if not isinstance(project_id, str) or not PROJECT_ID_RE.match(project_id):
            raise ProjectNotFound(f"Unknown project '{str(project_id)[:40]}'.")
        return self.root / project_id

    def lock(self, project_id: str) -> threading.RLock:
        key = str(self.folder(project_id).resolve())
        with _LOCKS_GUARD:
            return _LOCKS.setdefault(key, threading.RLock())

    # ---- read ----
    def list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return rows
        for entry in sorted(self.root.iterdir()):
            if not PROJECT_ID_RE.match(entry.name) or not (entry / PROJECT_FILE).is_file():
                continue
            try:
                project = self.load(entry.name)
            except StoreError as exc:
                rows.append({"id": entry.name, "error": str(exc)})
                continue
            rows.append({"id": project.id, "name": project.name, "revision": project.revision,
                         "shots": len(project.shots), "createdAt": project.createdAt})
        return rows

    def load(self, project_id: str) -> Project:
        folder = self.folder(project_id)
        path = folder / PROJECT_FILE
        if path.is_symlink() or not path.is_file():
            raise ProjectNotFound(f"Unknown project '{project_id}'.")
        project = parse_project(path.read_bytes())
        if project.id != project_id:
            raise StoreError("project.json id does not match its folder.")
        key = str(folder.resolve())
        if key not in _RECOVERED:
            with self.lock(project_id):
                if key not in _RECOVERED:
                    project = parse_project(path.read_bytes())
                    if self._recover(project):
                        project.revision += 1
                        atomic_write(path, _encode(project))
                    _RECOVERED.add(key)
        check_media(project, folder)
        deps.apply_stale(project)
        return project

    @staticmethod
    def _recover(project: Project) -> bool:
        changed = False
        for shot in project.shots:
            if shot.status in ("queued", "running"):
                shot.status = "idle"
                changed = True
        for batch in project.batches:
            if batch.state in ("queued", "running") and batch.id not in LIVE_BATCHES:
                batch.state = "interrupted"
                changed = True
        return changed

    # ---- write ----
    def create(self, name: str, defaults: dict[str, Any] | None = None) -> Project:
        self.root.mkdir(parents=True, exist_ok=True)
        project_id = new_project_id()
        project = Project(id=project_id, name=name, createdAt=time.time(), defaults=Defaults(**(defaults or {})))
        folder = self.folder(project_id)
        for sub in ("media/audio", "cache", "history", "exports"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        atomic_write(folder / PROJECT_FILE, _encode(project))
        _RECOVERED.add(str(folder.resolve()))
        return project

    def delete(self, project_id: str, expected_revision: int) -> dict[str, Any]:
        """Delete the project folder. Refused while a batch or render runs. ``project.json`` goes first,
        so a file another program holds open (Windows) cannot leave a half-project; such files are
        counted in ``leftover`` and named in ``message``."""
        from backend_service.storyboard.render_adapter import RENDERS

        folder = self.folder(project_id)
        with self.lock(project_id):
            current = self.load(project_id)
            if current.revision != expected_revision:
                raise RevisionConflict(current)
            if any(b.id in LIVE_BATCHES and b.state in ("queued", "running") for b in current.batches):
                raise ProjectBusy("A batch is running for this project. Cancel it or wait, then delete.")
            job = RENDERS.status(project_id)
            if job is not None and not job.done.is_set():
                raise ProjectBusy("A render is running for this project. Cancel it or wait, then delete.")
            if folder.is_symlink() or not folder.resolve().is_relative_to(self.root.resolve()):
                raise StoreError("Refused: the project folder is not inside the storyboards folder.")
            try:
                (folder / PROJECT_FILE).unlink()
            except OSError as exc:
                raise StoreError(f"Could not delete the project: project.json is in use ({exc.strerror or exc}). "
                                 "Close what is using it and try again.") from exc
            _RECOVERED.discard(str(folder.resolve()))
            failed: list[str] = []
            # Count files only: their folders then fail to go too (``rmdir`` of a non-empty folder).
            shutil.rmtree(folder, onerror=lambda fn, path, info: None if getattr(fn, "__name__", "") == "rmdir"
                          else failed.append(Path(path).name))
        result: dict[str, Any] = {"deleted": project_id, "leftover": len(failed)}
        if failed:
            result["message"] = (f"The project was deleted, but {len(failed)} file(s) could not be removed "
                                 f"(probably in use), for example {failed[0]}. They are in {folder}; remove them later.")
        return result

    def mutate(
        self,
        project_id: str,
        expected_revision: int | None,
        change: Callable[[Project, Path], Any],
        op: str,
        reversible: bool = True,
    ) -> Project:
        """Apply ``change(project, folder)`` under the lock. ``expected_revision`` None skips the check (internal use)."""
        folder = self.folder(project_id)
        with self.lock(project_id):
            project = self.load(project_id)
            if expected_revision is not None and project.revision != expected_revision:
                raise RevisionConflict(project)
            before = dump(project)
            change(project, folder)
            for index, shot in enumerate(project.shots):
                shot.order = index
            check_media(project, folder)
            deps.apply_stale(project)
            # Re-validate the whole project: a change must not leave it in a state a load would refuse.
            project = Project.model_validate(dump(project))
            return self._commit(project, folder, before, op, reversible)

    def _commit(self, project: Project, folder: Path, before: dict[str, Any], op: str, reversible: bool,
                stacks: Callable[[Project], None] | None = None, extra: dict[str, Any] | None = None) -> Project:
        project.revision += 1
        old, new = _diff(before, dump(project))
        if stacks is not None:
            stacks(project)
        elif reversible and (old or new):
            project.undoStack = (project.undoStack + [project.revision])[-200:]
            project.redoStack = []
        history = folder / "history"
        history.mkdir(exist_ok=True)
        entry = {"revision": project.revision, "op": op, "at": time.time(), "reversible": reversible, "before": old, "after": new,
                 **(extra or {})}
        atomic_write(history / f"{project.revision:08d}.json", json.dumps(entry, ensure_ascii=False).encode("utf-8"))
        atomic_write(folder / PROJECT_FILE, _encode(project))
        if project.revision % 50 == 0:
            self._prune_history(project, history)
        return project

    @staticmethod
    def _prune_history(project: Project, history: Path) -> None:
        files = sorted(history.glob("*.json"))
        for old in files[:-HISTORY_KEEP]:
            try:
                old.unlink()
            except OSError:
                pass

    def _history_entry(self, folder: Path, revision: int) -> dict[str, Any]:
        path = folder / "history" / f"{revision:08d}.json"
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StoreError("That change is no longer in the history.") from exc
        if not isinstance(entry, dict) or not isinstance(entry.get("before"), dict) or not isinstance(entry.get("after"), dict):
            raise StoreError("That history entry is damaged.")
        return entry

    def undo(self, project_id: str, expected_revision: int) -> Project:
        return self._replay(project_id, expected_revision, undo=True)

    def redo(self, project_id: str, expected_revision: int) -> Project:
        return self._replay(project_id, expected_revision, undo=False)

    def _replay(self, project_id: str, expected_revision: int, undo: bool) -> Project:
        folder = self.folder(project_id)
        with self.lock(project_id):
            project = self.load(project_id)
            if project.revision != expected_revision:
                raise RevisionConflict(project)
            stack = project.undoStack if undo else project.redoStack
            if not stack:
                raise StoreError("Nothing to undo." if undo else "Nothing to redo.")
            target = stack[-1]
            entry = self._history_entry(folder, target)
            before = dump(project)
            data = dump(project)
            side, other = (entry["before"], entry["after"]) if undo else (entry["after"], entry["before"])
            _apply_side(data, side, other, None if undo else self._removed_by_undo(folder, project.revision, target))
            try:
                replayed = Project.model_validate(data)
            except ValidationError as exc:
                raise StoreError(f"Cannot {'undo' if undo else 'redo'} that change: {exc.errors()[0]['msg']}") from exc
            check_media(replayed, folder)
            deps.apply_stale(replayed)

            def stacks(p: Project) -> None:
                if undo:
                    p.undoStack = p.undoStack[:-1]
                    p.redoStack = (p.redoStack + [target])[-200:]
                else:
                    p.redoStack = p.redoStack[:-1]
                    p.undoStack = (p.undoStack + [target])[-200:]

            return self._commit(replayed, folder, before, f"{'undo' if undo else 'redo'} {entry.get('op', '')}".strip(),
                                reversible=False, stacks=stacks, extra={"undid": target} if undo else None)

    def _removed_by_undo(self, folder: Path, revision: int, target: int) -> dict[str, Any]:
        """Shots, as they were, that the latest undo of ``target`` removed (its ``before`` side keeps them whole)."""
        for rev in range(revision, target, -1):
            path = folder / "history" / f"{rev:08d}.json"
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(entry, dict) and entry.get("undid") == target:
                before = (entry.get("before") or {}).get("shots") or {}
                after = (entry.get("after") or {}).get("shots") or {}
                return {i: item for i, item in (before.get("items") or {}).items()
                        if isinstance(item, dict) and (after.get("items") or {}).get(i, "x") is None}
        return {}


_DEFAULT: dict[str, ProjectStore] = {}


def default_store() -> ProjectStore:
    root = storyboards_dir()
    key = str(root)
    if key not in _DEFAULT:
        _DEFAULT[key] = ProjectStore(root)
    return _DEFAULT[key]
