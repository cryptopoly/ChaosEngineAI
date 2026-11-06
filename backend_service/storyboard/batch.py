"""Batch generation for a storyboard: the shots that need work, ordered by kind, one Cancel for all.

A batch makes every still first, then every clip, and calls the injected ``unload`` between the two kinds
so a Mac does not hold an image and a video model at once. Each run adds a NEW take per shot (never an
overwrite) and makes it active when the shot's last step finishes; the take records the fingerprint the
dependency check compares. Locked shots are skipped, and naming a locked shot directly is refused,
unless ``force``. The generators are injected callables, so tests run without a model:
``make_still(shot, project) -> path`` and ``make_clip(shot, project, start_image) -> path``; either may
return ``(path, info)`` where info can carry ``measuredSeconds`` and ``settings``. Progress and results
are written into ``project.batches`` so a restart shows what finished.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service.progress import GenerationCancelled
from backend_service.storyboard import deps, media
from backend_service.storyboard.models import BatchRecord, BatchResult, Project, Shot, Take, new_id
from backend_service.storyboard.store import LIVE_BATCHES, ProjectStore, RevisionConflict, StoreError

GenResult = Any  # str | Path | tuple[str | Path, dict]
MakeStill = Callable[[Shot, Project], GenResult]
MakeClip = Callable[[Shot, Project, "Path | None"], GenResult]
_SETTING_KEYS = ("model", "seed", "steps", "width", "height", "frames", "fps", "guidance")
KEEP_BATCHES = 20
BATCHES: dict[str, "Batch"] = {}
_BATCHES_LOCK = threading.Lock()


class BatchRefused(StoreError):
    pass


def needs_work(shot: Shot) -> bool:
    if shot.startFrom == "upload" or shot.status == "missing":
        return False
    return shot.redo or shot.active() is None or shot.status in ("idle", "stale", "failed")


def _split(result: GenResult) -> tuple[Path, dict[str, Any]]:
    if isinstance(result, tuple):
        return Path(result[0]), dict(result[1] or {})
    return Path(result), {}


def _clean_settings(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _SETTING_KEYS:
        value = raw.get(key)
        if isinstance(value, (bool, int, float)) or (isinstance(value, str) and "/" not in value and "\\" not in value):
            out[key] = value if not isinstance(value, str) else value[:200]
        elif isinstance(value, str) and key == "model":
            out[key] = value.replace("\\", "/")[:200]  # a repo id like org/name, never a folder
    return out


class Batch:
    def __init__(
        self,
        store: ProjectStore,
        project_id: str,
        what: str,
        make_still: MakeStill,
        make_clip: MakeClip,
        unload: Callable[[str], None] = lambda kind: None,
        last_frame: Callable[[Path], Path] | None = None,
        stop_running: Callable[[str], bool] | None = None,
        shot_ids: list[str] | None = None,
        force: bool = False,
    ) -> None:
        if what not in ("still", "clip", "both"):
            raise BatchRefused("what must be still, clip or both.")
        self.id = new_id("b")
        self.store, self.project_id, self.what = store, project_id, what
        self.make_still, self.make_clip, self.unload = make_still, make_clip, unload
        self.last_frame, self.stop_running = last_frame, stop_running
        self.shot_ids, self.force = shot_ids, force
        self.cancel_event = threading.Event()
        self.done = threading.Event()
        self.steps: list[tuple[str, str]] = []  # (shot id, still|clip), stills first
        self._takes: dict[str, str] = {}  # shot id -> take made in this run

    # ---- planning ----
    def plan(self, project: Project) -> list[tuple[str, str, str | None]]:
        """(shot id, kind, skip reason) for every step, stills first."""
        if self.shot_ids is not None:
            unknown = [i for i in self.shot_ids if project.shot(i) is None]
            if unknown:
                raise BatchRefused(f"Unknown shot ids: {', '.join(unknown)}.")
            locked = [i for i in self.shot_ids if project.shot(i).locked]  # type: ignore[union-attr]
            if locked and not self.force:
                raise BatchRefused(f"Shots {', '.join(locked)} are locked. Unlock them or pass force.")
            shots = [s for s in project.shots if s.id in set(self.shot_ids)]
        else:
            shots = [s for s in project.shots if needs_work(s) and (self.force or not s.locked)]
        stills, clips = [], []
        for shot in shots:
            if self.what in ("still", "both"):
                if shot.startFrom == "generate":
                    stills.append((shot.id, "still", None))
                elif self.what == "still":
                    stills.append((shot.id, "still", f"starts from {shot.startFrom}; no still to make"))
            if self.what in ("clip", "both"):
                clips.append((shot.id, "clip", "upload shot; import a clip" if shot.startFrom == "upload" else None))
        return stills + clips

    def start(self, expected_revision: int | None) -> Project:
        """Record the batch and mark its shots queued (revision-checked); then call ``run`` (or ``start_thread``)."""

        def change(p: Project, folder: Path) -> None:
            steps = self.plan(p)
            if not any(reason is None for _, _, reason in steps):
                raise BatchRefused("Nothing to generate: every shot is done, locked or an upload.")
            results = [BatchResult(shotId=s, kind=k, state="skipped" if r else "queued", error=r) for s, k, r in steps]  # type: ignore[arg-type]
            self.steps = [(s, k) for s, k, r in steps if r is None]
            for shot_id in {s for s, _ in self.steps}:
                p.shot(shot_id).status = "queued"  # type: ignore[union-attr]
            record = BatchRecord(id=self.id, what=self.what, force=self.force, state="running",  # type: ignore[arg-type]
                                 createdAt=time.time(), results=results)
            p.batches = (p.batches + [record])[-KEEP_BATCHES:]

        LIVE_BATCHES.add(self.id)
        try:
            project = self.store.mutate(self.project_id, expected_revision, change, "batch start", reversible=False)
        except (StoreError, RevisionConflict):
            LIVE_BATCHES.discard(self.id)
            raise
        with _BATCHES_LOCK:
            for old in [b for b in BATCHES.values() if b.done.is_set()][:-KEEP_BATCHES]:
                BATCHES.pop(old.id, None)
            BATCHES[self.id] = self
        return project

    def start_thread(self) -> None:
        threading.Thread(target=self.run, name=f"storyboard-{self.id}", daemon=True).start()

    def cancel(self) -> bool:
        if self.done.is_set():
            return False
        self.cancel_event.set()
        from backend_service.helpers.job_queue import HEAVY_JOBS

        HEAVY_JOBS.cancel_batch(self.id, self.stop_running)
        return True

    # ---- running ----
    def _update(self, change: Callable[[Project, Path], None], op: str = "batch") -> Project:
        return self.store.mutate(self.project_id, None, change, op, reversible=False)

    def _result(self, p: Project, shot_id: str, kind: str) -> BatchResult | None:
        record = next((b for b in p.batches if b.id == self.id), None)
        if record is None:
            return None
        return next((r for r in record.results if r.shotId == shot_id and r.kind == kind and r.state != "skipped"), None)

    def _last_step(self, shot_id: str, kind: str) -> bool:
        return kind == "clip" or (shot_id, "clip") not in self.steps

    def run(self) -> None:
        state = "done"
        try:
            ran_still = False
            for shot_id, kind in self.steps:
                if self.cancel_event.is_set():
                    state = "cancelled"
                    break
                if kind == "clip" and ran_still:
                    self.unload("image")
                    ran_still = False
                if kind == "still":
                    ran_still = True
                if not self._step(shot_id, kind):
                    state = "cancelled"
                    break
            if any(r.state == "failed" for r in self.record().results) and state == "done":
                state = "failed"
        finally:
            self._finish(state)

    def record(self) -> BatchRecord:
        project = self.store.load(self.project_id)
        found = next((b for b in project.batches if b.id == self.id), None)
        if found is None:
            raise StoreError("This batch is no longer in the project.")
        return found

    def _step(self, shot_id: str, kind: str) -> bool:
        """Run one step; False when the batch was cancelled."""

        def running(p: Project, folder: Path) -> None:
            shot, result = p.shot(shot_id), self._result(p, shot_id, kind)
            if shot is not None:
                shot.status = "running"
            if result is not None:
                result.state = "running"

        try:
            project = self._update(running)
            shot = project.shot(shot_id)
            if shot is None:
                raise StoreError("the shot was removed")
            if kind == "still":
                path, info = _split(self.make_still(shot, project))
            else:
                path, info = _split(self.make_clip(shot, project, self._start_image(project, shot)))
            self._update(lambda p, folder: self._store_output(p, folder, shot_id, kind, path, info))
            return True
        except GenerationCancelled:
            self._update(lambda p, folder: self._mark(p, shot_id, kind, "cancelled", None))
            return False
        except Exception as exc:  # noqa: BLE001 - recorded on the shot, the batch goes on
            message = media.scrub(getattr(exc, "detail", None) or str(exc) or type(exc).__name__)
            self._update(lambda p, folder: self._mark(p, shot_id, kind, "failed", message))
            return not self.cancel_event.is_set()

    def _start_image(self, project: Project, shot: Shot) -> Path | None:
        folder = self.store.folder(self.project_id)
        if shot.startFrom == "previous-shot":
            prev = deps.previous_shot(project, shot)
            take = prev.active() if prev else None
            if take is None or not take.clip:
                raise StoreError("the previous shot has no clip to continue from")
            if self.last_frame is None:
                raise StoreError("no last-frame extractor is set up")
            return self.last_frame(media.inside(folder, take.clip))
        take = shot.take(self._takes.get(shot.id)) or shot.active()
        if take is not None and take.still:
            return media.inside(folder, take.still)
        return None

    def _store_output(self, p: Project, folder: Path, shot_id: str, kind: str, path: Path, info: dict[str, Any]) -> None:
        shot = p.shot(shot_id)
        if shot is None:
            return
        take = shot.take(self._takes.get(shot_id))
        if take is None:
            take = Take(id=new_id("t"), createdAt=time.time(), origin="batch")
            if kind == "clip" and shot.active() is not None and shot.active().still:  # type: ignore[union-attr]
                old = shot.active()
                take.still, take.hashes = old.still, {"still": old.hashes["still"]} if "still" in old.hashes else {}  # type: ignore[union-attr]
            shot.takes.append(take)
            self._takes[shot_id] = take.id
        src = media.resolve_source(str(path), {"image"} if kind == "still" else {"video"})
        rel = f"media/{shot.id}/{take.id}/{kind}{src.suffix.lower()}"
        digest = media.ingest(folder, src, rel)
        settings = {**take.settings, **_clean_settings(info.get("settings") or {})}
        if kind == "still":
            take.still, take.hashes = rel, {**take.hashes, "still": digest}
        else:
            measured = info.get("measuredSeconds") or media.probe_seconds(media.inside(folder, rel))
            take.clip, take.hashes, take.measuredSeconds = rel, {**take.hashes, "clip": digest}, measured
        take.settings = settings
        result = self._result(p, shot_id, kind)
        if result is not None:
            result.state, result.takeId = "done", take.id
        if self._last_step(shot_id, kind):
            shot.activeTake = take.id
            take.basis = deps.fingerprint(p, shot)
            shot.status, shot.redo = "done", False
        else:
            shot.status = "queued"

    def _mark(self, p: Project, shot_id: str, kind: str, state: str, error: str | None) -> None:
        shot, result = p.shot(shot_id), self._result(p, shot_id, kind)
        if result is not None:
            result.state, result.error = state, error  # type: ignore[assignment]
        if shot is not None:
            shot.status = "failed" if state == "failed" else ("done" if shot.active() else "idle")

    def _finish(self, state: str) -> None:
        def change(p: Project, folder: Path) -> None:
            record = next((b for b in p.batches if b.id == self.id), None)
            if record is not None:
                record.state, record.finishedAt = state, time.time()  # type: ignore[assignment]
                for result in record.results:
                    if result.state in ("queued", "running"):
                        result.state = "cancelled"
            for shot_id in {s for s, _ in self.steps}:
                shot = p.shot(shot_id)
                if shot is not None and shot.status in ("queued", "running"):
                    shot.status = "done" if shot.active() else "idle"

        try:
            self._update(change, "batch end")
        finally:
            LIVE_BATCHES.discard(self.id)
            self.done.set()


def get_batch(batch_id: str) -> Batch | None:
    with _BATCHES_LOCK:
        return BATCHES.get(batch_id)
