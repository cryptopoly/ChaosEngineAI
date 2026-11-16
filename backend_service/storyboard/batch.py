"""Batch generation for a storyboard: the shots that need work, ordered by kind, one Cancel for all.

A batch makes every still first, then every clip, and calls the injected ``unload`` between the two kinds
so a Mac does not hold an image and a video model at once. Each run adds a NEW take per shot (never an
overwrite) and makes it active when the shot's last step finishes; the take records the fingerprint the
dependency check compares. Locked shots are skipped, and naming a locked shot directly is refused,
unless ``force``. The generators are injected callables, so tests run without a model:
``make_still(shot, project) -> path`` and ``make_clip(shot, project, start_image) -> path``; either may
return ``(path, info)`` where info can carry ``measuredSeconds`` and ``settings``. Progress and results
are written into ``project.batches`` so a restart shows what finished.

Passes (S6): a ``draft`` run hands the generators a copy of the project at the draft size, and its takes are
marked ``draft``. ``what="final"`` redoes only the shots marked ``kept`` that have no finished final take: the
still with the quality model (``final_image_model``) at the project size, then the clip from that still.

Redo one half (S11): ``what="redo-clip"`` makes a new take that keeps the active still and has a new clip;
``what="redo-still"`` makes a new take with a new still that keeps the old clip until it is remade (the clip records
the still it started from, so ``deps.clip_outdated`` marks the shot stale and the next Make remakes only the clip).
Both ask the makers for a new seed (the shot's own seed is left alone) and keep the pass of the take they redo.

Pause and resume (S-M1): ``pause`` stops the run after the step in progress; the record becomes ``paused`` and its
``queued`` results are the steps left. A record a dead process left running is made ``paused`` by the store's
reconcile. ``resume`` continues a paused record under its own id from those steps; a still made before the pause
still gets its clip in the same take (the done result keeps its ``takeId``). Progress counts the whole batch.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service.progress import GenerationCancelled
from backend_service.storyboard import deps, media
from backend_service.storyboard.deps import clip_outdated
from backend_service.storyboard.models import BatchRecord, BatchResult, Project, Shot, Take, new_id
from backend_service.storyboard.store import LIVE_BATCHES, ProjectStore, RevisionConflict, StoreError

GenResult = Any  # str | Path | tuple[str | Path, dict]
MakeStill = Callable[[Shot, Project], GenResult]
MakeClip = Callable[[Shot, Project, "Path | None"], GenResult]
# ``stillSeconds`` / ``clipSeconds``: how long making that still or clip took here (the estimate reads them).
_SETTING_KEYS = ("model", "seed", "steps", "width", "height", "frames", "fps", "guidance", "stillSeconds", "clipSeconds",
                 "stillModel", "clipModel", "stillSeed", "clipSeed", "style")
KEEP_BATCHES = 20
BATCHES: dict[str, "Batch"] = {}
_BATCHES_LOCK = threading.Lock()


class BatchRefused(StoreError):
    pass


REDO = ("redo-clip", "redo-still")
NOTHING_TO_REDO = ("Nothing to redo: no unlocked shot has a take with that half yet (or it is an upload). "
                   "To make shots that have no take, use what: still, clip or both.")
WHATS = ("still", "clip", "both", "final", *REDO)
__all__ = ["clip_outdated"]


def draft_size(width: int, height: int, min_side: int | None = None) -> tuple[int, int]:
    """The size of a draft: half the project size in multiples of 64 (at least 256 a side), never below the
    model's minimum short side and never above the project size."""
    factor = 0.5
    if min_side:
        factor = min(1.0, max(factor, min_side / min(width, height)))
    if factor >= 1.0:
        return width, height
    return max(256, round(width * factor / 64) * 64), max(256, round(height * factor / 64) * 64)


def needs_final(shot: Shot) -> bool:
    """A kept shot that has no finished final take yet (a final pass with no shot ids works on these)."""
    if not shot.kept or shot.startFrom == "upload" or shot.status == "missing":
        return False
    take = shot.active()
    finished = take is not None and take.renderPass == "final" and bool(take.clip)
    return not finished or shot.redo or shot.status in ("stale", "failed")


def needs_work(shot: Shot, what: str = "both") -> bool:
    """A shot a batch works on when no shot ids are given. A still-only take needs its clip (not for a stills batch)."""
    if what == "final":
        return needs_final(shot)
    if shot.startFrom == "upload" or shot.status == "missing":
        return False
    if what in REDO:  # a take to redo half of; a new still needs a shot that makes one
        return shot.active() is not None and (what == "redo-clip" or shot.startFrom == "generate")
    if what != "still" and still_only(shot):
        return True
    if what == "still" and clip_outdated(shot) and not shot.redo:
        return False  # its still is the new one: only the clip is behind
    return shot.redo or shot.active() is None or shot.status in ("idle", "stale", "failed")


def still_only(shot: Shot) -> bool:
    """The active take has a still and no clip yet (an imported still, or a batch that made only stills)."""
    take = shot.active()
    return take is not None and bool(take.still) and not take.clip


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
        elif isinstance(value, str) and key in ("model", "stillModel", "clipModel"):
            out[key] = value.replace("\\", "/")[:200]  # a repo id like org/name, never a folder
    return out


def _remove_frame(frame: Path) -> None:
    """Delete an extracted last frame and its folder when that folder is now empty (never anything else)."""
    try:
        frame.unlink(missing_ok=True)
        frame.parent.rmdir()
    except OSError:
        pass


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
        clip_image_refusal: Callable[[Project], str | None] | None = None,
        draft: bool = False,
        draft_sizes: dict[str, tuple[int, int]] | None = None,
        final_image_model: str | None = None,
        draft_image_model: str | None = None,
    ) -> None:
        if what not in WHATS:
            raise BatchRefused("what must be still, clip, both or final.")
        if draft and what in ("final", *REDO):
            raise BatchRefused("A final pass or a redo is not a draft (a redo keeps the pass of the take it redoes).")
        self.id = new_id("b")
        self.store, self.project_id, self.what = store, project_id, what
        self.make_still, self.make_clip, self.unload = make_still, make_clip, unload
        self.last_frame, self.stop_running = last_frame, stop_running
        self.shot_ids, self.force = shot_ids, force
        # Passes (S6): drafts are made at ``draft_sizes`` ({"still"|"clip": (w, h)}); a final pass uses the quality model.
        self.draft, self.draft_sizes, self.final_image_model = draft, dict(draft_sizes or {}), final_image_model
        self.draft_image_model = draft_image_model  # a draft still uses this (fast) model when set, else the project's
        self.render_pass = "draft" if draft else "final"
        self._shot_pass: dict[str, str] = {}  # redo kinds: shot id -> the pass of the take being redone
        # Why the project's video model cannot start a clip from an image (None when it can or is unknown).
        self.clip_image_refusal = clip_image_refusal
        self._refusal: tuple[str | None, str | None] | None = None  # (video model, reason) cache
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self.done = threading.Event()
        self._prior = {"still": 0, "clip": 0}  # steps of each kind that ended before a resume (S-M1)
        self.steps: list[tuple[str, str]] = []  # (shot id, still|clip), stills first
        self._takes: dict[str, str] = {}  # shot id -> take made in this run
        self._began: float | None = None  # monotonic start of ``run``
        self._finished_steps: list[tuple[str, float]] = []  # (kind, seconds) of steps that made their output
        self._current: tuple[str, str, float] | None = None  # (shot id, kind, monotonic start) of the running step
        self._processed = 0  # steps that ended, made or not
        # The models a run would use, for the catalogue price of a step no earlier step has measured (FU-153).
        self._variants: tuple[dict[str, Any] | None, dict[str, Any] | None] = (None, None)
        self._prices: dict[tuple[str, str], tuple[float, str] | None] = {}  # (shot id, kind) -> (seconds, basis)

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
            shots = [s for s in project.shots if needs_work(s, self.what) and (self.force or not s.locked)]
        if self.what == "final":
            return self._plan_final(project, shots)
        if self.what in REDO:
            return self._plan_redo(project, shots)
        refusal = self._image_refusal(project) if self.what in ("clip", "both") else None
        stills, clips = [], []
        for shot in shots:
            # A clip that would ignore its still or its chain is refused, and its still is not made for nothing.
            blocked = refusal if shot.startFrom in ("generate", "previous-shot") else None
            if self.what in ("still", "both"):
                if self.what == "both" and (still_only(shot) or clip_outdated(shot)) and not shot.redo:
                    pass  # its still is there: only the clip is missing (or behind), made from that still
                elif shot.startFrom == "generate":
                    stills.append((shot.id, "still", blocked))
                elif self.what == "still":
                    stills.append((shot.id, "still", f"starts from {shot.startFrom}; no still to make"))
            if self.what in ("clip", "both"):
                clips.append((shot.id, "clip", "upload shot; import a clip" if shot.startFrom == "upload" else blocked))
        return stills + clips

    def _plan_final(self, project: Project, shots: list[Shot]) -> list[tuple[str, str, str | None]]:
        """A new still for every kept shot that starts from one, then every clip (from the new still)."""
        refusal = self._image_refusal(project)
        stills, clips = [], []
        for shot in shots:
            if shot.startFrom == "upload":
                clips.append((shot.id, "clip", "upload shot; import a clip"))
                continue
            blocked = refusal if shot.startFrom in ("generate", "previous-shot") else None
            if shot.startFrom == "generate":
                stills.append((shot.id, "still", blocked))
            clips.append((shot.id, "clip", blocked))
        return stills + clips

    def _plan_redo(self, project: Project, shots: list[Shot]) -> list[tuple[str, str, str | None]]:
        """One step per shot: a new still (shots that start from one) or a new clip (from the active still)."""
        if self.what == "redo-still":
            # Named shots get the same rule as the form without ids: a redo needs a take to redo (FU-188 item 7).
            return [(s.id, "still", f"starts from {s.startFrom}; no still to make" if s.startFrom != "generate"
                     else None if s.active() is not None else "no take yet; make the shot first")
                    for s in shots]
        refusal = self._image_refusal(project)
        steps = []
        for shot in shots:
            take = shot.active()
            if shot.startFrom == "upload":
                reason = "upload shot; import a clip"
            elif shot.startFrom == "generate" and not (take and take.still):
                reason = "no still yet; make the shot first"
            else:
                reason = refusal if shot.startFrom in ("generate", "previous-shot") else None
            steps.append((shot.id, "clip", reason))
        return steps

    def _pass(self, shot_id: str | None) -> str:
        return self._shot_pass.get(shot_id or "", self.render_pass)

    def run_project(self, project: Project, kind: str, shot_id: str | None = None) -> Project:
        """The project the generators see for one step: the draft size for a draft (or a redo of a draft take), the
        quality model for a final pass. A copy; the stored project never changes."""
        changes: dict[str, Any] = {}
        if self._pass(shot_id) == "draft" and kind in self.draft_sizes:
            changes["width"], changes["height"] = self.draft_sizes[kind]
            if kind == "still" and self.draft_image_model:
                changes["imageModel"] = self.draft_image_model
        if self.what == "final" and kind == "still":
            changes["imageModel"] = self.final_image_model or project.defaults.finalImageModel or project.defaults.imageModel
        if not changes:
            return project
        return project.model_copy(update={"defaults": project.defaults.model_copy(update=changes)})

    def _image_refusal(self, project: Project) -> str | None:
        if self.clip_image_refusal is None:
            return None
        model = project.defaults.videoModel
        if self._refusal is None or self._refusal[0] != model:
            self._refusal = (model, self.clip_image_refusal(project))
        return self._refusal[1]

    def start(self, expected_revision: int | None) -> Project:
        """Record the batch and mark its shots queued (revision-checked); then call ``run`` (or ``start_thread``)."""

        def change(p: Project, folder: Path) -> None:
            steps = self.plan(p)
            if not any(reason is None for _, _, reason in steps):
                refusal = self._image_refusal(p) if self.what in ("clip", "both", "final", "redo-clip") else None
                if refusal and any(reason == refusal for _, _, reason in steps):
                    raise BatchRefused(refusal)
                if self.what == "final":
                    raise BatchRefused("Nothing for a final pass: mark the shots to keep first (or every kept shot already has a final take).")
                if self.what in REDO:
                    raise BatchRefused(NOTHING_TO_REDO)
                raise BatchRefused("Nothing to generate: every shot is done, locked or an upload.")
            results = [BatchResult(shotId=s, kind=k, state="skipped" if r else "queued", error=r) for s, k, r in steps]  # type: ignore[arg-type]
            self.steps = [(s, k) for s, k, r in steps if r is None]
            for shot_id in {s for s, _ in self.steps}:
                shot = p.shot(shot_id)
                shot.status = "queued"  # type: ignore[union-attr]
                if self.what in REDO and shot.active() is not None:  # type: ignore[union-attr]
                    self._shot_pass[shot_id] = shot.active().renderPass  # type: ignore[union-attr]
            record = BatchRecord(id=self.id, what=self.what, force=self.force, draft=self.draft,  # type: ignore[arg-type]
                                 state="running", createdAt=time.time(), results=results)
            p.batches = (p.batches + [record])[-KEEP_BATCHES:]

        if self.what in ("clip", "both", "final", "redo-clip"):
            self._image_refusal(self.store.load(self.project_id))  # look the model up before taking the lock
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

    def resume(self, batch_id: str, expected_revision: int | None) -> Project:
        """Take over the paused record ``batch_id`` and queue its remaining steps (revision-checked); then ``run``."""
        from backend_service.storyboard.batch_queue import resume_record

        return resume_record(self, batch_id, expected_revision)

    def pause(self) -> bool:
        """Stop after the step in progress; the record becomes ``paused`` with its remaining steps kept."""
        if self.done.is_set():
            return False
        self.pause_event.set()
        return True

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

    def use_models(self, image_variant: dict[str, Any] | None, video_variant: dict[str, Any] | None) -> None:
        """Name the models this run uses, so a step no earlier step has measured is priced from the catalogue."""
        self._variants = (image_variant, video_variant)
        self._prices.clear()

    def _catalog_price(self, shot_id: str, kind: str) -> tuple[float, str] | None:
        """(seconds, basis) for one step from ``estimate.estimate`` (this project's takes, else the catalogue scaled
        to the project's size and the shot's frames), or None when no model is known or it has no figure."""
        key = (shot_id, kind)
        if key not in self._prices:
            price = None
            variant = self._variants[0 if kind == "still" else 1]
            if variant is not None:
                try:
                    from backend_service.storyboard.estimate import estimate

                    draft = self._pass(shot_id) == "draft"
                    got = estimate(
                        self.store.load(self.project_id), self.store.folder(self.project_id), what=kind,
                        shot_ids=[shot_id], image_variant=self._variants[0], video_variant=self._variants[1], missing=[],
                        render_pass="draft" if draft else "final", sizes=self.draft_sizes if draft else None,
                    )
                    if isinstance(got.get("seconds"), (int, float)) and got["seconds"] > 0:
                        price = (float(got["seconds"]), "clips" if got.get("basis") == "measured" else "catalog")
                except Exception:  # noqa: BLE001 - a price is a nicety; no price means elapsed time only
                    price = None
            self._prices[key] = price
        return self._prices[key]

    def progress(self) -> dict[str, Any]:
        """Where the run is, for ``job_status`` and the UI: "clip 2 of 4", elapsed, and time left.

        Each remaining step is priced at the average of the finished steps of the same kind in this run
        (basis ``clips``). A kind with no finished step yet (the first clip of a batch) is priced from the
        catalogue figure for that shot (basis ``catalog``) when the models are known (``use_models``); with
        neither there is no ``etaSeconds``, elapsed time only. The basis is ``catalog`` if any price used it.
        """
        steps, now = list(self.steps), time.monotonic()
        current = self._current
        finished = list(self._finished_steps)
        prior = dict(self._prior)
        info: dict[str, Any] = {"totalSteps": len(steps) + sum(prior.values())}
        if self._began is not None:
            info["elapsedSeconds"] = round(now - self._began, 1)
        done_by_kind = {k: sum(1 for kk, _ in finished if kk == k) for k in ("still", "clip")}
        total_by_kind = {k: sum(1 for _, kk in steps if kk == k) + prior[k] for k in ("still", "clip")}
        processed = min(self._processed, len(steps))
        info["step"] = min(len(steps), processed + (1 if current else 0)) + sum(prior.values())
        if current is not None:
            kind = current[1]
            index = sum(1 for _, kk in steps[:processed] if kk == kind) + 1 + prior[kind]
            info["phase"] = f"{kind} {index} of {total_by_kind[kind]}"
            info["kind"], info["kindIndex"], info["kindTotal"], info["shotId"] = kind, index, total_by_kind[kind], current[0]
            info["stepElapsedSeconds"] = round(now - current[2], 1)
            inner = _inner_step(kind)
            if inner is not None:
                info["innerStep"], info["innerTotal"] = inner
        mean = {k: sum(sec for kk, sec in finished if kk == k) / done_by_kind[k] for k in ("still", "clip") if done_by_kind[k]}
        if mean:
            info["finishedSeconds"] = {k: round(v, 1) for k, v in mean.items()}
        remaining = steps[processed + (1 if current else 0):]
        left: float | None = 0.0
        basis = "clips"

        def price(shot_id: str, kind: str) -> float | None:
            nonlocal basis
            if kind in mean:
                return mean[kind]
            got = self._catalog_price(shot_id, kind)
            if got is None:
                return None
            basis = "catalog" if got[1] == "catalog" or basis == "catalog" else "clips"
            return got[0]

        if current is not None:
            each = price(current[0], current[1])
            left = None if each is None else max(0.0, each - (now - current[2]))
        for shot_id, kind in remaining:
            each = price(shot_id, kind)
            if left is None or each is None:
                left = None
                break
            left += each
        if left is not None and (remaining or current is not None):
            info["etaSeconds"], info["etaBasis"] = round(left), basis
        return info

    def run(self) -> None:
        state = "done"
        self._began = time.monotonic()
        try:
            ran_still = False
            for shot_id, kind in self.steps:
                if self.cancel_event.is_set():
                    state = "cancelled"
                    break
                if self.pause_event.is_set():
                    state = "paused"
                    break
                if kind == "clip" and ran_still:
                    self.unload("image")
                    ran_still = False
                if kind == "still":
                    ran_still = True
                ok = self._step(shot_id, kind)
                self._processed += 1
                if not ok:
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
            began = time.monotonic()
            self._current = (shot_id, kind, began)
            # A redo asks for a new seed; the shot keeps its own (the fingerprint and the next Make still use it).
            made_shot = shot.model_copy(update={"seed": None}) if self.what in REDO else shot
            if kind == "still":
                path, info = _split(self.make_still(made_shot, self.run_project(project, kind, shot_id)))
            else:
                start = self._start_image(project, shot)
                try:
                    path, info = _split(self.make_clip(made_shot, self.run_project(project, kind, shot_id), start))
                finally:
                    if start is not None and shot.startFrom == "previous-shot":
                        _remove_frame(start)
            # The measured time of this step rides with the take; the estimate uses it for this model next time.
            made = dict(info.get("settings") or {})
            made[f"{kind}Seconds"] = round(time.monotonic() - began, 2)
            self._finished_steps.append((kind, time.monotonic() - began))
            if made.get("model"):
                made[f"{kind}Model"] = made["model"]
            if made.get("seed") is not None:
                made[f"{kind}Seed"] = made["seed"]
            info = {**info, "settings": made}
            self._update(lambda p, folder: self._store_output(p, folder, shot_id, kind, path, info))
            self._current = None
            return True
        except GenerationCancelled:
            self._current = None
            self._update(lambda p, folder: self._mark(p, shot_id, kind, "cancelled", None))
            return False
        except Exception as exc:  # noqa: BLE001 - recorded on the shot, the batch goes on
            self._current = None
            if self.cancel_event.is_set():
                # The studio reports a stopped run as an error of its own; the user cancelled, so the shot keeps its last take.
                self._update(lambda p, folder: self._mark(p, shot_id, kind, "cancelled", None))
                return False
            detail = getattr(exc, "detail", None)
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("localized")
            message = media.scrub(detail or str(exc) or type(exc).__name__)
            self._update(lambda p, folder: self._mark(p, shot_id, kind, "failed", message))
            return not self.cancel_event.is_set()

    def _start_image(self, project: Project, shot: Shot) -> Path | None:
        folder = self.store.folder(self.project_id)
        if shot.startFrom == "text":
            return None  # the shot asked for no picture, even when an older take left a still behind
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
        active = shot.active()
        render_pass = self._pass(shot_id)
        # A clip is always made from the active take's still, so it keeps that still's pass: a clip made after
        # "Make drafts" belongs to a draft take and Export still warns, whatever pass the clip batch itself is.
        from_still = take is None and kind == "clip" and active is not None and bool(active.still)
        if from_still and not active.clip:  # type: ignore[union-attr]
            take = active  # the clip completes the still-only take instead of starting a new one
            self._takes[shot_id] = take.id
        if take is None:
            take = Take(id=new_id("t"), createdAt=time.time(), origin="batch", renderPass=render_pass)  # type: ignore[arg-type]
            if from_still:
                take.renderPass = shot.active().renderPass  # type: ignore[union-attr]
                old = shot.active()
                take.still, take.hashes = old.still, {"still": old.hashes["still"]} if "still" in old.hashes else {}  # type: ignore[union-attr]
                take.settings = {k: v for k, v in old.settings.items() if k in ("stillModel", "stillSeed")}  # type: ignore[union-attr]
            elif kind == "still" and self.what == "redo-still" and active is not None and active.clip:
                # The old clip stays until it is remade; it records the still it came from, so the shot shows stale.
                take.clip, take.measuredSeconds, take.settings = active.clip, active.measuredSeconds, dict(active.settings)
                take.hashes = {"clip": active.hashes["clip"]} if "clip" in active.hashes else {}
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
            if take.still and "still" in take.hashes and shot.startFrom == "generate":
                settings["clipStill"] = take.hashes["still"]  # the still this clip started from (deps.clip_outdated)
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
            if record is not None and state == "paused":
                record.state, record.pausedAt, record.pauseReason = "paused", time.time(), "user"
                for result in record.results:
                    if result.state == "running":
                        result.state = "queued"
            elif record is not None:
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


def _inner_step(kind: str) -> tuple[int, int] | None:
    """(step, total) of the denoise loop the running still or clip is in, from the shared progress tracker, else None.

    A still takes minutes on a big model and its batch step only moves when it ends; this is the part in between."""
    from backend_service.progress import IMAGE_PROGRESS, VIDEO_PROGRESS

    try:
        snap = (IMAGE_PROGRESS if kind == "still" else VIDEO_PROGRESS).snapshot()
    except Exception:  # noqa: BLE001 - a missing figure is fine, the bar then shows elapsed time only
        return None
    step, total = snap.get("step"), snap.get("totalSteps")
    if snap.get("active") and snap.get("phase") in ("diffusing", "compose") and isinstance(step, int) and isinstance(total, int) and total > 0:
        return min(step, total), total
    return None


def get_batch(batch_id: str) -> Batch | None:
    with _BATCHES_LOCK:
        return BATCHES.get(batch_id)
