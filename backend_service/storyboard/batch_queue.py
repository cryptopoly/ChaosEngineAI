"""The durable side of a storyboard batch (S-M1): resume a paused record, cancel one, and its batch bar data.

A batch record keeps one result per step; the ``queued`` ones of a ``paused`` record are the steps left. A record
gets there through ``Batch.pause`` (``pauseReason`` "user") or the store's reconcile after the app stopped while it
ran ("restart"). ``resume_record`` hands those steps to a fresh ``Batch`` under the record's own id, so the record,
its finished takes and its results stay one batch; ``paused_progress`` is what the batch bar shows meanwhile.
Split out of ``batch`` to keep that module near its size cap; ``Batch.resume`` calls in here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend_service.storyboard.models import BatchRecord, BatchResult, Project
from backend_service.storyboard.store import CANNOT_RESUME_NOTE, LIVE_BATCHES, ProjectBusy, ProjectStore, StoreError

if TYPE_CHECKING:
    from backend_service.storyboard.batch import Batch

NOTHING_TO_RESUME = "Nothing left to resume: every remaining step's shot is gone or locked. Cancel this batch."
LOCKED_SINCE = "locked since the pause; unlock it and make it again"


def step_seconds(project: Project, result: BatchResult) -> float | None:
    """How long a finished step took, from the seconds its take recorded (``stillSeconds`` / ``clipSeconds``)."""
    shot = project.shot(result.shotId)
    take = shot.take(result.takeId) if shot is not None else None
    value = (take.settings or {}).get(f"{result.kind}Seconds") if take is not None else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def resume_record(batch: "Batch", batch_id: str, expected_revision: int | None) -> Project:
    """Make ``batch`` the run of the paused record ``batch_id``: its queued steps, the takes its stills started,
    and the counts and times of the steps that ended before, so progress covers the whole batch."""
    from backend_service.storyboard.batch import _BATCHES_LOCK, BATCHES, REDO, BatchRefused

    if batch_id in LIVE_BATCHES:
        raise ProjectBusy("This batch is already running.")

    def change(p: Project, folder: Path) -> None:
        record = next((b for b in p.batches if b.id == batch_id), None)
        if record is None:
            raise StoreError(f"Unknown batch '{batch_id}'.")
        if record.state != "paused":
            raise ProjectBusy(f"This batch is {record.state}, not paused; only a paused batch can resume.")
        steps: list[tuple[str, str]] = []
        for result in record.results:
            shot = p.shot(result.shotId)
            if result.state == "done" and result.kind == "still" and result.takeId and shot is not None \
                    and shot.take(result.takeId) is not None:
                batch._takes[result.shotId] = result.takeId  # the clip completes the take this still started
            if result.state in ("done", "failed"):
                batch._prior[result.kind] += 1
                seconds = step_seconds(p, result)
                if seconds is not None:
                    batch._finished_steps.append((result.kind, seconds))
            if result.state != "queued":
                continue
            if shot is None:
                result.state, result.error = "cancelled", CANNOT_RESUME_NOTE
            elif shot.locked and not batch.force:
                result.state, result.error = "skipped", LOCKED_SINCE
            else:
                steps.append((result.shotId, result.kind))
        if not steps:
            raise BatchRefused(NOTHING_TO_RESUME)
        batch.steps = steps
        for shot_id in {s for s, _ in steps}:
            shot = p.shot(shot_id)
            shot.status = "queued"  # type: ignore[union-attr]
            if batch.what in REDO and shot.active() is not None:  # type: ignore[union-attr]
                batch._shot_pass[shot_id] = shot.active().renderPass  # type: ignore[union-attr]
        record.state, record.resumedAt, record.resumes = "running", time.time(), record.resumes + 1

    batch.id = batch_id
    if batch.what in ("clip", "both", "final", "redo-clip"):
        batch._image_refusal(batch.store.load(batch.project_id))  # look the model up before taking the lock
    LIVE_BATCHES.add(batch_id)
    try:
        project = batch.store.mutate(batch.project_id, expected_revision, change, "batch resume", reversible=False)
    except BaseException:
        LIVE_BATCHES.discard(batch_id)
        raise
    with _BATCHES_LOCK:
        BATCHES[batch_id] = batch
    return project


def cancel_paused(store: ProjectStore, project_id: str, batch_id: str) -> bool:
    """Cancel a paused batch (no thread runs it): the record and its remaining steps become ``cancelled``."""

    def change(p: Project, folder: Path) -> None:
        record = next((b for b in p.batches if b.id == batch_id), None)
        if record is None:
            raise StoreError(f"Unknown batch '{batch_id}'.")
        if record.state != "paused":
            raise ProjectBusy(f"This batch is {record.state}; only a running or paused batch can be cancelled.")
        record.state, record.finishedAt = "cancelled", time.time()
        for result in record.results:
            if result.state in ("queued", "running"):
                result.state = "cancelled"

    if batch_id in LIVE_BATCHES:
        raise ProjectBusy("This batch is running; cancel its run.")
    store.mutate(project_id, None, change, "batch cancel", reversible=False)
    return True


def paused_progress(project: Project, record: BatchRecord) -> dict[str, Any]:
    """The batch bar data of a paused batch: steps finished and left, and the time left when there is a basis.

    Each remaining step is priced at the mean measured seconds of this batch's finished steps of its kind (basis
    ``clips``), else of this project's other takes of that kind (basis ``project``); with neither, no ``etaSeconds``."""
    counted = [r for r in record.results if r.state != "skipped"]
    finished = [r for r in counted if r.state in ("done", "failed")]
    left = [r for r in counted if r.state in ("queued", "running")]
    info: dict[str, Any] = {
        "paused": True, "pauseReason": record.pauseReason, "totalSteps": len(finished) + len(left),
        "step": len(finished), "remainingSteps": len(left),
        "remaining": {k: sum(1 for r in left if r.kind == k) for k in ("still", "clip")},
    }
    own: dict[str, list[float]] = {"still": [], "clip": []}
    for result in finished:
        seconds = step_seconds(project, result)
        if seconds is not None:
            own[result.kind].append(seconds)
    other: dict[str, list[float]] = {"still": [], "clip": []}
    for shot in project.shots:
        for take in shot.takes:
            for kind in ("still", "clip"):
                value = (take.settings or {}).get(f"{kind}Seconds")
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    other[kind].append(float(value))
    total, basis = 0.0, "clips"
    for result in left:
        rates = own[result.kind] or other[result.kind]
        if not rates:
            return info  # no basis for this kind: steps only, no time left
        if not own[result.kind]:
            basis = "project"
        total += sum(rates) / len(rates)
    if left:
        info["etaSeconds"], info["etaBasis"] = round(total), basis
    return info
