"""Agent jobs: each runs on the shared heavy-job queue and is polled by id.

Every job keeps a small JSON record under ``<data dir>/agent_jobs`` for 7 days, so job ids survive an app
restart: a job that was queued or running then reads ``interrupted`` (the storyboard rule). Each token may
have at most ``MAX_UNFINISHED_PER_TOKEN`` unfinished jobs.
"""

from __future__ import annotations

import contextvars
import itertools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service.helpers.job_queue import HEAVY_JOBS
from backend_service.progress import GenerationCancelled, IMAGE_PROGRESS, MUSIC_PROGRESS, VIDEO_PROGRESS

MAX_KEPT = 100
KEEP_SECONDS = 7 * 24 * 3600
MAX_UNFINISHED_PER_TOKEN = 20
FINISHED = ("done", "failed", "cancelled", "interrupted")
# Kinds whose running job can be cancelled through their progress tracker (music: the worker's
# cancel, then a kill after 3 s). Voice is handled by ``_cancel_voice``.
_TRACKERS = {"image": IMAGE_PROGRESS, "video": VIDEO_PROGRESS, "music": MUSIC_PROGRESS}
# The token that is making the current tool call; set by the protocol layer, read by ``submit``.
CURRENT_OWNER: contextvars.ContextVar[str] = contextvars.ContextVar("agent_job_owner", default="app")
# A runner returns (output, notes).
Runner = Callable[[], tuple[dict[str, Any], list[str]]]


class JobLimitError(Exception):
    """Too many unfinished jobs for one token; shown to the agent as a readable tool error."""


def jobs_dir() -> Path | None:
    """Where job records live. None under pytest (unless a test patches this), so tests never touch real data."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    from backend_service.app import SETTINGS_DIR

    return Path(SETTINGS_DIR) / "agent_jobs"


class Job:
    def __init__(self, job_id: str, kind: str, label: str, ticket: Any, cancel: Callable[[], bool] | None = None,
                 owner: str = "app", estimate_seconds: float | None = None) -> None:
        self.id = job_id
        self.cancel_fn = cancel  # set for a job that queues its own work (a storyboard batch or render)
        self.kind = kind
        self.label = label
        self.ticket = ticket
        self.owner = owner
        self.estimate_seconds = estimate_seconds
        self.state = "queued"  # queued | running | done | failed | cancelled | interrupted
        self.output: dict[str, Any] | None = None
        self.error: str | None = None
        self.notes: list[str] = []
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.cancel_requested = False
        self.done = threading.Event()

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "jobId": self.id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "notes": list(self.notes),
        }
        if self.state == "queued" and self.ticket is not None:
            info["queuePosition"] = max(HEAVY_JOBS.position(self.ticket), 0)
        if self.state == "running":
            tracker = _TRACKERS.get(self.kind)
            if tracker is not None:
                snap = tracker.snapshot()
                info["progress"] = {
                    "phase": snap.get("phase"),
                    "step": snap.get("step"),
                    "totalSteps": snap.get("totalSteps"),
                    "elapsedSeconds": snap.get("elapsedSeconds"),
                }
        if self.output is not None:
            info["output"] = self.output
        if self.error:
            info["error"] = self.error
        return info

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "state": self.state, "owner": self.owner,
            "output": self.output, "error": self.error, "notes": self.notes,
            "createdAt": self.created_at, "finishedAt": self.finished_at,
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)
        self._loaded = False

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        """Read the records of earlier runs once. Caller holds no lock."""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            directory = jobs_dir()
            if directory is None or not directory.is_dir():
                return
            highest = 0
            for path in directory.glob("job-*.json"):
                try:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                    stamp = float(rec.get("finishedAt") or rec["createdAt"])
                    if time.time() - stamp > KEEP_SECONDS:
                        path.unlink(missing_ok=True)
                        continue
                    job = Job(str(rec["id"]), str(rec["kind"]), str(rec["label"]), None, owner=str(rec.get("owner") or "app"))
                    job.state, job.output, job.error = str(rec["state"]), rec.get("output"), rec.get("error")
                    job.notes = list(rec.get("notes") or [])
                    job.created_at, job.finished_at = float(rec["createdAt"]), rec.get("finishedAt")
                    highest = max(highest, int(job.id.split("-")[1]))
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    continue
                if job.state not in FINISHED:
                    job.state = "interrupted"
                    job.error = "The app restarted while this job was waiting or running. Start it again."
                    job.finished_at = time.time()
                    self._write(job)
                job.done.set()
                self._jobs[job.id] = job
            self._ids = itertools.count(highest + 1)

    @staticmethod
    def _write(job: Job) -> None:
        directory = jobs_dir()
        if directory is None:
            return
        try:
            directory.mkdir(parents=True, exist_ok=True)
            tmp = directory / f"{job.id}.tmp"
            tmp.write_text(json.dumps(job.record(), default=str), encoding="utf-8")
            os.replace(tmp, directory / f"{job.id}.json")
        except OSError:
            pass  # a record that cannot be written only costs restart-safety

    @staticmethod
    def _forget(job_id: str) -> None:
        directory = jobs_dir()
        if directory is not None:
            (directory / f"{job_id}.json").unlink(missing_ok=True)

    # -- submit / run ----------------------------------------------------

    def unfinished(self, owner: str) -> int:
        self._load()
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.owner == owner and not j.done.is_set())

    def check_capacity(self, owner: str | None = None) -> None:
        owner = owner or CURRENT_OWNER.get()
        if self.unfinished(owner) >= MAX_UNFINISHED_PER_TOKEN:
            raise JobLimitError(
                f"This token already has {MAX_UNFINISHED_PER_TOKEN} unfinished jobs (the limit). Wait for some "
                "to finish with wait_for_job, or cancel_job the ones you do not need, then try again."
            )

    def submit(self, kind: str, label: str, runner: Runner, cancel: Callable[[], bool] | None = None,
               estimate_seconds: float | None = None) -> Job:
        """Run ``runner`` on the heavy queue. With ``cancel``, take no ticket: the runner queues its own work."""
        owner = CURRENT_OWNER.get()
        self.check_capacity(owner)
        ticket = HEAVY_JOBS.enqueue(kind, "agent", label) if cancel is None else None
        with self._lock:
            job = Job(f"job-{next(self._ids)}", kind, label, ticket, cancel, owner, estimate_seconds)
            self._jobs[job.id] = job
            finished = sorted((j for j in self._jobs.values() if j.done.is_set()), key=lambda j: j.created_at)
            stale = [j for j in finished if time.time() - (j.finished_at or j.created_at) > KEEP_SECONDS]
            for old in stale + finished[: max(0, len(finished) - len(stale) - MAX_KEPT)]:
                if self._jobs.pop(old.id, None) is not None:
                    self._forget(old.id)
        self._write(job)
        threading.Thread(target=self._run, args=(job, runner), name=job.id, daemon=True).start()
        return job

    def _run(self, job: Job, runner: Runner) -> None:
        try:
            if job.ticket is not None:
                HEAVY_JOBS.wait_turn(job.ticket)
            job.state = "running"
            job.started_at = time.time()
            self._write(job)
            output, notes = runner()
            if job.cancel_requested:
                raise GenerationCancelled("cancelled")
            if output.get("path"):
                from backend_service.mcp_bridge.files import download_info

                try:
                    info = download_info(str(output["path"]))
                except OSError:
                    info = None
                if info:
                    output = {**output, "download": info}
            job.output, job.notes = output, notes
            job.state = "done"
        except GenerationCancelled:
            job.state = "cancelled"
        except Exception as exc:  # noqa: BLE001 - shown to the agent as text
            if job.cancel_requested:
                job.state = "cancelled"  # a killed voice worker raises; that is the cancel working
            else:
                job.state = "failed"
                job.error = _readable(exc)
        finally:
            if job.ticket is not None:
                HEAVY_JOBS.release(job.ticket)
            job.finished_at = time.time()
            self._write(job)
            job.done.set()

    # -- read / cancel ---------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        self._load()
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, state: str | None = None, limit: int = 20) -> list[Job]:
        self._load()
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j for j in jobs if state is None or j.state == state][:limit]

    def cancel(self, job: Job) -> bool:
        if job.done.is_set():
            return False
        if job.cancel_fn is not None:
            return bool(job.cancel_fn())
        if job.state == "queued":
            return HEAVY_JOBS.cancel_waiting(ticket_id=job.ticket.id) > 0
        if job.kind == "voice":
            return _cancel_voice(job)
        tracker = _TRACKERS.get(job.kind)
        return bool(tracker and tracker.request_cancel())

    # -- queue view (system_status) ---------------------------------------

    def queue_view(self) -> dict[str, Any]:
        """The whole heavy queue, running first, with positions and labelled-estimated times."""
        snap = HEAVY_JOBS.snapshot()
        with self._lock:
            by_ticket = {j.ticket.id: j for j in self._jobs.values() if j.ticket is not None and not j.done.is_set()}
        now = time.time()
        entries: list[dict[str, Any]] = []
        wait: float | None = 0.0  # seconds until the next entry starts; None once an estimate is missing
        rows = ([snap["running"]] if snap["running"] else []) + list(snap["waiting"])
        for index, row in enumerate(rows):
            job = by_ticket.get(row["id"])
            estimate = job.estimate_seconds if job is not None else None
            running = index == 0 and snap["running"] is not None
            entry: dict[str, Any] = {
                "kind": row["kind"], "source": row["source"], "label": row["label"],
                "position": 0 if running else index + (0 if snap["running"] else 1) ,
                "jobId": job.id if job is not None else None,
                "estimatedSeconds": round(estimate) if estimate else None,
                "estimatedStartInSeconds": None if running or wait is None else round(wait),
            }
            if running and estimate:
                entry["estimatedRemainingSeconds"] = max(0, round(estimate - (now - (row.get("startedAt") or now))))
            entries.append(entry)
            if wait is not None:
                spent = (entry.get("estimatedRemainingSeconds") if running else estimate) if (estimate or running) else None
                wait = None if spent is None else wait + spent
        return {
            "jobs": entries,
            "estimateNote": "Times are estimated from the model catalogue for this machine; the first run of a model "
            "is slower (loading). A missing time means no estimate exists.",
        }


def _cancel_voice(job: Job) -> bool:
    """Stop a running speak or transcribe by killing the voice worker; it restarts on the next voice job."""
    from backend_service import voice_client

    job.cancel_requested = True
    worker = voice_client._WORKER  # noqa: SLF001 - request() holds the worker lock, so stop() would wait for it
    if worker is None:
        return True
    worker._kill()  # noqa: SLF001 - the blocked request then fails and _run reports "cancelled"
    return True


def _readable(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)  # HTTPException from the shared route code
    return str(detail if detail else exc) or type(exc).__name__


STORE = JobStore()
