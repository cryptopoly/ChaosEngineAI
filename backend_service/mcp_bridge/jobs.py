"""Agent jobs: each runs on the shared heavy-job queue and is polled by id."""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any, Callable

from backend_service.helpers.job_queue import HEAVY_JOBS
from backend_service.progress import GenerationCancelled, IMAGE_PROGRESS, VIDEO_PROGRESS

MAX_KEPT = 100
# A runner returns (output, notes).
Runner = Callable[[], tuple[dict[str, Any], list[str]]]


class Job:
    def __init__(self, job_id: str, kind: str, label: str, ticket: Any) -> None:
        self.id = job_id
        self.kind = kind
        self.label = label
        self.ticket = ticket
        self.state = "queued"  # queued | running | done | failed | cancelled
        self.output: dict[str, Any] | None = None
        self.error: str | None = None
        self.notes: list[str] = []
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.done = threading.Event()

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "jobId": self.id,
            "kind": self.kind,
            "label": self.label,
            "state": self.state,
            "notes": list(self.notes),
        }
        if self.state == "queued":
            info["queuePosition"] = max(HEAVY_JOBS.position(self.ticket), 0)
        if self.state == "running":
            tracker = {"image": IMAGE_PROGRESS, "video": VIDEO_PROGRESS}.get(self.kind)
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


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._ids = itertools.count(1)

    def submit(self, kind: str, label: str, runner: Runner) -> Job:
        ticket = HEAVY_JOBS.enqueue(kind, "agent", label)
        with self._lock:
            job = Job(f"job-{next(self._ids)}", kind, label, ticket)
            self._jobs[job.id] = job
            for old in [j for j in self._jobs.values() if j.done.is_set()][:-MAX_KEPT]:
                self._jobs.pop(old.id, None)
        threading.Thread(target=self._run, args=(job, runner), name=job.id, daemon=True).start()
        return job

    def _run(self, job: Job, runner: Runner) -> None:
        try:
            HEAVY_JOBS.wait_turn(job.ticket)
            job.state = "running"
            output, notes = runner()
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
            job.state = "failed"
            job.error = _readable(exc)
        finally:
            HEAVY_JOBS.release(job.ticket)
            job.finished_at = time.time()
            job.done.set()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job: Job) -> bool:
        if job.done.is_set():
            return False
        if job.state == "queued":
            return HEAVY_JOBS.cancel_waiting(ticket_id=job.ticket.id) > 0
        tracker = {"image": IMAGE_PROGRESS, "video": VIDEO_PROGRESS}.get(job.kind)
        return bool(tracker and tracker.request_cancel())


def _readable(exc: Exception) -> str:
    detail = getattr(exc, "detail", None)  # HTTPException from the shared route code
    return str(detail if detail else exc) or type(exc).__name__


STORE = JobStore()
