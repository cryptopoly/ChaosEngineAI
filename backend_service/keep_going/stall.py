"""Stall guard for Keep going (FU-207): a job the run waits on that makes no progress stops the run.

A ``wait_for_job`` call blocks up to 30 minutes and a repeat wait costs no tool step, so a hung job once kept
one turn alive for 76 minutes. ``TurnWatch`` is handed to one chat turn's studio tools: while ``wait_for_job``
blocks it reads the job every few seconds and remembers when its progress last changed (state, queue place,
phase or step, output, or another chat job finishing). After the run's ``stall_minutes`` with no change it
stops the run with ``job_stalled`` (the job is left as it is) and the wait returns. It also tells the run which
job it waits on, for the chip. A chat without an active run gets no stall guard.
"""

from __future__ import annotations

import json
from typing import Any

CHECK_SECONDS = 5.0


def job_signature(job: Any, finished: list[str]) -> str:
    """What counts as progress for a job: everything in its state but the clock (elapsed time, time left)."""
    info = job.describe() if hasattr(job, "describe") else {"state": getattr(job, "state", None)}
    progress = dict(info.get("progress") or {})
    for key in ("elapsedSeconds", "etaSeconds", "etaBasis"):
        progress.pop(key, None)
    seen = {"state": info.get("state"), "queue": info.get("queuePosition"), "progress": progress,
            "output": info.get("output"), "finished": finished}
    return json.dumps(seen, sort_keys=True, default=str)


class TurnWatch:
    """One chat turn's view of the stall rule; ``controller`` is the app's ``KeepGoing``."""

    def __init__(self, controller: Any, session_id: str) -> None:
        self.controller = controller
        self.session_id = session_id
        self._marks: dict[str, tuple[str, float]] = {}  # job id -> (signature, when it last changed)
        self._checked: float | None = None
        self.reason = "A job has made no progress for too long."  # the run's stop reason once it stalled

    def stalled(self, job: Any) -> bool:
        """Called while a wait blocks. True once the run's limit passed with no progress (the run is then
        stopped). Reads the job at most every ``CHECK_SECONDS``."""
        clock = self.controller.clock
        now = clock()
        if self._checked is not None and now - self._checked < CHECK_SECONDS:
            return False
        self._checked = now
        limit = self.controller.stall_seconds(self.session_id)
        if limit is None:
            return False
        finished = sorted(str(j.id) for j in self.controller.hooks.chat_jobs()
                          if getattr(j, "state", "") in ("done", "failed", "cancelled", "interrupted"))
        sig = job_signature(job, finished)
        mark = self._marks.get(job.id)
        if mark is None or mark[0] != sig:
            self._marks[job.id] = mark = (sig, now)
        self.controller.set_waiting(self.session_id, {"jobId": job.id, "label": job.label, "since": mark[1]})
        if now - mark[1] < limit:
            return False
        self.controller.job_stalled(self.session_id, job.label)
        run = self.controller.get(self.session_id)
        self.reason = (run.stop_reason if run is not None else None) or self.reason
        return True

    def finish(self) -> None:
        """The wait ended: the chip stops saying what it waits on."""
        self._checked = None
        self.controller.set_waiting(self.session_id, None)


def waiting_stalled(controller: Any, session_id: str) -> bool:
    """A run waiting between turns on chat jobs: True (and the run stopped) once none of them made progress for
    the run's limit. ``controller.wait_marks`` keeps (signature, when it last changed) per chat."""
    from backend_service.keep_going.run import _finished_ids

    jobs = controller.hooks.chat_jobs()
    busy = [j for j in jobs if getattr(j, "state", "") in ("queued", "running")]
    now = controller.clock()
    run = controller.get(session_id)
    if run is None or run.state != "waiting" or not busy:
        controller.wait_marks.pop(session_id, None)
        return False
    finished = _finished_ids(jobs)
    sig = "|".join(job_signature(j, finished) for j in busy)
    mark = controller.wait_marks.get(session_id)
    if mark is None or mark[0] != sig:
        controller.wait_marks[session_id] = mark = (sig, now)
    first_id = str(getattr(busy[0], "id", ""))
    label = str(getattr(busy[0], "label", "") or first_id)
    controller.set_waiting(session_id, {"jobId": first_id, "label": label, "since": mark[1]})
    if now - mark[1] < run.stall_minutes * 60.0:
        return False
    controller.wait_marks.pop(session_id, None)
    controller.stop(session_id, "job_stalled", label, cancel_turn=False)
    return True
