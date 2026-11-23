"""Sync-scoring add-on for lip-sync takes (D-132): status and the one-click install the Video Studio offers.

* ``GET  /api/setup/lipsync-score/status``: installed or not, size, and the install job's state.
* ``POST /api/setup/install-lipsync-score``: installs it in a background thread (``lipsync_score_addon.install``);
  a second call while it runs returns the running job.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter

from backend_service import lipsync_score_addon as addon

router = APIRouter()


class _Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.phase = "idle"  # idle | packages | model | check | done | error
        self.error: str | None = None
        self.started_at = 0.0
        self.finished_at = 0.0

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {"phase": self.phase, "error": self.error, "running": self.running(),
                    "done": self.phase in ("done", "error"), "startedAt": self.started_at,
                    "finishedAt": self.finished_at}


_JOB = _Job()


def _set_phase(phase: str) -> None:
    with _JOB.lock:
        _JOB.phase = phase


def _work() -> None:
    try:
        addon.install(_set_phase)
    except Exception as exc:  # noqa: BLE001 - the thread has no caller; the reason goes to the status
        with _JOB.lock:
            _JOB.phase, _JOB.error = "error", str(exc)
    else:
        _set_phase("done")
    finally:
        with _JOB.lock:
            _JOB.finished_at = time.time()


@router.get("/api/setup/lipsync-score/status")
def lipsync_score_status() -> dict[str, Any]:
    return {**addon.status(), "job": _JOB.to_dict()}


@router.post("/api/setup/install-lipsync-score")
def install_lipsync_score() -> dict[str, Any]:
    with _JOB.lock:
        if not (_JOB.thread is not None and _JOB.thread.is_alive()):
            _JOB.phase, _JOB.error, _JOB.started_at, _JOB.finished_at = "packages", None, time.time(), 0.0
            _JOB.thread = threading.Thread(target=_work, name="chaosengine-lipsync-score-install", daemon=True)
            _JOB.thread.start()
    return {**addon.status(), "job": _JOB.to_dict()}
