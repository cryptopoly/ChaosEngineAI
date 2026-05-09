"""LongLive installer endpoints.

Background-job pattern: a single in-memory ``_LongLiveJobState`` tracks
the running install. POST starts a thread; GET polls. A second POST
while the job is running returns the running state instead of starting
a duplicate.

Extracted from ``routes/setup.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@dataclass
class _LongLiveJobState:
    """In-memory status for the currently-running or most-recent LongLive install.

    Same single-job semantics as the GPU bundle: a second POST while
    running returns the running job's state. State sticks around after
    completion so a late status poll sees the final outcome.
    """

    id: str = ""
    phase: str = "idle"  # idle | preflight | downloading | verifying | done | error
    message: str = ""
    package_current: str | None = None
    package_index: int = 0
    package_total: int = 0
    percent: float = 0.0
    target_dir: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "message": self.message,
            "packageCurrent": self.package_current,
            "packageIndex": self.package_index,
            "packageTotal": self.package_total,
            "percent": round(self.percent, 1),
            "targetDir": self.target_dir,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": self.attempts,
            "done": self.done,
        }


_LONGLIVE_JOB = _LongLiveJobState()
_LONGLIVE_LOCK = threading.Lock()


# Friendly labels for the phases declared in
# ``backend_service.longlive_installer.INSTALL_PHASES``. Used by the
# job worker to populate ``package_current`` so the UI shows
# "Step 4/9: Installing requirements" instead of "pip-requirements".
_LONGLIVE_PHASE_LABELS: dict[str, str] = {
    "clone": "Clone LongLive repo",
    "venv": "Create isolated venv",
    "pip-upgrade": "Upgrade pip / setuptools / wheel",
    "pip-requirements": "Install LongLive requirements",
    "flash-attn": "Build flash-attn (optional)",
    "pip-hub": "Install huggingface-hub",
    "weights-longlive": "Download LongLive checkpoints (~5 GB)",
    "weights-wan": "Download Wan 2.1 base (~3 GB)",
    "marker": "Write ready marker",
}


def _longlive_job_worker() -> None:
    """Run ``longlive_installer.install`` and stream its output into the job state.

    All exceptions land in ``job.error`` rather than escaping — the daemon
    thread has no parent to receive them. The job's ``phase`` flips to
    ``error`` on any failure and ``done`` on a clean run.
    """
    # Local import: ``backend_service.longlive_installer`` pulls in
    # ``longlive_engine`` which imports torch/transformers shims at module
    # load time on some paths. Deferring the import keeps the route module
    # importable in environments where those aren't installed (notably the
    # macOS test box and CI).
    from backend_service import longlive_installer  # noqa: PLC0415

    job = _LONGLIVE_JOB

    # Buffer of streamed log lines. We chunk the buffer into one
    # ``output`` field per phase so each row in the InstallLogPanel
    # holds the lines that landed during that phase. The current
    # phase's buffer accumulates until ``progress()`` fires.
    phase_buffer: list[str] = []
    current_phase: dict[str, object] = {"name": "preflight"}
    total_phases = len(longlive_installer.INSTALL_PHASES)

    def push_attempt(phase: str, ok: bool) -> None:
        """Emit one row for ``InstallLogPanel`` and reset the buffer."""
        job.attempts.append(
            {
                "phase": phase,
                "package": _LONGLIVE_PHASE_LABELS.get(phase, phase),
                "ok": ok,
                "output": "\n".join(phase_buffer)[-8000:],
            }
        )
        phase_buffer.clear()

    def stream_log(line: str) -> None:
        # Each line of installer output appends to the active phase's
        # buffer. ``InstallLogPanel`` tail-limits to 80 lines per
        # attempt anyway, but we cap the raw buffer at 8000 chars so
        # a misbehaving subprocess can't blow the response payload.
        phase_buffer.append(line)
        if len(phase_buffer) > 400:  # ~80 displayed + headroom for filter
            del phase_buffer[: len(phase_buffer) - 400]

    def report_progress(event: dict[str, object]) -> None:
        # Phase transition: flush the current phase's buffer as one
        # attempt row, advance the counter, set the new label.
        phase_name = str(event.get("phase") or "")
        ok = bool(event.get("ok"))
        push_attempt(phase_name, ok)
        if not ok:
            job.phase = "error"
            return
        try:
            idx = longlive_installer.INSTALL_PHASES.index(phase_name)
        except ValueError:
            return
        next_idx = idx + 1
        job.package_index = next_idx
        job.percent = (next_idx / total_phases) * 100.0
        if next_idx < total_phases:
            next_phase = longlive_installer.INSTALL_PHASES[next_idx]
            current_phase["name"] = next_phase
            job.package_current = _LONGLIVE_PHASE_LABELS.get(next_phase, next_phase)
            job.message = f"Running: {job.package_current}"

    job.phase = "downloading"  # mirror the GPU bundle phase name so the UI label matches
    job.message = "Starting LongLive install"
    job.package_current = _LONGLIVE_PHASE_LABELS.get("clone", "clone")
    job.package_total = total_phases

    try:
        longlive_installer.install(
            logger=stream_log,
            progress=report_progress,
        )
    except longlive_installer.LongLiveInstallError as exc:
        # Flush whatever was buffered during the failing phase, then
        # mark the job errored. ``current_phase['name']`` is the phase
        # that was running when the exception fired.
        if phase_buffer:
            push_attempt(str(current_phase["name"]), ok=False)
        job.phase = "error"
        job.error = str(exc)
        job.message = f"LongLive install failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface unexpected failures
        if phase_buffer:
            push_attempt(str(current_phase["name"]), ok=False)
        job.phase = "error"
        job.error = f"Unexpected error: {exc}"
        job.message = job.error
    else:
        job.phase = "done"
        job.percent = 100.0
        job.package_index = total_phases
        job.package_current = None
        job.message = "LongLive install complete."
    finally:
        job.finished_at = time.time()
        job.done = True


@router.post("/api/setup/install-longlive")
def start_install_longlive(request: Request) -> dict[str, Any]:
    """Kick off a background LongLive install.

    Returns the current job state immediately. Poll
    ``/api/setup/install-longlive/status`` for progress. Calling this
    endpoint again while a job is running returns the running job's
    state rather than starting a new one.
    """
    state_chaosengine = request.app.state.chaosengine

    # Resolve the install root the same way the installer does so the
    # UI can show it in the ``InstallLogPanel`` meta line. We import
    # locally for the same reason as the worker above.
    from backend_service.longlive_engine import resolve_install  # noqa: PLC0415

    info = resolve_install()

    with _LONGLIVE_LOCK:
        if _LONGLIVE_JOB.phase in {"preflight", "downloading", "verifying"}:
            return _LONGLIVE_JOB.to_dict()

        # Reset state for a fresh run.
        from backend_service import longlive_installer  # noqa: PLC0415

        _LONGLIVE_JOB.id = f"longlive-{int(time.time() * 1000)}"
        _LONGLIVE_JOB.phase = "preflight"
        _LONGLIVE_JOB.message = "Starting install"
        _LONGLIVE_JOB.package_current = _LONGLIVE_PHASE_LABELS["clone"]
        _LONGLIVE_JOB.package_index = 0
        _LONGLIVE_JOB.package_total = len(longlive_installer.INSTALL_PHASES)
        _LONGLIVE_JOB.percent = 0.0
        _LONGLIVE_JOB.target_dir = str(info.root)
        _LONGLIVE_JOB.error = None
        _LONGLIVE_JOB.started_at = time.time()
        _LONGLIVE_JOB.finished_at = 0.0
        _LONGLIVE_JOB.attempts = []
        _LONGLIVE_JOB.done = False

        thread = threading.Thread(
            target=_longlive_job_worker,
            name="chaosengine-longlive-install",
            daemon=True,
        )
        thread.start()

    state_chaosengine.add_log(
        "server", "info",
        f"LongLive install started (job={_LONGLIVE_JOB.id}, target={info.root})",
    )
    return _LONGLIVE_JOB.to_dict()


@router.get("/api/setup/install-longlive/status")
def install_longlive_status() -> dict[str, Any]:
    """Snapshot of the current LongLive install job. Safe to poll at 1-2 Hz."""
    return _LONGLIVE_JOB.to_dict()
