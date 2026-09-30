"""TensorFold install and status endpoints.

Background-job pattern (same as ``mtplx.py``): a single in-memory
``_TensorfoldJobState`` tracks the running install. POST starts a daemon
thread; GET polls. A second POST while the job is running returns the
running state instead of starting another.

Phases driven by ``scripts/install-tensorfold.sh`` ``PHASE:`` markers:
  idle → preflight → creating-venv → installing → verifying → done | error

``/api/setup/tensorfold-status`` is a lightweight probe (version file + venv
files, no subprocess). The launch settings use it to decide between the
TensorFold toggle and its install button.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_CHAOSENGINE_BIN_DIR = Path.home() / ".chaosengine" / "bin"
_TENSORFOLD_VENV_DIR = Path.home() / ".chaosengine" / "tensorfold-venv"
_TENSORFOLD_VERSION_FILE = _CHAOSENGINE_BIN_DIR / "tensorfold.version"
_INSTALL_SCRIPT = Path(__file__).parents[3] / "scripts" / "install-tensorfold.sh"

_PHASE_LABELS: dict[str, str] = {
    "preflight": "Checking Python environment",
    "creating-venv": "Creating isolated venv",
    "installing": "Installing TensorFold",
    "verifying": "Verifying install",
}

_TOTAL_PHASES = len(_PHASE_LABELS)
_ACTIVE_PHASES = ("idle", *_PHASE_LABELS)


@dataclass
class _TensorfoldJobState:
    phase: str = "idle"
    message: str = ""
    package_current: str | None = None
    package_index: int = 0
    package_total: int = _TOTAL_PHASES
    percent: float = 0.0
    target_dir: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": "tensorfold-install",
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
            "attempts": list(self.attempts),
            "done": self.done,
        }


_JOB = _TensorfoldJobState()
_JOB_LOCK = threading.Lock()


def _read_version_file() -> tuple[str | None, str | None, str | None]:
    """(version, installed_at, pinned ref) from the version file, or Nones."""
    if not _TENSORFOLD_VERSION_FILE.exists():
        return None, None, None
    try:
        lines = [line.strip() for line in _TENSORFOLD_VERSION_FILE.read_text(encoding="utf-8").splitlines()]
    except OSError:
        return None, None, None
    version = lines[0] if lines and lines[0] else None
    installed_at = lines[1] if len(lines) > 1 and lines[1] else None
    ref = lines[2] if len(lines) > 2 and lines[2] else None
    return version, installed_at, ref


def _is_installed() -> bool:
    python = _TENSORFOLD_VENV_DIR / "bin" / "python"
    script = _TENSORFOLD_VENV_DIR / "bin" / "tensorfold"
    return _TENSORFOLD_VERSION_FILE.exists() and python.exists() and script.exists()


def _apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _job_worker() -> None:
    """Run install-tensorfold.sh and stream its output into the job state."""
    from backend_service.inference._utils import _isolated_child_env

    job = _JOB
    phase_buffer: list[str] = []
    phase_index = 0

    def push_attempt(phase: str, ok: bool) -> None:
        job.attempts.append(
            {
                "phase": phase,
                "package": _PHASE_LABELS.get(phase, phase),
                "ok": ok,
                "output": "\n".join(phase_buffer)[-8000:],
            }
        )
        phase_buffer.clear()

    def advance_phase(name: str) -> None:
        nonlocal phase_index
        if job.phase not in _ACTIVE_PHASES:
            return
        if phase_index > 0:
            push_attempt(job.phase, ok=True)
        phase_index += 1
        job.phase = name
        job.package_current = _PHASE_LABELS.get(name, name)
        job.package_index = phase_index
        job.percent = round((phase_index - 1) / _TOTAL_PHASES * 100, 1)

    try:
        # The packaged app exports PYTHONHOME / PYTHONPATH / DYLD_* for its
        # embedded runtime; the installer builds a venv and must not see them.
        proc = subprocess.Popen(
            ["bash", str(_INSTALL_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_isolated_child_env(),
        )

        for raw_line in proc.stdout:  # type: ignore[union-attr]
            line = raw_line.rstrip("\n")
            if line.startswith("PHASE:"):
                advance_phase(line[len("PHASE:"):].strip())
            elif line.startswith("FAIL:"):
                job.error = line[len("FAIL:"):].strip() or "Install failed"
                phase_buffer.append(line)
            else:
                phase_buffer.append(line)
                if len(phase_buffer) > 400:
                    del phase_buffer[: len(phase_buffer) - 400]

        proc.wait()

        if proc.returncode == 0 and not job.error:
            push_attempt(job.phase, ok=True)
            job.phase = "done"
            job.percent = 100.0
            version, _installed_at, _ref = _read_version_file()
            job.message = f"TensorFold {version or 'installed'} ready in {_TENSORFOLD_VENV_DIR}"
            job.done = True
            _refresh_capabilities()
        else:
            push_attempt(job.phase, ok=False)
            job.phase = "error"
            job.error = job.error or f"install-tensorfold.sh exited with code {proc.returncode}"
            job.done = True

    except Exception as exc:  # noqa: BLE001
        push_attempt(job.phase, ok=False)
        job.phase = "error"
        job.error = str(exc)
        job.done = True
    finally:
        job.finished_at = time.time()


def _refresh_capabilities() -> None:
    """Make the running backend see the fresh install without a full re-probe."""
    try:
        from backend_service.inference.capabilities import refresh_install_detection

        refresh_install_detection()
    except Exception:  # noqa: BLE001 - best effort; the next probe picks it up
        pass


@router.get("/api/setup/tensorfold-status")
def tensorfold_status() -> dict[str, Any]:
    """Lightweight probe: is TensorFold installed, and can this machine run it?"""
    installed = _is_installed()
    version, installed_at, ref = _read_version_file()
    return {
        "installed": installed,
        "version": version if installed else None,
        "installedAt": installed_at if installed else None,
        "ref": ref if installed else None,
        "venvPath": str(_TENSORFOLD_VENV_DIR) if installed else None,
        # The Mac engine needs Apple Silicon; the UI hides the whole
        # affordance elsewhere, this lets a caller explain why.
        "supported": _apple_silicon(),
    }


@router.post("/api/setup/install-tensorfold")
def start_tensorfold_install() -> dict[str, Any]:
    """Start the TensorFold install job. Returns immediately; poll the status endpoint."""
    with _JOB_LOCK:
        if _JOB.phase not in ("idle", "done", "error"):
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        _JOB.phase = "preflight"
        _JOB.started_at = time.time()
        _JOB.target_dir = str(_TENSORFOLD_VENV_DIR)
        _JOB.package_current = _PHASE_LABELS["preflight"]

    thread = threading.Thread(target=_job_worker, daemon=True)
    thread.start()
    return _JOB.to_dict()


@router.get("/api/setup/install-tensorfold/status")
def tensorfold_install_status() -> dict[str, Any]:
    """Poll the running install job."""
    return _JOB.to_dict()
