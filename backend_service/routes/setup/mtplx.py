"""MTPLX install and status endpoints.

Background-job pattern: a single in-memory ``_MtplxJobState`` tracks the
running install. POST starts a daemon thread; GET polls. A second POST while
the job is running returns the running state rather than starting a new job.

Phases driven by ``scripts/install-mtplx.sh`` PHASE: markers:
  idle → preflight → creating-venv → installing → verifying → done | error

The ``/api/setup/mtplx-status`` endpoint is a lightweight probe that checks
whether MTPLX is installed (version file + import smoke-test) without
triggering a full install. Used by RuntimeControls to decide whether to show
the MTPLX toggle or the install chip.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()

_CHAOSENGINE_BIN_DIR = Path.home() / ".chaosengine" / "bin"
_MTPLX_VENV_DIR = Path.home() / ".chaosengine" / "mtplx-venv"
_MTPLX_VERSION_FILE = _CHAOSENGINE_BIN_DIR / "mtplx.version"
_INSTALL_SCRIPT = Path(__file__).parents[3] / "scripts" / "install-mtplx.sh"

_PHASE_LABELS: dict[str, str] = {
    "preflight": "Checking Python environment",
    "creating-venv": "Creating isolated venv",
    "installing": "Installing MTPLX",
    "verifying": "Verifying install",
}

_TOTAL_PHASES = len(_PHASE_LABELS)


@dataclass
class _MtplxJobState:
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
            "id": "mtplx-install",
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


_JOB = _MtplxJobState()
_JOB_LOCK = threading.Lock()


def _read_version() -> tuple[str | None, str | None]:
    """Return (version, installed_at) from the version file, or (None, None)."""
    if not _MTPLX_VERSION_FILE.exists():
        return None, None
    try:
        lines = _MTPLX_VERSION_FILE.read_text().strip().splitlines()
        version = lines[0].strip() if lines else None
        installed_at = lines[1].strip() if len(lines) > 1 else None
        return version, installed_at
    except OSError:
        return None, None


def _is_installed() -> bool:
    python = _MTPLX_VENV_DIR / "bin" / "python"
    return _MTPLX_VERSION_FILE.exists() and python.exists()


def _job_worker() -> None:
    """Run install-mtplx.sh and stream output into job state."""
    job = _JOB
    phase_buffer: list[str] = []
    phase_index = 0

    def push_attempt(phase: str, ok: bool) -> None:
        job.attempts.append({
            "phase": phase,
            "package": _PHASE_LABELS.get(phase, phase),
            "ok": ok,
            "output": "\n".join(phase_buffer)[-8000:],
        })
        phase_buffer.clear()

    def advance_phase(name: str) -> None:
        nonlocal phase_index
        if job.phase not in ("idle", "preflight", "creating-venv", "installing", "verifying"):
            return
        if phase_index > 0:
            push_attempt(job.phase, ok=True)
        phase_index += 1
        job.phase = name
        job.package_current = _PHASE_LABELS.get(name, name)
        job.package_index = phase_index
        job.percent = round((phase_index - 1) / _TOTAL_PHASES * 100, 1)

    try:
        proc = subprocess.Popen(
            ["bash", str(_INSTALL_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
            version, _ = _read_version()
            job.message = f"MTPLX {version or 'installed'} ready in {_MTPLX_VENV_DIR}"
            job.done = True
        else:
            push_attempt(job.phase, ok=False)
            job.phase = "error"
            job.error = job.error or f"install-mtplx.sh exited with code {proc.returncode}"
            job.done = True

    except Exception as exc:  # noqa: BLE001
        push_attempt(job.phase, ok=False)
        job.phase = "error"
        job.error = str(exc)
        job.done = True
    finally:
        job.finished_at = time.time()


def _detect_fan_control() -> dict[str, Any]:
    """Look for fan-control daemons MTPLX's ``--max`` burst can drive.

    MTPLX warns ``FAN CONTROL DID NOT TAKE EFFECT`` when ``--max`` runs
    without ThermalForge or TG Pro present, falling back to silent
    operation but losing the fan-boost that lets the GPU hold full
    clocks for longer. Surface this so the Setup tab can suggest the
    install rather than silently capping speedup.
    """
    import os as _os
    import shutil as _shutil

    thermalforge = _shutil.which("thermalforge") is not None or Path(
        "/Applications/ThermalForge.app"
    ).exists()
    tg_pro = Path("/Applications/TG Pro.app").exists() or Path(
        _os.path.expanduser("~/Applications/TG Pro.app")
    ).exists()
    return {
        "thermalforge": thermalforge,
        "tgPro": tg_pro,
        "anyAvailable": thermalforge or tg_pro,
        "recommendedAction": (
            None
            if thermalforge or tg_pro
            else "Install ThermalForge (free, MTPLX-recommended) to enable "
                 "fan boost during burst-mode generation: run "
                 "`~/.chaosengine/mtplx-venv/bin/mtplx max --install`. "
                 "Without it, --max requests succeed but the GPU thermally "
                 "throttles after a few seconds of sustained load."
        ),
    }


@router.get("/api/setup/mtplx-status")
def mtplx_status() -> dict[str, Any]:
    """Lightweight probe: is MTPLX installed and what version?"""
    installed = _is_installed()
    version, installed_at = _read_version()
    return {
        "installed": installed,
        "version": version,
        "installedAt": installed_at,
        "venvPath": str(_MTPLX_VENV_DIR) if installed else None,
        # FU-MTPLX-thermal: surface fan-control availability so the UI
        # can prompt for ThermalForge install before the user hits the
        # silent-throttle ceiling on burst-mode runs.
        "fanControl": _detect_fan_control(),
    }


@router.post("/api/setup/install-mtplx")
def start_mtplx_install() -> dict[str, Any]:
    """Start the MTPLX install job. Returns immediately; poll status endpoint."""
    with _JOB_LOCK:
        if _JOB.phase not in ("idle", "done", "error"):
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        _JOB.phase = "preflight"
        _JOB.started_at = time.time()
        _JOB.target_dir = str(_MTPLX_VENV_DIR)
        _JOB.package_current = _PHASE_LABELS["preflight"]

    thread = threading.Thread(target=_job_worker, daemon=True)
    thread.start()
    return _JOB.to_dict()


@router.get("/api/setup/install-mtplx/status")
def mtplx_install_status() -> dict[str, Any]:
    """Poll the running install job."""
    return _JOB.to_dict()
