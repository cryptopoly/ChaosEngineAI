"""vLLM-in-WSL installer endpoint (FU-056 Phase 8).

vLLM ships no native Windows wheels; the practical path on a Windows
+ CUDA box is to install vLLM inside a WSL2 Ubuntu distro and run it
there. This module provides the in-app installer + status poll so
users never have to drop to PowerShell to type
``wsl -- pip install vllm``.

The install runs three steps inside the user's default WSL distro:

  1. **venv** — ``python3 -m venv ~/.chaosengine/vllm-venv`` (idempotent;
     skips when already present). The venv is rooted in the WSL user's
     ``$HOME`` (ext4-backed) so CUDA torch wheels don't pay the
     ~10× IO penalty of being on ``/mnt/c/...``.
  2. **pip upgrade** — ``pip install --upgrade pip setuptools wheel``.
     Stops pip falling back to ancient resolver shapes on Ubuntu 22.04.
  3. **vllm** — ``pip install vllm``. Pulls torch CUDA + flash-attn +
     friends. ~2 GB download, ~5-15 min wall time on a warm box.
  4. **verify** — ``python -c "import vllm"`` confirms the install is
     functional (catches half-baked builds the way Phase 1's
     ``_safe_version`` does for the embedded runtime).

Same single-job semantics as the LongLive installer: a second POST
while running returns the running job state; completion state sticks
around for a late status poll. Mirrors that module's structure on
purpose so the frontend's ``InstallLogPanel`` can render WSL-vLLM
attempts using the same job shape.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.i18n import localized_detail

router = APIRouter()


_WSL_VLLM_VENV_PATH = "~/.chaosengine/vllm-venv"

# Order matches the user-visible progress: preflight is the first
# attempt row that surfaces "checking WSL", venv writes the dir,
# pip-upgrade refreshes packaging plumbing, pip-vllm is the long
# download, verify proves import works.
_INSTALL_PHASES: tuple[str, ...] = (
    "preflight",
    "venv",
    "pip-upgrade",
    "pip-vllm",
    "verify",
)

_PHASE_LABELS: dict[str, str] = {
    "preflight": "Check WSL + CUDA",
    "venv": "Create isolated venv",
    "pip-upgrade": "Upgrade pip / setuptools / wheel",
    "pip-vllm": "Install vllm (~2 GB)",
    "verify": "Verify import",
}

# Total wall-time budget per step. The pip-vllm step gets the lion's
# share — fresh CUDA torch wheel can be 20+ min on a slow link.
_STEP_TIMEOUTS_SEC: dict[str, float] = {
    "preflight": 10.0,
    "venv": 60.0,
    "pip-upgrade": 180.0,
    "pip-vllm": 1800.0,
    "verify": 30.0,
}


@dataclass
class _VllmWslJobState:
    id: str = ""
    phase: str = "idle"  # idle | preflight | installing | done | error
    message: str = ""
    package_current: str | None = None
    package_index: int = 0
    package_total: int = len(_INSTALL_PHASES)
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


_JOB = _VllmWslJobState()
_LOCK = threading.Lock()


def _run_wsl_step(
    bash_command: str,
    timeout_sec: float,
) -> tuple[int, str]:
    """Run ``wsl -- bash -c "<command>"`` and return ``(exit_code, output)``.

    Captures stdout + stderr into a single string truncated to ~8000
    characters — keeps the response payload bounded. ``wsl`` itself
    emits UTF-16 on some paths but ``bash -c`` output comes back as
    UTF-8, so we decode permissively to avoid a corrupt-locale crash.
    """
    if sys.platform != "win32":
        return 127, "WSL bridge install only runs on Windows hosts."
    try:
        result = subprocess.run(
            ["wsl", "--", "bash", "-c", bash_command],
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError:
        return 127, "wsl.exe not found on PATH."
    except subprocess.TimeoutExpired:
        return 124, f"Step timed out after {timeout_sec:.0f}s."
    output = (result.stdout + result.stderr).decode("utf-8", errors="ignore")
    return result.returncode, output[-8000:]


def _push_attempt(job: _VllmWslJobState, phase: str, ok: bool, output: str) -> None:
    job.attempts.append({
        "phase": phase,
        "package": _PHASE_LABELS.get(phase, phase),
        "ok": ok,
        "output": output,
    })


def _advance(job: _VllmWslJobState, next_phase_index: int) -> None:
    job.package_index = next_phase_index
    job.percent = (next_phase_index / job.package_total) * 100.0
    if next_phase_index < job.package_total:
        next_phase = _INSTALL_PHASES[next_phase_index]
        job.package_current = _PHASE_LABELS.get(next_phase, next_phase)
        job.message = f"Running: {job.package_current}"


def _job_worker() -> None:
    """Run the install steps sequentially, streaming each into ``job.attempts``.

    Any subprocess returning non-zero flips the job to ``error`` and
    stops the chain. Late status polls see the failing attempt's
    captured output so the UI can surface the pip error without a
    separate log fetch.
    """
    job = _JOB
    job.phase = "installing"
    job.package_current = _PHASE_LABELS["preflight"]
    job.target_dir = _WSL_VLLM_VENV_PATH

    # Step 1 — preflight. Confirm WSL responds + CUDA passthrough works
    # before paying for the venv + pip download. Fails fast if the user
    # tried to install on a box where ``nvidia-smi -L`` doesn't work
    # inside WSL (the NVIDIA WSL driver kicker hasn't been installed
    # on the Windows host).
    code, output = _run_wsl_step(
        "nvidia-smi -L",
        _STEP_TIMEOUTS_SEC["preflight"],
    )
    _push_attempt(job, "preflight", ok=(code == 0), output=output)
    if code != 0:
        job.phase = "error"
        job.error = (
            "CUDA isn't reachable inside WSL. Install the NVIDIA WSL "
            "driver on Windows first: https://docs.nvidia.com/cuda/wsl-user-guide/"
        )
        job.message = job.error
        job.finished_at = time.time()
        job.done = True
        return
    _advance(job, 1)

    # Step 2 — venv. ``python3 -m venv`` is idempotent: if the dir
    # already exists Python silently re-creates the pyvenv.cfg shim
    # without nuking site-packages. We still wrap in ``mkdir -p`` so
    # the parent ``~/.chaosengine`` exists on a clean WSL host.
    code, output = _run_wsl_step(
        (
            f"mkdir -p $HOME/.chaosengine && "
            f"python3 -m venv {_WSL_VLLM_VENV_PATH}"
        ),
        _STEP_TIMEOUTS_SEC["venv"],
    )
    _push_attempt(job, "venv", ok=(code == 0), output=output)
    if code != 0:
        job.phase = "error"
        job.error = "Failed to create the WSL venv. See output above."
        job.message = job.error
        job.finished_at = time.time()
        job.done = True
        return
    _advance(job, 2)

    # Step 3 — pip upgrade. Ubuntu 22.04 ships pip 22.x; the vllm
    # wheel resolution wants pip ≥ 23.0 to pick the right CUDA tag.
    code, output = _run_wsl_step(
        (
            f"{_WSL_VLLM_VENV_PATH}/bin/python -m pip install "
            "--upgrade pip setuptools wheel"
        ),
        _STEP_TIMEOUTS_SEC["pip-upgrade"],
    )
    _push_attempt(job, "pip-upgrade", ok=(code == 0), output=output)
    if code != 0:
        job.phase = "error"
        job.error = "Failed to upgrade pip in the WSL venv."
        job.message = job.error
        job.finished_at = time.time()
        job.done = True
        return
    _advance(job, 3)

    # Step 4 — the actual pip install. Long step (~2 GB download +
    # extraction). The InstallLogPanel will show pip's progress lines
    # in the attempt row as they accumulate.
    code, output = _run_wsl_step(
        f"{_WSL_VLLM_VENV_PATH}/bin/pip install vllm",
        _STEP_TIMEOUTS_SEC["pip-vllm"],
    )
    _push_attempt(job, "pip-vllm", ok=(code == 0), output=output)
    if code != 0:
        job.phase = "error"
        job.error = "pip install vllm failed. See output above."
        job.message = job.error
        job.finished_at = time.time()
        job.done = True
        return
    _advance(job, 4)

    # Step 5 — verify the install is functional. Catches the
    # half-baked-install failure mode we hit with torch on Windows
    # (DLLs present but Python source missing).
    code, output = _run_wsl_step(
        (
            f"{_WSL_VLLM_VENV_PATH}/bin/python -c "
            "'import vllm; print(vllm.__version__)'"
        ),
        _STEP_TIMEOUTS_SEC["verify"],
    )
    _push_attempt(job, "verify", ok=(code == 0), output=output)
    if code != 0:
        job.phase = "error"
        job.error = "vllm installed but ``import vllm`` failed inside the WSL venv."
        job.message = job.error
        job.finished_at = time.time()
        job.done = True
        return

    job.phase = "done"
    job.percent = 100.0
    job.message = f"vLLM ready in WSL ({output.strip() or 'version unknown'})."
    job.finished_at = time.time()
    job.done = True


@router.post("/api/setup/install-vllm-wsl")
def start_install_vllm_wsl(request: Request) -> dict[str, Any]:
    """Kick off the WSL vLLM install on a background thread.

    Idempotent: a second POST while a job is in flight returns the
    running state instead of double-booting the install. Same shape
    as ``install-gpu-bundle`` so the frontend pattern stays uniform.
    """
    if sys.platform != "win32":
        raise HTTPException(
            status_code=400,
            detail=localized_detail(
                request,
                "vLLM-in-WSL install only runs on Windows hosts.",
            ),
        )

    with _LOCK:
        if _JOB.phase in {"preflight", "installing"}:
            return _JOB.to_dict()

        _JOB.id = f"vllm-wsl-{int(time.time() * 1000)}"
        _JOB.phase = "preflight"
        _JOB.message = "Starting vLLM install in WSL..."
        _JOB.package_current = _PHASE_LABELS["preflight"]
        _JOB.package_index = 0
        _JOB.package_total = len(_INSTALL_PHASES)
        _JOB.percent = 0.0
        _JOB.target_dir = _WSL_VLLM_VENV_PATH
        _JOB.error = None
        _JOB.started_at = time.time()
        _JOB.finished_at = 0.0
        _JOB.attempts = []
        _JOB.done = False

        thread = threading.Thread(
            target=_job_worker,
            name="chaosengine-vllm-wsl-install",
            daemon=True,
        )
        thread.start()

    return _JOB.to_dict()


@router.get("/api/setup/install-vllm-wsl/status")
def vllm_wsl_status() -> dict[str, Any]:
    """Snapshot of the most-recent WSL vLLM install attempt.

    Safe to poll at 1-2 Hz. Returns ``phase="idle"`` before any
    install has been started in this backend session.
    """
    return _JOB.to_dict()
