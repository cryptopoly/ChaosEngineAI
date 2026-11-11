"""Install and status endpoints for the NVIDIA CUDA llama-server on Linux and Windows.

Background-job pattern (same as ``ltx2.py``): POST starts a daemon thread that
downloads ggml-org's CUDA build of the pinned llama.cpp tag (see
``helpers/llama_cuda.py``); GET polls it. ``/api/setup/llama-cuda-status``
reports whether this host can use it and whether GGUF models run on it now.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter

from backend_service.helpers import llama_cuda

router = APIRouter()


@dataclass
class _JobState:
    phase: str = "idle"
    message: str = ""
    current: str | None = None
    percent: float = 0.0
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    log: list[str] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": "llama-cuda-install",
            "phase": self.phase,
            "message": self.message,
            "packageCurrent": self.current,
            "percent": round(self.percent, 1),
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "log": list(self.log[-200:]),
            "done": self.done,
        }


_JOB = _JobState()
_JOB_LOCK = threading.Lock()


def _worker() -> None:
    job = _JOB
    assets_seen: list[str] = []

    def log(message: str) -> None:
        job.log.append(message)
        job.message = message

    def progress(name: str, done: int, total: int | None) -> None:
        if name not in assets_seen:
            assets_seen.append(name)
        job.current = name
        if total:
            # Two archives: the build first, then the CUDA runtime.
            share = (len(assets_seen) - 1 + done / total) / 2
            job.percent = min(99.0, share * 100)

    try:
        job.phase = "downloading"
        llama_cuda.install(log=log, progress=progress)
        job.phase = "verifying"
        usable = llama_cuda.usable_binary()
        try:
            # Point the running backend at the new server without a restart.
            from backend_service.inference.capabilities import refresh_install_detection

            refresh_install_detection()
        except Exception:  # noqa: BLE001 - the install itself succeeded
            pass
        job.phase = "done"
        job.percent = 100.0
        job.message = (
            "CUDA llama-server installed. GGUF models now run on the NVIDIA GPU."
            if usable
            else "CUDA llama-server installed, but it found no CUDA device. "
            "Check that the NVIDIA driver is installed (nvidia-smi); the bundled build stays in use."
        )
    except Exception as exc:  # noqa: BLE001 - reported to the UI
        job.phase = "error"
        job.error = str(exc)
    finally:
        job.done = True
        job.finished_at = time.time()


@router.get("/api/setup/llama-cuda-status")
def llama_cuda_status() -> dict[str, Any]:
    chosen = llama_cuda.plan()
    installed = llama_cuda.installed_binary()
    info = llama_cuda.installed_info() if installed else None
    in_use = bool(installed and llama_cuda.usable_binary())
    state = chosen["state"]
    message = chosen["message"]
    if installed:
        state = "in-use" if in_use else "installed-no-device"
        message = (
            "GGUF models run on the NVIDIA GPU."
            if in_use
            else "Installed, but it found no CUDA device. Check the NVIDIA driver (nvidia-smi); the bundled build stays in use."
        )
    elif chosen["ok"]:
        state, message = "not-installed", "Not installed. GGUF models use the bundled build."
    return {
        # Shown on Linux and Windows hosts that have an NVIDIA driver (even one that is too old,
        # so the panel can say why it cannot install).
        "supported": chosen["state"] in ("ready", "driver-too-old"),
        "canInstall": bool(chosen["ok"]),
        "state": state,
        "message": message,
        "os": chosen["os"],
        "arch": chosen["arch"],
        "driverMajor": chosen["driverMajor"],
        "driverCuda": chosen["driverCuda"],
        "driverTooOld": chosen["state"] == "driver-too-old",
        "cudaVersion": chosen["cuda"],
        "tag": llama_cuda.LLAMA_CPP_TAG,
        "installed": installed is not None,
        "installedCuda": (info or {}).get("cuda"),
        "verified": bool(info and info.get("verified") and all(info["verified"].values())),
        "path": str(installed) if installed else None,
        "inUse": in_use,
        "disabled": os.environ.get("CHAOSENGINE_LLAMA_CUDA") == "0",
    }


@router.post("/api/setup/install-llama-cuda")
def start_llama_cuda_install() -> dict[str, Any]:
    with _JOB_LOCK:
        if _JOB.phase not in ("idle", "done", "error"):
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        _JOB.phase = "preflight"
        _JOB.started_at = time.time()
    threading.Thread(target=_worker, daemon=True).start()
    return _JOB.to_dict()


@router.get("/api/setup/install-llama-cuda/status")
def llama_cuda_install_status() -> dict[str, Any]:
    return _JOB.to_dict()
