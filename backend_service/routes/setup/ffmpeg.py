"""Video tools (ffmpeg) status and install endpoints for Setup and the first-use prompt (D-49).

macOS, and Windows and Linux by default, install ``imageio-ffmpeg`` through the existing
``/api/setup/install-package`` route. On Windows and Linux a pinned LGPL download is also offered when its
pins are verified (``ffmpeg_fetch.pins``); it runs as a background job polled by ``/install/status``.
"""

from __future__ import annotations

import importlib
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend_service.ffmpeg_fetch import install as fetch
from backend_service.ffmpeg_fetch.pins import Pin, pin_for_host
from backend_service.storyboard import ffmpeg_bin

router = APIRouter()

_PHASE_SHARE = {"download": (0.0, 0.85), "verify": (0.85, 0.9), "extract": (0.9, 0.95), "install": (0.95, 1.0)}


class _Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.phase = "idle"
        self.percent = 0.0
        self.error: str | None = None
        self.done = False
        self.started_at = 0.0

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {"phase": self.phase, "percent": round(self.percent, 1), "error": self.error,
                    "done": self.done, "running": self.running(), "startedAt": self.started_at}


_JOB = _Job()


class InstallBody(BaseModel):
    source: str = "download"


class CustomBody(BaseModel):
    path: str


def _pin_dict(pin: Pin | None) -> dict[str, Any] | None:
    if pin is None:
        return None
    return {"url": pin.url, "sha256": pin.sha256, "size": pin.size, "version": pin.version,
            "licenceUrl": pin.licence_url, "sourceUrl": pin.source_url}


def _progress(phase: str, done: int, total: int) -> None:
    lo, hi = _PHASE_SHARE.get(phase, (0.0, 1.0))
    with _JOB.lock:
        _JOB.phase = phase
        _JOB.percent = 100.0 * (lo + (hi - lo) * (done / total if total else 0.0))


def _start(work: Any) -> None:
    with _JOB.lock:
        if _JOB.running():
            return
        _JOB.cancel = threading.Event()
        _JOB.phase, _JOB.percent, _JOB.error, _JOB.done = "download", 0.0, None, False
        _JOB.started_at = time.time()

    def run() -> None:
        try:
            work()
            with _JOB.lock:
                _JOB.phase, _JOB.percent, _JOB.done = "done", 100.0, True
        except fetch.FetchError as exc:
            with _JOB.lock:
                _JOB.error, _JOB.phase = str(exc), "error"
        except Exception as exc:  # noqa: BLE001 - shown to the user, not raised into a daemon thread
            with _JOB.lock:
                _JOB.error, _JOB.phase = f"Install failed: {exc}", "error"

    _JOB.thread = threading.Thread(target=run, name="ffmpeg-install", daemon=True)
    _JOB.thread.start()


@router.get("/api/setup/ffmpeg")
def ffmpeg_status() -> dict[str, Any]:
    """Where ffmpeg comes from, its version and encoder, the pinned download for this host, and the job."""
    importlib.invalidate_caches()
    try:
        from backend_service.runtime_paths import ensure_extras_on_sys_path

        ensure_extras_on_sys_path()  # a package installed since start-up
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        info = ffmpeg_bin.ffmpeg_info()
    except Exception as exc:  # noqa: BLE001 - a broken binary is a status, not a 500
        info = {"installed": False, "path": None, "source": None, "version": None, "licence": None,
                "encoder": None, "error": str(exc)}
    pin = pin_for_host()
    return {**info, "pin": _pin_dict(pin), "canDownload": pin is not None, "job": _JOB.to_dict()}


@router.post("/api/setup/ffmpeg/install")
def ffmpeg_install(body: InstallBody) -> dict[str, Any]:
    """Start the pinned download (Windows and Linux, only when the pins are verified)."""
    if body.source != "download":
        raise HTTPException(status_code=400, detail="Use /api/setup/install-package for imageio-ffmpeg.")
    pin = pin_for_host()
    if pin is None:
        raise HTTPException(status_code=400, detail="There is no verified download for this system. Install the video tools package instead.")
    _start(lambda: fetch.install_pin(pin, ffmpeg_bin.managed_dir(), _progress, _JOB.cancel))
    return {"job": _JOB.to_dict()}


@router.get("/api/setup/ffmpeg/install/status")
def ffmpeg_install_status() -> dict[str, Any]:
    return {"job": _JOB.to_dict()}


@router.post("/api/setup/ffmpeg/install/cancel")
def ffmpeg_install_cancel() -> dict[str, Any]:
    _JOB.cancel.set()
    return {"job": _JOB.to_dict()}


@router.post("/api/setup/ffmpeg/custom")
def ffmpeg_custom(body: CustomBody) -> dict[str, Any]:
    """Use a file the user chose: an ffmpeg program, or (offline) the pinned archive they fetched elsewhere."""
    path = Path(body.path).expanduser()
    pin = pin_for_host()
    if pin is not None and path.name.lower().endswith((".zip", ".tar.xz")):
        _start(lambda: fetch.install_from_file(pin, path, ffmpeg_bin.managed_dir(), _progress))
        return {"job": _JOB.to_dict(), "version": None}
    try:
        version = fetch.validate_custom(path.resolve())
    except fetch.FetchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    root = ffmpeg_bin.managed_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / "custom").write_text(str(path.resolve()), encoding="utf-8")
    return {"job": _JOB.to_dict(), "version": version}


@router.delete("/api/setup/ffmpeg")
def ffmpeg_remove() -> dict[str, Any]:
    """Remove the app-managed download and any chosen file (the imageio-ffmpeg package stays)."""
    if _JOB.running():
        raise HTTPException(status_code=409, detail="An install is running.")
    root = ffmpeg_bin.managed_dir()
    if root.is_dir():
        shutil.rmtree(root, ignore_errors=True)
    return ffmpeg_status()
