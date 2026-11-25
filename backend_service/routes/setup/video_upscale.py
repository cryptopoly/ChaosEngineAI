"""Video upscale add-on (fast tier): status and the install Setup offers. Nothing is installed at startup.

* ``GET  /api/setup/video-upscale/status``: packages and each model's weights, installed or not, and the install job.
* ``POST /api/setup/install-video-upscale``: installs in a background thread (``video_upscale.addon.install``); the
  body may name ``models`` (catalog ids; the default row when absent). A second call while it runs returns the
  running job.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend_service.catalog.video_upscale_models import find_model
from backend_service.video_upscale import addon

router = APIRouter()


class InstallRequest(BaseModel):
    models: list[str] | None = Field(default=None, max_length=8)


class _Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.phase = "idle"  # idle | packages | weights | check | done | error
        self.error: str | None = None
        self.models: list[str] = []
        self.started_at = 0.0
        self.finished_at = 0.0

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {"phase": self.phase, "error": self.error, "running": self.running(), "models": list(self.models),
                    "done": self.phase in ("done", "error"), "startedAt": self.started_at,
                    "finishedAt": self.finished_at}


_JOB = _Job()


def _set_phase(phase: str) -> None:
    with _JOB.lock:
        _JOB.phase = phase


def _work(models: list[str]) -> None:
    try:
        addon.install(_set_phase, models)
    except Exception as exc:  # noqa: BLE001 - the thread has no caller; the reason goes to the status
        with _JOB.lock:
            _JOB.phase, _JOB.error = "error", str(exc)
    else:
        _set_phase("done")
    finally:
        with _JOB.lock:
            _JOB.finished_at = time.time()


@router.get("/api/setup/video-upscale/status")
def video_upscale_status() -> dict[str, Any]:
    return {**addon.status(), "job": _JOB.to_dict()}


@router.post("/api/setup/install-video-upscale")
def install_video_upscale(body: InstallRequest | None = None) -> dict[str, Any]:
    models = list((body.models if body else None) or [])
    unknown = [model for model in models if find_model(model) is None]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown video upscale model: {', '.join(unknown)}")
    with _JOB.lock:
        if not (_JOB.thread is not None and _JOB.thread.is_alive()):
            _JOB.phase, _JOB.error, _JOB.models = "packages", None, models
            _JOB.started_at, _JOB.finished_at = time.time(), 0.0
            _JOB.thread = threading.Thread(target=_work, args=(models or None,), name="chaosengine-video-upscale-install",
                                           daemon=True)
            _JOB.thread.start()
    return {**addon.status(), "job": _JOB.to_dict()}
