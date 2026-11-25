"""Quality upscale add-on (FlashVSR+ Turbo, D0145): status, install and cancel. Nothing is installed at startup.

* ``GET  /api/setup/video-upscale-quality/status``: whether it can run here (NVIDIA on Windows or Linux), whether it
  is installed, the download size, and the install job (phase, bytes of the weights, the last log lines).
* ``POST /api/setup/install-video-upscale-quality``: starts the install in a background thread
  (``video_upscale.quality_addon``); a second call while it runs returns the running job. On a computer where it
  cannot run the job ends at once with the reason, and nothing is created or downloaded.
* ``POST /api/setup/video-upscale-quality/cancel``: stops the install; a weight file being fetched resumes from where
  it stopped on the next install.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from backend_service.video_upscale import quality_addon

router = APIRouter()


@router.get("/api/setup/video-upscale-quality/status")
def video_upscale_quality_status() -> dict[str, Any]:
    return quality_addon.status()


@router.post("/api/setup/install-video-upscale-quality")
def install_video_upscale_quality() -> dict[str, Any]:
    quality_addon.start_install()
    return quality_addon.status()


@router.post("/api/setup/video-upscale-quality/cancel")
def cancel_video_upscale_quality() -> dict[str, Any]:
    quality_addon.cancel_install()
    return quality_addon.status()
