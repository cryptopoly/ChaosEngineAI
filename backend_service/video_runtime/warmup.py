"""Torch + dep prewarm for the video runtime.

Importing torch for the first time is expensive (30-60s on a cold Windows
SSD). Because probe() is a sync FastAPI route that calls ``import torch``,
the first probe blew past the frontend's 30s fetch timeout and surfaced as
"Video runtime did not respond" with every downstream endpoint cascading to
"Failed to fetch". We warm torch on a background thread at sidecar startup
so probe() can return a fast "initializing" status while the import is in
flight, and an accurate status the moment it completes. The import lock
means any in-flight probe still ends up serialized behind the warmup
anyway — the fast-path here is purely to keep the probe route itself from
blocking so the rest of the video API stays responsive.

Extracted from ``video_runtime/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from typing import Any

from backend_service.video_runtime.defaults import (
    _CORE_DEPS,
    _VIDEO_MODEL_DEPS,
    _VIDEO_OUTPUT_DEPS,
)


_torch_warmup_lock = threading.Lock()
_torch_warmup_state: dict[str, Any] = {
    "status": "not_started",  # "not_started" | "in_progress" | "ready" | "failed"
    "error": None,  # exception message when status == "failed"
    "started_at": None,
}


def _torch_warmup_worker() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        with _torch_warmup_lock:
            _torch_warmup_state["status"] = "failed"
            _torch_warmup_state["error"] = f"{type(exc).__name__}: {exc}"
        return
    # Pre-warm anything else the first probe() call would otherwise pay for
    # inline. On Windows the nvidia-smi shell-out adds 1-2s per probe when
    # uncached, and importlib.util.find_spec on a cold NTFS volume with
    # antivirus scanning can be slow enough to push a probe past the
    # frontend's fetch timeout. Doing both here keeps probe() a hashmap
    # lookup in the common case.
    try:
        from backend_service.helpers.gpu import get_device_vram_total_gb
        get_device_vram_total_gb()
    except Exception:
        pass
    try:
        for _pkg, module_name in _CORE_DEPS + _VIDEO_OUTPUT_DEPS + _VIDEO_MODEL_DEPS:
            try:
                importlib.util.find_spec(module_name)
            except Exception:
                pass
    except Exception:
        pass
    with _torch_warmup_lock:
        _torch_warmup_state["status"] = "ready"
        _torch_warmup_state["error"] = None


def start_torch_warmup() -> None:
    """Kick off a one-shot background import of torch.

    Called from ``create_app()`` at sidecar startup. Safe to call repeatedly —
    only the first call spawns a thread. If torch is already importable
    cheaply (e.g. the interpreter has seen it before in this process), the
    worker finishes almost immediately.
    """
    with _torch_warmup_lock:
        if _torch_warmup_state["status"] != "not_started":
            return
        _torch_warmup_state["status"] = "in_progress"
        _torch_warmup_state["started_at"] = time.monotonic()
    thread = threading.Thread(
        target=_torch_warmup_worker,
        name="chaosengine-torch-warmup",
        daemon=True,
    )
    thread.start()


def torch_warmup_status() -> dict[str, Any]:
    """Snapshot of the warmup state. Used by ``probe()`` to avoid blocking."""
    with _torch_warmup_lock:
        return dict(_torch_warmup_state)
