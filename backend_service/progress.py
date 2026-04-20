"""Real-time generation progress tracking for image + video runtimes.

The diffusers pipelines used by ``image_runtime`` and ``video_runtime`` each
take 30 seconds to several minutes to finish. The frontend used to render an
arbitrary "estimated seconds" bar that drifted out of sync with reality on
slower hardware. This module gives the runtimes a tiny thread-safe scratchpad
they can update as they progress, and the routes a way to report that state
back to the UI so the progress bar reflects what's actually happening.

A tracker exposes four operations:

* ``begin(...)`` — call when generation starts. Resets state and stamps the
  start time.
* ``set_phase(...)`` — call when the runtime moves into a new phase
  (``loading``, ``encoding``, ``diffusing``, ``decoding``, ``saving``). The
  string is opaque to the backend — the frontend maps it onto the same phase
  IDs the modal already understands.
* ``set_step(step, total)`` — call inside ``callback_on_step_end`` to publish
  per-step progress during diffusion.
* ``finish(...)`` — call after the pipeline returns (or raises). Marks the
  tracker idle so the next poll cycle stops showing stale values.

``snapshot()`` returns a JSON-serialisable dict the routes hand back to the
frontend. ``active=False`` means "no run in flight" — callers should fall
back to client-side estimates.

Two module-level singletons (`IMAGE_PROGRESS`, `VIDEO_PROGRESS`) are exposed
so the runtimes and routes share the same instance without the
``ChaosEngineState`` plumbing having to know about it.
"""

from __future__ import annotations

import time
from threading import RLock
from typing import Any


# Phase IDs the frontend expects. Keep these in sync with the modal's phase
# list — adding a new phase here without updating the modal will just show up
# as "unknown phase" in the UI but won't crash.
PHASE_IDLE = "idle"
PHASE_LOADING = "loading"
PHASE_ENCODING = "encoding"
PHASE_DIFFUSING = "diffusing"
PHASE_DECODING = "decoding"
PHASE_SAVING = "saving"


class ProgressTracker:
    """Thread-safe scratchpad for one in-flight generation at a time.

    The runtimes are already serialised through their own ``RLock``s — only
    one image (or one video) can render at a time per process — so we don't
    need to multiplex multiple runs. We just need the GET endpoint and the
    pipeline callback to read/write the same state without tearing.
    """

    def __init__(self, *, kind: str) -> None:
        self._lock = RLock()
        # ``kind`` is included in the snapshot so logs can tell image and
        # video apart at a glance.
        self._kind = kind
        self._active = False
        self._phase = PHASE_IDLE
        self._message = ""
        self._step = 0
        self._total_steps = 0
        self._started_at = 0.0
        self._updated_at = 0.0
        # Optional run-shape metadata so the UI can render labels like
        # "Diffusing 3 images" without a separate request.
        self._run_label: str | None = None

    def begin(
        self,
        *,
        run_label: str | None = None,
        total_steps: int = 0,
        phase: str = PHASE_LOADING,
        message: str = "",
    ) -> None:
        with self._lock:
            now = time.time()
            self._active = True
            self._phase = phase
            self._message = message
            self._step = 0
            self._total_steps = max(0, int(total_steps))
            self._started_at = now
            self._updated_at = now
            self._run_label = run_label

    def set_phase(self, phase: str, message: str = "") -> None:
        """Move into a new phase. Resets ``step`` so per-phase progress is
        measured from zero rather than carrying over the previous phase's
        counter."""
        with self._lock:
            if not self._active:
                # Setting a phase before ``begin()`` is meaningless — it would
                # leave ``started_at`` at 0 and the elapsed time would be
                # nonsense. Treat it as an implicit ``begin`` so callers don't
                # have to remember the order on simple paths.
                self._active = True
                self._started_at = time.time()
                self._step = 0
                self._total_steps = 0
                self._run_label = None
            self._phase = phase
            self._message = message
            self._step = 0
            self._updated_at = time.time()

    def set_step(self, step: int, total: int | None = None) -> None:
        with self._lock:
            if not self._active:
                return
            self._step = max(0, int(step))
            if total is not None:
                self._total_steps = max(0, int(total))
            self._updated_at = time.time()

    def finish(self, *, message: str = "") -> None:
        with self._lock:
            self._active = False
            self._phase = PHASE_IDLE
            self._message = message
            self._step = 0
            self._total_steps = 0
            self._updated_at = time.time()
            self._run_label = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            elapsed = max(0.0, now - self._started_at) if self._active else 0.0
            return {
                "kind": self._kind,
                "active": self._active,
                "phase": self._phase,
                "message": self._message,
                "step": self._step,
                "totalSteps": self._total_steps,
                "startedAt": self._started_at if self._active else 0.0,
                "updatedAt": self._updated_at,
                "elapsedSeconds": round(elapsed, 3),
                "runLabel": self._run_label,
            }


# Module-level singletons. The runtime managers and the route handlers both
# import these directly so we don't have to thread the tracker through
# ``ChaosEngineState`` constructor signatures.
IMAGE_PROGRESS = ProgressTracker(kind="image")
VIDEO_PROGRESS = ProgressTracker(kind="video")
