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
from threading import Event, RLock
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
        # Cooperative cancel signal — the UI's Cancel button sets this via
        # /api/{images,video}/cancel; the pipeline's step-end callback reads
        # it and raises to abort the run. ``Event`` (not a plain bool)
        # because the pipeline callback runs on a non-main thread and we
        # want the set-from-request-thread → read-from-pipeline-thread
        # synchronisation the stdlib primitive gives us for free.
        self._cancel_event = Event()

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
            # Clear any cancel flag from a previous run — otherwise a user
            # who cancelled yesterday's gen would have today's first click
            # abort before it started.
            self._cancel_event.clear()

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
            # Leave ``_cancel_event`` alone — the route handler needs to be
            # able to check whether the just-finished run was cancelled so
            # it can return the right status. ``begin()`` clears it for the
            # next run.

    def request_cancel(self) -> bool:
        """Signal the running pipeline to abort at the next step boundary.

        Returns True when the signal was accepted (a run was in flight),
        False when there was nothing to cancel. Idempotent — calling twice
        is fine. The actual abort happens cooperatively: the pipeline's
        step-end callback reads ``is_cancelled()`` and raises.
        """
        with self._lock:
            if not self._active:
                return False
            self._cancel_event.set()
            self._message = "Cancelling..."
            self._updated_at = time.time()
            return True

    def is_cancelled(self) -> bool:
        """Fast poll used from the pipeline's step-end callback. No lock —
        ``Event.is_set()`` is already thread-safe and we call it every step."""
        return self._cancel_event.is_set()

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
                "cancelRequested": self._cancel_event.is_set(),
            }


class GenerationCancelled(RuntimeError):
    """Raised by the pipeline callback when the user clicks Cancel.

    Distinct from generic ``RuntimeError`` so route handlers can catch it
    and return a 499-style "cancelled" response rather than a 500 error,
    and so logs can tell legitimate cancels apart from pipeline crashes.
    """


# Module-level singletons. The runtime managers and the route handlers both
# import these directly so we don't have to thread the tracker through
# ``ChaosEngineState`` constructor signatures.
IMAGE_PROGRESS = ProgressTracker(kind="image")
VIDEO_PROGRESS = ProgressTracker(kind="video")
