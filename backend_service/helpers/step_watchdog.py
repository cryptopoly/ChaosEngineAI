"""Per-step watchdog for image and video generation.

The cancel flag is only read between steps, so a step that never returns (a card full
past its memory spilling into system memory, a kernel that hangs) cannot be cancelled: one
run on an RTX 4090 sat on step 1 for over an hour. This watches the progress tracker from a
side thread:

1. A step that takes longer than ``limit`` seconds raises the cancel flag with a clear
   message. If the step does return, the pipeline's step-end callback raises and the run
   stops cleanly.
2. If the run is still on the same step ``grace`` seconds later, nothing in-process can end
   a call that is stuck inside a GPU kernel, so the hard stop runs: the message is written
   to the log and to ``last_stall.json``, and the backend process exits so the card is
   released (the desktop shell's "Restart Backend" brings it back).

The limit scales with the run: ``factor`` times the median of the steps seen so far, never
below ``floor`` seconds; before two steps have finished it is ``first_floor`` seconds, since
the first step carries warm-up (a compile, offload swaps). Nothing here reads the card.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service.progress import PHASE_DIFFUSING, ProgressTracker

_LOG = logging.getLogger(__name__)

DEFAULT_FACTOR = 8.0
DEFAULT_FLOOR_SECONDS = 120.0
DEFAULT_FIRST_FLOOR_SECONDS = 600.0
DEFAULT_GRACE_SECONDS = 120.0
_EXIT_CODE = 75


def stall_limit_seconds(
    step_seconds: list[float],
    *,
    factor: float = DEFAULT_FACTOR,
    floor: float = DEFAULT_FLOOR_SECONDS,
    first_floor: float = DEFAULT_FIRST_FLOOR_SECONDS,
) -> float:
    """How long one step may take before it counts as stuck, given the steps finished so far."""
    if len(step_seconds) < 2:
        return first_floor
    return max(floor, factor * statistics.median(step_seconds))


def stall_message(kind: str, step: int, total: int, waited: float, limit: float) -> str:
    return (
        f"The {kind} stopped: step {step + 1} of {total} ran for {waited:.0f} s without finishing "
        f"(limit {limit:.0f} s). This usually means the GPU ran out of memory and spilled into "
        "system memory; pick a smaller size or fewer frames and try again."
    )


def _write_stall_record(message: str) -> None:
    try:
        folder = Path(os.environ.get("CHAOSENGINE_DATA_DIR") or (Path.home() / ".chaosengine"))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "last_stall.json").write_text(json.dumps({"at": time.time(), "message": message}), encoding="utf-8")
    except OSError:
        pass


def exit_backend(message: str) -> None:
    """The hard stop: nothing in-process can end a hung GPU call, so release the card by exiting."""
    _LOG.critical("step watchdog hard stop: %s", message)
    _write_stall_record(message)
    logging.shutdown()
    os._exit(_EXIT_CODE)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "")) if os.environ.get(name) else default
    except ValueError:
        return default


class StepWatchdog:
    """Use as a context manager around the denoising call.

    The limits can be tuned for a machine or a test with ``CHAOSENGINE_STALL_FACTOR``, ``_FLOOR``,
    ``_FIRST_FLOOR`` and ``_GRACE`` (seconds), and the hard stop switched off with
    ``CHAOSENGINE_STALL_HARD_STOP=0`` (the run is then only cancelled, never the process).
    """

    def __init__(
        self,
        progress: ProgressTracker,
        *,
        kind: str,
        factor: float | None = None,
        floor: float | None = None,
        first_floor: float | None = None,
        grace: float | None = None,
        poll: float = 1.0,
        hard_stop: Callable[[str], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._progress = progress
        self._kind = kind
        self._factor = DEFAULT_FACTOR if factor is None else factor
        self._floor = DEFAULT_FLOOR_SECONDS if floor is None else floor
        self._first_floor = DEFAULT_FIRST_FLOOR_SECONDS if first_floor is None else first_floor
        self._grace = DEFAULT_GRACE_SECONDS if grace is None else grace
        if factor is None:
            self._factor = _env_float("CHAOSENGINE_STALL_FACTOR", DEFAULT_FACTOR)
        if floor is None:
            self._floor = _env_float("CHAOSENGINE_STALL_FLOOR", DEFAULT_FLOOR_SECONDS)
        if first_floor is None:
            self._first_floor = _env_float("CHAOSENGINE_STALL_FIRST_FLOOR", DEFAULT_FIRST_FLOOR_SECONDS)
        if grace is None:
            self._grace = _env_float("CHAOSENGINE_STALL_GRACE", DEFAULT_GRACE_SECONDS)
        self._poll = poll
        if hard_stop is None:
            hard_stop = (lambda _message: None) if os.environ.get("CHAOSENGINE_STALL_HARD_STOP") == "0" else exit_backend
        self._hard_stop = hard_stop
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stalled = False
        self.hard_stopped = False
        self.message: str | None = None

    def __enter__(self) -> "StepWatchdog":
        self._thread = threading.Thread(target=self._run, name=f"{self._kind}-step-watchdog", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        last_step = -1
        last_change = self._clock()
        durations: list[float] = []
        stalled_at: float | None = None
        while not self._stop.wait(self._poll):
            snap = self._progress.snapshot()
            now = self._clock()
            if not snap["active"]:
                last_step, last_change, durations, stalled_at = -1, now, [], None
                continue
            if snap["phase"] != PHASE_DIFFUSING:
                last_change = now  # loading, encoding and decoding have their own pace
                continue
            step = int(snap["step"])
            if snap["totalSteps"] and step >= int(snap["totalSteps"]):
                last_step, last_change = step, now  # every step is done; what is left is decoding
                continue
            if step != last_step:
                if last_step >= 0 and step > last_step:
                    durations.append((now - last_change) / (step - last_step))
                last_step, last_change = step, now
                continue
            waited = now - last_change
            limit = stall_limit_seconds(durations, factor=self._factor, floor=self._floor, first_floor=self._first_floor)
            total = int(snap["totalSteps"]) or step + 1
            if stalled_at is None and waited > limit:
                self.stalled = True
                self.message = stall_message(self._kind, step, total, waited, limit)
                stalled_at = now
                _LOG.error("step watchdog: %s", self.message)
                self._progress.mark_stalled(self.message)
                self._progress.request_cancel()
            elif stalled_at is not None and now - stalled_at > self._grace:
                self.hard_stopped = True
                self._hard_stop(self.message or "step stalled")
                return
