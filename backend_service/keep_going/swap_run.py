"""The GPU swap sequence of a Keep going run (FU-191 phase 2): unload the chat model, run the renders kept from
the turn, wait for their jobs, load the same model again, then nudge.

The chat model is always loaded again once it was unloaded, also when the run is stopped while a render runs,
unless the same model is already back. A model that does not come back stops the run with
``gpu_reload_failed``; nothing is retried. Kept separate from the controller so its file stays readable.
"""

from __future__ import annotations

import time
from typing import Any

FINISHED = ("done", "failed", "cancelled", "interrupted")


def run_swap(ctrl: Any, session_id: str) -> None:
    with ctrl._lock:
        run = ctrl._runs.get(session_id)
        if run is None or run.state not in ("running", "waiting") or run.dispatching or not run.deferred:
            return
        run.dispatching = True
        run.state = "waiting"
        run.swap_phase = "unloading"
        calls = list(run.deferred)
        run.deferred = []
        app_request = run.app_request
        ctrl._save()
    swap = ctrl.swap
    fields: dict[str, Any] | None = None
    job_ids: list[str] = []
    errors: list[str] = []
    reload_error: str | None = None
    try:
        fields = swap.unload()
        ctrl.hooks.log("info", f"Keep going unloaded the chat model in chat {session_id} for {len(calls)} render(s).")
        _phase(ctrl, run, "rendering")
        for name, args in calls:
            if not _active(ctrl, run):
                break
            try:
                job_id, error = swap.start_render(app_request, name, args)
            except Exception as exc:  # noqa: BLE001 - a render that cannot start is reported, the model comes back
                job_id, error = None, f"{name} failed to start: {type(exc).__name__}: {exc}"
            if job_id:
                job_ids.append(job_id)
            if error:
                errors.append(f"{name}: {error}")
        while job_ids and _active(ctrl, run):
            states = {str(getattr(j, "id", "")): getattr(j, "state", "") for j in ctrl.hooks.chat_jobs()}
            if all(states.get(j) in FINISHED or j not in states for j in job_ids):
                break
            time.sleep(ctrl.poll_seconds)
    finally:
        _phase(ctrl, run, "reloading")
        if fields is not None:
            try:
                swap.reload(fields)
                ctrl.hooks.log("info", f"Keep going loaded {fields.get('modelName')} again in chat {session_id}.")
            except Exception as exc:  # noqa: BLE001 - reported once as the stop reason, never retried
                reload_error = f"{fields.get('modelName') or fields.get('modelRef')}: {exc}"
        with ctrl._lock:
            run.dispatching = False
            run.swap_phase = None
            if reload_error is not None:
                if run.state in ("armed", "running", "waiting"):
                    ctrl._stop_locked(run, "gpu_reload_failed", reload_error)
                return
            if run.state not in ("running", "waiting"):
                ctrl._save()
                return
            if errors:
                run.last_error = ["render", errors[-1]]
            run.state = "running"
            run.nudge_why = "swap_done"
            ctrl._save()
    ctrl._send_nudge(session_id)


def _active(ctrl: Any, run: Any) -> bool:
    with ctrl._lock:
        return run.state in ("running", "waiting")


def _phase(ctrl: Any, run: Any, phase: str) -> None:
    with ctrl._lock:
        run.swap_phase = phase
