"""The one door to the assembler, plus a one-at-a-time background render with progress and cancel.

``render(plan, progress_cb, cancel_event)`` imports ``render_plan`` and ``assemble`` lazily (they are
heavy and live in their own modules) and hands them the plain-dict plan from ``timeline.build_render_plan``.
Tests replace ``render`` with a fake. ``RENDERS`` runs one render at a time in a daemon thread; the UI
and the agent poll its status, and Cancel sets the event the assembler checks between stages.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service.storyboard.models import new_id


class RenderBusy(RuntimeError):
    pass


def render(plan: dict[str, Any], progress_cb: Callable[..., None], cancel_event: threading.Event) -> Any:
    from backend_service.storyboard import assemble
    from backend_service.storyboard.render_plan import RenderPlan

    return assemble.render(RenderPlan.from_dict(plan), progress_cb, cancel_event)


class RenderJob:
    def __init__(self, project_id: str, plan: dict[str, Any]) -> None:
        self.id = new_id("r")
        self.project_id = project_id
        self.preview = bool(plan.get("preview"))
        self.out_path = Path(plan["out_path"])
        self.plan = plan
        self.state = "running"  # running | done | failed | cancelled
        self.fraction = 0.0
        self.message = ""
        self.error: str | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.cancel_event = threading.Event()
        self.done = threading.Event()

    def progress(self, *args: Any, **kwargs: Any) -> None:
        """Tolerant progress callback: a number is the fraction (0..1), a string is the message."""
        for value in [*args, *kwargs.values()]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.fraction = max(0.0, min(1.0, float(value)))
            elif isinstance(value, str):
                self.message = value[:200]

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "renderId": self.id,
            "projectId": self.project_id,
            "preview": self.preview,
            "state": self.state,
            "progress": round(self.fraction, 3),
            "message": self.message,
        }
        if self.state == "done":
            info["file"] = f"exports/{self.out_path.name}"
        if self.error:
            info["error"] = self.error
        return info


class RenderManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: RenderJob | None = None
        self.last: dict[str, RenderJob] = {}

    def start(self, project_id: str, plan: dict[str, Any]) -> RenderJob:
        with self._lock:
            if self.current is not None and not self.current.done.is_set():
                raise RenderBusy("A render is already running. Wait for it or cancel it.")
            job = RenderJob(project_id, plan)
            self.current = job
            self.last[project_id] = job
        threading.Thread(target=self._run, args=(job,), name=f"storyboard-render-{job.id}", daemon=True).start()
        return job

    def _run(self, job: RenderJob) -> None:
        from backend_service.storyboard import render_adapter
        from backend_service.storyboard.media import scrub

        try:
            Path(job.plan["out_path"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job.plan["work_dir"]).mkdir(parents=True, exist_ok=True)
            render_adapter.render(job.plan, job.progress, job.cancel_event)
            if job.cancel_event.is_set():
                job.state = "cancelled"
            elif not job.out_path.is_file():
                job.state, job.error = "failed", "The render finished without writing the film."
            else:
                job.state, job.fraction = "done", 1.0
        except Exception as exc:  # noqa: BLE001 - shown in the status
            if job.cancel_event.is_set():
                job.state = "cancelled"
            else:
                job.state, job.error = "failed", scrub(f"{type(exc).__name__}: {exc}")
        finally:
            job.finished_at = time.time()
            job.done.set()

    def status(self, project_id: str) -> RenderJob | None:
        with self._lock:
            return self.last.get(project_id)

    def cancel(self, project_id: str) -> bool:
        job = self.status(project_id)
        if job is None or job.done.is_set():
            return False
        job.cancel_event.set()
        return True


RENDERS = RenderManager()
