"""The one door to the assembler, plus a one-at-a-time background render with progress, cancel and resume.

``render(plan, progress_cb, cancel_event)`` imports ``render_plan`` and ``assemble`` lazily (they are
heavy and live in their own modules) and hands them the plain-dict plan from ``timeline.build_render_plan``.
Tests replace ``render`` with a fake. ``RENDERS`` runs one render at a time in a daemon thread; the UI
and the agent poll its status, and Cancel sets the event the assembler checks between stages.

A render writes a small manifest (``cache/render/manifest-<preview|export>.json``): its state, progress and
the per-shot intermediates it will use. The finished intermediates are already cached by content, so when
the backend is killed mid-render the next status read finds a manifest still saying ``running`` with no
live job and reports it as ``interrupted`` and resumable; starting the render again reuses every finished
shot and redoes only what is missing. While a render runs the machine is kept awake, and a long one ends
with a system banner (``longrun``).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service.storyboard import longrun
from backend_service.storyboard.models import new_id

MANIFEST_VERSION = 1
MANIFEST_EVERY_S = 1.5


class RenderBusy(RuntimeError):
    pass


def render(plan: dict[str, Any], progress_cb: Callable[..., None], cancel_event: threading.Event) -> Any:
    from backend_service.storyboard import assemble
    from backend_service.storyboard.render_plan import RenderPlan

    return assemble.render(RenderPlan.from_dict(plan), progress_cb, cancel_event)


def manifest_path(work_dir: str | Path, preview: bool) -> Path:
    return Path(work_dir) / f"manifest-{'preview' if preview else 'export'}.json"


def _intermediates(plan: dict[str, Any]) -> list[str]:
    try:
        from backend_service.storyboard import assemble
        from backend_service.storyboard.render_plan import RenderPlan

        return assemble.intermediate_names(RenderPlan.from_dict(plan))
    except Exception:  # noqa: BLE001 - a plan that does not validate fails in the render, with its own message
        return []


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)


class RenderJob:
    def __init__(self, project_id: str, plan: dict[str, Any], notify: bool = False) -> None:
        self.id = new_id("r")
        self.project_id = project_id
        self.preview = bool(plan.get("preview"))
        self.out_path = Path(plan["out_path"])
        self.plan = plan
        self.notify = notify
        self.state = "running"  # running | done | failed | cancelled | interrupted
        self.fraction = 0.0
        self.message = ""
        self.error: str | None = None
        self.warnings: list[str] = []
        self.extras: list[str] = []  # project-relative files written next to the film (captions, stems)
        self.shots: list[str] = []  # the per-shot intermediates this render uses
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.cancel_event = threading.Event()
        self.done = threading.Event()
        self._saved_at = 0.0

    @classmethod
    def from_manifest(cls, data: dict[str, Any], work_dir: Path, project_id: str) -> "RenderJob":
        """A finished or interrupted render read back from its manifest (nothing is running)."""
        job = cls.__new__(cls)
        job.id = str(data.get("renderId", ""))
        job.project_id = project_id
        job.preview = bool(data.get("preview"))
        job.out_path = Path(str(data.get("out", "")))
        job.plan = {"work_dir": str(work_dir)}
        job.notify = False
        state = str(data.get("state", "interrupted"))
        # A manifest still saying "running" belongs to a process that is gone.
        job.state = "interrupted" if state == "running" else state
        job.fraction = float(data.get("progress", 0.0))
        job.message = str(data.get("message", ""))
        job.error = data.get("error")
        job.warnings = list(data.get("warnings", []))
        job.extras = list(data.get("extras", []))
        job.shots = list(data.get("shots", []))
        job.started_at = float(data.get("startedAt", 0.0))
        job.finished_at = data.get("finishedAt")
        job.cancel_event = threading.Event()
        job.done = threading.Event()
        job.done.set()
        job._saved_at = 0.0
        return job

    def manifest(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "renderId": self.id,
            "preview": self.preview,
            "state": self.state,
            "progress": round(self.fraction, 3),
            "message": self.message,
            "out": self.out_path.name,
            "shots": self.shots,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "warnings": self.warnings,
            "extras": self.extras,
            "error": self.error,
        }

    def save(self) -> None:
        work = Path(self.plan["work_dir"])
        if work.is_dir():
            _atomic_json(manifest_path(work, self.preview), self.manifest())
            self._saved_at = time.monotonic()

    def progress(self, *args: Any, **kwargs: Any) -> None:
        """Tolerant progress callback: a number is the fraction (0..1), a string is the message."""
        for value in [*args, *kwargs.values()]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.fraction = max(0.0, min(1.0, float(value)))
            elif isinstance(value, str):
                self.message = value[:200]
        if time.monotonic() - self._saved_at >= MANIFEST_EVERY_S:
            self.save()

    def shots_ready(self) -> int:
        work = Path(self.plan.get("work_dir", ""))
        return sum(1 for name in self.shots if (work / name).is_file())

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "renderId": self.id,
            "projectId": self.project_id,
            "preview": self.preview,
            "state": self.state,
            "progress": round(self.fraction, 3),
            "message": self.message,
            "startedAt": self.started_at,
        }
        if self.state == "done":
            info["file"] = f"exports/{self.out_path.name}"
        if self.error:
            info["error"] = self.error
        if self.warnings:
            info["warnings"] = self.warnings
        if self.extras:
            info["extras"] = self.extras
        if self.state in ("interrupted", "failed", "cancelled") and self.shots:
            info["resumable"] = True
            info["shotsReady"] = self.shots_ready()
            info["shotsTotal"] = len(self.shots)
        return info


class RenderManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current: RenderJob | None = None
        self.last: dict[str, RenderJob] = {}

    def start(self, project_id: str, plan: dict[str, Any], notify: bool = False) -> RenderJob:
        with self._lock:
            if self.current is not None and not self.current.done.is_set():
                raise RenderBusy("A render is already running. Wait for it or cancel it.")
            job = RenderJob(project_id, plan, notify)
            self.current = job
            self.last[project_id] = job
        threading.Thread(target=self._run, args=(job,), name=f"storyboard-render-{job.id}", daemon=True).start()
        return job

    def _run(self, job: RenderJob) -> None:
        from backend_service.storyboard import assemble
        from backend_service.storyboard.media import scrub

        result: Any = None
        try:
            out_dir, work = Path(job.plan["out_path"]).parent, Path(job.plan["work_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            work.mkdir(parents=True, exist_ok=True)
            assemble.remove_leftovers(work, out_dir)  # what a killed render left half-written
            job.shots = _intermediates(job.plan)
            job.save()
            with longrun.keep_awake("Rendering a Storyboard film"):
                result = render(job.plan, job.progress, job.cancel_event)
            if job.cancel_event.is_set():
                job.state = "cancelled"
            elif not job.out_path.is_file():
                job.state, job.error = "failed", "The render finished without writing the film."
            else:
                job.state, job.fraction = "done", 1.0
                self._after_film(job, result)
        except Exception as exc:  # noqa: BLE001 - shown in the status
            if job.cancel_event.is_set():
                job.state = "cancelled"
            else:
                job.state, job.error = "failed", scrub(f"{type(exc).__name__}: {exc}")
        finally:
            job.finished_at = time.time()
            job.save()
            job.done.set()
            self._announce(job)

    @staticmethod
    def _after_film(job: RenderJob, result: Any) -> None:
        """Record the warnings and the files written next to the film; write the caption sidecar."""
        job.warnings = [str(w) for w in (getattr(result, "warnings", None) or [])]
        for name in getattr(result, "stems", None) or []:
            job.extras.append(f"exports/{name}")
        sidecar = job.plan.get("sidecar_srt")
        if sidecar:
            target = Path(sidecar["path"])
            tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                tmp.write_text(str(sidecar["text"]), encoding="utf-8")
                os.replace(tmp, target)
                job.extras.append(f"exports/{target.name}")
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                job.warnings.append(f"The caption file could not be written: {exc}")

    @staticmethod
    def _announce(job: RenderJob) -> None:
        if not job.notify or job.preview or (job.finished_at or 0) - job.started_at < longrun.LONG_RENDER_SECONDS:
            return
        if job.state == "done":
            longrun.notify("Your film is ready", job.out_path.name)
        elif job.state == "failed":
            longrun.notify("The render stopped", (job.error or "It failed.")[:160])

    def status(self, project_id: str, folder: Path | None = None) -> RenderJob | None:
        """The newest render of the project: the live or last one of this process, else the newest manifest."""
        with self._lock:
            job = self.last.get(project_id)
        if job is not None or folder is None:
            return job
        work = folder / "cache" / "render"
        found: list[RenderJob] = []
        for preview in (False, True):
            try:
                data = json.loads(manifest_path(work, preview).read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("version") == MANIFEST_VERSION:
                    found.append(RenderJob.from_manifest(data, work, project_id))
            except (OSError, ValueError):
                continue
        return max(found, key=lambda j: j.started_at) if found else None

    def cancel(self, project_id: str) -> bool:
        job = self.status(project_id)
        if job is None or job.done.is_set():
            return False
        job.cancel_event.set()
        return True


RENDERS = RenderManager()
