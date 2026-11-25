"""Upscale jobs: a background thread per job on the shared heavy-job slot, the warm worker that runs the model, and
resume from the job's manifest.

A job takes the ``HEAVY_JOBS`` slot (kind ``upscale``) before it touches the GPU, so it never runs beside an image,
video, music or voice job, and while it waits it can be cancelled. Its id is ``JobSpec.key()``: the same request
always finds the same work folder (``~/.chaosengine/cache/video-upscale/<id>/``), so starting it again, or
``resume`` after a cancel, a failure or a backend restart, skips every finished segment. The work folder is removed
once the film is written. Segments run in ``worker.py`` through a ``WarmWorker`` (kind ``upscale``: a video, image or
music job starting stops it when it idles, and an upscale job stops their idle workers).
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from backend_service.catalog.video_upscale_models import find_model
from backend_service.video_upscale import addon, pipeline
from backend_service.video_upscale.engine import UpscaleError

SLOT_KIND = "upscale"
_WORKER_LOG = Path.home() / ".chaosengine" / "logs" / "video-upscale-worker.log"
_SEGMENT_TIMEOUT_S = 3 * 3600.0  # one segment; the slow row at 4K is minutes, not hours
_WORKER_STALL_S = 600.0  # the first load and the first batch send no step
_WORKER_HARD_CANCEL_S = 20.0


def work_root() -> Path:
    return Path.home() / ".chaosengine" / "cache" / "video-upscale"


# ---------------------------------------------------------------------------
# the warm worker
# ---------------------------------------------------------------------------

_WORKER: Any = None
_WORKER_LOCK = threading.Lock()


def _worker_env() -> dict[str, str]:
    from backend_service.inference._utils import _isolated_child_env  # noqa: PLC0415

    env = dict(_isolated_child_env())
    root = str(Path(__file__).resolve().parents[2])
    parts = [root, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []), str(addon.site_dir())]
    env["PYTHONPATH"] = os.pathsep.join(parts)  # the add-on last: it must never shadow the backend's own packages
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def worker() -> Any:
    global _WORKER
    from backend_service.warm_worker_client import WarmWorker  # noqa: PLC0415

    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = WarmWorker(
                name="video-upscale",
                kind=SLOT_KIND,
                command=lambda: [sys.executable, "-m", "backend_service.video_upscale.worker"],
                env=_worker_env,
                log_path=_WORKER_LOG,
                hard_cancel_after_s=_WORKER_HARD_CANCEL_S,
            )
        return _WORKER


def worker_runner() -> Callable[..., dict[str, Any]]:
    """A ``pipeline.run_job`` runner that sends each segment to the warm worker."""
    from backend_service.warm_worker_client import WarmWorkerCancelled, WarmWorkerError  # noqa: PLC0415

    def run(req: dict[str, Any], on_frames: Callable[[int], None], is_cancelled: Callable[[], bool]) -> dict[str, Any]:
        def on_event(event: dict[str, Any]) -> None:
            if event.get("event") == "step":
                on_frames(int(event.get("step") or 0))

        try:
            return worker().request("generate", timeout=_SEGMENT_TIMEOUT_S, stall_s=_WORKER_STALL_S, on_event=on_event,
                                    cancel_check=is_cancelled, job=f"up-{uuid.uuid4().hex[:12]}", **req)
        except WarmWorkerCancelled as exc:
            raise pipeline.Cancelled("cancelled") from exc
        except WarmWorkerError as exc:
            raise RuntimeError(f"The video upscaler failed: {exc}") from exc

    return run


# ---------------------------------------------------------------------------
# building a job
# ---------------------------------------------------------------------------


def check_ready(model_id: str) -> dict[str, Any]:
    """The catalog row, or ``UpscaleError`` naming the install step (409) or the unknown model (400)."""
    row = find_model(model_id)
    if row is None:
        raise UpscaleError(f"Unknown video upscale model '{model_id}'.")
    if not addon.site_installed() or not addon.model_installed(row):
        raise UpscaleError(addon.NOT_INSTALLED, 409)
    return row


def build_spec(src: Path, dst: Path, model_id: str, size: tuple[int, int], *, deinterlace: bool = False,
               avtb: bool = False, segment_seconds: float = pipeline.SEGMENT_SECONDS) -> pipeline.JobSpec:
    """Check the add-on and the ffmpeg build, then probe the input: the spec a job runs."""
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    row = check_ready(model_id)
    exe = ffmpeg_bin.ffmpeg_exe()  # FfmpegMissing: the app offers the video tools install
    if deinterlace and pipeline.DEINTERLACE_FILTER not in ffmpeg_bin.ffmpeg_filters():
        raise UpscaleError(f"This ffmpeg has no '{pipeline.DEINTERLACE_FILTER}' filter, so it cannot deinterlace. "
                           "Turn deinterlacing off or choose another ffmpeg in Setup.")
    spec = pipeline.make_spec(exe, src, dst, model_id=model_id, weights=addon.weights_path(row), size=size,
                              tile=int(row.get("tile") or 0), batch=int(row.get("batch") or 1), deinterlace=deinterlace,
                              segment_seconds=segment_seconds, avtb=avtb)
    # The Storyboard's final quality with the encoder this build has (libx264 CRF 18 when present).
    spec.codec_args = ffmpeg_bin.h264_args(False, width=size[0], height=size[1],
                                           fps=round(pipeline.fps_value(spec.source["fps"])))
    return spec


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


class UpscaleJob:
    def __init__(self, spec: pipeline.JobSpec, source: str = "ui") -> None:
        self.id = spec.key()
        self.spec = spec
        self.source = source
        self.state = "queued"  # queued | running | done | failed | cancelled | interrupted
        self.frames_done = 0
        self.frames_total = int(spec.source["frames"])
        self.segment = 0
        self.segments = 0
        self.error: str | None = None
        self.notes: list[str] = []
        self.resumed_segments = 0
        self.started_at = time.time()
        self.running_since: float | None = None
        self.finished_at: float | None = None
        self.ticket_id: int | None = None
        self.cancel_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._first_frames: int | None = None

    def progress(self, done: int, total: int, segment: int, segments: int) -> None:
        if self._first_frames is None:
            self._first_frames = done
        self.frames_done, self.frames_total, self.segment, self.segments = done, total, segment, segments

    def live(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def to_dict(self) -> dict[str, Any]:
        rate = None
        if self.running_since and self._first_frames is not None and self.state == "running":
            elapsed = time.time() - self.running_since
            rate = round((self.frames_done - self._first_frames) / elapsed, 2) if elapsed > 1 else None
        return {
            "id": self.id, "state": self.state, "input": self.spec.input, "output": self.spec.output,
            "modelId": self.spec.model_id, "width": self.spec.width, "height": self.spec.height,
            "framesDone": self.frames_done, "framesTotal": self.frames_total,
            "fraction": round(min(1.0, self.frames_done / self.frames_total), 4) if self.frames_total else 0.0,
            "framesPerSecond": rate, "segment": self.segment, "segments": self.segments,
            "resumedSegments": self.resumed_segments, "error": self.error, "notes": list(self.notes),
            "startedAt": self.started_at, "finishedAt": self.finished_at,
            "resumable": self.state in ("failed", "cancelled", "interrupted"),
        }


class UpscaleJobs:
    def __init__(self, runner_factory: Callable[[], Callable[..., dict[str, Any]]] | None = None,
                 root: Callable[[], Path] = work_root, queue: Any = None) -> None:
        self.runner_factory = runner_factory or worker_runner
        self.root = root
        self._queue = queue
        self._jobs: dict[str, UpscaleJob] = {}
        self._lock = threading.Lock()

    @property
    def queue(self) -> Any:
        if self._queue is None:
            from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415

            self._queue = HEAVY_JOBS
        return self._queue

    def start(self, spec: pipeline.JobSpec, source: str = "ui") -> UpscaleJob:
        key = spec.key()
        with self._lock:
            current = self._jobs.get(key)
            if current is not None and current.live():
                return current
            job = UpscaleJob(spec, source)
            self._jobs[key] = job
            job.thread = threading.Thread(target=self._run, args=(job,), name=f"video-upscale-{key}", daemon=True)
            job.thread.start()
        return job

    def _run(self, job: UpscaleJob) -> None:
        from backend_service.progress import GenerationCancelled  # noqa: PLC0415

        label = f"upscale {Path(job.spec.input).name} to {job.spec.width}x{job.spec.height}"
        ticket = self.queue.enqueue(SLOT_KIND, job.source, label)
        job.ticket_id = ticket.id
        work = self.root() / job.id
        try:
            if job.cancel_event.is_set():
                raise pipeline.Cancelled("cancelled")
            self.queue.wait_turn(ticket)
            job.state, job.running_since = "running", time.time()
            result = pipeline.run_job(job.spec, self.runner_factory(), exe=_ffmpeg(), work=work,
                                      progress=job.progress, is_cancelled=job.cancel_event.is_set)
            job.notes, job.resumed_segments = list(result.get("notes") or []), int(result.get("resumedSegments") or 0)
            job.state = "done"
            shutil.rmtree(work, ignore_errors=True)
        except (pipeline.Cancelled, GenerationCancelled):
            job.state = "cancelled"
        except Exception as exc:  # noqa: BLE001 - shown in the status
            job.state, job.error = ("cancelled", None) if job.cancel_event.is_set() else ("failed", str(exc)[:400])
        finally:
            self.queue.release(ticket)
            job.finished_at = time.time()

    def get(self, job_id: str) -> UpscaleJob | None:
        """The job of this process, else one read back from its manifest (``interrupted`` when it was running)."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        if not _safe_id(job_id):
            return None
        data = pipeline.read_manifest(self.root() / job_id)
        if data is None:
            return None
        job = UpscaleJob(pipeline.JobSpec.from_dict(data["spec"]))
        state = data.get("state") or "interrupted"
        job.state = "interrupted" if state == "running" else state
        job.error = data.get("error")
        done = [s for s in data["segments"] if s.get("done")]
        job.frames_done, job.segments = sum(int(s.get("made", 0)) for s in done), len(data["segments"])
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or not job.live():
            return False
        job.cancel_event.set()
        if job.ticket_id is not None:
            self.queue.cancel_waiting(ticket_id=job.ticket_id)
        return True

    def resume(self, job_id: str) -> UpscaleJob:
        job = self.get(job_id)
        if job is None:
            raise UpscaleError(f"No upscale job '{job_id}'.", 404)
        if job.live() or job.state == "done":
            return job
        spec = job.spec
        try:
            stat = Path(spec.input).stat()
        except OSError as exc:
            raise UpscaleError(f"The input {Path(spec.input).name} is gone, so the job cannot resume.", 409) from exc
        if stat.st_size != spec.input_bytes or stat.st_mtime != spec.input_mtime:
            raise UpscaleError("The input file changed since the job started; start a new job instead.", 409)
        check_ready(spec.model_id)
        return self.start(spec, job.source)

    def run_sync(self, spec: pipeline.JobSpec, *, source: str, label: str, is_cancelled: Callable[[], bool],
                 progress: Callable[[int, int, int, int], None] | None = None) -> dict[str, Any]:
        """Run a job in the calling thread on the heavy-job slot (the Storyboard export); its work folder is removed
        when it finishes."""
        work = self.root() / spec.key()
        with self.queue.slot(SLOT_KIND, source, label):
            result = pipeline.run_job(spec, self.runner_factory(), exe=_ffmpeg(), work=work, progress=progress,
                                      is_cancelled=is_cancelled)
        shutil.rmtree(work, ignore_errors=True)
        return result


def _safe_id(job_id: str) -> bool:
    return len(job_id) == 16 and all(c in "0123456789abcdef" for c in job_id)


def _ffmpeg() -> str:
    from backend_service.storyboard import ffmpeg_bin  # noqa: PLC0415

    return ffmpeg_bin.ffmpeg_exe()


JOBS = UpscaleJobs()
