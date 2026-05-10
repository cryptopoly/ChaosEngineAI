"""mlx-video Wan installer endpoints (FU-025).

Mirror of the LongLive install pattern but for the Apple Silicon
Wan2.x → MLX conversion path. Phases: preflight, download-raw,
convert, verify. Same single-job semantics, same InstallLogPanel
attempt-row shape, same status poll cadence.

Extracted from ``routes/setup.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


@dataclass
class _WanInstallJobState:
    id: str = ""
    phase: str = "idle"  # idle | preflight | downloading | converting | verifying | done | error
    message: str = ""
    repo: str | None = None
    package_current: str | None = None
    package_index: int = 0
    package_total: int = 0
    percent: float = 0.0
    output_dir: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase,
            "message": self.message,
            "repo": self.repo,
            "packageCurrent": self.package_current,
            "packageIndex": self.package_index,
            "packageTotal": self.package_total,
            "percent": round(self.percent, 1),
            "outputDir": self.output_dir,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": self.attempts,
            "done": self.done,
        }


_WAN_INSTALL_JOB = _WanInstallJobState()
_WAN_INSTALL_LOCK = threading.Lock()


_WAN_PHASE_LABELS: dict[str, str] = {
    "preflight": "Verify Apple Silicon + mlx-video",
    "download-raw": "Download raw Wan checkpoint",
    "convert": "Convert weights to MLX",
    "verify": "Verify converted output",
}


class _WanInstallRequest(BaseModel):
    repo: str = Field(min_length=1, max_length=128)
    dtype: str = Field(default="bfloat16")
    quantize: bool = Field(default=False)
    bits: int = Field(default=4)
    groupSize: int = Field(default=64)
    cleanupRaw: bool = Field(default=False)


def _wan_install_job_worker(
    repo: str,
    *,
    dtype: str,
    quantize: bool,
    bits: int,
    group_size: int,
    cleanup_raw: bool,
) -> None:
    """Run the Wan installer + stream output into the shared job state.

    Same buffering pattern as ``_longlive_job_worker``: per-phase line
    accumulation flushed to an attempt row on each progress event,
    capped at 8000 chars to bound the response payload size.
    """
    from backend_service import mlx_video_wan_installer  # noqa: PLC0415

    job = _WAN_INSTALL_JOB
    phase_buffer: list[str] = []
    current_phase: dict[str, object] = {"name": "preflight"}
    total_phases = len(mlx_video_wan_installer.INSTALL_PHASES)

    def push_attempt(phase: str, ok: bool) -> None:
        job.attempts.append({
            "phase": phase,
            "package": _WAN_PHASE_LABELS.get(phase, phase),
            "ok": ok,
            "output": "\n".join(phase_buffer)[-8000:],
        })
        phase_buffer.clear()

    def stream_log(line: str) -> None:
        phase_buffer.append(line)
        if len(phase_buffer) > 400:
            del phase_buffer[: len(phase_buffer) - 400]

    def report_progress(event: dict[str, object]) -> None:
        phase_name = str(event.get("phase") or "")
        ok = bool(event.get("ok"))
        # Phase event marks the START of that phase; flush prior buffer
        # as a completed attempt only when transitioning from a real
        # phase. The first event (preflight) has no prior buffer.
        if current_phase.get("name") and current_phase.get("name") != phase_name:
            push_attempt(str(current_phase["name"]), ok=True)
        if not ok:
            push_attempt(phase_name, ok=False)
            job.phase = "error"
            return
        current_phase["name"] = phase_name
        try:
            idx = mlx_video_wan_installer.INSTALL_PHASES.index(phase_name)
        except ValueError:
            return
        job.package_index = idx
        job.percent = (idx / total_phases) * 100.0
        job.package_current = _WAN_PHASE_LABELS.get(phase_name, phase_name)
        job.message = f"Running: {job.package_current}"
        # Update job phase label for the UI status badge.
        job.phase = {
            "preflight": "preflight",
            "download-raw": "downloading",
            "convert": "converting",
            "verify": "verifying",
        }.get(phase_name, "preflight")

    job.message = f"Starting Wan install for {repo}"
    job.package_current = _WAN_PHASE_LABELS["preflight"]
    job.package_total = total_phases

    try:
        mlx_video_wan_installer.install(
            repo,
            dtype=dtype,
            quantize=quantize,
            bits=bits,
            group_size=group_size,
            keep_raw=not cleanup_raw,
            logger=stream_log,
            progress=report_progress,
        )
    except mlx_video_wan_installer.WanInstallError as exc:
        if phase_buffer:
            push_attempt(str(current_phase["name"]), ok=False)
        job.phase = "error"
        job.error = str(exc)
        job.message = f"Wan install failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        if phase_buffer:
            push_attempt(str(current_phase["name"]), ok=False)
        job.phase = "error"
        job.error = f"Unexpected error: {exc}"
        job.message = job.error
    else:
        if phase_buffer:
            # Flush the verify-phase buffer that wasn't followed by a
            # phase-transition event.
            push_attempt(str(current_phase["name"]), ok=True)
        job.phase = "done"
        job.percent = 100.0
        job.package_index = total_phases
        job.package_current = None
        job.message = f"Wan install complete: {repo}"
    finally:
        job.finished_at = time.time()
        job.done = True


@router.post("/api/setup/install-mlx-video-wan")
def start_install_mlx_video_wan(
    body: _WanInstallRequest, request: Request
) -> dict[str, Any]:
    """Kick off a background Wan install (download raw HF weights +
    convert to MLX).

    Returns the current job state immediately. Poll
    ``/api/setup/install-mlx-video-wan/status`` for progress.
    Calling again while a job runs returns the running state without
    starting a duplicate.
    """
    state_chaosengine = request.app.state.chaosengine

    from backend_service import mlx_video_wan_convert, mlx_video_wan_installer  # noqa: PLC0415

    if not mlx_video_wan_installer.is_supported_raw_repo(body.repo):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported Wan repo {body.repo!r}. Supported: "
                f"{sorted(mlx_video_wan_installer.SUPPORTED_RAW_REPOS)}"
            ),
        )

    output_dir = mlx_video_wan_convert.output_dir_for(body.repo)

    with _WAN_INSTALL_LOCK:
        if _WAN_INSTALL_JOB.phase in {"preflight", "downloading", "converting", "verifying"}:
            return _WAN_INSTALL_JOB.to_dict()

        _WAN_INSTALL_JOB.id = f"wan-mlx-{int(time.time() * 1000)}"
        _WAN_INSTALL_JOB.phase = "preflight"
        _WAN_INSTALL_JOB.repo = body.repo
        _WAN_INSTALL_JOB.message = "Starting install"
        _WAN_INSTALL_JOB.package_current = _WAN_PHASE_LABELS["preflight"]
        _WAN_INSTALL_JOB.package_index = 0
        _WAN_INSTALL_JOB.package_total = len(mlx_video_wan_installer.INSTALL_PHASES)
        _WAN_INSTALL_JOB.percent = 0.0
        _WAN_INSTALL_JOB.output_dir = str(output_dir)
        _WAN_INSTALL_JOB.error = None
        _WAN_INSTALL_JOB.started_at = time.time()
        _WAN_INSTALL_JOB.finished_at = 0.0
        _WAN_INSTALL_JOB.attempts = []
        _WAN_INSTALL_JOB.done = False

        thread = threading.Thread(
            target=_wan_install_job_worker,
            name="chaosengine-wan-install",
            kwargs={
                "repo": body.repo,
                "dtype": body.dtype,
                "quantize": body.quantize,
                "bits": body.bits,
                "group_size": body.groupSize,
                "cleanup_raw": body.cleanupRaw,
            },
            daemon=True,
        )
        thread.start()

    state_chaosengine.add_log(
        "server", "info",
        f"Wan install started (job={_WAN_INSTALL_JOB.id}, repo={body.repo}, "
        f"target={output_dir})",
    )
    return _WAN_INSTALL_JOB.to_dict()


@router.get("/api/setup/install-mlx-video-wan/status")
def install_mlx_video_wan_status() -> dict[str, Any]:
    """Snapshot of the current Wan install job. Safe to poll at 1-2 Hz."""
    return _WAN_INSTALL_JOB.to_dict()


@router.get("/api/setup/mlx-video-wan/inventory")
def mlx_video_wan_inventory() -> dict[str, Any]:
    """List every Wan repo: supported + converted-on-disk + approx size.

    The Setup-page panel uses this to render a per-variant install
    table without poking at every status endpoint individually."""
    from backend_service import mlx_video_wan_convert, mlx_video_wan_installer  # noqa: PLC0415

    converted_repos = {s.repo for s in mlx_video_wan_convert.list_converted()}
    items: list[dict[str, Any]] = []
    for repo in sorted(mlx_video_wan_installer.SUPPORTED_RAW_REPOS):
        status = mlx_video_wan_convert.status_for(repo)
        items.append({
            "repo": repo,
            "approxRawSizeGb": mlx_video_wan_installer.approx_raw_size_gb(repo),
            "converted": repo in converted_repos,
            "status": status.to_dict(),
        })
    return {
        "items": items,
        "convertRoot": str(mlx_video_wan_convert.CONVERT_ROOT),
        "rawRoot": str(mlx_video_wan_installer.RAW_ROOT),
    }
