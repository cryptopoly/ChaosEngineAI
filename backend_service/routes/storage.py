"""Model storage settings — HF cache location, disk usage, background move job.

The user can redirect Hugging Face downloads to a different drive when the
default (``~/.cache/huggingface``) runs out of space. The chosen path is
stored in ``settings.json`` under ``hfCachePath``; the Tauri shell reads it
before spawning the backend and injects ``HF_HOME`` so every subsequent
``snapshot_download`` lands on the new drive.

This module exposes three things:

1. ``GET /api/settings/storage`` — current + effective path, disk usage on
   both the current path's drive and the configured override's drive (so
   the UI can warn before the user picks a path with no headroom), and
   total bytes currently under the HF hub tree so the Move button can show
   "Move 163 GB".

2. ``POST /api/settings/storage/move`` — kick off a background copy of the
   current HF hub tree to a new root, preserving the ``blobs/`` +
   ``snapshots/<commit>/`` symlink layout that HuggingFace relies on.
   Returns immediately with a job handle; progress is polled via...

3. ``GET /api/settings/storage/move/status`` — snapshot of the running
   move's phase + byte progress. Safe to poll at 1 Hz.

The move worker is deliberately copy-then-delete rather than ``os.rename``:
the source and destination are typically on different drives, which forces
``rename`` to fall back to a full copy anyway, and the two-phase approach
lets us verify the copy landed before touching the source.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend_service.helpers.discovery import _path_size_bytes
from backend_service.helpers.huggingface import _hf_hub_cache_root


router = APIRouter()


# ------------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------------


def _default_hf_home() -> Path:
    """Platform default for HF_HOME when the user hasn't overridden it.

    Mirrors huggingface_hub's own logic so our "what you'd get by default"
    preview matches exactly what the library will create.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("USERPROFILE") or Path.home())
    else:
        base = Path.home()
    return base / ".cache" / "huggingface"


def _effective_hf_home(settings_value: str | None) -> Path:
    """Resolve ``settings.hfCachePath`` to an absolute Path, or the default."""
    raw = (settings_value or "").strip()
    if raw:
        return Path(os.path.expanduser(raw)).resolve()
    # Respect a live HF_HOME env var (e.g. set by Tauri after reading the
    # setting) over the hard default, so the preview matches what the
    # current process is actually writing.
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(os.path.expanduser(env_home)).resolve()
    return _default_hf_home()


def _free_bytes_on_drive(path: Path) -> int | None:
    """Best-effort free-space probe; walks up to the nearest existing dir."""
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return None
        probe = parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _hub_tree(hf_home: Path) -> Path:
    """The ``hub/`` subdir under any HF_HOME — where all model blobs live."""
    return hf_home / "hub"


def _directory_size(path: Path) -> int:
    """Dedupe-by-inode size walk, or 0 when path is missing."""
    try:
        if not path.exists():
            return 0
    except OSError:
        return 0
    try:
        return _path_size_bytes(path)
    except OSError:
        return 0


def _validate_target_path(raw: str) -> Path:
    """Return the resolved ``Path`` or raise 400 with a user-actionable message.

    Accepts absolute POSIX, absolute Windows (``D:\\foo``), or ``~``-relative.
    Bare relative paths are rejected so we don't silently write model blobs
    to the backend's cwd (which on packaged Windows builds is the app's
    install directory).
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise HTTPException(
            status_code=400,
            detail="Provide an absolute path (e.g. D:\\AI\\huggingface or ~/ai-models).",
        )
    if not (
        cleaned.startswith("/")
        or cleaned.startswith("~")
        or (len(cleaned) >= 2 and cleaned[1] == ":")
    ):
        raise HTTPException(
            status_code=400,
            detail="Path must be absolute or start with ~.",
        )
    try:
        resolved = Path(os.path.expanduser(cleaned)).resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc
    return resolved


# ------------------------------------------------------------------
# Read-only status endpoint
# ------------------------------------------------------------------


@router.get("/api/settings/storage")
def storage_settings(request: Request) -> dict[str, Any]:
    """Snapshot of the HF cache's current location, size, and disk headroom.

    ``configuredPath`` is the raw user setting (empty = default); ``effectivePath``
    is what HF is actually writing to right now, resolved through env vars and
    the platform default. They differ when the user has saved a new path but
    hasn't restarted the backend — the UI uses this gap to surface the
    "Restart required" prompt.
    """
    state = request.app.state.chaosengine
    configured = str(state.settings.get("hfCachePath") or "").strip()
    effective = _effective_hf_home(configured)
    hub_dir = _hub_tree(effective)
    # Live HF_HUB_CACHE wins over our computed hub path — the backend helper
    # respects env overrides even when HF_HOME is unset.
    hub_override = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if hub_override:
        hub_dir = Path(os.path.expanduser(hub_override)).resolve()
    current_hub_size = _directory_size(hub_dir)
    current_free = _free_bytes_on_drive(hub_dir)
    return {
        "configuredPath": configured,
        "effectivePath": str(effective),
        "effectiveHubPath": str(hub_dir),
        "defaultPath": str(_default_hf_home()),
        "currentHubSizeBytes": current_hub_size,
        "currentFreeBytes": current_free,
        "moveJob": _MOVE_JOB.to_dict(),
    }


class UpdateStoragePath(BaseModel):
    hfCachePath: str


@router.post("/api/settings/storage")
def update_storage_path(request: Request, body: UpdateStoragePath) -> dict[str, Any]:
    """Save a new HF cache path. The Tauri shell picks this up on next backend
    launch. Empty string clears the override and restores the default.

    Also returns ``restartRequired`` whenever the saved value differs from what
    the current process is actually using — the UI renders a "Restart backend"
    banner based on this.
    """
    state = request.app.state.chaosengine
    cleaned = (body.hfCachePath or "").strip()
    if cleaned:
        resolved = _validate_target_path(cleaned)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not create {resolved}: {exc}",
            ) from exc
        cleaned = str(resolved)

    # Route through the existing update_settings plumbing so the settings.json
    # write + log emission stays in one place.
    from backend_service.models import UpdateSettingsRequest
    state.update_settings(UpdateSettingsRequest(hfCachePath=cleaned))

    # Re-read so the response reflects the actually-persisted value.
    new_configured = str(state.settings.get("hfCachePath") or "").strip()
    effective = _effective_hf_home(new_configured)
    live_hub = Path(os.path.expanduser(
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HF_HUB_CACHE")
        or (str(effective) + "/hub")
    )).resolve()
    running_hub = _hf_hub_cache_root()
    restart_required = Path(str(running_hub)).resolve() != live_hub
    return {
        "configuredPath": new_configured,
        "effectivePath": str(effective),
        "restartRequired": restart_required,
    }


# ------------------------------------------------------------------
# Move job — dataclass + thread (mirrors _GpuBundleJobState)
# ------------------------------------------------------------------


@dataclass
class _ModelMoveJobState:
    id: str = ""
    phase: str = "idle"  # idle | preflight | copying | cleanup | done | error
    message: str = ""
    sourcePath: str | None = None
    destinationPath: str | None = None
    bytesTotal: int = 0
    bytesCopied: int = 0
    filesTotal: int = 0
    filesCopied: int = 0
    currentEntry: str | None = None
    error: str | None = None
    startedAt: float = 0.0
    finishedAt: float = 0.0
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        percent = 0.0
        if self.bytesTotal > 0:
            percent = min(100.0, (self.bytesCopied / self.bytesTotal) * 100.0)
        return {
            "id": self.id,
            "phase": self.phase,
            "message": self.message,
            "sourcePath": self.sourcePath,
            "destinationPath": self.destinationPath,
            "bytesTotal": self.bytesTotal,
            "bytesCopied": self.bytesCopied,
            "percent": round(percent, 1),
            "filesTotal": self.filesTotal,
            "filesCopied": self.filesCopied,
            "currentEntry": self.currentEntry,
            "error": self.error,
            "startedAt": self.startedAt,
            "finishedAt": self.finishedAt,
            "done": self.done,
        }


_MOVE_JOB = _ModelMoveJobState()
_MOVE_LOCK = threading.Lock()


class StartMoveRequest(BaseModel):
    destinationPath: str
    # When true, the source tree is deleted after a successful copy. Default
    # True because the typical ask is "free up space on C:". Users who want a
    # copy (e.g. backup before migration) can pass false.
    deleteSourceAfter: bool = True


def _count_bytes_and_files(root: Path) -> tuple[int, int]:
    """Walk ``root`` and sum up real blob sizes — dedupes symlinks by target.

    HF's hub tree is ~80% symlinks (``snapshots/<commit>/filename`` →
    ``blobs/<hash>``), so the naive ``getsize`` pass would double-count
    every blob. We count each inode exactly once to get an accurate
    "total bytes to copy" estimate.
    """
    total_bytes = 0
    total_files = 0
    seen_inodes: set[tuple[int, int]] = set()
    if not root.exists():
        return 0, 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                stat = path.stat(follow_symlinks=False)
            except OSError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key in seen_inodes:
                continue
            seen_inodes.add(key)
            total_files += 1
            try:
                total_bytes += stat.st_size
            except OSError:
                continue
    return total_bytes, total_files


def _copy_hub_tree(
    source_hub: Path,
    dest_hub: Path,
    state: _ModelMoveJobState,
) -> None:
    """Walk ``source_hub``, recreating it at ``dest_hub`` one file at a time.

    We use ``copy2`` for regular files (preserving mtime) and re-create
    symlinks with ``os.symlink`` (not ``copy2``, which would resolve them
    into duplicate copies and explode the disk footprint). Directory
    structure is created lazily via ``mkdir(parents=True)``.
    """
    for dirpath, dirnames, filenames in os.walk(source_hub, followlinks=False):
        rel = Path(dirpath).relative_to(source_hub)
        target_dir = dest_hub / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in dirnames:
            # Create subdir early so symlinks whose targets sit under them
            # don't race ahead of the parent directory existing.
            (target_dir / name).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            src = Path(dirpath) / name
            dst = target_dir / name
            state.currentEntry = str(rel / name) if str(rel) != "." else name
            try:
                if src.is_symlink():
                    link_target = os.readlink(src)
                    # Re-point relative symlinks at the new blob location —
                    # HF snapshots use paths like "../../blobs/<hash>" which
                    # are stable under relocation as long as the tree
                    # structure is preserved.
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    os.symlink(link_target, dst)
                    state.filesCopied += 1
                else:
                    shutil.copy2(src, dst)
                    try:
                        size = src.stat().st_size
                    except OSError:
                        size = 0
                    state.bytesCopied += size
                    state.filesCopied += 1
            except OSError as exc:
                # Surface the first real failure as the job error — pip-style
                # silent-skip would leave the dest incomplete and the user
                # would only discover the gap on next preload.
                raise RuntimeError(f"Copy failed for {src}: {exc}") from exc


def _move_job_worker(
    source_hub: Path,
    dest_root: Path,
    dest_hub: Path,
    delete_source: bool,
    state: _ModelMoveJobState,
) -> None:
    """Background worker for the HF hub tree move.

    Phases:
      1. preflight — total bytes + free-space check on destination drive
      2. copying — walk source, create symlinks / copy blobs
      3. cleanup — ``rmtree`` the source (if ``delete_source`` is True)

    On any OSError mid-copy we abort and surface the error; the partial
    destination tree is left in place so the user can inspect / resume.
    """
    try:
        state.phase = "preflight"
        state.message = "Measuring current HF hub size"
        total_bytes, total_files = _count_bytes_and_files(source_hub)
        state.bytesTotal = total_bytes
        state.filesTotal = total_files

        if total_bytes == 0:
            # Nothing to copy — just create the dest tree so subsequent
            # downloads have somewhere to land, and mark done.
            dest_hub.mkdir(parents=True, exist_ok=True)
            state.phase = "done"
            state.message = "Source HF cache was empty — nothing to move."
            state.done = True
            state.finishedAt = time.time()
            return

        free = _free_bytes_on_drive(dest_root)
        # Leave 1 GB headroom so we don't fill the drive exactly and block
        # other writes mid-copy.
        headroom = 1_000_000_000
        if free is not None and free < (total_bytes + headroom):
            required_gb = (total_bytes + headroom) / 1_000_000_000
            free_gb = free / 1_000_000_000
            raise RuntimeError(
                f"Destination drive has {free_gb:.1f} GB free but the move needs "
                f"{required_gb:.1f} GB (models + 1 GB headroom). Free up space "
                "and try again."
            )

        state.phase = "copying"
        state.message = f"Copying {total_files:,} files to {dest_hub}"
        dest_hub.mkdir(parents=True, exist_ok=True)
        _copy_hub_tree(source_hub, dest_hub, state)

        if delete_source:
            state.phase = "cleanup"
            state.message = f"Removing original tree at {source_hub}"
            state.currentEntry = None
            shutil.rmtree(source_hub, ignore_errors=False)

        state.phase = "done"
        state.message = (
            f"Moved {total_files:,} files ({state.bytesCopied / 1_000_000_000:.1f} GB) "
            f"to {dest_hub}. Restart the backend to start downloading there."
        )
        state.done = True
        state.finishedAt = time.time()
    except Exception as exc:  # noqa: BLE001 — surface every failure via state
        state.phase = "error"
        state.message = str(exc) or f"{type(exc).__name__} (no message)"
        state.error = state.message
        state.done = True
        state.finishedAt = time.time()


@router.post("/api/settings/storage/move")
def start_model_move(request: Request, body: StartMoveRequest) -> dict[str, Any]:
    """Kick off a background copy of the running backend's HF hub tree to a
    new root. Returns immediately with the job state; poll ``/move/status``
    for progress. Calling again while a move is running returns the running
    job's state rather than starting a new one — we deliberately can't have
    two moves fighting over the same source tree.
    """
    dest_root = _validate_target_path(body.destinationPath)
    dest_hub = _hub_tree(dest_root)

    # Source = whatever HF hub the running process is actually writing to.
    # This is deliberately independent of the saved ``hfCachePath`` — users
    # frequently change the setting, then click Move, and expect "copy my
    # current models to the new path I just saved".
    source_hub = _hf_hub_cache_root()

    try:
        source_hub_resolved = source_hub.resolve()
    except OSError:
        source_hub_resolved = source_hub
    if source_hub_resolved == dest_hub.resolve() if dest_hub.exists() else dest_hub:
        raise HTTPException(
            status_code=400,
            detail="Source and destination resolve to the same path — nothing to move.",
        )

    with _MOVE_LOCK:
        if _MOVE_JOB.phase in {"preflight", "copying", "cleanup"}:
            return _MOVE_JOB.to_dict()

        _MOVE_JOB.id = f"model-move-{int(time.time() * 1000)}"
        _MOVE_JOB.phase = "preflight"
        _MOVE_JOB.message = "Starting model move"
        _MOVE_JOB.sourcePath = str(source_hub)
        _MOVE_JOB.destinationPath = str(dest_hub)
        _MOVE_JOB.bytesTotal = 0
        _MOVE_JOB.bytesCopied = 0
        _MOVE_JOB.filesTotal = 0
        _MOVE_JOB.filesCopied = 0
        _MOVE_JOB.currentEntry = None
        _MOVE_JOB.error = None
        _MOVE_JOB.startedAt = time.time()
        _MOVE_JOB.finishedAt = 0.0
        _MOVE_JOB.done = False

        thread = threading.Thread(
            target=_move_job_worker,
            args=(source_hub, dest_root, dest_hub, body.deleteSourceAfter, _MOVE_JOB),
            name="chaosengine-model-move",
            daemon=True,
        )
        thread.start()

    state_chaos = request.app.state.chaosengine
    state_chaos.add_log(
        "settings",
        "info",
        f"Model move started: {source_hub} -> {dest_hub} (delete_source={body.deleteSourceAfter})",
    )
    return _MOVE_JOB.to_dict()


@router.get("/api/settings/storage/move/status")
def model_move_status() -> dict[str, Any]:
    """Snapshot of the running (or last-completed) move job. Safe to poll."""
    return _MOVE_JOB.to_dict()
