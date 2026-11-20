"""The "Copy now" job for music weights a link could not place (FU-276, D-100).

When symlinks and hard links are both refused (no Developer Mode, cache on another drive) the weight files
must be copied into the checkpoints layout. This module runs that copy as one tracked job in a worker thread:
free-space check first, progress in bytes, cancel, and resume-safe (a finished file is never copied again;
each file goes through ``<name>.partial``, is size-checked, then renamed, so a final name is always whole).
A cancelled or crashed partial file is started again, not appended to.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service import music_env

CHUNK_BYTES = 8 * 1024 * 1024
_MIN_MARGIN_BYTES = 1024 ** 3


class WeightCopyError(RuntimeError):
    """Refused or failed with a plain message (not enough space, a source file went missing)."""


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}
_CANCEL = threading.Event()


def _reset() -> None:
    _STATE.clear()
    _STATE.update(
        phase="idle", bytesDone=0, totalBytes=0, fileCount=0, filesDone=0, currentFile="",
        error=None, startedAt=0.0, finishedAt=0.0, etaSeconds=None, source="", target="",
    )


_reset()


def is_running() -> bool:
    return _STATE["phase"] == "running"


def status() -> dict[str, Any]:
    with _LOCK:
        data = dict(_STATE)
    data["running"] = data["phase"] == "running"
    return data


def _pending_files() -> list[tuple[Path, Path]]:
    from backend_service.catalog.music_models import music_variants

    _placed, pending = music_env.mirror_variants(music_variants())
    return pending


def required_bytes(pending: list[tuple[Path, Path]]) -> int:
    """Disk the copy needs: every file still to copy, plus a margin (the larger of 1 GiB and 5 %)."""
    total = sum(real.stat().st_size for real, _ in pending)
    return total + max(_MIN_MARGIN_BYTES, total // 20)


def check_space(pending: list[tuple[Path, Path]]) -> None:
    target = music_env.checkpoints_dir()
    target.mkdir(parents=True, exist_ok=True)
    need, free = required_bytes(pending), shutil.disk_usage(target).free
    if free < need:
        raise WeightCopyError(
            f"Not enough free space on {music_env._drive(target)} to copy the music model: it needs "
            f"{need / 1024 ** 3:.1f} GB free (the files plus a safety margin) and has {free / 1024 ** 3:.1f} GB. "
            "Free some space, or turn on Developer Mode, or keep the model cache on the same drive."
        )


def _copy_file(real: Path, target: Path, on_bytes: Callable[[int], None]) -> None:
    size = real.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        with real.open("rb") as src, partial.open("wb") as dst:
            while True:
                if _CANCEL.is_set():
                    raise InterruptedError("Copy cancelled.")
                chunk = src.read(CHUNK_BYTES)
                if not chunk:
                    break
                dst.write(chunk)
                on_bytes(len(chunk))
        if partial.stat().st_size != size:
            raise WeightCopyError(f"{real.name} was not copied completely ({partial.stat().st_size} of {size} bytes).")
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _run(pending: list[tuple[Path, Path]], on_done: Callable[[dict[str, Any]], None] | None) -> None:
    started = time.time()

    def add(n: int) -> None:
        with _LOCK:
            _STATE["bytesDone"] += n
            elapsed = time.time() - started
            left = _STATE["totalBytes"] - _STATE["bytesDone"]
            _STATE["etaSeconds"] = round(left / (_STATE["bytesDone"] / elapsed)) if elapsed > 0.5 else None

    try:
        for real, target in pending:
            with _LOCK:
                _STATE["currentFile"] = target.name
            if target.exists() and target.stat().st_size == real.stat().st_size:
                add(real.stat().st_size)  # already whole (a retry after a cancel)
            else:
                _copy_file(real, target, add)
            with _LOCK:
                _STATE["filesDone"] += 1
        music_env._copy_notice_path().unlink(missing_ok=True)
        phase, error = "done", None
    except InterruptedError:
        phase, error = "cancelled", None
    except (OSError, WeightCopyError) as exc:
        phase, error = "error", f"The copy stopped: {exc}"
    with _LOCK:
        _STATE.update(phase=phase, error=error, finishedAt=time.time(), currentFile="", etaSeconds=None)
    if phase == "done" and on_done is not None:
        on_done(status())


def start(on_done: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Start the copy (or report the running one). Raises ``WeightCopyError`` when the disk is too full."""
    if is_running():
        return status()
    pending = _pending_files()
    plan = music_env.copy_plan(pending) if pending else {}
    if pending:
        check_space(pending)
    with _LOCK:
        if _STATE["phase"] == "running":
            return {**_STATE, "running": True}
        _CANCEL.clear()
        _reset()
        _STATE.update(
            phase="running" if pending else "done", fileCount=len(pending), totalBytes=plan.get("totalBytes", 0),
            source=plan.get("source", ""), target=plan.get("target", ""),
            startedAt=time.time(), finishedAt=0.0 if pending else time.time(),
        )
    if pending:
        threading.Thread(target=_run, args=(pending, on_done), daemon=True).start()
    return status()


def cancel() -> dict[str, Any]:
    _CANCEL.set()
    return status()
