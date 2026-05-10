"""Hugging Face download lifecycle for ``ChaosEngineState``.

Six helpers + one inner thread worker lifted out of
``state/__init__.py`` covering the full download flow:

* ``start_download`` — preflight repo size, install a tracking entry
  in ``state._downloads``, and spawn a background thread that runs
  ``snapshot_download`` while a sibling progress thread polls the
  filesystem to update bytes-downloaded → progress percent.
* ``download_status`` — list current download entries (UI poll).
* ``cancel_download`` — flip the cancel flag, terminate the live
  process, refresh the byte count, mark the entry ``cancelled``.
* ``delete_download`` — cancel + unload from runtimes + ``rmtree``
  the repo cache dir + clear bookkeeping.
* ``loaded_model_matches_repo_cache`` — predicate used by
  ``unload_repo_from_runtimes`` to decide whether the active or warm
  model should be evicted before deleting the cache.
* ``unload_repo_from_runtimes`` — purge the repo from the LLM
  runtime, image runtime, video runtime, and the warm engine pool.

All take the ``ChaosEngineState`` instance as the first argument so
the class methods stay 1-3 line wrappers. The ``_download_worker``
inner thread is closed over ``(state, repo, allow_patterns,
download_token, validation_error_fn)`` instead of ``self``.

Extracted as part of the v0.8.0 Phase 1a-10 refactor.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fastapi import HTTPException

from backend_service.helpers.formatting import _bytes_to_gb
from backend_service.helpers.huggingface import (
    _HF_REPO_PATTERN,
    _hf_repo_cache_dir,
    _known_repo_size_gb,
)
from backend_service.helpers.images import (
    _friendly_image_download_error,
    _image_download_validation_error,
    _image_repo_allow_patterns,
)
from backend_service.helpers.video import (
    _video_download_validation_error,
    _video_repo_allow_patterns,
)
from backend_service.state._helpers import _read_text_tail, _spawn_snapshot_download


if TYPE_CHECKING:
    from backend_service.state import ChaosEngineState


def start_download(
    state: ChaosEngineState,
    repo: str,
    allow_patterns: list[str] | None = None,
    validation_error_fn: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    from backend_service.helpers.huggingface import (
        _friendly_hf_download_error,
        _hf_repo_downloaded_bytes,
        _hf_repo_preflight_size_gb,
    )

    if not _HF_REPO_PATTERN.match(repo):
        raise HTTPException(
            status_code=400,
            detail="Invalid repo format. Expected 'owner/model-name'.",
        )
    if repo in state._downloads and state._downloads[repo].get("state") == "downloading":
        return state._downloads[repo]

    total_gb = _known_repo_size_gb(repo)
    downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
    try:
        preflight_total_gb = _hf_repo_preflight_size_gb(repo)
    except Exception as exc:
        friendly_error = _friendly_image_download_error(
            repo,
            _friendly_hf_download_error(repo, str(exc)),
        )
        failed_status = {
            "repo": repo,
            "state": "failed",
            "progress": 0.0,
            "downloadedGb": downloaded_gb,
            "totalGb": total_gb,
            "error": friendly_error,
        }
        with state._lock:
            state._downloads[repo] = failed_status
            state._download_cancel.pop(repo, None)
            state._download_processes.pop(repo, None)
            state._download_tokens.pop(repo, None)
        state.add_log("library", "error", f"Download preflight failed for {repo}: {friendly_error}")
        return failed_status
    if isinstance(preflight_total_gb, (int, float)) and preflight_total_gb > 0:
        total_gb = float(preflight_total_gb)

    initial_progress = 0.0
    if isinstance(total_gb, (int, float)) and total_gb > 0 and downloaded_gb > 0:
        initial_progress = min(0.99, downloaded_gb / float(total_gb))
    elif downloaded_gb > 0:
        initial_progress = 0.01
    download_token = uuid.uuid4().hex
    state._downloads[repo] = {
        "repo": repo,
        "state": "downloading",
        "progress": initial_progress,
        "downloadedGb": downloaded_gb,
        "totalGb": total_gb,
        "error": None,
    }
    state._download_cancel[repo] = False
    state._download_tokens[repo] = download_token
    state.add_log(
        "library",
        "info",
        f"{'Resuming' if downloaded_gb > 0 else 'Starting'} download: {repo}",
    )

    def _download_worker():
        stop_progress = threading.Event()
        process: subprocess.Popen[str] | None = None
        process_log_path: str | None = None

        def _progress_worker() -> None:
            while not stop_progress.wait(1.0):
                downloaded_bytes = _hf_repo_downloaded_bytes(repo)
                downloaded_gb_local = _bytes_to_gb(downloaded_bytes)
                with state._lock:
                    current = state._downloads.get(repo)
                    if (
                        current is None
                        or current.get("state") != "downloading"
                        or state._download_tokens.get(repo) != download_token
                    ):
                        return
                    current["downloadedGb"] = downloaded_gb_local
                    total = current.get("totalGb")
                    if isinstance(total, (int, float)) and total > 0:
                        current["progress"] = min(0.99, downloaded_gb_local / float(total))
                    elif downloaded_gb_local > 0:
                        current["progress"] = max(float(current.get("progress") or 0.0), 0.01)

        monitor = threading.Thread(target=_progress_worker, daemon=True)
        monitor.start()
        try:
            with state._lock:
                if state._download_tokens.get(repo) != download_token:
                    return
            env = os.environ.copy()
            env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            env.setdefault("PYTHONUNBUFFERED", "1")
            # Force the standard Hub download path for app-managed downloads.
            # Xet-backed transfers can keep most activity outside the per-repo
            # cache tree we use for progress tracking, which makes large repos
            # look permanently stuck at 0-1% in the UI.
            env["HF_HUB_DISABLE_XET"] = "1"
            temp_log = tempfile.NamedTemporaryFile(
                prefix="chaosengine-download-",
                suffix=".log",
                delete=False,
            )
            process_log_path = temp_log.name
            temp_log.close()
            with open(process_log_path, "w", encoding="utf-8", errors="replace") as process_log:
                # Diffusers repos (image + video) get a component-folder
                # allowlist so we skip legacy single-file checkpoints the
                # pipelines never load. Both helpers return None for repos
                # outside their catalog, so only one ever applies.
                effective_allow_patterns = allow_patterns or (
                    _video_repo_allow_patterns(repo)
                    or _image_repo_allow_patterns(repo)
                )
                process = _spawn_snapshot_download(
                    repo,
                    env,
                    process_log,
                    allow_patterns=effective_allow_patterns,
                )
                with state._lock:
                    if state._download_tokens.get(repo) == download_token:
                        state._download_processes[repo] = process

                while True:
                    with state._lock:
                        cancel_requested = state._download_cancel.get(repo, False)
                        token_matches = state._download_tokens.get(repo) == download_token
                    if not token_matches:
                        return
                    if cancel_requested:
                        if process.poll() is None:
                            try:
                                process.terminate()
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=5)
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.5)

            returncode = process.returncode if process.returncode is not None else process.wait()
            stderr_output = _read_text_tail(process_log_path)

            with state._lock:
                if state._download_tokens.get(repo) != download_token:
                    return
                cancelled = state._download_cancel.get(repo, False)
            if cancelled:
                downloaded_gb_local = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
                with state._lock:
                    current = state._downloads.get(repo)
                    if current is None or state._download_tokens.get(repo) != download_token:
                        return
                    current["state"] = "cancelled"
                    current["error"] = None
                    current["downloadedGb"] = downloaded_gb_local
                    total = current.get("totalGb")
                    if isinstance(total, (int, float)) and total > 0:
                        current["progress"] = min(0.99, downloaded_gb_local / float(total))
                    elif downloaded_gb_local > 0:
                        current["progress"] = max(float(current.get("progress") or 0.0), 0.01)
                return

            if returncode != 0:
                raise RuntimeError(
                    stderr_output or f"snapshot_download exited with status {returncode}"
                )

            # Image catalog validation first; fall through to video so
            # a successful video download isn't flagged for missing image
            # shape. Each validator returns None for repos outside its
            # catalog.
            video_validation_error = (
                validation_error_fn(repo)
                if validation_error_fn is not None
                else _video_download_validation_error(repo)
            )
            validation_error = _image_download_validation_error(repo) or video_validation_error
            if validation_error:
                with state._lock:
                    if state._download_tokens.get(repo) != download_token:
                        return
                    state._downloads[repo]["state"] = "failed"
                    state._downloads[repo]["error"] = validation_error
                    state.add_log("library", "error", validation_error)
                return
            downloaded_gb_local = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
            with state._lock:
                if state._download_tokens.get(repo) != download_token:
                    return
                state._downloads[repo]["state"] = "completed"
                state._downloads[repo]["progress"] = 1.0
                state._downloads[repo]["downloadedGb"] = downloaded_gb_local
                if downloaded_gb_local > 0:
                    current_total = state._downloads[repo].get("totalGb")
                    if not isinstance(current_total, (int, float)) or current_total <= 0:
                        state._downloads[repo]["totalGb"] = downloaded_gb_local
                    else:
                        state._downloads[repo]["totalGb"] = max(
                            float(current_total), downloaded_gb_local
                        )
                state._library_cache = None
                state.add_log("library", "info", f"Download completed: {repo}")
        except Exception as exc:
            with state._lock:
                if state._download_tokens.get(repo) != download_token:
                    return
                state._downloads[repo]["state"] = "failed"
                friendly_error = _friendly_image_download_error(
                    repo,
                    _friendly_hf_download_error(repo, str(exc)),
                )
                state._downloads[repo]["error"] = friendly_error
                state.add_log(
                    "library", "error", f"Download failed for {repo}: {friendly_error}"
                )
        finally:
            stop_progress.set()
            monitor.join(timeout=1.0)
            with state._lock:
                if process is not None and state._download_processes.get(repo) is process:
                    state._download_processes.pop(repo, None)
                if (
                    state._download_tokens.get(repo) == download_token
                    and state._downloads.get(repo, {}).get("state") != "downloading"
                ):
                    state._download_tokens.pop(repo, None)
                    state._download_cancel.pop(repo, None)
            if process_log_path:
                try:
                    os.unlink(process_log_path)
                except OSError:
                    pass

    t = threading.Thread(target=_download_worker, daemon=True)
    t.start()
    return state._downloads[repo]


def download_status(state: ChaosEngineState) -> list[dict[str, Any]]:
    return list(state._downloads.values())


def loaded_model_matches_repo_cache(
    loaded: Any, repo: str, repo_cache_dir: Path
) -> bool:
    if loaded is None:
        return False
    if repo in {
        getattr(loaded, "ref", None),
        getattr(loaded, "runtimeTarget", None),
    }:
        return True
    path_value = getattr(loaded, "path", None)
    if not path_value:
        return False
    try:
        resolved = Path(str(path_value)).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    try:
        return resolved == repo_cache_dir or resolved.is_relative_to(repo_cache_dir)
    except AttributeError:
        return False


def unload_repo_from_runtimes(
    state: ChaosEngineState, repo: str, repo_cache_dir: Path
) -> None:
    try:
        loaded = getattr(state.runtime, "loaded_model", None)
        if loaded_model_matches_repo_cache(loaded, repo, repo_cache_dir):
            state.runtime.unload_model()
    except Exception:
        pass

    try:
        state.image_runtime.unload(repo)
    except Exception:
        pass

    try:
        state.video_runtime.unload(repo)
    except Exception:
        pass

    warm_pool = getattr(state.runtime, "_warm_pool", None)
    if not isinstance(warm_pool, dict):
        return
    pool_lock = getattr(state.runtime, "_pool_lock", None)
    pool_context = (
        pool_lock
        if hasattr(pool_lock, "__enter__") and hasattr(pool_lock, "__exit__")
        else nullcontext()
    )
    with pool_context:
        stale_keys = [
            key
            for key, (_engine, info) in warm_pool.items()
            if loaded_model_matches_repo_cache(info, repo, repo_cache_dir)
        ]
        for key in stale_keys:
            engine, _info = warm_pool.pop(key)
            try:
                engine.unload_model()
            except Exception:
                pass


def cancel_download(state: ChaosEngineState, repo: str) -> dict[str, Any]:
    from backend_service.helpers.huggingface import _hf_repo_downloaded_bytes

    with state._lock:
        current = state._downloads.get(repo)
        if current is None:
            return {"repo": repo, "state": "not_found"}
        if current.get("state") == "completed":
            return current
        state._download_cancel[repo] = True
        process = state._download_processes.get(repo)

    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass

    downloaded_gb = _bytes_to_gb(_hf_repo_downloaded_bytes(repo))
    with state._lock:
        current = state._downloads.get(repo)
        if current is None:
            return {"repo": repo, "state": "not_found"}
        current["state"] = "cancelled"
        current["error"] = None
        current["downloadedGb"] = downloaded_gb
        total = current.get("totalGb")
        if isinstance(total, (int, float)) and total > 0:
            current["progress"] = min(0.99, downloaded_gb / float(total))
        elif downloaded_gb > 0:
            current["progress"] = max(float(current.get("progress") or 0.0), 0.01)
        state.add_log("library", "info", f"Download paused: {repo}")
        return current
    return {"repo": repo, "state": "not_found"}


def delete_download(state: ChaosEngineState, repo: str) -> dict[str, Any]:
    if not _HF_REPO_PATTERN.match(repo):
        raise HTTPException(
            status_code=400,
            detail="Invalid repo format. Expected 'owner/model-name'.",
        )

    repo_cache_dir = _hf_repo_cache_dir(repo)
    with state._lock:
        current = state._downloads.get(repo)
        process = state._download_processes.get(repo)
        if current is not None:
            state._download_cancel[repo] = True

    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass

    with state._lock:
        unload_repo_from_runtimes(state, repo, repo_cache_dir)

        removed_local_data = False
        try:
            if repo_cache_dir.exists():
                import shutil as _shutil

                _shutil.rmtree(repo_cache_dir)
                removed_local_data = True
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete cached download for {repo}: {exc}",
            ) from exc

        removed_status = repo in state._downloads
        state._downloads.pop(repo, None)
        state._download_cancel.pop(repo, None)
        state._download_processes.pop(repo, None)
        state._download_tokens.pop(repo, None)
        state._library_cache = None

        if removed_local_data or removed_status:
            action = "download data" if removed_local_data else "download record"
            state.add_log("library", "info", f"Deleted {action}: {repo}")
            return {"repo": repo, "state": "deleted"}

        return {"repo": repo, "state": "not_found"}
