"""Deleting voice models from disk: a speech model with its companion repos, or a dictation model (D-73 M1, M2).

A speech model is its pinned repo plus the companion repos its loader needs (``catalog/voice_models.py``). Delete
removes the model's repo from the Hugging Face cache and each companion that no other downloaded speech model still
uses; the watermark component (AudioSeal weights) stays while any downloaded model can clone. A dictation model is
one repo, removed through the generic download delete so its download record goes too.

Every delete is refused while the speech worker is answering a request or a voice job runs (the model may be loading
or speaking) and while a speech model downloads; otherwise the worker is stopped first when it holds the model, so
no file is removed under a loaded model. Paths are resolved and must sit directly in the Hugging Face cache. Cloned
voices and Library items are never touched.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelInUse(RuntimeError):
    """The model cannot be deleted right now (in use, loading or downloading); the message says why."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_cache_dir(repo: str) -> Path:
    """The repo's folder in the Hugging Face cache, checked: a valid repo id, resolved, directly in the cache root."""
    from backend_service.helpers.hf_cache_paths import _hf_hub_cache_root, _hf_repo_cache_dir  # noqa: PLC0415

    if not _REPO.match(repo or "") or ".." in repo:
        raise ValueError(f"Invalid repo id {repo!r}.")
    root = _hf_hub_cache_root().resolve()
    folder = _hf_repo_cache_dir(repo).resolve()
    if folder.parent != root or not folder.name.startswith("models--"):
        raise ValueError(f"{repo} does not resolve to a folder in the Hugging Face cache.")
    return folder


def _size_gb(folder: Path) -> float:
    from backend_service.helpers.discovery import _path_size_bytes  # noqa: PLC0415

    try:
        return round(_path_size_bytes(folder) / 1e9, 2) if folder.exists() else 0.0
    except OSError:
        return 0.0


def _remove_repo(repo: str) -> float:
    """Remove the repo's cache folder; returns the GB freed (0 when nothing was there)."""
    folder = repo_cache_dir(repo)
    if not folder.exists():
        return 0.0
    size = _size_gb(folder)
    shutil.rmtree(folder)
    return size


# ---------------------------------------------------------------------------
# In use
# ---------------------------------------------------------------------------

def voice_busy_reason() -> str | None:
    """Why voice files cannot change now, or None. A job in the voice queue or a request in the speech worker."""
    from backend_service import voice_client  # noqa: PLC0415
    from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415

    if HEAVY_JOBS.running_source("voice") is not None:
        return "Voice is working on a job (speech or a transcript). Try again when it has finished."
    if voice_client.worker_busy():
        return "A voice model is loading or speaking (read-aloud, a call or Speak). Try again when it has finished."
    return None


def _holds(source: str, repos: list[str]) -> bool:
    """Whether a worker ``source`` (repo id or snapshot folder) is one of ``repos``."""
    for repo in repos:
        if source == repo or f"models--{repo.replace('/', '--')}" in source:
            return True
    return False


def _stop_worker_if_holding(repos: list[str]) -> bool:
    from backend_service import voice_client  # noqa: PLC0415

    held = voice_client.loaded_sources()
    if any(_holds(source, repos) for source in held.values()):
        voice_client.stop_speech_worker()
        return True
    return False


# ---------------------------------------------------------------------------
# Speech models
# ---------------------------------------------------------------------------

def _on_disk(row: dict[str, Any]) -> bool:
    """The row's own weights are downloaded (companions aside)."""
    from backend_service import voice_clone  # noqa: PLC0415

    return voice_clone.weights_ready(voice_clone._revision_dir(row["repo"], row["revision"]), row.get("files"))


def _repos_of(row: dict[str, Any]) -> list[str]:
    return [row["repo"], *(c["repo"] for c in row.get("companions", []))]


def removal_plan(key: str) -> dict[str, Any]:
    """What deleting ``key`` removes and keeps. Raises ValueError for an unknown key."""
    from backend_service import voice_env  # noqa: PLC0415
    from backend_service.catalog import voice_models as catalog  # noqa: PLC0415

    row = catalog.voice_model(key)
    others = [r for r in catalog.VOICE_MODELS if r["key"] != row["key"] and _on_disk(r)]
    remove: list[str] = []
    kept: list[dict[str, Any]] = []
    for repo in dict.fromkeys(_repos_of(row)):
        users = [r["name"] for r in others if repo in _repos_of(r)]
        if users:
            kept.append({"repo": repo, "usedBy": users})
        else:
            remove.append(repo)
    # The watermark stays while any other downloaded model can clone (cloned speech is always watermarked).
    watermark = bool(
        row["capabilities"].get("clone")
        and not any(r["capabilities"].get("clone") for r in others)
        and voice_env.watermark_model_paths() is not None
    )
    size = sum(_size_gb(repo_cache_dir(repo)) for repo in remove)
    return {"key": row["key"], "name": row["name"], "remove": remove, "kept": kept, "watermark": watermark,
            "sizeGb": round(size, 2)}


def remove_speech_model(key: str) -> dict[str, Any]:
    """Delete a speech model and the companions only it uses. Raises ValueError (unknown key) or ModelInUse."""
    from backend_service import voice_clone, voice_env  # noqa: PLC0415

    plan = removal_plan(key)
    reason = voice_busy_reason()
    if reason:
        raise ModelInUse(reason)
    if any(job.get("state") == "downloading" for job in voice_clone._DOWNLOADS.values()):
        raise ModelInUse("A speech model is downloading. Wait for it to finish before deleting a model.")
    stopped = _stop_worker_if_holding([*plan["remove"], *(k["repo"] for k in plan["kept"])])
    freed = 0.0
    removed: list[str] = []
    for repo in plan["remove"]:
        if repo_cache_dir(repo).exists():
            freed += _remove_repo(repo)
            removed.append(repo)
    watermark_removed = False
    if plan["watermark"]:
        freed += _remove_repo(voice_env.WATERMARK_REPO)
        watermark_removed = True
    voice_clone._DOWNLOADS.pop(plan["key"], None)
    return {
        "key": plan["key"],
        "deleted": True,
        "removed": removed,
        "kept": plan["kept"],
        "watermarkRemoved": watermark_removed,
        "workerStopped": stopped,
        "freedGb": round(freed, 2),
    }


# ---------------------------------------------------------------------------
# Dictation models
# ---------------------------------------------------------------------------

def stt_repo(key: str) -> str:
    """The repo of a listed dictation model, by catalog key or repo id. Raises ValueError for anything else."""
    from backend_service.voice_runtime import list_stt_models  # noqa: PLC0415

    wanted = (key or "").strip()
    for model in list_stt_models():
        if wanted and wanted in (model.get("key"), model.get("id")):
            return str(model["id"])
    raise ValueError(f"Unknown dictation model '{wanted}'.")


def prepare_stt_delete(key: str) -> str:
    """Check a dictation model may be deleted now and stop the worker if it holds it; returns the repo to delete."""
    repo = stt_repo(key)
    repo_cache_dir(repo)  # path check before anything changes
    reason = voice_busy_reason()
    if reason:
        raise ModelInUse(reason)
    _stop_worker_if_holding([repo])
    return repo
