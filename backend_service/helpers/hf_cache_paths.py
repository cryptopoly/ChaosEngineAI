"""Hugging Face cache filesystem helpers.

Resolve the platform's HF cache root, build per-repo cache directories
matching the ``models--owner--name`` layout, sum the on-disk download
size, and pick the active snapshot directory (``refs/main`` revision
when available, falling back to the most-recent snapshot).

Extracted from ``backend_service/helpers/huggingface.py`` as part of
the v0.8.0 refactor. Re-exported from ``helpers.huggingface`` so
existing ``from backend_service.helpers.huggingface import _hf_repo_snapshot_dir``
imports keep working.
"""

from __future__ import annotations

import os
from pathlib import Path


def _hf_hub_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(os.path.expanduser(explicit)).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(os.path.expanduser(hf_home)).expanduser() / "hub"
    # Use huggingface_hub's own cache constant when available -- it handles
    # platform differences (Windows uses LOCALAPPDATA or userprofile).
    try:
        from huggingface_hub import constants as _hf_constants
        return Path(_hf_constants.HF_HUB_CACHE)
    except Exception:
        pass
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return _hf_hub_cache_root() / f"models--{repo_id.replace('/', '--')}"


def _hf_repo_downloaded_bytes(repo_id: str) -> int:
    from backend_service.helpers.discovery import _path_size_bytes

    cache_dir = _hf_repo_cache_dir(repo_id)
    try:
        if not cache_dir.exists():
            return 0
    except OSError:
        return 0
    try:
        return _path_size_bytes(cache_dir)
    except OSError:
        return 0


def _hf_repo_snapshot_dir(repo_id: str) -> Path | None:
    cache_dir = _hf_repo_cache_dir(repo_id)
    snapshots_dir = cache_dir / "snapshots"
    ref_path = cache_dir / "refs" / "main"
    try:
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                candidate = snapshots_dir / revision
                if candidate.exists():
                    return candidate
    except OSError:
        pass

    try:
        snapshots = sorted(
            [candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return snapshots[0] if snapshots else None
