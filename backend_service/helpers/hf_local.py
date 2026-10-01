"""Resolve a Hugging Face repo to its cached snapshot without touching the network."""

from __future__ import annotations

from typing import Any


def local_snapshot_path(repo_id: str, **kwargs: Any) -> str:
    """``snapshot_download(local_files_only=True)`` that accepts a filtered cache.

    Recent ``huggingface_hub`` releases raise ``IncompleteSnapshotError`` when the
    cached file listing names files that are not on disk. The app's download
    allow-lists skip such files on purpose (duplicate single-file checkpoints,
    sample images), so the error carries the usable snapshot path and callers
    check the files they actually need.
    """
    from huggingface_hub import snapshot_download  # type: ignore

    try:
        return snapshot_download(repo_id=repo_id, local_files_only=True, **kwargs)
    except Exception as exc:
        snapshot_path = getattr(exc, "snapshot_path", None)
        if type(exc).__name__ == "IncompleteSnapshotError" and snapshot_path:
            return str(snapshot_path)
        raise
