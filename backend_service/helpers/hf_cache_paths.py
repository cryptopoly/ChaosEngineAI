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


def hub_dir_for_home(hf_home: Path) -> Path:
    """The hub directory for a chosen cache folder.

    Huggingface_hub appends ``hub`` to ``HF_HOME``. A folder the user picked
    that already holds ``models--*`` repos (and has no ``hub`` folder) is the
    hub itself, so use it as is instead of looking in an empty ``hub``.
    """
    hub = hf_home / "hub"
    try:
        if hub.is_dir():
            return hub
        if hf_home.is_dir() and any(p.name.startswith("models--") for p in hf_home.iterdir()):
            return hf_home
    except OSError:
        pass
    return hub


def _hf_hub_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(os.path.expanduser(explicit)).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return hub_dir_for_home(Path(os.path.expanduser(hf_home)).expanduser())
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


def _hf_repo_downloaded_bytes(repo_id: str, allow_patterns: list[str] | None = None) -> int:
    """Bytes of ``repo_id`` in the cache.

    With ``allow_patterns`` (a download that fetches only some files), count
    just the snapshot files those patterns select plus in-progress blobs, so a
    repo that also holds other quants does not show more than its own total.
    """
    from backend_service.helpers.discovery import _path_size_bytes

    cache_dir = _hf_repo_cache_dir(repo_id)
    if allow_patterns:
        return _matching_cache_bytes(cache_dir, allow_patterns)
    try:
        if not cache_dir.exists():
            return 0
    except OSError:
        return 0
    try:
        every, live = _incomplete_sizes(cache_dir)
        return max(0, _path_size_bytes(cache_dir) - every + live)
    except OSError:
        return 0


def _incomplete_sizes(cache_dir: Path) -> tuple[int, int]:
    """(all ``.incomplete`` bytes, bytes of the largest ``.incomplete`` part per blob).

    A download that was stopped and restarted leaves older partial files for the same blob next to the new
    one; only the largest is the live copy, so the progress bar must not add the stale ones.
    """
    every = 0
    largest: dict[str, int] = {}
    try:
        for blob in (cache_dir / "blobs").glob("*.incomplete"):
            try:
                size = blob.stat().st_size
            except OSError:
                continue
            key = blob.name.split(".", 1)[0]
            if os.path.lexists(cache_dir / "blobs" / key):
                # The blob finished (or links to the shared store): this partial is a dead leftover.
                continue
            every += size
            largest[key] = max(largest.get(key, 0), size)
    except OSError:
        pass
    return every, sum(largest.values())


def prune_stale_partials(cache_dir: Path) -> int:
    """Delete ``.incomplete`` files nothing can resume; return the bytes freed.

    Call it only while no download of this repo runs. A partial whose blob already finished is dead; of
    several partials for one blob only the largest is kept, because the resuming download takes the largest
    (FU-275).
    """
    groups: dict[str, list[tuple[int, Path]]] = {}
    try:
        for blob in (cache_dir / "blobs").glob("*.incomplete"):
            try:
                groups.setdefault(blob.name.split(".", 1)[0], []).append((blob.stat().st_size, blob))
            except OSError:
                continue
    except OSError:
        return 0
    freed = 0
    for key, parts in groups.items():
        parts.sort(key=lambda item: item[0], reverse=True)
        doomed = parts if os.path.lexists(cache_dir / "blobs" / key) else parts[1:]
        for size, path in doomed:
            try:
                path.unlink()
                freed += size
            except OSError:
                continue
    return freed


def _matching_cache_bytes(cache_dir: Path, allow_patterns: list[str]) -> int:
    import fnmatch

    seen: set[str] = set()
    total = 0
    snapshots = cache_dir / "snapshots"
    try:
        for entry in snapshots.rglob("*"):
            relative = entry.relative_to(snapshots).as_posix().split("/", 1)
            name = relative[1] if len(relative) > 1 else relative[0]
            if not any(fnmatch.fnmatch(name, pattern) for pattern in allow_patterns):
                continue
            try:
                real = entry.resolve()
                if not real.is_file() or str(real) in seen:
                    continue
                seen.add(str(real))
                total += real.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total + _incomplete_sizes(cache_dir)[1]


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
