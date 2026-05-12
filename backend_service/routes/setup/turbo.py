"""llama-server-turbo update-check endpoint.

Reads ``~/.chaosengine/bin/llama-server-turbo.version`` (3-line file:
commit / branch / build-date written by the build-* scripts) and
compares against the live HEAD of the upstream branch via
``git ls-remote``. Result is hour-cached so the Setup tab can poll
without rate-limiting GitHub.

Extracted from ``routes/setup/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter()


_CHAOSENGINE_BIN_DIR = Path.home() / ".chaosengine" / "bin"

_TURBO_REPO = "https://github.com/TheTom/llama-cpp-turboquant.git"
_TURBO_BRANCH = "feature/turboquant-kv-cache"
_TURBO_VERSION_FILE = _CHAOSENGINE_BIN_DIR / "llama-server-turbo.version"

# Cache the remote HEAD for an hour. Setup-tab polls roughly per-minute
# while the user has it open, and ``git ls-remote`` against GitHub takes
# ~600ms cold + counts toward unauthenticated rate limits.
_turbo_remote_cache: tuple[str | None, float] = (None, 0.0)
_TURBO_REMOTE_CACHE_TTL = 3600.0  # 1 hour


def _read_turbo_version() -> tuple[str | None, str | None, str | None]:
    """Read the installed turbo binary version file.

    Returns ``(commit_hash, branch, build_date)`` or ``(None, None, None)``.
    """
    if not _TURBO_VERSION_FILE.exists():
        return None, None, None
    try:
        lines = _TURBO_VERSION_FILE.read_text().strip().splitlines()
        commit = lines[0].strip() if len(lines) > 0 else None
        branch = lines[1].strip() if len(lines) > 1 else None
        build_date = lines[2].strip() if len(lines) > 2 else None
        return commit, branch, build_date
    except OSError:
        return None, None, None


def _fetch_turbo_remote_head() -> str | None:
    """Get the latest commit hash on the turbo fork's branch via ``git ls-remote``.

    Results are cached for 1 hour to avoid excessive GitHub API calls.
    """
    global _turbo_remote_cache
    cached_hash, cached_at = _turbo_remote_cache
    if cached_hash is not None and (time.time() - cached_at) < _TURBO_REMOTE_CACHE_TTL:
        return cached_hash

    try:
        result = subprocess.run(
            ["git", "ls-remote", _TURBO_REPO, f"refs/heads/{_TURBO_BRANCH}"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            remote_hash = result.stdout.strip().split()[0]
            _turbo_remote_cache = (remote_hash, time.time())
            return remote_hash
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cached_hash  # return stale cache on failure


@router.get("/api/setup/turbo-update-check")
def turbo_update_check() -> dict[str, Any]:
    """Check if llama-server-turbo is installed and whether an update is available."""
    installed_commit, branch, build_date = _read_turbo_version()
    installed = _TURBO_VERSION_FILE.exists() and (_CHAOSENGINE_BIN_DIR / "llama-server-turbo").exists()

    remote_commit = _fetch_turbo_remote_head() if installed else None
    update_available = (
        installed
        and installed_commit is not None
        and remote_commit is not None
        and installed_commit != remote_commit
    )

    return {
        "installed": installed,
        "installedCommit": installed_commit[:12] if installed_commit else None,
        "remoteCommit": remote_commit[:12] if remote_commit else None,
        "updateAvailable": update_available,
        "branch": branch,
        "buildDate": build_date,
    }
