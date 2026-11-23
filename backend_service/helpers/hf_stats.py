"""Hugging Face likes and downloads for the recommended chat rows ("Most popular" pick).

The numbers come from the public model API, are kept in ``<dataDir>/hf-stats.json`` with the date they were
fetched, and are refreshed at most once a day by a background thread. Nothing here ever waits on the network:
``load_stats`` only reads the file, and with no file (offline, first run) the rows simply carry no numbers and
the pick is not offered.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

STATS_FILENAME = "hf-stats.json"
STALE_AFTER_SECONDS = 24 * 3600
_FIELDS = ("likes", "downloads", "downloadsAllTime")
_refresh_lock = threading.Lock()


def default_path() -> Path:
    from backend_service.app import DATA_LOCATION

    return DATA_LOCATION.data_dir / STATS_FILENAME


def load_stats(path: Path | None = None) -> dict[str, Any]:
    """``{"fetchedAt": iso | None, "repos": {repo: {likes, downloads, downloadsAllTime}}}``; never raises."""
    try:
        raw = json.loads((path or default_path()).read_text(encoding="utf-8"))
        repos = raw.get("repos")
        if isinstance(repos, dict):
            return {"fetchedAt": raw.get("fetchedAt"), "repos": repos}
    except Exception:
        pass
    return {"fetchedAt": None, "repos": {}}


def fetch_repo_stats(repo: str, timeout: float = 6.0) -> dict[str, int] | None:
    """One repo's counters from the Hub, or None when it cannot be read."""
    query = "expand[]=likes&expand[]=downloads&expand[]=downloadsAllTime"
    url = f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}?{query}"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return {field: int(data.get(field) or 0) for field in _FIELDS}


def _is_fresh(stats: dict[str, Any], repos: list[str], now: datetime) -> bool:
    try:
        age = (now - datetime.fromisoformat(str(stats["fetchedAt"]))).total_seconds()
    except Exception:
        return False
    return age < STALE_AFTER_SECONDS and all(repo in stats["repos"] for repo in repos)


def refresh_stats(
    repos: list[str],
    path: Path | None = None,
    *,
    fetch: Callable[[str], dict[str, int] | None] = fetch_repo_stats,
    now: datetime | None = None,
) -> bool:
    """Fetch ``repos`` and merge them into the file. Returns False when nothing could be fetched."""
    target = path or default_path()
    stats = load_stats(target)
    fetched = {repo: values for repo in repos if (values := fetch(repo)) is not None}
    if not fetched:
        return False
    stats["repos"].update(fetched)
    stats["fetchedAt"] = (now or datetime.now(timezone.utc)).isoformat()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(stats), encoding="utf-8")
    temp.replace(target)
    return True


def refresh_in_background(repos: list[str], path: Path | None = None) -> bool:
    """Start a daemon thread that refreshes stale stats. Returns True when a thread was started.

    Skipped when the file is fresh, when the Hub is switched off (``HF_HUB_OFFLINE``) or when a refresh is
    already running.
    """
    repos = [repo for repo in dict.fromkeys(repos) if repo]
    if not repos or os.environ.get("HF_HUB_OFFLINE") == "1" or "PYTEST_CURRENT_TEST" in os.environ:
        return False
    target = path or default_path()
    if _is_fresh(load_stats(target), repos, datetime.now(timezone.utc)):
        return False
    if not _refresh_lock.acquire(blocking=False):
        return False

    def run() -> None:
        try:
            refresh_stats(repos, target)
        except Exception:
            logger.debug("Hugging Face stats refresh failed", exc_info=True)
        finally:
            _refresh_lock.release()

    threading.Thread(target=run, name="hf-stats-refresh", daemon=True).start()
    return True
