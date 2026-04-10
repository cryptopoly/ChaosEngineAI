"""HuggingFace model download management service."""
from __future__ import annotations

from typing import Any


class DownloadService:
    """Thin facade over the download state held by ChaosEngineState.

    Delegates to the existing ``_downloads`` / ``_download_cancel`` /
    ``_download_processes`` dicts on the state object so that all current
    route handlers keep working unchanged.
    """

    def __init__(self, state: Any) -> None:
        self._state = state

    # -- Queries -------------------------------------------------------------

    def get_status(self, repo: str) -> dict[str, Any] | None:
        """Return download status dict for *repo*, or ``None``."""
        return self._state._downloads.get(repo)

    def get_all_downloads(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the full downloads dict."""
        return dict(self._state._downloads)

    @property
    def active_repos(self) -> list[str]:
        """Return repos that currently have an active download."""
        return [
            repo
            for repo, info in self._state._downloads.items()
            if info.get("status") == "downloading"
        ]

    # -- Mutations -----------------------------------------------------------

    def start_download(self, repo: str) -> dict[str, Any]:
        """Mark *repo* as downloading and return its status entry.

        The actual subprocess launch is still handled by the state method;
        this just initializes the tracking dict entry.
        """
        entry: dict[str, Any] = {
            "repo": repo,
            "status": "downloading",
            "progress": 0.0,
            "error": None,
        }
        self._state._downloads[repo] = entry
        self._state._download_cancel[repo] = False
        return entry

    def cancel_download(self, repo: str) -> bool:
        """Request cancellation for *repo*.  Returns True if it was active."""
        if repo not in self._state._downloads:
            return False
        self._state._download_cancel[repo] = True
        proc = self._state._download_processes.get(repo)
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        status = self._state._downloads.get(repo)
        if status:
            status["status"] = "cancelled"
        return True

    def mark_complete(self, repo: str, *, error: str | None = None) -> None:
        """Update a download entry to finished (or failed) status."""
        entry = self._state._downloads.get(repo)
        if entry is None:
            return
        if error:
            entry["status"] = "error"
            entry["error"] = error
        else:
            entry["status"] = "complete"
            entry["progress"] = 100.0

    def remove(self, repo: str) -> None:
        """Clean up all tracking state for *repo*."""
        self._state._downloads.pop(repo, None)
        self._state._download_cancel.pop(repo, None)
        self._state._download_processes.pop(repo, None)
        self._state._download_tokens.pop(repo, None)
