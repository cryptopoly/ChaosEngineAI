"""Centralized log management service."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from threading import RLock
from typing import Any


class LogService:
    """Manages application logs with subscription support.

    Provides thread-safe log storage using a bounded deque
    and async subscription for real-time log streaming.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._lock = RLock()
        self._logs: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

    def add_log(
        self,
        source: str,
        level: str,
        message: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a log entry and notify all subscribers."""
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "source": source,
            "level": level,
            "message": message,
        }
        if extra:
            entry["extra"] = extra

        with self._lock:
            self._logs.append(entry)

        # Fan-out to async subscribers (best-effort, drop if full)
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                pass

        return entry

    def get_logs(
        self,
        *,
        limit: int | None = None,
        level: str | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return stored logs, optionally filtered."""
        with self._lock:
            items = list(self._logs)

        if level:
            items = [e for e in items if e["level"] == level]
        if source:
            items = [e for e in items if e["source"] == source]
        if limit and limit > 0:
            items = items[-limit:]
        return items

    def subscribe_logs(self, maxsize: int = 64) -> asyncio.Queue[dict[str, Any]]:
        """Create a new async queue that receives future log entries."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    def clear(self) -> None:
        """Remove all stored logs."""
        with self._lock:
            self._logs.clear()
