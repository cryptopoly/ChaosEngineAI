"""Log + activity buffers for ChaosEngineState.

Extracted from the monolithic ``state.py`` as part of the v0.8.0 refactor.
The public surface (``state.logs``, ``state.activity``, ``state.add_log``,
``state.subscribe_logs``, ``state.unsubscribe_logs``, ``state.add_activity``)
is preserved by the facade in ``state/__init__.py``.

Two ring buffers:
- ``logs`` — system / runtime / chat / server entries; max 120, polled by
  ``GET /api/server/logs/stream`` for the in-app log panel.
- ``activity`` — user-visible event ticker for the dashboard widget; max 60,
  shorter and curated for "what happened recently".

Subscribers register a ``queue.Queue`` via ``subscribe_logs()`` and get
every new log entry pushed to it. The streaming SSE endpoint owns the
queue's lifecycle and unsubscribes in its ``finally`` clause; ``put_nowait``
silently drops entries on a full queue rather than blocking the writer.
"""

from __future__ import annotations

import queue as _queue_mod
import time
from collections import deque
from typing import Any


def _time_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _relative_label() -> str:
    return time.strftime("%H:%M")


class LogManager:
    """Owns the two ring buffers + subscriber list."""

    def __init__(self, *, log_capacity: int = 120, activity_capacity: int = 60) -> None:
        self.logs: deque[dict[str, Any]] = deque(maxlen=log_capacity)
        self.activity: deque[dict[str, Any]] = deque(maxlen=activity_capacity)
        self._subscribers: list[_queue_mod.Queue] = []

    def add_log(self, source: str, level: str, message: str) -> None:
        entry = {
            "ts": _time_label(),
            "source": source,
            "level": level,
            "message": message,
        }
        self.logs.appendleft(entry)
        for q in self._subscribers:
            try:
                q.put_nowait(entry)
            except _queue_mod.Full:
                pass

    def subscribe(self) -> _queue_mod.Queue:
        q: _queue_mod.Queue = _queue_mod.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: _queue_mod.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def add_activity(self, title: str, detail: str) -> None:
        self.activity.appendleft(
            {
                "time": "Now",
                "title": title,
                "detail": detail,
            }
        )
