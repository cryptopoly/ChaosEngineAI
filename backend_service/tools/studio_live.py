"""The in-progress reply of a running Studio turn, for a window that does not stream it (FU-208).

A Studio turn runs on its own thread and the response only relays its SSE frames (S-J). A Keep going nudge has
no response at all, so the open chat showed only "This reply is still being made" until the turn ended. Each
running turn now feeds its frames to a ``LiveTurn`` as well: the same events the page reads from the stream
(token, reasoning, tool call start, progress and result) folded into one partial assistant message. The
detached-turn poll (``GET /api/chat/studio/sessions/{id}/turn``) returns it as ``partial``, so the page draws
the same text and tool cards as for a turn it streams. Nothing is saved: the thread gets the full reply when
the turn ends.
"""

from __future__ import annotations

import copy
import json
import threading
from typing import Any

MAX_TEXT_CHARS = 20_000  # the tail of a long reply; the saved message has all of it
MAX_TOOL_CALLS = 60

_LOCK = threading.Lock()
_LIVE: dict[str, "LiveTurn"] = {}


class LiveTurn:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.text = ""
        self.reasoning = ""
        self.tool_calls: list[dict[str, Any]] = []

    def feed(self, frame: str) -> None:
        """Fold one SSE frame (``data: {...}``) into the partial reply; comments and other frames are ignored."""
        if not frame.startswith("data: "):
            return
        try:
            event = json.loads(frame[6:])
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        with self._lock:
            if isinstance(event.get("token"), str):
                self.text = (self.text + event["token"])[-MAX_TEXT_CHARS:]
            if isinstance(event.get("reasoning"), str):
                self.reasoning = (self.reasoning + event["reasoning"])[-MAX_TEXT_CHARS:]
            start = event.get("toolCallStart")
            if isinstance(start, dict):
                self._upsert({"id": str(start.get("id") or ""), "name": str(start.get("name") or "tool"),
                              "arguments": _arguments(start.get("arguments")),
                              "result": "", "elapsed": 0, "startedAt": start.get("startedAt"), "running": True})
            progress = event.get("toolCallProgress")
            if isinstance(progress, dict) and isinstance(progress.get("data"), dict):
                call = self._find(str(progress.get("id") or ""))
                if call is not None and call.get("running"):
                    call["renderAs"] = "studio"
                    call["data"] = {**(call.get("data") or {}), **progress["data"]}
            result = event.get("toolCallResult")
            if isinstance(result, dict):
                self._upsert({"id": str(result.get("id") or ""), "name": str(result.get("name") or "tool"),
                              "arguments": _arguments(result.get("arguments")),
                              "result": str(result.get("result") or ""), "elapsed": result.get("elapsed") or 0,
                              "startedAt": result.get("startedAt"), "renderAs": result.get("renderAs"),
                              "data": result.get("data")})

    def _find(self, call_id: str) -> dict[str, Any] | None:
        return next((c for c in self.tool_calls if c["id"] == call_id), None)

    def _upsert(self, call: dict[str, Any]) -> None:
        found = self._find(call["id"])
        if found is None:
            self.tool_calls.append(call)
            del self.tool_calls[:-MAX_TOOL_CALLS]
            return
        started = call.get("startedAt") if call.get("startedAt") is not None else found.get("startedAt")
        found.clear()
        found.update(call, startedAt=started)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"text": self.text, "reasoning": self.reasoning, "toolCalls": copy.deepcopy(self.tool_calls)}


def _arguments(value: Any) -> dict[str, Any]:
    """A call's arguments as an object; a start event may carry them as JSON text (the page parses it the same way)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def begin(session_id: str) -> LiveTurn:
    live = LiveTurn()
    with _LOCK:
        _LIVE[session_id] = live
    return live


def end(session_id: str, live: LiveTurn) -> None:
    with _LOCK:
        if _LIVE.get(session_id) is live:
            del _LIVE[session_id]


def partial(session_id: str) -> dict[str, Any] | None:
    """The running turn's reply so far, or None when no turn of this chat runs."""
    with _LOCK:
        live = _LIVE.get(session_id)
    return None if live is None else live.snapshot()
