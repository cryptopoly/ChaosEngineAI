"""MCP progress notifications (A7): a short event stream, only when the client sent a ``progressToken``.

Without a token ``/mcp`` answers one plain JSON body exactly as before. With a token on a ``tools/call``,
the answer is ``text/event-stream``: zero or more ``notifications/progress`` events, then the one
JSON-RPC response, then the stream ends. Only ``wait_for_job`` reports progress (it is the call that
waits); ``report`` is a no-op everywhere else.
"""

from __future__ import annotations

import contextvars
import json
import queue
import threading
from typing import Any, Callable, Iterator

Reporter = Callable[[float, float | None, str | None], None]
PROGRESS: contextvars.ContextVar[Reporter | None] = contextvars.ContextVar("mcp_progress", default=None)


def token_of(message: Any) -> Any:
    """The ``progressToken`` of a tools/call, or None."""
    if not isinstance(message, dict) or message.get("method") != "tools/call":
        return None
    meta = (message.get("params") or {}).get("_meta")
    token = meta.get("progressToken") if isinstance(meta, dict) else None
    return token if isinstance(token, (str, int)) and not isinstance(token, bool) else None


def report(progress: float, total: float | None = None, message: str | None = None) -> None:
    reporter = PROGRESS.get()
    if reporter is not None:
        reporter(progress, total, message)


def _event(payload: dict[str, Any]) -> bytes:
    return b"event: message\ndata: " + json.dumps(payload, default=str).encode("utf-8") + b"\n\n"


def stream(handle: Callable[[Any, Any], dict[str, Any] | None], message: dict[str, Any], request: Any, token: Any) -> Iterator[bytes]:
    """Run ``handle(message, request)`` in a thread; yield its progress events and then the response."""
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    last = [-1.0]

    def reporter(progress: float, total: float | None, text: str | None) -> None:
        if progress <= last[0]:  # progress must rise with every notification
            progress = last[0] + 0.001
        last[0] = progress
        params: dict[str, Any] = {"progressToken": token, "progress": progress}
        if total is not None:
            params["total"] = total
        if text:
            params["message"] = text
        events.put(("event", {"jsonrpc": "2.0", "method": "notifications/progress", "params": params}))

    def work() -> None:
        PROGRESS.set(reporter)
        try:
            events.put(("final", handle(message, request)))
        except Exception as exc:  # noqa: BLE001 - always end the stream with an answer
            events.put(("final", {"jsonrpc": "2.0", "id": message.get("id"),
                                  "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}))

    threading.Thread(target=work, name="mcp-progress", daemon=True).start()
    while True:
        kind, payload = events.get()
        if payload is not None:
            yield _event(payload)
        if kind == "final":
            return
