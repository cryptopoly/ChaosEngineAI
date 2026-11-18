from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import threading
from typing import Any

from fastapi import APIRouter, Query, Request
from starlette.responses import StreamingResponse

router = APIRouter()


@router.get("/api/server/status")
def server_status(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.server_status()


@router.post("/api/server/shutdown")
def shutdown_server(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    state.add_log("server", "info", "Shutdown requested via API.")
    # Schedule a graceful shutdown after responding
    def _delayed_shutdown():
        time.sleep(0.5)
        sig = signal.SIGTERM if hasattr(signal, "SIGTERM") else signal.SIGINT
        os.kill(os.getpid(), sig)
    threading.Thread(target=_delayed_shutdown, daemon=True).start()
    return {"status": "shutting_down"}


@router.get("/api/server/logs/stream")
async def stream_server_logs(request: Request):
    import queue as _queue_mod

    state = request.app.state.chaosengine
    q = state.subscribe_logs()

    async def event_stream():
        try:
            # Send the 50 newest lines first, oldest of them first (the ring is newest first; skip debug noise)
            for entry in reversed(list(state.logs)[:50]):
                if entry.get("level") == "debug":
                    continue
                yield f"data: {json.dumps(entry)}\n\n"
            # Then stream new entries by polling the thread-safe queue
            while True:
                try:
                    entry = q.get(block=False)
                    yield f"data: {json.dumps(entry)}\n\n"
                except _queue_mod.Empty:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            state.unsubscribe_logs(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })


def _model_event_log(request: Request):
    return getattr(request.app.state.chaosengine, "model_events", None)


@router.get("/api/server/events")
def server_events(
    request: Request,
    model: str | None = None,
    level: str | None = None,
    kind: str | None = None,
    requestId: str | None = None,
    since: int | None = None,
    limit: int = Query(default=500, ge=0, le=5000),
) -> dict[str, Any]:
    """Structured model events (D-88), oldest first: requested, loading, loaded, first token, generation, unloaded."""
    log = _model_event_log(request)
    if log is None:
        return {"events": [], "capacity": 0, "dropped": 0}
    events = log.events(model=model, level=level, kind=kind, request_id=requestId, since=since, limit=limit)
    return {"events": events, "capacity": log.capacity, "dropped": log.dropped}


@router.get("/api/server/events/stream")
async def stream_server_events(request: Request):
    """The newest 200 model events, then each new one as it happens (server-sent events)."""
    import queue as _queue_mod

    log = _model_event_log(request)

    async def event_stream():
        if log is None:
            yield ": no event log\n\n"
            return
        q = log.subscribe()
        try:
            for event in log.events(limit=200):
                yield f"data: {json.dumps(event)}\n\n"
            while True:
                try:
                    event = q.get(block=False)
                    yield f"data: {json.dumps(event)}\n\n"
                except _queue_mod.Empty:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            log.unsubscribe(q)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
