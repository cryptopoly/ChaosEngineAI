from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import threading
from typing import Any

from fastapi import APIRouter, Request
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
            # Send recent logs first (skip debug noise)
            for entry in reversed(list(state.logs)[-50:]):
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
