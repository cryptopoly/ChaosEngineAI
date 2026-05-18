#!/usr/bin/env python3
"""Stub MTPLX server for integration tests.

Mimics the surface MtplxEngine talks to:
  - ``mtplx start --model <path> --port N`` CLI shape
  - ``GET /health`` returns 200 OK
  - ``POST /v1/chat/completions`` returns OpenAI-compatible response
    (non-streaming JSON, or SSE when ``stream: true``)

Does not implement actual MTP speculative decoding — just enough surface to
prove the engine boots, proxies a prompt, and parses the response.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_FAKE_REPLY = "stub-mtplx says hi"
_FAKE_STREAM_CHUNKS = ["stub", "-mtplx", " says", " hi"]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args, **_kwargs) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "bad JSON")
            return

        if payload.get("stream"):
            self._stream_response(payload)
        else:
            self._json_response(payload)

    def _json_response(self, payload: dict) -> None:
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in payload.get("messages", []))
        completion_tokens = len(_FAKE_REPLY.split())
        body = json.dumps({
            "id": "stub-mtplx-1",
            "model": payload.get("model", "stub"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": _FAKE_REPLY},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream_response(self, payload: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in _FAKE_STREAM_CHUNKS:
            data = json.dumps({
                "id": "stub-mtplx-1",
                "model": payload.get("model", "stub"),
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            })
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in payload.get("messages", []))
        completion_tokens = len(_FAKE_STREAM_CHUNKS)
        final = json.dumps({
            "id": "stub-mtplx-1",
            "model": payload.get("model", "stub"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })
        self.wfile.write(f"data: {final}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _serve(port: int, *, fail_mode: str | None = None) -> None:
    if fail_mode == "crash-before-ready":
        sys.stderr.write("stub crash: simulated startup failure\n")
        sys.stderr.flush()
        sys.exit(2)
    if fail_mode == "delay":
        time.sleep(0.5)
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    # MtplxEngine now invokes ``mtplx quickstart``; ``start`` kept for
    # backwards-compat with older test fixtures that called the original
    # subcommand. Both behave identically here — just accept the flags
    # MtplxEngine emits and start the HTTP stub.
    for sub_name in ("quickstart", "start"):
        sp = sub.add_parser(sub_name)
        sp.add_argument("--model", required=True)
        sp.add_argument("--port", type=int, required=True)
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--mtp", action="store_true")
        sp.add_argument("--no-mtp", action="store_true")
        sp.add_argument("--depth", type=int, default=3)
        sp.add_argument("--profile", default=None)
        sp.add_argument("--max", action="store_true")
        sp.add_argument("--yes", action="store_true")
        sp.add_argument("--fail-mode", default=None)
    args = parser.parse_args()
    if args.cmd in {"quickstart", "start"}:
        _serve(args.port, fail_mode=args.fail_mode)


if __name__ == "__main__":
    main()
