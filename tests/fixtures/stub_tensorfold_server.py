#!/usr/bin/env python3
"""Stub ``tensorfold serve`` for integration tests.

Mimics the surface ``TensorFoldEngine`` talks to, with the wire shapes taken
from TensorFold 0.5.0's own server (``server/http.py``, ``server/app.py``):

  - the ``tensorfold serve <model dir> --host --port --name ...`` CLI shape
  - ``GET /health`` -> ``{"status": "ok", "model", "model_ids", "warming", ...}``
  - ``POST /v1/chat/completions``: JSON, or SSE (``role`` chunk -> ``content`` /
    ``reasoning_content`` / ``tool_calls`` deltas -> a final chunk carrying
    ``finish_reason``, ``usage``, ``tensorfold`` runtime stats and ``speculative``
    stats -> ``[DONE]``); a refusal is ``{"error": {...}}`` (HTTP 400 non-streamed,
    an in-stream payload once the stream has opened)

It does no speculative decoding. Every request body is appended, one JSON
line each, to ``$STUB_TENSORFOLD_RECORD`` so tests can assert what the engine
sent; the argv and the environment the server started with are written to
``$STUB_TENSORFOLD_RECORD.start``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_REPLY_CHUNKS = ["stub", "-tensorfold", " says", " hi"]
_CONTEXT_WINDOW = 1000

_STATE: dict = {"name": "stub", "drafts": True}


def _has_image(body: dict) -> bool:
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def _record(payload: dict) -> None:
    path = os.environ.get("STUB_TENSORFOLD_RECORD")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args, **_kwargs) -> None:
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0].rstrip("/")
        if route in {"", "/health"}:
            self._send_json({
                "status": "ok",
                "model": _STATE["name"],
                "model_ids": [_STATE["name"]],
                "max_batch_size": 8,
                "warming": False,
                "memory": {},
            })
            return
        self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)

    def do_POST(self) -> None:
        if not self.path.split("?", 1)[0].rstrip("/").endswith("/chat/completions"):
            self._send_json({"error": {"message": f"unknown path {self.path}"}}, status=404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": {"message": "bad JSON"}}, status=400)
            return
        _record(body)

        refusal = self._refusal(body)
        wants_stream = bool(body.get("stream"))
        if refusal and not wants_stream:
            self._send_json({"error": {"message": refusal, "type": "invalid_request_error"}}, status=400)
            return
        if wants_stream:
            self._stream(body, refusal)
        else:
            self._json(body)

    # -- behaviours -----------------------------------------------------

    @staticmethod
    def _refusal(body: dict) -> str | None:
        limit = int(body.get("max_tokens") or 0)
        if limit > _CONTEXT_WINDOW:
            return (
                f"the rendered prompt has 10 tokens and requests {limit} reply tokens; "
                f"this server's context window is {_CONTEXT_WINDOW}. Reduce the prompt to at most "
                f"{max(0, _CONTEXT_WINDOW - limit)} prompt tokens or request at most "
                f"{_CONTEXT_WINDOW - 10} reply tokens, including chat template and thinking tokens."
            )
        if _has_image(body) and not _STATE.get("vision"):
            return "image input is off on this server; start it with --vision"
        response_format = body.get("response_format")
        if response_format and os.environ.get("STUB_TENSORFOLD_GRAMMAR") != "1":
            return "response_format needs xgrammar, which this build does not include"
        return None

    @staticmethod
    def _last_user(body: dict) -> str:
        for message in reversed(body.get("messages") or []):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def _thinking(body: dict) -> bool:
        kwargs = body.get("chat_template_kwargs") or {}
        return kwargs.get("enable_thinking") is not False

    @staticmethod
    def _tool_call(body: dict) -> dict | None:
        if not body.get("tools") or "USE_TOOL" not in _Handler._last_user(body):
            return None
        name = body["tools"][0]["function"]["name"]
        return {"id": "call_0", "type": "function", "function": {"name": name, "arguments": '{"query": "cats"}'}}

    @staticmethod
    def _stats(body: dict, completion_tokens: int) -> dict:
        drafts = bool(_STATE["drafts"]) and body.get("draft") is not False
        extras: dict = {
            "exact_mode": "target-verified",
            "tensorfold": {
                "enable_thinking": _Handler._thinking(body),
                "tokens_per_second": 42.5,
                "seconds": 0.5,
                "time_to_first_token": 0.1,
                "drafts": drafts,
            },
        }
        if drafts:
            extras["speculative"] = {
                "rounds": 4,
                "drafted": 12,
                "accepted": 8,
                "acceptance_rate": 8 / 12,
                "tokens_per_round": completion_tokens / 4,
            }
        return extras

    def _usage(self, body: dict, completion_tokens: int) -> dict:
        prompt_tokens = sum(len(str(m.get("content", "")).split()) for m in body.get("messages", []))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

    def _json(self, body: dict) -> None:
        call = self._tool_call(body)
        message: dict = {"role": "assistant", "content": None if call else "".join(_REPLY_CHUNKS)}
        if call:
            message["tool_calls"] = [call]
        elif self._thinking(body):
            message["reasoning_content"] = "planning a reply"
        completion_tokens = 1 if call else len(_REPLY_CHUNKS)
        self._send_json({
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": _STATE["name"],
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if call else "stop",
            }],
            "usage": self._usage(body, completion_tokens),
            **self._stats(body, completion_tokens),
        })

    def _stream(self, body: dict, refusal: str | None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(payload: dict) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {
                "id": "chatcmpl-stub",
                "object": "chat.completion.chunk",
                "model": _STATE["name"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        if os.environ.get("STUB_TENSORFOLD_ROLE_CHUNK", "1") == "1":
            emit(chunk({"role": "assistant"}))
        if refusal:
            emit({"error": {"message": refusal, "type": "invalid_request_error"}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        call = self._tool_call(body)
        if call:
            emit(chunk({"tool_calls": [{
                "index": 0, "id": call["id"], "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""},
            }]}))
            emit(chunk({"tool_calls": [{"index": 0, "function": {"arguments": call["function"]["arguments"]}}]}))
            completion_tokens = 1
            finish = "tool_calls"
        else:
            if self._thinking(body):
                for piece in ("planning", " a reply"):
                    emit(chunk({"reasoning_content": piece}))
            for piece in _REPLY_CHUNKS:
                emit(chunk({"content": piece}))
                time.sleep(0.01)
            completion_tokens = len(_REPLY_CHUNKS)
            finish = "stop"
        final = chunk({}, finish)
        final.update(self._stats(body, completion_tokens))
        final["usage"] = self._usage(body, completion_tokens)
        emit(final)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _serve(args: argparse.Namespace) -> None:
    if args.fail_mode == "crash-before-ready":
        # The real CLI reports a refusal it expects as ``tensorfold: <reason>``.
        sys.stderr.write(
            "tensorfold: Ternary Bonsai 2's weights (30.0 GiB) do not fit this server's 25.0 GiB memory budget\n"
        )
        sys.stderr.flush()
        sys.exit(1)
    print(f"[tensorfold] memory budget 100.0 GiB", flush=True)
    print("[tensorfold] loading stub weights", flush=True)
    if args.fail_mode == "delay":
        time.sleep(1.5)
    _STATE["name"] = args.name or "stub"
    # ``--drafter none`` only drops the external draft model; a checkpoint's own
    # MTP head still drafts. Only ``--no-drafts`` is the serial reference.
    _STATE["drafts"] = not args.no_drafts
    _STATE["vision"] = args.vision
    record = os.environ.get("STUB_TENSORFOLD_RECORD")
    if record:
        with open(record + ".start", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "argv": sys.argv[1:],
                    "cwd": os.getcwd(),
                    "env": {
                        key: os.environ.get(key)
                        for key in (
                            "HF_HUB_OFFLINE",
                            "TENSORFOLD_NO_UPDATE_CHECK",
                            "PYTHONHOME",
                            "PYTHONPATH",
                            "DYLD_LIBRARY_PATH",
                            "VIRTUAL_ENV",
                            "CHAOSENGINE_KEEP_ME",
                        )
                    },
                },
                handle,
            )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(f"[tensorfold] serving {_STATE['name']} on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("model")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--name", default="")
    serve.add_argument("--snapshot-dir", default=None)
    serve.add_argument("--no-update-check", action="store_true")
    serve.add_argument("--context", type=int, default=None)
    serve.add_argument("--no-drafts", action="store_true")
    serve.add_argument("--vision", action="store_true")
    serve.add_argument("--vision-urls", action="store_true")
    serve.add_argument("--drafter", default=None)
    serve.add_argument("--drafter-bits", type=int, default=4)
    serve.add_argument("--fail-mode", default=None)
    args = parser.parse_args()
    _serve(args)


if __name__ == "__main__":
    main()
