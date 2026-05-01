"""Minimal stdio MCP client — JSON-RPC 2.0 over a subprocess pipe.

The client speaks the bare-minimum slice of the Model Context Protocol
needed for tool discovery + invocation:

  - `initialize` / `initialized` handshake (protocolVersion + capabilities)
  - `tools/list` to enumerate available tools
  - `tools/call` to run a tool

Everything else (resources, prompts, sampling, roots) is ignored.
Servers that depend on these features will still load — we just don't
surface them. Adding support is a forward-compatible extension.

Errors are wrapped in `McpClientError`. Servers that crash, hang, or
return malformed JSON are isolated: the client raises, the registry
falls back to whatever it had before, and the chat agent loop still
runs with the built-in tools intact.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any


# Conservative defaults. Stdio MCP servers are local subprocesses, so a
# multi-second ceiling is plenty — anything slower is a hung server we
# want to abort rather than wait on.
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_INITIALIZE_TIMEOUT_S = 15.0


class McpClientError(RuntimeError):
    """Raised on any client-side failure — protocol, timeout, or process."""


@dataclass(frozen=True)
class McpServerConfig:
    """User-supplied configuration for one MCP server.

    `id` is a short opaque key (e.g. "filesystem", "search-perplexity")
    used in tool provenance and the settings UI. `command` + `args` is
    the subprocess to spawn; `env` overlays the parent environment.
    """

    id: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "McpServerConfig":
        if not isinstance(payload, dict):
            raise McpClientError(f"MCP server config must be a dict, got {type(payload).__name__}")
        server_id = str(payload.get("id") or "").strip()
        command = str(payload.get("command") or "").strip()
        if not server_id or not command:
            raise McpClientError("MCP server config requires non-empty `id` and `command`")
        raw_args = payload.get("args") or []
        if not isinstance(raw_args, list):
            raise McpClientError("MCP server config `args` must be a list")
        env_payload = payload.get("env") or {}
        if not isinstance(env_payload, dict):
            raise McpClientError("MCP server config `env` must be an object")
        return cls(
            id=server_id,
            command=command,
            args=tuple(str(a) for a in raw_args),
            env={str(k): str(v) for k, v in env_payload.items()},
            enabled=bool(payload.get("enabled", True)),
        )


@dataclass(frozen=True)
class McpToolDescriptor:
    """Metadata for one tool exported by an MCP server."""

    server_id: str
    name: str
    description: str
    input_schema: dict[str, Any]


class McpClient:
    """One open client per MCP server. Thread-safe for sequential RPCs.

    Construct via `McpClient(config)` then call `initialize()` exactly
    once before `list_tools()` / `call_tool()`. Always close via
    `close()` (or use as a context manager) so the subprocess pipes are
    drained — leaking pipes wedges the parent app on exit.
    """

    def __init__(self, config: McpServerConfig, *, request_timeout: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
        self.config = config
        self._timeout = request_timeout
        self._proc: subprocess.Popen | None = None
        self._stdout_queue: Queue[str | None] = Queue()
        self._stdout_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._initialized = False

    def __enter__(self) -> "McpClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def start(self) -> None:
        """Spawn the subprocess. Idempotent."""
        if self._proc is not None and self._proc.poll() is None:
            return
        env = os.environ.copy()
        env.update(self.config.env)
        try:
            self._proc = subprocess.Popen(
                [self.config.command, *self.config.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as exc:
            raise McpClientError(
                f"MCP server '{self.config.id}' command not found: {self.config.command}"
            ) from exc

        # Drain stdout in a worker thread so reads don't block on the
        # main thread when the server is busy producing output.
        def _drain() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                self._stdout_queue.put(line.rstrip("\n"))
            self._stdout_queue.put(None)

        self._stdout_thread = threading.Thread(target=_drain, daemon=True)
        self._stdout_thread.start()

    def initialize(self, *, timeout: float = DEFAULT_INITIALIZE_TIMEOUT_S) -> dict[str, Any]:
        """Run the initialize handshake. Must complete before any RPCs."""
        self.start()
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "ChaosEngineAI",
                    "version": "0.7.x",
                },
            },
            timeout=timeout,
        )
        # Per spec, send the `initialized` notification after the
        # response. Notifications have no `id` and expect no response.
        self._notify("notifications/initialized", {})
        self._initialized = True
        return result

    def list_tools(self, *, timeout: float | None = None) -> list[McpToolDescriptor]:
        """Enumerate the server's tools. Requires `initialize()` first."""
        if not self._initialized:
            raise McpClientError(
                f"MCP server '{self.config.id}' not initialised — call initialize() first"
            )
        result = self._request("tools/list", {}, timeout=timeout)
        raw_tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(raw_tools, list):
            return []
        descriptors: list[McpToolDescriptor] = []
        for entry in raw_tools:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            schema = entry.get("inputSchema") or {"type": "object", "properties": {}}
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            descriptors.append(McpToolDescriptor(
                server_id=self.config.id,
                name=name,
                description=str(entry.get("description") or ""),
                input_schema=schema,
            ))
        return descriptors

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        """Invoke a tool. Returns the text representation of the result.

        MCP tool results are a structured list of content parts (text,
        image, embedded resources, etc.). For chat-agent integration we
        flatten the parts into a single string by concatenating text
        parts and stringifying anything else, matching the contract
        every existing built-in tool already follows.
        """
        if not self._initialized:
            raise McpClientError(
                f"MCP server '{self.config.id}' not initialised — call initialize() first"
            )
        result = self._request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout=timeout,
        )
        return _flatten_tool_result(result)

    def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # JSON-RPC plumbing
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Any:
        with self._lock:
            assert self._proc is not None and self._proc.stdin is not None, "client not started"
            request_id = self._next_id
            self._next_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except OSError as exc:
                raise McpClientError(
                    f"MCP server '{self.config.id}' stdin failed: {exc}"
                ) from exc

            deadline_seconds = timeout if timeout is not None else self._timeout
            while True:
                try:
                    line = self._stdout_queue.get(timeout=deadline_seconds)
                except Empty as exc:
                    raise McpClientError(
                        f"MCP server '{self.config.id}' timed out waiting for {method}"
                    ) from exc
                if line is None:
                    stderr_tail = self._read_stderr_tail()
                    raise McpClientError(
                        f"MCP server '{self.config.id}' exited mid-request: {stderr_tail}"
                    )
                parsed = _parse_json_rpc_line(line)
                if parsed is None:
                    continue  # progress / log line — keep reading
                # Skip notifications + responses for other request ids
                if parsed.get("id") != request_id:
                    continue
                if "error" in parsed and parsed["error"]:
                    err = parsed["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise McpClientError(
                        f"MCP server '{self.config.id}' returned error for {method}: {msg}"
                    )
                return parsed.get("result")

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        with self._lock:
            if self._proc is None or self._proc.stdin is None:
                return
            payload = {"jsonrpc": "2.0", "method": method, "params": params}
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except OSError:
                pass

    def _read_stderr_tail(self) -> str:
        if self._proc is None or self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read()[-500:]
        except OSError:
            return ""


# ----------------------------------------------------------------------
# Pure helpers (testable without a subprocess)
# ----------------------------------------------------------------------


def _parse_json_rpc_line(line: str) -> dict[str, Any] | None:
    """Parse a single line of JSON-RPC. Returns None for unparseable / empty.

    Some servers print log lines to stdout alongside JSON-RPC frames;
    the client tolerates them by returning None and continuing the
    read loop. A frame must be a JSON object with `jsonrpc: "2.0"`.
    """
    stripped = line.strip()
    if not stripped:
        return None
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("jsonrpc") != "2.0":
        return None
    return payload


def _flatten_tool_result(result: Any) -> str:
    """Convert an MCP `tools/call` result into a single string.

    The MCP spec returns ``{"content": [{"type": "text", "text": "..."}, ...]}``
    plus optional `isError`. We concatenate text parts; anything else
    is JSON-stringified so the caller still sees the data.
    """
    if not isinstance(result, dict):
        return str(result) if result is not None else ""
    if result.get("isError"):
        prefix = "[MCP error] "
    else:
        prefix = ""
    content = result.get("content")
    if not isinstance(content, list):
        return prefix + (str(result) if result else "")
    parts: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            parts.append(str(entry))
            continue
        if entry.get("type") == "text":
            parts.append(str(entry.get("text") or ""))
        else:
            parts.append(json.dumps(entry, sort_keys=True))
    return prefix + "\n".join(parts).strip()
