"""Structured model events: one record per step of a model's life (D-88).

A model is requested, resolved to an engine, loaded, answers (first token,
then a generation summary), goes idle and is unloaded. Each step is one event
in a bounded in-memory ring and in a rotated JSONL file under the data
directory, so the Server page's Activity view can draw a timeline and the
history survives a restart. ``emit`` never blocks on disk: a daemon thread
writes, and a full queue drops the disk copy (counted), never the ring entry.
Strings are redacted when the event is made (tokens, secret file paths, the
home folder). The controller's hooks live in ``model_event_hooks.py``.
"""

from __future__ import annotations

import itertools
import json
import os
import queue as _queue_mod
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

EVENT_KINDS = (
    "requested",
    "resolving",
    "loading",
    "loaded",
    "load_failed",
    "parked",
    "first_token",
    "generation",
    "generation_failed",
    "idle",
    "unloaded",
)
# ``idle`` is reserved: nothing unloads an idle model on its own today.
UNLOAD_REASONS = ("user", "swap", "memory", "idle", "error")

DEFAULT_CAPACITY = 1000
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_BACKUPS = 3
DEFAULT_QUEUE_SIZE = 2000

# ---------------------------------------------------------------- redaction

_HOME = str(Path.home())
_SECRET_ENV_NAME = re.compile(r"TOKEN|SECRET|KEY|PASSWORD|PASSWD|AUTH|CREDENTIAL", re.IGNORECASE)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhf_[A-Za-z0-9]{8,}"), "hf_***"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{4,}"), r"\1***"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    (re.compile(r"(?i)(--(?:api[-_]key|hf[-_]token|token|password|secret)(?:=|\s+))(\S+)"), r"\1***"),
    (re.compile(r"(?i)\b((?:[A-Z0-9_]*(?:token|api_key|apikey|password|secret))=)([^&\s\"']+)"), r"\1***"),
    (
        re.compile(
            r"(?:~|/)[^\s\"':]*?(?:/huggingface/token|/\.netrc|/\.ssh/[^\s\"':]+|\.pem|/[^\s\"'/:]+\.key)(?=$|[\s\"':,)])"
        ),
        "<secret path>",
    ),
)


def redact(text: str) -> str:
    """``text`` with tokens, secret file paths and the home folder hidden."""
    if not text:
        return text
    if _HOME and _HOME != "/" and _HOME in text:
        text = text.replace(_HOME, "~")
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


_ENV_PREFIXES = ("CHAOSENGINE_", "MLX_", "LLAMA_", "GGML_", "HF_", "CUDA_", "PYTORCH_", "TORCH_")
_ENV_NAMES = {"OMP_NUM_THREADS", "PYTHONPATH", "VIRTUAL_ENV", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"}


def engine_environment(env: dict[str, str] | None = None) -> list[str]:
    """The engine-relevant environment as ``NAME=value`` lines, secrets hidden."""
    source = os.environ if env is None else env
    lines: list[str] = []
    for name in sorted(source):
        if not (name.startswith(_ENV_PREFIXES) or name in _ENV_NAMES):
            continue
        value = "***" if _SECRET_ENV_NAME.search(name) else redact(str(source[name]))
        lines.append(f"{name}={value}")
    return lines


# ---------------------------------------------------------------- the log


class ModelEventLog:
    """Bounded ring plus a rotated JSONL file of model events."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        capacity: int = DEFAULT_CAPACITY,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backups: int = DEFAULT_BACKUPS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        load_history: bool = True,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.capacity = capacity
        self.max_bytes = max(1024, int(max_bytes))
        self.backups = max(1, int(backups))
        self.dropped = 0
        self._ring: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._subscribers: list[_queue_mod.Queue] = []
        self._inflight: dict[str, int] = {}
        self._writes: _queue_mod.Queue | None = _queue_mod.Queue(maxsize=queue_size) if self.path else None
        self._writer: threading.Thread | None = None
        next_id = 1
        if self.path is not None and load_history:
            next_id = self._load_history() + 1
        self._ids = itertools.count(next_id)

    # -- reading

    def events(
        self,
        *,
        model: str | None = None,
        level: str | None = None,
        kind: str | None = None,
        request_id: str | None = None,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Matching events, oldest first; ``limit`` keeps the newest."""
        with self._lock:
            items = list(self._ring)
        if since is not None:
            items = [e for e in items if e["id"] > since]
        if model:
            items = [e for e in items if model in (e.get("model"), e.get("modelRef"))]
        if level:
            items = [e for e in items if e.get("level") == level]
        if kind:
            items = [e for e in items if e.get("kind") == kind]
        if request_id:
            items = [e for e in items if e.get("requestId") == request_id]
        if limit is not None and limit >= 0:
            items = items[-limit:] if limit else []
        return items

    def subscribe(self, maxsize: int = 500) -> _queue_mod.Queue:
        q: _queue_mod.Queue = _queue_mod.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: _queue_mod.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    # -- writing

    def emit(
        self,
        kind: str,
        *,
        level: str = "info",
        model: str | None = None,
        model_ref: str | None = None,
        request_id: str | None = None,
        engine: str | None = None,
        cache_strategy: str | None = None,
        speculative: str | None = None,
        message: str = "",
        data: dict[str, Any] | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        event: dict[str, Any] = {
            "id": 0,
            "ts": round(now, 3),
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "kind": kind,
            "level": level,
            "model": redact(model) if model else model,
            "modelRef": redact(model_ref) if model_ref else model_ref,
            "requestId": request_id,
            "engine": engine,
            "cacheStrategy": cache_strategy,
            "speculative": speculative,
            "message": redact(message),
            "data": dict(data or {}),
            "detail": _redact_value(dict(detail or {})),
        }
        with self._lock:
            event["id"] = next(self._ids)
            self._ring.append(event)
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except _queue_mod.Full:
                pass
        if self._writes is not None:
            self._ensure_writer()
            try:
                self._writes.put_nowait(event)
            except _queue_mod.Full:
                self.dropped += 1
        return event

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait until the disk queue is written (tests and shutdown)."""
        if self._writes is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._writes.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return False

    # -- generation book-keeping (idle detection)

    def begin(self, key: str) -> None:
        with self._lock:
            self._inflight[key] = self._inflight.get(key, 0) + 1

    def end(self, key: str) -> int:
        with self._lock:
            left = max(0, self._inflight.get(key, 0) - 1)
            if left:
                self._inflight[key] = left
            else:
                self._inflight.pop(key, None)
            return left

    # -- disk

    def _ensure_writer(self) -> None:
        if self._writer is not None and self._writer.is_alive():
            return
        with self._lock:
            if self._writer is not None and self._writer.is_alive():
                return
            self._writer = threading.Thread(target=self._write_loop, name="model-events-writer", daemon=True)
            self._writer.start()

    def _write_loop(self) -> None:
        assert self.path is not None and self._writes is not None
        handle = None
        try:
            while True:
                event = self._writes.get()
                try:
                    if handle is None:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        handle = self.path.open("a", encoding="utf-8")
                    handle.write(json.dumps(event, separators=(",", ":")) + "\n")
                    handle.flush()
                    if handle.tell() >= self.max_bytes:
                        handle.close()
                        handle = None
                        self._rotate()
                except OSError:
                    self.dropped += 1
                finally:
                    self._writes.task_done()
        finally:
            if handle is not None:
                handle.close()

    def _rotate(self) -> None:
        assert self.path is not None
        for index in range(self.backups - 1, 0, -1):
            older = self.path.with_name(f"{self.path.name}.{index}")
            if older.exists():
                older.replace(self.path.with_name(f"{self.path.name}.{index + 1}"))
        if self.path.exists():
            self.path.replace(self.path.with_name(f"{self.path.name}.1"))

    def _load_history(self) -> int:
        """Fill the ring from the newest lines on disk; return the highest id seen."""
        assert self.path is not None
        lines: list[str] = []
        for candidate in (self.path.with_name(f"{self.path.name}.1"), self.path):
            try:
                with candidate.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 2 * self.max_bytes))
                    lines.extend(handle.read().decode("utf-8", errors="replace").splitlines())
            except OSError:
                continue
        highest = 0
        for line in lines[-self.capacity:]:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or not isinstance(event.get("id"), int):
                continue
            self._ring.append(event)
            highest = max(highest, event["id"])
        return highest
