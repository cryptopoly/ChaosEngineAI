"""Structured model events: one record per step of a model's life (D-88).

A model is requested, resolved to an engine, loaded, answers (first token,
then a generation summary), goes idle and is unloaded. Each step is one event
in a bounded in-memory ring and in a rotated JSONL file under the data
directory, so the Server page's Activity view can draw a timeline and the
history survives a restart. ``emit`` never blocks on disk: a daemon thread
writes, and a full queue drops the disk copy (counted), never the ring entry.
Strings are redacted when the event is made (tokens, secret file paths, the
home folder). The controller calls the small helpers at the bottom; they do
nothing when no log is attached (tests, fake runtimes).
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import queue as _queue_mod
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator

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
STDERR_TAIL_LINES = 20

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


# ---------------------------------------------------------------- helpers for the controller


def new_request_id() -> str:
    return uuid.uuid4().hex[:10]


def speculative_method(info: Any) -> str | None:
    """The speculative decoding method a loaded model runs, or ``None``."""
    if info is None or not getattr(info, "speculativeDecoding", False):
        return None
    engine = str(getattr(info, "engine", "") or "").lower()
    if "mtp" in engine:
        return "mtp"
    if int(getattr(info, "treeBudget", 0) or 0) > 0:
        return "ddtree"
    if engine == "llama.cpp":
        return "draft"
    return "dflash"


def model_fields(info: Any) -> dict[str, Any]:
    """The model columns of an event, from a ``LoadedModelInfo``."""
    if info is None:
        return {}
    return {
        "model": getattr(info, "name", None),
        "model_ref": getattr(info, "ref", None),
        "engine": getattr(info, "engine", None),
        "cache_strategy": getattr(info, "cacheStrategy", None),
        "speculative": speculative_method(info),
    }


def _model_key(info: Any) -> str:
    return f"{getattr(info, 'ref', '')}|{getattr(info, 'engine', '')}"


def _tail_text(path: Any, lines: int = STDERR_TAIL_LINES) -> list[str]:
    if not path:
        return []
    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 16_384))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()][-lines:]


def engine_detail(engine: Any, error: BaseException | None = None) -> dict[str, Any]:
    """Command line, environment and last stderr lines of an engine, as far as it exposes them."""
    detail: dict[str, Any] = {}
    command = getattr(engine, "last_command", None)
    if command is None:
        command = getattr(getattr(engine, "worker", None), "command", None)
    if isinstance(command, (list, tuple)) and command:
        detail["command"] = " ".join(str(part) for part in command)
    detail["env"] = engine_environment()
    stderr = _tail_text(getattr(engine, "log_path", None))
    if error is not None:
        detail["error"] = str(error)
        if not stderr:
            stderr = [line for line in str(error).splitlines() if line.strip()][-STDERR_TAIL_LINES:]
    if stderr:
        detail["stderr"] = stderr
    return detail


def emit_unloaded(log: ModelEventLog | None, info: Any, reason: str, *, message: str = "") -> None:
    if log is None or info is None:
        return
    log.emit(
        "unloaded",
        message=message or f"Unloaded ({reason}).",
        data={"reason": reason},
        **model_fields(info),
    )


def emit_parked(log: ModelEventLog | None, info: Any) -> None:
    if log is None or info is None:
        return
    log.emit("parked", message="Kept warm for a quick switch back.", **model_fields(info))


class _GenerationTrace:
    """One generation's events: first token, the summary or the failure, then idle."""

    def __init__(self, log: ModelEventLog, info: Any, *, streaming: bool) -> None:
        self.log = log
        self.info = info
        self.fields = model_fields(info)
        self.request_id = new_request_id()
        self.streaming = streaming
        self.started = time.perf_counter()
        self.first_at: float | None = None
        self.closed = False
        log.begin(_model_key(info))

    def first_token(self) -> None:
        if self.first_at is not None:
            return
        self.first_at = time.perf_counter()
        prefill = round(self.first_at - self.started, 3)
        self.log.emit(
            "first_token",
            request_id=self.request_id,
            message=f"First token after {prefill:.2f} s.",
            data={"prefillSeconds": prefill},
            **self.fields,
        )

    def done(self, *, prompt_tokens: int, completion_tokens: int, tok_s: float, finish_reason: str | None) -> None:
        seconds = round(time.perf_counter() - self.started, 3)
        data: dict[str, Any] = {
            "seconds": seconds,
            "promptTokens": int(prompt_tokens or 0),
            "completionTokens": int(completion_tokens or 0),
            "tokS": round(float(tok_s or 0.0), 2),
            "finishReason": finish_reason,
            "streaming": self.streaming,
        }
        if self.first_at is not None:
            data["prefillSeconds"] = round(self.first_at - self.started, 3)
        self.log.emit(
            "generation",
            request_id=self.request_id,
            message=f"{data['completionTokens']} tokens at {data['tokS']:.1f} tok/s in {seconds:.2f} s.",
            data=data,
            **self.fields,
        )
        self.close()

    def failed(self, error: BaseException) -> None:
        if self.closed:
            return
        self.log.emit(
            "generation_failed",
            level="error",
            request_id=self.request_id,
            message=str(error).splitlines()[0][:300] if str(error) else type(error).__name__,
            data={"seconds": round(time.perf_counter() - self.started, 3)},
            detail={"error": f"{type(error).__name__}: {error}"},
            **self.fields,
        )
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.log.end(_model_key(self.info)) == 0:
            self.log.emit("idle", request_id=self.request_id, message="Idle.", **self.fields)


def _trace_info(controller: Any, engine: Any) -> Any:
    """The model a generation runs on: the active one, or the warm model owning ``engine``."""
    if engine is None or engine is getattr(controller, "engine", None):
        return getattr(controller, "loaded_model", None)
    for pooled_engine, info in list(getattr(controller, "_warm_pool", {}).values()):
        if pooled_engine is engine:
            return info
    return getattr(controller, "loaded_model", None)


def _start(controller: Any, kwargs: dict[str, Any], *, streaming: bool) -> _GenerationTrace | None:
    log = getattr(controller, "events", None)
    info = _trace_info(controller, kwargs.get("engine"))
    if log is None or info is None:
        return None
    try:
        return _GenerationTrace(log, info, streaming=streaming)
    except Exception:  # noqa: BLE001 - events never break a generation
        return None


def traced_generation(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``RuntimeController.generate``: a generation event, or a failure, then idle."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        trace = _start(self, kwargs, streaming=False)
        try:
            result = method(self, *args, **kwargs)
        except BaseException as exc:
            if trace is not None:
                trace.failed(exc)
            raise
        if trace is not None:
            trace.done(
                prompt_tokens=getattr(result, "promptTokens", 0),
                completion_tokens=getattr(result, "completionTokens", 0),
                tok_s=getattr(result, "tokS", 0.0),
                finish_reason=getattr(result, "finishReason", None),
            )
        return result

    return wrapper


def traced_stream(method: Callable[..., Iterator[Any]]) -> Callable[..., Iterator[Any]]:
    """Wrap ``RuntimeController.stream_generate``: first token, summary or failure, then idle."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        trace = _start(self, kwargs, streaming=True)
        inner = method(self, *args, **kwargs)
        try:
            for chunk in inner:
                if trace is not None:
                    if getattr(chunk, "text", None) or getattr(chunk, "reasoning", None):
                        trace.first_token()
                    if getattr(chunk, "done", False):
                        trace.done(
                            prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                            completion_tokens=getattr(chunk, "completion_tokens", 0),
                            tok_s=getattr(chunk, "tok_s", 0.0),
                            finish_reason=getattr(chunk, "finish_reason", None),
                        )
                yield chunk
        except GeneratorExit:
            if trace is not None and not trace.closed:
                trace.done(prompt_tokens=0, completion_tokens=0, tok_s=0.0, finish_reason="cancelled")
            raise
        except BaseException as exc:
            if trace is not None:
                trace.failed(exc)
            raise
        finally:
            inner.close()
            if trace is not None:
                trace.close()

    return wrapper


# The load in flight on this thread: its request id and when the engine started loading.
_current = threading.local()


def traced_load(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``RuntimeController.load_model``: requested, then loaded or load_failed."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        log: ModelEventLog | None = getattr(self, "events", None)
        if log is None:
            return method(self, *args, **kwargs)
        request_id = new_request_id()
        ref = kwargs.get("model_ref")
        name = kwargs.get("model_name") or ref
        base = {
            "model": name,
            "model_ref": ref,
            "request_id": request_id,
            "cache_strategy": kwargs.get("cache_strategy"),
        }
        _current.request_id = request_id
        _current.load_started = None
        started = time.perf_counter()
        log.emit(
            "requested",
            message=f"Load requested ({kwargs.get('backend') or 'auto'}).",
            data={
                "backend": kwargs.get("backend") or "auto",
                "contextTokens": kwargs.get("context_tokens"),
                "speculativeDecoding": bool(kwargs.get("speculative_decoding")),
                "source": kwargs.get("source"),
            },
            **base,
        )
        try:
            info = method(self, *args, **kwargs)
        except BaseException as exc:
            log.emit(
                "load_failed",
                level="error",
                message=str(exc).splitlines()[0][:300] if str(exc) else type(exc).__name__,
                data={"seconds": round(time.perf_counter() - started, 3), "reason": "error"},
                detail=engine_detail(getattr(self, "engine", None), exc),
                engine=getattr(getattr(self, "engine", None), "engine_name", None),
                **base,
            )
            raise
        load_started = getattr(_current, "load_started", None)
        seconds = round(time.perf_counter() - (load_started or started), 3)
        memory = 0
        try:
            memory = max(self._model_resident_bytes(info), self._engine_resident_bytes(self.engine))
        except Exception:  # noqa: BLE001
            pass
        fields = model_fields(info)
        fields.pop("model_ref", None)
        log.emit(
            "loaded",
            request_id=request_id,
            model_ref=ref,
            message=f"Loaded in {seconds:.1f} s via {getattr(info, 'engine', '?')}.",
            data={
                "seconds": seconds,
                "totalSeconds": round(time.perf_counter() - started, 3),
                "memoryBytes": memory,
                "contextTokens": getattr(info, "contextTokens", None),
                "warm": load_started is None,
            },
            detail=engine_detail(self.engine),
            **fields,
        )
        return info

    return wrapper


def emit_resolving(controller: Any, **kwargs: Any) -> None:
    log = getattr(controller, "events", None)
    if log is None:
        return
    log.emit(
        "resolving",
        model=kwargs.get("model_name") or kwargs.get("model_ref"),
        model_ref=kwargs.get("model_ref"),
        request_id=getattr(_current, "request_id", None),
        cache_strategy=kwargs.get("cache_strategy"),
        message="Choosing an engine.",
    )


def emit_loading(controller: Any, *, engine: Any, model_ref: str, model_name: str, cache_strategy: str, weights_bytes: int) -> None:
    log = getattr(controller, "events", None)
    if log is None:
        return
    _current.load_started = time.perf_counter()
    log.emit(
        "loading",
        model=model_name,
        model_ref=model_ref,
        request_id=getattr(_current, "request_id", None),
        engine=getattr(engine, "engine_name", None),
        cache_strategy=cache_strategy,
        message=f"Loading on {getattr(engine, 'engine_name', '?')}.",
        data={"bytes": int(weights_bytes or 0)},
    )
