"""Controller hooks for the structured model events (D-88).

Decorators wrap ``RuntimeController.load_model``, ``generate`` and
``stream_generate``; one-line calls mark resolving, loading, parking and
unloads. Each does nothing when the controller has no event log attached
(tests and fake runtimes), and none runs per token: a stream emits at most
its first token and its summary.
"""

from __future__ import annotations

import functools
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator

from backend_service.model_events import ModelEventLog, engine_environment

STDERR_TAIL_LINES = 20


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
