"""Shared JSON-lines loop for the warm image and video workers (``mflux_worker.py``, ``ltx2_worker.py``).

Standard library only: it runs inside the mflux and ltx-2-mlx venvs, which never import
``backend_service``; each worker loads this file by path. The protocol is the music worker's
(``music_worker.py``): one JSON object per line each way, the first line out is the readiness line.
``ping``, ``status`` and ``cancel`` are answered at once; ``generate`` and ``unload`` run one at a time on
one engine thread (MLX streams belong to the thread that made them) and a second one while busy gets
``busy``. A ``generate`` sends ``stage``, ``step`` and ``line`` events with its request id and job.
Anything a library prints goes to stderr (the worker log), so it cannot corrupt the protocol.

An engine is an object with ``generate(request, hooks) -> dict``, ``unload()`` and ``loaded() -> dict``.
``CHAOSENGINE_WARM_ENGINE_MODULE`` points the worker at a fake engine file (tests).
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ENGINE_MODULE_ENV = "CHAOSENGINE_WARM_ENGINE_MODULE"
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Cancelled(Exception):
    """Raised inside a generation when its job's cancel flag is set."""


class Protocol:
    """The protocol stream (the real stdout) and a lock so engine and stdin threads never interleave."""

    def __init__(self) -> None:
        self.out = sys.stdout
        sys.stdout = sys.stderr  # library prints go to the log, never into the protocol
        self.lock = threading.Lock()

    def reply(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, default=str) + "\n"
        with self.lock:
            self.out.write(line)
            self.out.flush()


def footprint_gb() -> float | None:
    """``phys_footprint`` on macOS (includes Metal buffers), else peak RSS; GB."""
    value: int | None = None
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            buf = (ctypes.c_uint64 * 40)()
            if libc.proc_pid_rusage(os.getpid(), 2, ctypes.byref(buf)) == 0:  # RUSAGE_INFO_V2
                value = int(buf[9])
        except (OSError, AttributeError):
            value = None
    if value is None:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            value = int(peak if sys.platform == "darwin" else peak * 1024)
        except (ImportError, OSError, ValueError):
            return None
    return round(value / 1e9, 2) if value else None


class PeakSampler:
    """Samples this process's footprint every 0.5 s while a job runs."""

    def __init__(self) -> None:
        self.peak = footprint_gb() or 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.peak = max(self.peak, footprint_gb() or 0.0)

    def __enter__(self) -> "PeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak = max(self.peak, footprint_gb() or 0.0)


class Hooks:
    """What an engine calls while it generates: stages, steps, raw output lines and the cancel check."""

    def __init__(self, protocol: Protocol, request_id: Any, job: str, cancel: threading.Event) -> None:
        self._protocol, self.request_id, self.job, self._cancel = protocol, request_id, job, cancel
        self._stages: set[str] = set()
        self.started = time.perf_counter()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("cancelled")

    def _event(self, payload: dict[str, Any]) -> None:
        self._protocol.reply({"id": self.request_id, "job": self.job, **payload})

    def stage(self, name: str) -> None:
        """Report a stage once (``load``, ``encode``, ``denoise``, ``decode``, ``write``)."""
        if name in self._stages:
            return
        self._stages.add(name)
        self._event({"event": "stage", "stage": name, "elapsed_s": round(time.perf_counter() - self.started, 2)})

    def step(self, step: int, total: int) -> None:
        self.stage("denoise")
        self._event({"event": "step", "step": int(step), "total": int(total)})

    def line(self, text: str) -> None:
        self._event({"event": "line", "text": text})


class LineTee:
    """Replaces ``sys.stderr``: everything still reaches the log, and while a job runs each output line
    (tqdm redraws split on ``\\r``) also goes to ``on_line``. A write from the engine thread raises
    ``Cancelled`` once the job's cancel flag is set, which stops engines that have no step hook."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self._buffer = ""
        self.on_line: Callable[[str], None] | None = None
        self.cancel: threading.Event | None = None
        self.engine_thread: threading.Thread | None = None

    def write(self, text: str) -> int:
        self._target.write(text)
        if self.on_line is not None:
            self._buffer += text
            parts = re.split(r"[\r\n]", self._buffer)
            self._buffer = parts.pop()
            for part in parts:
                if part.strip():
                    self.on_line(part.rstrip())
        if self.cancel is not None and self.cancel.is_set() and threading.current_thread() is self.engine_thread:
            raise Cancelled("cancelled")
        return len(text)

    def flush(self) -> None:
        self._target.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def load_engine(default_factory: Callable[[], Any]) -> Any:
    path = os.environ.get(ENGINE_MODULE_ENV)
    if not path:
        return default_factory()
    spec = importlib.util.spec_from_file_location("chaosengine_warm_fake_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No engine at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_engine()


def serve(engine_factory: Callable[[], Any], *, engine_version: str = "unknown", tee_lines: bool = False) -> None:
    """Run the worker until stdin closes."""
    protocol = Protocol()
    tee = LineTee(sys.stderr) if tee_lines else None
    if tee is not None:
        sys.stderr = tee
        sys.stdout = tee
    state: dict[str, Any] = {"busy": None, "engine": None}
    lock = threading.Lock()
    tasks: "queue.Queue[dict[str, Any]]" = queue.Queue()

    def engine() -> Any:
        if state["engine"] is None:
            state["engine"] = load_engine(engine_factory)
        return state["engine"]

    def run_generate(request: dict[str, Any], busy: dict[str, Any]) -> dict[str, Any]:
        hooks = Hooks(protocol, request.get("id"), str(request["job"]), busy["cancel"])
        if tee is not None:
            tee.on_line, tee.cancel, tee.engine_thread = hooks.line, busy["cancel"], threading.current_thread()
        started = time.perf_counter()
        try:
            with PeakSampler() as sampler:
                try:
                    result = engine().generate(request, hooks) or {}
                except Cancelled:
                    result = None
        finally:
            if tee is not None:
                tee.on_line, tee.cancel = None, None
            gc.collect()
        if hooks.cancelled or result is None:
            return {"ok": False, "cancelled": True}
        return {
            "ok": True,
            **result,
            "seconds": round(time.perf_counter() - started, 2),
            "peak_footprint_gb": round(sampler.peak, 2),
            "footprint_gb": footprint_gb(),
        }

    def run_unload(_request: dict[str, Any], _busy: dict[str, Any]) -> dict[str, Any]:
        if state["engine"] is not None:
            state["engine"].unload()
        gc.collect()
        return {"ok": True, "footprint_gb": footprint_gb()}

    commands = {"generate": run_generate, "unload": run_unload}

    def engine_thread() -> None:
        while True:
            request = tasks.get()
            out: dict[str, Any] = {"id": request.get("id")}
            if request.get("job"):
                out["job"] = request["job"]
            try:
                out.update(commands[request["cmd"]](request, state["busy"]))
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - every failure goes back; the thread lives on
                traceback.print_exc()
                out.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                with lock:
                    state["busy"] = None
            protocol.reply(out)

    def status() -> dict[str, Any]:
        loaded = state["engine"].loaded() if state["engine"] is not None else {}
        busy = state["busy"]
        return {"loaded": loaded, "busy": (busy.get("job") or busy.get("cmd")) if busy else None,
                "footprint_gb": footprint_gb(), "engine_version": engine_version, "pid": os.getpid()}

    def handle(request: dict[str, Any]) -> None:
        request_id, cmd = request.get("id"), request.get("cmd")
        if cmd in commands:
            if cmd == "generate" and not _JOB_ID.fullmatch(str(request.get("job") or "")):
                raise ValueError("job must be 1 to 64 letters, digits, '-' or '_'.")
            with lock:
                if state["busy"] is not None:
                    protocol.reply({"id": request_id, "job": request.get("job"), "ok": False, "error": "busy"})
                    return
                state["busy"] = {"cmd": cmd, "job": request.get("job"), "cancel": threading.Event()}
            tasks.put(request)
            return
        if cmd == "ping":
            protocol.reply({"id": request_id, "ok": True, "pong": True})
        elif cmd == "status":
            protocol.reply({"id": request_id, "ok": True, **status()})
        elif cmd == "cancel":
            job = str(request.get("job") or "")
            with lock:
                busy = state["busy"]
                hit = bool(busy and busy.get("job") == job)
                if hit:
                    busy["cancel"].set()
            protocol.reply({"id": request_id, "ok": True, "job": job, "state": "cancelling" if hit else "not-running"})
        else:
            raise ValueError(f"Unknown command {cmd!r}.")

    threading.Thread(target=engine_thread, name="warm-engine", daemon=True).start()
    protocol.reply({"id": None, "ok": True, "ready": True, "pid": os.getpid(), "engine_version": engine_version})
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("A request must be a JSON object.")
            handle(request)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            protocol.reply({"id": request.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"})


def strip_own_dir(script: str) -> Path:
    """Take the script's folder off ``sys.path`` (our siblings must never shadow a library) and return it."""
    here = Path(script).resolve().parent

    def same(entry: str) -> bool:
        try:
            return Path(entry or ".").resolve() == here
        except OSError:
            return False

    sys.path[:] = [entry for entry in sys.path if not same(entry)]
    return here
