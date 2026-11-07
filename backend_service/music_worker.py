"""Music worker: ACE-Step generation in its own process, driven by JSON lines.

Runs from the music venv (``~/.chaosengine/music-venv``, built by ``backend_service.music_env``), never
in the FastAPI backend, and imports nothing from ``backend_service``: its own folder is taken off
``sys.path`` and the engine file next to it is loaded by path. Protocol (``docs/plans/music-spike.md``
6.3): one JSON object per line each way; the first line out is the readiness line. ``ping``, ``status``
and ``cancel`` are answered at once by the stdin loop; ``load``, ``generate`` and ``unload`` run on one
engine thread (MLX streams belong to the thread that made them), reply when done, and a second one while
busy gets ``busy``. ``generate`` sends ``stage`` and ``step`` events with its request id and job.
Anything a library prints goes to stderr, so it cannot corrupt the protocol.
"""

from __future__ import annotations

import ctypes
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

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr  # the planner LM path prints to stdout (spike section 7.4)

_HERE = Path(__file__).resolve().parent


def _same_dir(entry: str) -> bool:
    try:
        return Path(entry or ".").resolve() == _HERE
    except OSError:
        return False


# Our siblings (``progress.py``, ``models/``) must never shadow a library's imports.
sys.path[:] = [entry for entry in sys.path if not _same_dir(entry)]

ENGINE_MODULE_ENV = "CHAOSENGINE_MUSIC_ENGINE_MODULE"  # tests point this at a fake engine file
DATA_DIR = Path(os.environ.get("CHAOSENGINE_MUSIC_DATA") or Path.home() / ".chaosengine" / "music").resolve()
ENGINE_VERSION = os.environ.get("CHAOSENGINE_MUSIC_ENGINE_VERSION", "unknown")
TASKS = ("text2music",)  # an engine file may declare its own ``TASKS`` and ``SECONDS_RANGE``
SECONDS_RANGE = (10, 600)
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PATH_FIELDS = ("out_dir", "src_audio", "reference_audio")
# VAE decode has no hook; it takes about 0.05 to 0.07 s per second of audio (spike section 6.1).
DECODE_SECONDS_PER_AUDIO_SECOND = 0.06

_write_lock = threading.Lock()


class Cancelled(Exception):
    """Raised from the step hook when the job's cancel flag is set."""


def reply(payload: dict[str, Any]) -> None:
    line = json.dumps(payload, default=str) + "\n"
    with _write_lock:
        PROTOCOL_OUT.write(line)
        PROTOCOL_OUT.flush()


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _macos_footprint_bytes(pid: int) -> int | None:
    """``phys_footprint`` from ``proc_pid_rusage`` (what Activity Monitor shows; includes Metal buffers)."""
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        buf = (ctypes.c_uint64 * 40)()
        if libc.proc_pid_rusage(pid, 2, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V2
            return None
        return int(buf[9])  # 16-byte uuid, then ri_phys_footprint is the 8th uint64 field
    except (OSError, AttributeError):
        return None


def _fallback_footprint_bytes() -> int | None:
    """Resident memory without psutil: ``/proc`` on Linux, else the peak from ``getrusage``."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except (ImportError, OSError, ValueError):
        return None


def footprint_gb() -> float | None:
    value = _macos_footprint_bytes(os.getpid()) if sys.platform == "darwin" else None
    if value is None:
        value = _fallback_footprint_bytes()
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


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Hooks:
    """What an engine calls while it generates: stages, steps and the cancel check."""

    def __init__(self, request_id: Any, job: str, cancel: threading.Event) -> None:
        self.request_id, self.job, self._cancel = request_id, job, cancel
        self._stages: set[str] = set()
        self._started = time.perf_counter()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled("cancelled")

    def stage(self, name: str, estimate_s: float | None = None) -> None:
        """Report a stage once (``load``, ``encode``, ``diffusion``, ``decode``, ``write``)."""
        if name in self._stages:
            return
        self._stages.add(name)
        reply({"id": self.request_id, "job": self.job, "event": "stage", "stage": name,
               "estimate_s": None if estimate_s is None else round(float(estimate_s), 2)})

    def step(self, step: int, total: int) -> None:
        self.stage("diffusion")
        reply({"id": self.request_id, "job": self.job, "event": "step", "step": int(step), "total": int(total),
               "elapsed_s": round(time.perf_counter() - self._started, 2)})


def _load_engine_module() -> Any:
    path = Path(os.environ.get(ENGINE_MODULE_ENV) or _HERE / "music_worker_acestep.py")
    spec = importlib.util.spec_from_file_location("chaosengine_music_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No music engine at {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_engine_cache: dict[str, Any] = {}
_engine_module_cache: dict[str, Any] = {}


def engine() -> Any:
    if "engine" not in _engine_cache:
        module = _engine_module_cache.get("module") or _load_engine_module()
        _engine_module_cache["module"] = module
        _engine_cache["engine"] = module.create_engine(DATA_DIR)
    return _engine_cache["engine"]


def engine_unavailable() -> str | None:
    try:
        return engine().unavailable_reason()
    except Exception as exc:  # noqa: BLE001 - a broken engine file is reported, not raised
        return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------

def _inside_data_dir(raw: Any, name: str) -> None:
    path = Path(str(raw))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path.")
    resolved = path.resolve()
    if resolved != DATA_DIR and DATA_DIR not in resolved.parents:
        raise ValueError(f"{name} must be inside the music data folder.")


def _engine_limits() -> tuple[tuple[str, ...], tuple[float, float]]:
    """(tasks, (min seconds, max seconds)) the loaded engine file allows; the music defaults without one."""
    module = _engine_module_cache.get("module")
    if module is None:
        try:
            module = _engine_module_cache["module"] = _load_engine_module()
        except Exception:  # noqa: BLE001 - a broken engine file is reported by engine_unavailable()
            return TASKS, SECONDS_RANGE
    return tuple(getattr(module, "TASKS", TASKS)), tuple(getattr(module, "SECONDS_RANGE", SECONDS_RANGE))  # type: ignore[return-value]


def check_generate(request: dict[str, Any]) -> None:
    job = str(request.get("job") or "")
    if not _JOB_ID.fullmatch(job):
        raise ValueError("job must be 1 to 64 letters, digits, '-' or '_'.")
    tasks, (min_seconds, max_seconds) = _engine_limits()
    if request.get("task", tasks[0]) not in tasks:
        raise ValueError(f"Task {request.get('task')!r} is not supported yet.")
    if not request.get("out_dir"):
        raise ValueError("out_dir is required.")
    for name in _PATH_FIELDS:
        if request.get(name):
            _inside_data_dir(request[name], name)
    seconds = float(request.get("seconds") or 0)
    if not min_seconds <= seconds <= max_seconds:
        raise ValueError(f"seconds must be between {min_seconds:g} and {max_seconds:g}.")
    if int(request.get("variations") or 1) not in (1, 2, 4):
        raise ValueError("variations must be 1, 2 or 4.")
    if request.get("seed") is None:
        raise ValueError("seed must be resolved before generation.")


class State:
    def __init__(self) -> None:
        self.busy: dict[str, Any] | None = None  # {"cmd", "job", "id", "cancel"}
        self.lock = threading.Lock()
        self.tasks: "queue.Queue[tuple[dict[str, Any], Callable[[dict[str, Any]], dict[str, Any]]]]" = queue.Queue()


STATE = State()


def _run_load(request: dict[str, Any]) -> dict[str, Any]:
    reason = engine_unavailable()
    if reason:
        raise RuntimeError(f"engine unavailable: {reason}")
    started = time.perf_counter()
    info = engine().load(str(request.get("model") or ""), request.get("lm") or None) or {}
    return {**info, "load_s": round(time.perf_counter() - started, 2), "footprint_gb": footprint_gb()}


def _run_unload(_request: dict[str, Any]) -> dict[str, Any]:
    if "engine" in _engine_cache:
        engine().unload()
    return {"footprint_gb": footprint_gb()}


def _run_generate(request: dict[str, Any]) -> dict[str, Any]:
    reason = engine_unavailable()
    if reason:
        raise RuntimeError(f"engine unavailable: {reason}")
    busy = STATE.busy or {}
    hooks = Hooks(request.get("id"), str(request["job"]), busy.get("cancel") or threading.Event())
    Path(str(request["out_dir"])).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with PeakSampler() as sampler:
        try:
            result = engine().generate(request, hooks)
        except Cancelled:
            result = None
    if hooks.cancelled:
        return {"ok": False, "cancelled": True}
    if result is None:
        raise RuntimeError("The engine returned no result.")
    for track in result.get("tracks") or []:
        _inside_data_dir(track.get("path"), "track path")
    return {
        "ok": True,
        "seconds": round(time.perf_counter() - started, 2),
        "tracks": result.get("tracks") or [],
        "timings": result.get("timings") or {},
        "peak_footprint_gb": round(sampler.peak, 2),
    }


_ENGINE_COMMANDS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "load": _run_load,
    "generate": _run_generate,
    "unload": _run_unload,
}


def engine_thread() -> None:
    """Runs every engine command in one thread, one at a time, and sends its reply."""
    while True:
        request, command = STATE.tasks.get()
        out: dict[str, Any] = {"id": request.get("id")}
        if request.get("job"):
            out["job"] = request["job"]
        try:
            result = command(request)
            out.update({"ok": True, **result})
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller
            traceback.print_exc(file=sys.stderr)
            out.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            with STATE.lock:
                STATE.busy = None
        reply(out)


def status(_request: dict[str, Any]) -> dict[str, Any]:
    loaded = engine().loaded() if "engine" in _engine_cache else {"model": None, "lm": None}
    busy = STATE.busy
    return {
        "model": loaded.get("model"),
        "lm": loaded.get("lm"),
        "busy": (busy.get("job") or busy.get("cmd")) if busy else None,
        "footprint_gb": footprint_gb(),
        "engine_version": ENGINE_VERSION,
        "pid": os.getpid(),
    }


def cancel(request: dict[str, Any]) -> dict[str, Any]:
    job = str(request.get("job") or "")
    with STATE.lock:
        busy = STATE.busy
        if busy and busy.get("job") == job and busy.get("cancel") is not None:
            busy["cancel"].set()
            return {"job": job, "state": "cancelling"}
    return {"job": job, "state": "not-running"}


def handle(request: dict[str, Any]) -> None:
    """Answer one request line (engine commands are queued and answer later)."""
    request_id = request.get("id")
    cmd = request.get("cmd")
    if cmd in _ENGINE_COMMANDS:
        if cmd == "generate":
            check_generate(request)
        with STATE.lock:
            if STATE.busy is not None:
                out = {"id": request_id, "ok": False, "error": "busy"}
                if request.get("job"):
                    out["job"] = request["job"]
                reply(out)
                return
            STATE.busy = {"cmd": cmd, "job": request.get("job"), "id": request_id,
                          "cancel": threading.Event() if cmd == "generate" else None}
        STATE.tasks.put((request, _ENGINE_COMMANDS[cmd]))
        return
    simple = {"ping": lambda _r: {"pong": True}, "status": status, "cancel": cancel}.get(str(cmd))
    if simple is None:
        raise ValueError(f"Unknown command {cmd!r}.")
    reply({"id": request_id, "ok": True, **simple(request)})


def main() -> None:
    threading.Thread(target=engine_thread, name="music-engine", daemon=True).start()
    reply({"id": None, "ok": True, "ready": True, "pid": os.getpid(), "engine_version": ENGINE_VERSION})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("A request must be a JSON object.")
            handle(request)
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller
            traceback.print_exc(file=sys.stderr)
            out = {"id": request.get("id"), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if request.get("job"):
                out["job"] = request["job"]
            reply(out)


if __name__ == "__main__":
    main()
