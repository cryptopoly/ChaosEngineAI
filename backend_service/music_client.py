"""Backend side of the music worker: start it, talk to it, cancel a job, stop it when idle.

One worker per backend. Every request gets an id; a reader thread routes each reply line (and every
``stage`` / ``step`` event) to the request it belongs to, so ``status`` and ``cancel`` reach the worker
while a ``generate`` is running. A request that does not answer in time kills the worker (``generate``:
the model's own run estimate times ``SAFETY_FACTOR`` plus its cold start, so a slower Mac or the XL model is
not cut off). A stall watchdog kills sooner when no stage or step event has come for the current stage's
limit (``stall_limits``; generous for VAE decode, which sends none). Cancel is two-step: the cancel flag first, and if the
job has not ended 3 s later (it is in VAE decode, which has no hook) the worker is killed. The worker is
stopped after ``idle_minutes`` without use, which frees all of its memory.
"""

from __future__ import annotations

import atexit
import itertools
import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend_service import music_env

_WORKER_SCRIPT = Path(__file__).with_name("music_worker.py")
_READY_TIMEOUT_S = 120.0
HARD_CANCEL_AFTER_S = 3.0
_POLL_S = 0.2
_CLOSED: dict[str, Any] = {"__closed__": True}


class MusicWorkerError(RuntimeError):
    """The worker failed a request, timed out or could not start."""


class MusicCancelled(RuntimeError):
    """The job was cancelled (cooperatively or by stopping the worker)."""


SAFETY_FACTOR = 4.0  # the estimates come from an M4 Max; a slower Mac may take several times as long
_BASE_S = 30.0
STALL_DEFAULT_S = 180.0  # load, encode, diffusion steps and write all send events well inside this


def _estimate(model: str | None, seconds: float, variations: int) -> tuple[float, float, dict[str, Any]]:
    """(warm wall seconds, cold-start seconds, the row's estimates) from the catalog row."""
    from backend_service.music_runtime import estimate_run, resolve_variant

    run = estimate_run(model, seconds, variations, warm=False)
    return run["seconds"] - run["coldStartSeconds"], run["coldStartSeconds"], resolve_variant(model)["estimates"]


def generate_timeout_s(seconds: float, variations: int, model: str | None = None) -> float:
    """How long a ``generate`` may take in all before the worker is killed: estimate x safety + cold start."""
    warm, cold, _ = _estimate(model, seconds, variations)
    return round(_BASE_S + SAFETY_FACTOR * warm + cold, 1)


def stall_limits(seconds: float, variations: int, model: str | None = None) -> dict[str, float]:
    """Seconds without any event before a ``generate`` counts as stuck, per stage.

    ``load`` covers a model load inside the job (and the time before the first event); ``decode`` is the
    VAE decode, which sends no events and grows with the audio length; every other stage sends events.
    """
    _, cold, est = _estimate(model, seconds, variations)
    decode = float(est.get("vaeSecondsPerAudioSecond") or 0.1) * float(seconds) * max(1, int(variations))
    return {
        "load": round(max(STALL_DEFAULT_S, SAFETY_FACTOR * cold + 60.0), 1),
        "decode": round(max(STALL_DEFAULT_S, SAFETY_FACTOR * decode + 60.0), 1),
        "default": STALL_DEFAULT_S,
    }


class MusicWorker:
    """One worker process. ``pack`` is the module that owns its venv (``music_env`` by default, or
    ``music_sound_env`` for the sounds worker): it provides ``is_installed``, ``venv_python`` and ``worker_env``."""

    def __init__(self, pack: Any = None, log_name: str = "worker.log", label: str = "music") -> None:
        self._pack = pack
        self._label = label
        self._process: subprocess.Popen[str] | None = None
        self._start_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[Any, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._in_flight = 0
        self._heavy = 0  # load and generate requests in flight (status polls do not count)
        self._last_used = time.monotonic()
        self.idle_minutes = 10.0
        self._log_path = music_env.MUSIC_DATA_DIR / log_name
        threading.Thread(target=self._idle_watch, daemon=True).start()

    # -- lifecycle -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def running(self) -> bool:
        return self.pid is not None

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._kill()
            pack = self._pack or music_env
            if not pack.is_installed():
                raise MusicWorkerError(f"The {self._label} pack is not installed. Install it from Music > Models.")
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115 - owned by the process
            log.write(f"\n--- music worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            ready: queue.Queue[dict[str, Any]] = queue.Queue()
            with self._pending_lock:
                self._pending[None] = ready
            self._process = subprocess.Popen(
                [str(pack.venv_python()), str(_WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=pack.worker_env(),
                cwd=str(music_env.MUSIC_DATA_DIR),
            )
            log.close()
            threading.Thread(target=self._read, args=(self._process,), daemon=True).start()
            try:
                line = ready.get(timeout=_READY_TIMEOUT_S)
            except queue.Empty:
                line = None
            with self._pending_lock:
                self._pending.pop(None, None)
            if not line or not line.get("ready"):
                self._kill()
                raise MusicWorkerError(f"The music worker did not start. See {self._log_path}.")

    def _read(self, process: subprocess.Popen[str]) -> None:
        for raw in process.stdout or []:
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(line, dict):
                continue
            with self._pending_lock:
                target = self._pending.get(line.get("id"))
            if target is not None:
                target.put(line)
        with self._pending_lock:
            if self._process is not process and self._process is not None:
                return
            waiting = list(self._pending.values())
        for target in waiting:
            target.put(_CLOSED)  # stdout closed: the worker exited

    def _kill(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - kill whatever is left
            process.kill()
            try:
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - nothing more to do
                pass
        finally:
            if process.stdout:
                process.stdout.close()
        with self._pending_lock:
            waiting = list(self._pending.values())
        for target in waiting:
            target.put(_CLOSED)

    def stop(self) -> None:
        with self._start_lock:
            self._kill()

    def stop_if_idle(self) -> bool:
        """Stop the worker when it runs and nothing is in flight (another kind needs the memory)."""
        with self._pending_lock:
            busy = self._in_flight > 0
        if busy or not self.running():
            return False
        self.stop()
        return True

    def busy(self) -> bool:
        """A model load or a generation is running in the worker."""
        with self._pending_lock:
            return self._heavy > 0

    def _idle_watch(self) -> None:
        while True:
            time.sleep(30)
            if self.idle_expired():
                self.stop()

    def idle_expired(self, now: float | None = None) -> bool:
        if self._process is None or self.idle_minutes <= 0 or self._in_flight:
            return False
        return (now or time.monotonic()) - self._last_used > self.idle_minutes * 60

    # -- requests ------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise MusicWorkerError("The music worker is not running.")
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
        except (OSError, ValueError) as exc:
            self.stop()
            raise MusicWorkerError(f"The music worker stopped: {exc}") from exc

    def request(
        self,
        cmd: str,
        timeout: float = 120.0,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        start: bool = True,
        touch: bool = True,
        stall_limits: dict[str, float] | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Send one command and wait for its final reply; events go to ``on_event``.

        ``cancel_check`` is polled while waiting: when it turns true a ``cancel`` is sent for the job,
        and the worker is killed if the job has not ended ``HARD_CANCEL_AFTER_S`` later. ``touch=False``
        (status polls) leaves the idle clock alone, so a UI that polls never keeps the worker alive.
        ``stall_limits`` (stage -> seconds, with ``default``; ``load`` until the first stage) kills the
        worker when no event has arrived for that long.
        """
        if start:
            self._ensure_started()
        elif not self.running():
            raise MusicWorkerError("The music worker is not running.")
        request_id = next(self._ids)
        replies: queue.Queue[dict[str, Any]] = queue.Queue()
        heavy = cmd in ("load", "generate")
        with self._pending_lock:
            self._pending[request_id] = replies
            self._in_flight += 1
            self._heavy += heavy
        if touch:
            self._last_used = time.monotonic()
        try:
            self._send({"id": request_id, "cmd": cmd, **payload})
            return self._wait(request_id, replies, timeout, payload.get("job"), on_event, cancel_check, stall_limits)
        finally:
            if touch:
                self._last_used = time.monotonic()
            with self._pending_lock:
                self._in_flight -= 1
                self._heavy -= heavy
                self._pending.pop(request_id, None)

    def _wait(
        self,
        request_id: int,
        replies: queue.Queue[dict[str, Any]],
        timeout: float,
        job: Any,
        on_event: Callable[[dict[str, Any]], None] | None,
        cancel_check: Callable[[], bool] | None,
        stall: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        cancel_sent_at: float | None = None
        last_event, stage = time.monotonic(), "load"
        while True:
            now = time.monotonic()
            if stall is not None and cancel_sent_at is None:
                limit = stall.get(stage, stall.get("default", timeout))
                if now - last_event > limit:
                    self.stop()
                    raise MusicWorkerError(f"The music worker stopped: no progress for {int(limit)} s ({stage}). See {self._log_path}.")
            if cancel_check is not None and cancel_sent_at is None and cancel_check():
                cancel_sent_at = now
                if job:
                    self._send_cancel(str(job))
            if cancel_sent_at is not None and now - cancel_sent_at > HARD_CANCEL_AFTER_S:
                self.stop()
                raise MusicCancelled("Cancelled; the music worker was stopped because the job did not end in time.")
            if now > deadline:
                self.stop()
                raise MusicWorkerError(f"The music worker did not answer within {int(timeout)} s.")
            try:
                line = replies.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if line is _CLOSED or line.get("__closed__"):
                if cancel_sent_at is not None:
                    raise MusicCancelled("Cancelled.")
                raise MusicWorkerError(f"The music worker exited. See {self._log_path}.")
            if line.get("event"):
                last_event = time.monotonic()
                if line.get("event") == "stage":
                    stage = str(line.get("stage") or stage)
                elif line.get("event") == "step":
                    stage = "diffusion"
                if on_event is not None:
                    on_event(line)
                continue
            if line.get("cancelled"):
                raise MusicCancelled("Cancelled.")
            if not line.get("ok"):
                raise MusicWorkerError(str(line.get("error") or "Music request failed."))
            return line

    def _send_cancel(self, job: str) -> None:
        cancel_id = next(self._ids)
        try:
            self._send({"id": cancel_id, "cmd": "cancel", "job": job})
        except MusicWorkerError:
            pass  # the worker is gone already; the wait loop sees it

    def status(self) -> dict[str, Any] | None:
        """The worker's own status, or ``None`` when it is not running (never starts it)."""
        if not self.running():
            return None
        try:
            return self.request("status", timeout=5.0, start=False, touch=False)
        except MusicWorkerError:
            return None


_WORKER: MusicWorker | None = None
_WORKER_LOCK = threading.Lock()


def get_worker() -> MusicWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = MusicWorker()
            atexit.register(_WORKER.stop)
        return _WORKER


_SOUND_WORKER: MusicWorker | None = None


def get_sound_worker() -> MusicWorker:
    """The sounds worker: the same worker script in the sounds venv (``music_sound_env``)."""
    global _SOUND_WORKER
    with _WORKER_LOCK:
        if _SOUND_WORKER is None:
            from backend_service import music_sound_env

            _SOUND_WORKER = MusicWorker(pack=music_sound_env, log_name="sound-worker.log", label="sounds")
            atexit.register(_SOUND_WORKER.stop)
        return _SOUND_WORKER


def stop_all() -> None:
    for worker in (_WORKER, _SOUND_WORKER):
        if worker is not None:
            worker.stop()


def worker_pid() -> int | None:
    return _WORKER.pid if _WORKER is not None else None
