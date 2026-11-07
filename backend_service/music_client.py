"""Backend side of the music worker: start it, talk to it, cancel a job, stop it when idle.

One worker per backend. Every request gets an id; a reader thread routes each reply line (and every
``stage`` / ``step`` event) to the request it belongs to, so ``status`` and ``cancel`` reach the worker
while a ``generate`` is running. A request that does not answer in time kills the worker (``generate``:
30 s + 0.3 x seconds x variations, from the spike). Cancel is two-step: the cancel flag first, and if the
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


def generate_timeout_s(seconds: float, variations: int) -> float:
    """How long a ``generate`` may take before the worker is killed (spike section 6.3)."""
    return 30.0 + 0.3 * float(seconds) * max(1, int(variations))


class MusicWorker:
    def __init__(self) -> None:
        self._process: subprocess.Popen[str] | None = None
        self._start_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[Any, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._in_flight = 0
        self._last_used = time.monotonic()
        self.idle_minutes = 10.0
        self._log_path = music_env.MUSIC_DATA_DIR / "worker.log"
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
            if not music_env.is_installed():
                raise MusicWorkerError("The music pack is not installed. Install it from Music > Models.")
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115 - owned by the process
            log.write(f"\n--- music worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            ready: queue.Queue[dict[str, Any]] = queue.Queue()
            with self._pending_lock:
                self._pending[None] = ready
            self._process = subprocess.Popen(
                [str(music_env.venv_python()), str(_WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=music_env.worker_env(),
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
        **payload: Any,
    ) -> dict[str, Any]:
        """Send one command and wait for its final reply; events go to ``on_event``.

        ``cancel_check`` is polled while waiting: when it turns true a ``cancel`` is sent for the job,
        and the worker is killed if the job has not ended ``HARD_CANCEL_AFTER_S`` later.
        """
        if start:
            self._ensure_started()
        elif not self.running():
            raise MusicWorkerError("The music worker is not running.")
        request_id = next(self._ids)
        replies: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._pending_lock:
            self._pending[request_id] = replies
            self._in_flight += 1
        self._last_used = time.monotonic()
        try:
            self._send({"id": request_id, "cmd": cmd, **payload})
            return self._wait(request_id, replies, timeout, payload.get("job"), on_event, cancel_check)
        finally:
            self._last_used = time.monotonic()
            with self._pending_lock:
                self._in_flight -= 1
                self._pending.pop(request_id, None)

    def _wait(
        self,
        request_id: int,
        replies: queue.Queue[dict[str, Any]],
        timeout: float,
        job: Any,
        on_event: Callable[[dict[str, Any]], None] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        cancel_sent_at: float | None = None
        while True:
            now = time.monotonic()
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
            return self.request("status", timeout=5.0, start=False)
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


def stop_all() -> None:
    if _WORKER is not None:
        _WORKER.stop()


def worker_pid() -> int | None:
    return _WORKER.pid if _WORKER is not None else None
