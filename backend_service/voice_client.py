"""Backend side of the voice worker: start it, talk to it, stop it when idle.

One worker per backend. Requests are serialized (a speech model serves one
request at a time anyway); a request that does not answer in time kills the
worker, and the next request starts a fresh one. The worker exits after
``idle_minutes`` without use, which frees all of its memory.

Conversation mode runs voice activity detection in a second, small worker
(``get_vad_worker``) so a microphone frame never waits behind a sentence
being spoken or a recording being transcribed.
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
from typing import Any

from backend_service import voice_env

_WORKER_SCRIPT = Path(__file__).with_name("voice_worker.py")
_READY_TIMEOUT_S = 120.0


class VoiceWorkerError(RuntimeError):
    """The worker failed a request or could not start."""


class VoiceWorker:
    def __init__(self, log_name: str = "worker.log") -> None:
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._last_used = 0.0
        self.idle_minutes = 10.0
        self._log_path = voice_env.VOICE_DATA_DIR / log_name
        threading.Thread(target=self._idle_watch, daemon=True).start()

    # -- lifecycle -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def _start(self) -> None:
        if not voice_env.is_installed():
            raise VoiceWorkerError("The voice pack is not installed. Install it from Voice > Models.")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115 - owned by the process
        log.write(f"\n--- voice worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log.flush()
        self._lines = queue.Queue()
        self._process = subprocess.Popen(
            [str(voice_env.venv_python()), str(_WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=voice_env.worker_env(),
            cwd=str(voice_env.VOICE_DATA_DIR),
        )
        log.close()
        threading.Thread(target=self._read, args=(self._process,), daemon=True).start()
        ready = self._next_line(_READY_TIMEOUT_S)
        if not ready or not ready.get("ready"):
            self._kill()
            raise VoiceWorkerError(f"The voice worker did not start. See {self._log_path}.")

    def _read(self, process: subprocess.Popen[str]) -> None:
        for raw in process.stdout or []:
            try:
                self._lines.put(json.loads(raw))
            except ValueError:
                continue
        self._lines.put(None)  # stdout closed: the worker exited

    def _next_line(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

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

    def stop(self) -> None:
        with self._lock:
            self._kill()

    def _idle_watch(self) -> None:
        while True:
            time.sleep(30)
            if self._process is None or self.idle_minutes <= 0:
                continue
            if time.monotonic() - self._last_used > self.idle_minutes * 60 and self._lock.acquire(blocking=False):
                try:
                    if time.monotonic() - self._last_used > self.idle_minutes * 60:
                        self._kill()
                finally:
                    self._lock.release()

    # -- requests ------------------------------------------------------

    def request(self, cmd: str, timeout: float = 300.0, **payload: Any) -> dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()
            assert self._process is not None and self._process.stdin is not None
            request_id = next(self._ids)
            self._last_used = time.monotonic()
            try:
                self._process.stdin.write(json.dumps({"id": request_id, "cmd": cmd, **payload}) + "\n")
                self._process.stdin.flush()
            except OSError as exc:
                self._kill()
                raise VoiceWorkerError(f"The voice worker stopped: {exc}") from exc
            deadline = time.monotonic() + timeout
            while True:
                reply = self._next_line(max(0.0, deadline - time.monotonic()))
                if reply is None:
                    exited = self._process is None or self._process.poll() is not None
                    self._kill()
                    if exited:
                        raise VoiceWorkerError(f"The voice worker exited. See {self._log_path}.")
                    raise VoiceWorkerError(f"The voice worker did not answer within {int(timeout)} s.")
                if reply.get("id") != request_id:
                    continue  # a late reply to a request that timed out
                self._last_used = time.monotonic()
                if not reply.get("ok"):
                    raise VoiceWorkerError(str(reply.get("error") or "Voice request failed."))
                return reply


_WORKER: VoiceWorker | None = None
_WORKER_LOCK = threading.Lock()


def get_worker() -> VoiceWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = VoiceWorker()
            atexit.register(_WORKER.stop)
        return _WORKER


_VAD_WORKER: VoiceWorker | None = None


def get_vad_worker() -> VoiceWorker:
    global _VAD_WORKER
    with _WORKER_LOCK:
        if _VAD_WORKER is None:
            _VAD_WORKER = VoiceWorker(log_name="vad-worker.log")
            _VAD_WORKER.idle_minutes = 2.0
            atexit.register(_VAD_WORKER.stop)
        return _VAD_WORKER


def stop_all() -> None:
    for worker in (_WORKER, _VAD_WORKER):
        if worker is not None:
            worker.stop()


def worker_pid() -> int | None:
    return _WORKER.pid if _WORKER is not None else None
