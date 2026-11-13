"""Backend side of the voice worker: start it, talk to it, stop it when idle.

One worker per backend. Requests are serialized (a speech model serves one
request at a time anyway); a request that does not answer in time kills the
worker, and the next request starts a fresh one. The worker exits after
``idle_minutes`` without use, which frees all of its memory.

Conversation mode runs voice activity detection in a second, small worker
(``get_vad_worker``) so a microphone frame never waits behind a sentence
being spoken or a recording being transcribed.

Qwen3-TTS on an NVIDIA GPU runs in a third worker (``get_qwen_worker``): the same script and the same Python
and PyTorch, with the pinned transformers 4.57 that ``qwen-tts`` needs put first on ``PYTHONPATH``, because the
speech GPU add-on's own transformers 5.2 (which Chatterbox pins) breaks its generation. Only one of the two
speech workers holds a model at a time (``speech_worker_for`` frees the other).
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
# How long a worker whose stdin was closed gets to finish its request loop and run its atexit cleanup.
_GRACEFUL_EXIT_S = 5.0


def _end_children(pid: int) -> None:
    """End a worker's own child processes (Orpheus's llama-server) before the worker is killed: Windows does not
    take a killed process's children with it."""
    try:
        import psutil  # noqa: PLC0415 - in the backend's base install
    except ImportError:
        return
    try:
        children = psutil.Process(pid).children(recursive=True)
    except psutil.Error:
        return
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
# Put on the line queue when the worker's stdout closes, so a dead worker is told
# apart from a slow one even before the OS has reaped it (Windows lags here).
_CLOSED: dict[str, Any] = {}


class VoiceWorkerError(RuntimeError):
    """The worker failed a request or could not start."""


class VoiceWorker:
    def __init__(self, log_name: str = "worker.log", env_extra: dict[str, str] | None = None) -> None:
        self._env_extra = env_extra or {}
        self._process: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._last_used = 0.0
        self.idle_minutes = 10.0
        self._log_path = voice_env.VOICE_DATA_DIR / log_name
        # What the worker last loaded, per lane ("stt", "tts"): the ``source`` the request named (a repo id, a
        # snapshot folder or the Kokoro folder). Shown as "In memory now" and checked before a model is deleted.
        self.loaded: dict[str, str] = {}
        threading.Thread(target=self._idle_watch, daemon=True).start()

    # -- lifecycle -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def _start(self) -> None:
        if not voice_env.is_installed():
            raise VoiceWorkerError("The voice runtime is not installed. Install it from Voice > Models.")
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
            env={**voice_env.worker_env(), **self._env_extra},
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
        self._lines.put(_CLOSED)  # stdout closed: the worker exited

    def _next_line(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def _kill(self) -> None:
        process, self._process = self._process, None
        self.loaded = {}
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            # Closing stdin ends the worker's request loop; let it exit on its own so its atexit handlers run
            # (the Orpheus GGUF lane stops its llama-server there). terminate() on Windows is TerminateProcess,
            # which skips them and left llama-server running after Stop (RTX 4090 run).
            try:
                process.wait(timeout=_GRACEFUL_EXIT_S)
            except subprocess.TimeoutExpired:
                _end_children(process.pid)
                process.terminate()
                process.wait(timeout=5)
        except Exception:  # noqa: BLE001 - kill whatever is left
            _end_children(process.pid)
            process.kill()
            try:
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - nothing more to do
                pass
        finally:
            if process.stdout:
                process.stdout.close()

    def stop(self) -> None:
        with self._lock:
            self._kill()

    def stop_if_idle(self) -> bool:
        """Stop the worker unless a request is running. True when a running worker was stopped."""
        if self._process is None or not self._lock.acquire(blocking=False):
            return False
        try:
            if self._process is None:
                return False
            self._kill()
            return True
        finally:
            self._lock.release()

    def interrupt(self) -> None:
        """End a running request now. ``stop`` waits for the request lock; this does not.

        The reader sees the worker exit, so the waiting ``request`` raises and cleans up; the next request starts a fresh worker.
        """
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

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
                if reply is None or reply is _CLOSED:
                    exited = reply is _CLOSED or self._process is None or self._process.poll() is not None
                    self._kill()
                    if exited:
                        raise VoiceWorkerError(f"The voice worker exited. See {self._log_path}.")
                    raise VoiceWorkerError(f"The voice worker did not answer within {int(timeout)} s.")
                if reply.get("id") != request_id:
                    continue  # a late reply to a request that timed out
                self._last_used = time.monotonic()
                if not reply.get("ok"):
                    raise VoiceWorkerError(str(reply.get("error") or "Voice request failed."))
                lane = {"transcribe": "stt", "synthesize": "tts"}.get(cmd)
                if lane and payload.get("source"):
                    self.loaded[lane] = str(payload["source"])
                elif cmd == "unload":
                    what = payload.get("what") or "all"
                    self.loaded = {} if what == "all" else {k: v for k, v in self.loaded.items() if k != what}
                return reply

    @property
    def busy(self) -> bool:
        """A request is being answered right now (a sentence, a transcript, a model load)."""
        return self._lock.locked()


_WORKER: VoiceWorker | None = None
_WORKER_LOCK = threading.Lock()
# The user's "unload after N idle minutes" (Voice settings), for both speech workers. None until first read.
_SPEECH_IDLE_MINUTES: float | None = None


def _speech_idle_minutes() -> float:
    """The stored setting; read once, so a worker created after a backend restart honours it too."""
    global _SPEECH_IDLE_MINUTES
    if _SPEECH_IDLE_MINUTES is None:
        try:
            from backend_service.voice_runtime import load_settings  # noqa: PLC0415 - voice_runtime imports this module

            _SPEECH_IDLE_MINUTES = float(load_settings()["idleUnloadMinutes"])
        except Exception:  # noqa: BLE001 - a bad settings file keeps the default
            _SPEECH_IDLE_MINUTES = 10.0
    return _SPEECH_IDLE_MINUTES


def set_speech_idle_minutes(minutes: float) -> None:
    """Apply the idle-unload setting to both speech workers (the Qwen3-TTS one holds the most GPU memory)."""
    global _SPEECH_IDLE_MINUTES
    _SPEECH_IDLE_MINUTES = float(minutes)
    for worker in _speech_workers():
        worker.idle_minutes = _SPEECH_IDLE_MINUTES


def get_worker() -> VoiceWorker:
    global _WORKER
    minutes = _speech_idle_minutes()
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = VoiceWorker()
            _WORKER.idle_minutes = minutes
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


_QWEN_WORKER: VoiceWorker | None = None


def get_qwen_worker() -> VoiceWorker:
    """The worker that runs Qwen3-TTS on the GPU: the main worker's Python with the pinned transformers first."""
    global _QWEN_WORKER
    minutes = _speech_idle_minutes()
    with _WORKER_LOCK:
        if _QWEN_WORKER is None:
            _QWEN_WORKER = VoiceWorker(log_name="qwen-worker.log", env_extra=voice_env.qwen_worker_env())
            _QWEN_WORKER.idle_minutes = minutes
            atexit.register(_QWEN_WORKER.stop)
        return _QWEN_WORKER


def uses_qwen_worker(engine: str, family: str) -> bool:
    return engine == "torch" and family == "qwen3-tts"


def speech_worker_for(engine: str, family: str) -> VoiceWorker:
    """The worker that serves this speech model. A model held by the other speech worker is unloaded first, so two
    speech models never sit in GPU memory together (an unload of a worker that is not running costs nothing)."""
    wanted, other = ((get_qwen_worker(), _WORKER) if uses_qwen_worker(engine, family) else (get_worker(), _QWEN_WORKER))
    held = other.loaded.get("tts", "") if other is not None and other.pid is not None else ""
    if held and "kokoro" not in held.lower():  # Kokoro is small and runs on the CPU; it can stay
        if other.busy:
            raise VoiceWorkerError("Another voice is speaking. Try again when it has finished.")
        other.request("unload", timeout=60.0, what="tts")
    return wanted


def _speech_workers() -> list[VoiceWorker]:
    return [w for w in (_WORKER, _QWEN_WORKER) if w is not None]


def stop_all() -> None:
    for worker in (*_speech_workers(), _VAD_WORKER):
        if worker is not None:
            worker.stop()


def interrupt_all() -> None:
    for worker in _speech_workers():
        worker.interrupt()


def worker_pid() -> int | None:
    return _WORKER.pid if _WORKER is not None else None


def worker_busy() -> bool:
    """The speech worker is answering a request (read-aloud, a call turn, Speak, a transcript)."""
    return any(worker.busy for worker in _speech_workers())


def loaded_sources() -> dict[str, str]:
    """``{"stt": source, "tts": source}`` of the models the running worker holds; empty when it is not running."""
    held: dict[str, str] = {}
    for worker in _speech_workers():
        if worker.pid is not None:
            held.update(worker.loaded)
    return held


def release_gpu_for_media() -> list[str]:
    """Stop idle speech workers that hold a model on the NVIDIA GPU, before an image or video job loads its own.

    An idle Qwen3-TTS worker keeps about 4.7 GB of the card until its idle unload; an LTX-2.5 clip that started
    next to it spilled into shared system memory and took about 600 s for its first step instead of about 5 s
    (RTX 4090 24 GB). Kokoro runs on the CPU and is left alone; a worker answering a request is left alone. The next
    voice request starts a fresh worker (one model load). Returns the models that were released.
    """
    if not voice_env.voice_uses_cuda():
        return []
    released: list[str] = []
    for worker in _speech_workers():
        held = [source for source in worker.loaded.values() if "kokoro" not in source.lower()]
        if worker.pid is not None and held and worker.stop_if_idle():
            released.extend(held)
    return released


def stop_speech_worker() -> None:
    """Stop the speech workers (not the small turn-detection one), freeing every model they hold."""
    for worker in _speech_workers():
        worker.stop()
