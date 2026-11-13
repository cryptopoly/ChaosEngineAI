"""Backend side of the warm image and video workers (I1, V1) and their keep-warm policy (C2, C5).

One ``WarmWorker`` per engine (mflux, ltx-2-mlx). It starts the worker script in the engine's own venv,
sends JSON lines, routes each reply and event to its request (a reader thread), cancels a job (the
cancel flag first, a kill ``hard_cancel_after_s`` later) and kills a worker that stops answering.
Modelled on ``music_client.py``.

Keep-warm policy, without settings:

* idle stop after ``IDLE_MINUTES`` (10, as the music and voice workers);
* before a heavy job of another kind takes the ``HEAVY_JOBS`` slot (image, video, music), every idle
  warm worker of a different kind stops, so a storyboard keeps one model for all stills, then one for
  all clips, and two big models never sit in memory together;
* when free memory runs low while a worker idles, it stops;
* a worker whose footprint after a job is over ``KEEP_WARM_MAX_FRACTION`` of the Mac's memory stops at
  once (a 50 GB model is not worth keeping on a 64 GB Mac);
* a busy worker is never killed by the policy (a running job is cancelled by the user, not by us).

``CHAOSENGINE_WARM_WORKERS=0`` turns warm workers off; the engines then run one CLI process per job.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

IDLE_MINUTES = 10.0
KEEP_WARM_MAX_FRACTION = 0.5
LOW_MEMORY_AVAILABLE_GB = 6.0
HEAVY_KINDS = frozenset({"image", "video", "music"})
_READY_TIMEOUT_S = 120.0
_POLL_S = 0.2
_WATCH_S = 30.0
_CLOSED: dict[str, Any] = {"__closed__": True}


class WarmWorkerError(RuntimeError):
    """The worker failed a request, timed out or could not start."""


class WarmWorkerCancelled(RuntimeError):
    """The job was cancelled (cooperatively or by stopping the worker)."""


def warm_workers_enabled() -> bool:
    return os.environ.get("CHAOSENGINE_WARM_WORKERS", "1").strip().lower() not in ("0", "false", "no", "off")


def total_memory_gb() -> float:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


class WarmWorker:
    def __init__(
        self,
        *,
        name: str,
        kind: str,
        command: Callable[[], list[str]],
        env: Callable[[], dict[str, str]],
        log_path: Path,
        hard_cancel_after_s: float = 10.0,
    ) -> None:
        self.name, self.kind = name, kind
        self._command, self._env, self._log_path = command, env, log_path
        self.hard_cancel_after_s = hard_cancel_after_s
        self.idle_minutes = IDLE_MINUTES
        self._process: subprocess.Popen[str] | None = None
        # Set by the reader thread when a worker's output stream closes (the process is gone). On Windows
        # ``Popen.poll()`` can still read ``None`` for a worker that called ``os._exit``, so stream-close is
        # the reliable death signal: without it ``_ensure_started`` would reuse a dead process and the next
        # write would raise ``OSError 22``.
        self._crashed = False
        self._start_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[Any, queue.Queue[dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._in_flight = 0
        self._last_used = time.monotonic()
        self._watching = False
        self.last_stop_reason: str | None = None
        register(self)

    # -- lifecycle -----------------------------------------------------

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def running(self) -> bool:
        return self.pid is not None

    def busy(self) -> bool:
        with self._pending_lock:
            return self._in_flight > 0

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._process is not None and self._process.poll() is None and not self._crashed:
                return
            self._crashed = False
            self._kill()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(self._log_path, "a", encoding="utf-8")  # noqa: SIM115 - owned by the process
            log.write(f"\n--- {self.name} worker start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            log.flush()
            ready: queue.Queue[dict[str, Any]] = queue.Queue()
            with self._pending_lock:
                self._pending[None] = ready
            try:
                self._process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    env=self._env(),
                )
            except OSError as exc:
                with self._pending_lock:
                    self._pending.pop(None, None)
                raise WarmWorkerError(f"The {self.name} worker did not start: {exc}") from exc
            finally:
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
                raise WarmWorkerError(f"The {self.name} worker did not start. See {self._log_path}.")
            self._last_used = time.monotonic()
            if not self._watching:
                self._watching = True
                threading.Thread(target=self._watch, daemon=True).start()

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
            if self._process is process:
                # The stream closed, so this worker is gone; mark it so the next request restarts it
                # instead of reusing a dead process whose ``poll()`` may still read ``None`` on Windows.
                self._crashed = True
            waiting = list(self._pending.values())
        for target in waiting:
            target.put(_CLOSED)

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
            except Exception:  # noqa: BLE001
                pass
        finally:
            # On Windows closing a pipe the reader thread is still on can raise OSError 22; the process is
            # already gone, so a failed close must not turn a clean restart into a raw error.
            if process.stdout:
                try:
                    process.stdout.close()
                except (OSError, ValueError):
                    pass
        with self._pending_lock:
            waiting = list(self._pending.values())
        for target in waiting:
            target.put(_CLOSED)

    def stop(self, reason: str = "stopped") -> None:
        with self._start_lock:
            if self._process is not None:
                self.last_stop_reason = reason
            self._kill()

    def stop_if_idle(self, reason: str = "another job needs the memory") -> bool:
        """Stop the worker when it runs and nothing is in flight. Never kills a running job."""
        if self.busy() or not self.running():
            return False
        self.stop(reason)
        return True

    def idle_expired(self, now: float | None = None) -> bool:
        if self._process is None or self.idle_minutes <= 0 or self.busy():
            return False
        return (now or time.monotonic()) - self._last_used > self.idle_minutes * 60

    def _watch(self) -> None:
        while True:
            time.sleep(_WATCH_S)
            if not self.running():
                continue
            if self.idle_expired():
                self.stop(f"idle for {self.idle_minutes:g} minutes")
            elif not self.busy() and memory_is_low():
                self.stop("free memory ran low")

    # -- requests ------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise WarmWorkerError(f"The {self.name} worker is not running.")
        try:
            with self._write_lock:
                process.stdin.write(json.dumps(payload) + "\n")
                process.stdin.flush()
        except (OSError, ValueError) as exc:
            self.stop("its pipe broke")
            raise WarmWorkerError(f"The {self.name} worker stopped: {exc}") from exc

    def request(
        self,
        cmd: str,
        timeout: float = 120.0,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        start: bool = True,
        stall_s: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Send one command and wait for its final reply; events go to ``on_event``.

        ``cancel_check`` is polled while waiting: when it turns true a ``cancel`` is sent for the job, and
        the worker is killed if the job has not ended ``hard_cancel_after_s`` later. ``stall_s`` kills the
        worker when no event has arrived for that long.
        """
        if start:
            self._ensure_started()
        elif not self.running():
            raise WarmWorkerError(f"The {self.name} worker is not running.")
        request_id = next(self._ids)
        replies: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._pending_lock:
            self._pending[request_id] = replies
            self._in_flight += 1
        self._last_used = time.monotonic()
        try:
            self._send({"id": request_id, "cmd": cmd, **payload})
            return self._wait(replies, timeout, payload.get("job"), on_event, cancel_check, stall_s)
        finally:
            self._last_used = time.monotonic()
            with self._pending_lock:
                self._in_flight -= 1
                self._pending.pop(request_id, None)

    def _wait(
        self,
        replies: queue.Queue[dict[str, Any]],
        timeout: float,
        job: Any,
        on_event: Callable[[dict[str, Any]], None] | None,
        cancel_check: Callable[[], bool] | None,
        stall_s: float | None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        cancel_sent_at: float | None = None
        last_event = time.monotonic()
        while True:
            now = time.monotonic()
            if stall_s is not None and cancel_sent_at is None and now - last_event > stall_s:
                self.stop("it stalled")
                raise WarmWorkerError(f"The {self.name} worker stopped: no progress for {int(stall_s)} s. See {self._log_path}.")
            if cancel_check is not None and cancel_sent_at is None and cancel_check():
                cancel_sent_at = now
                if job:
                    try:
                        self._send({"id": next(self._ids), "cmd": "cancel", "job": str(job)})
                    except WarmWorkerError:
                        pass
            if cancel_sent_at is not None and now - cancel_sent_at > self.hard_cancel_after_s:
                self.stop("a cancelled job did not end in time")
                raise WarmWorkerCancelled("Cancelled; the worker was stopped because the job did not end in time.")
            if now > deadline:
                self.stop("it did not answer in time")
                raise WarmWorkerError(f"The {self.name} worker did not answer within {int(timeout)} s.")
            try:
                line = replies.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            if line is _CLOSED or line.get("__closed__"):
                if cancel_sent_at is not None:
                    raise WarmWorkerCancelled("Cancelled.")
                raise WarmWorkerError(f"The {self.name} worker exited. See {self._log_path}.")
            if line.get("event"):
                last_event = time.monotonic()
                if on_event is not None:
                    on_event(line)
                continue
            if line.get("cancelled"):
                raise WarmWorkerCancelled("Cancelled.")
            if not line.get("ok"):
                raise WarmWorkerError(str(line.get("error") or f"The {self.name} request failed."))
            return line

    def after_job(self, reply: dict[str, Any]) -> None:
        """Stop right away when the model is too big to keep (C2)."""
        footprint = float(reply.get("footprint_gb") or 0.0)
        total = total_memory_gb()
        if total and footprint > KEEP_WARM_MAX_FRACTION * total:
            self.stop_if_idle(f"its {footprint:.0f} GB footprint is over half of this Mac's memory")

    def status(self) -> dict[str, Any] | None:
        if not self.running():
            return None
        try:
            return self.request("status", timeout=5.0, start=False)
        except WarmWorkerError:
            return None


def available_memory_gb() -> float | None:
    try:
        from backend_service.helpers.memory_gate import snapshot_memory_signals

        available_gb, _pressure = snapshot_memory_signals()
    except Exception:  # noqa: BLE001
        return None
    return float(available_gb) if available_gb is not None else None


def can_start(worker: "WarmWorker | None", min_available_gb: float) -> bool:
    """A running worker already holds its memory; a new one starts only with this much free memory
    (its resident set plus its peak above the CLI's), so it never pushes a busy Mac into swap."""
    if worker is not None and worker.running():
        return True
    available = available_memory_gb()
    return available is None or available >= min_available_gb


def memory_is_low() -> bool:
    try:
        from backend_service.helpers.memory_gate import snapshot_memory_signals

        available_gb, _pressure = snapshot_memory_signals()
    except Exception:  # noqa: BLE001 - a broken memory read never stops a worker
        return False
    return available_gb is not None and float(available_gb) < LOW_MEMORY_AVAILABLE_GB


# -- registry and the HEAVY_JOBS hook ------------------------------------------

_WORKERS: list[WarmWorker] = []
_REGISTRY_LOCK = threading.Lock()
_HOOKED = False


def register(worker: WarmWorker) -> None:
    global _HOOKED
    with _REGISTRY_LOCK:
        _WORKERS.append(worker)
        if not _HOOKED:
            _HOOKED = True
            atexit.register(stop_all)
            try:
                from backend_service.helpers.job_queue import HEAVY_JOBS

                HEAVY_JOBS.add_start_listener(release_for_kind)
            except Exception:  # noqa: BLE001 - the idle stop still frees memory
                pass


def release_for_kind(kind: str) -> list[str]:
    """A heavy job of ``kind`` is starting: stop every idle warm worker of another kind (C2, C5)."""
    if kind not in HEAVY_KINDS:
        return []
    with _REGISTRY_LOCK:
        workers = list(_WORKERS)
    return [w.name for w in workers if w.kind != kind and w.stop_if_idle(f"a {kind} job started")]


def stop_all() -> None:
    with _REGISTRY_LOCK:
        workers = list(_WORKERS)
    for worker in workers:
        worker.stop("the backend stopped")
