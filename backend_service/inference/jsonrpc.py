"""Subprocess JSON-RPC bridge for the MLX worker.

The MLX inference path runs in a subprocess (mlx_lm import is heavy
and segfaulty if it cohabits with PyTorch in the same address space).
``JsonRpcProcess`` owns the worker's lifecycle:

- ``start()`` — spawn or no-op if already alive.
- ``request()`` / ``stream_request()`` — send a JSON line on stdin,
  wait for matching response lines on stdout. ``stream_request`` is
  used for token-stream generation; ``request`` is used for one-shot
  ops (load, status, unload).
- ``close()`` — graceful terminate, escalate to SIGKILL after 5 s.
- ``is_alive()`` — for health probes.

Stdout reads run on a daemon ``Thread`` so the request-side timeout
fires precisely instead of blocking on a slow line. ``_lock`` serialises
requests so two callers can't interleave their JSON-RPC traffic on the
same pipe.

Extracted from ``inference.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import json
import subprocess
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Iterator

from backend_service.inference._constants import (
    DEFAULT_MLX_TIMEOUT_SECONDS,
    WORKSPACE_ROOT,
)


class JsonRpcProcess:
    def __init__(self, command: list[str], *, timeout: float = DEFAULT_MLX_TIMEOUT_SECONDS) -> None:
        self.command = command
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._stdout_queue: Queue[str | None] = Queue()
        self._reader_thread: Thread | None = None
        self._lock = Lock()

    def _pump_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self._stdout_queue.put(line.rstrip("\n"))
        self._stdout_queue.put(None)

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self.process = subprocess.Popen(
            self.command,
            cwd=str(WORKSPACE_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = Queue()
        self._reader_thread = Thread(target=self._pump_stdout, daemon=True)
        self._reader_thread.start()

    def close(self, *, force: bool = False) -> None:
        if self.process is None:
            return

        try:
            if self.process.stdin is not None and self.process.poll() is None:
                self.process.stdin.close()
        except OSError:
            pass

        if self.process.poll() is None:
            if force:
                self.process.kill()
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def is_alive(self) -> bool:
        """Return True if the worker process is running."""
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def _exit_detail(stderr: str, return_code: int | None) -> str:
        if stderr:
            return stderr
        if return_code == -9:
            return (
                "worker was SIGKILLed by the OS. This usually means memory pressure "
                "during MLX model load or startup."
            )
        return f"worker exited with code {return_code}"

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return self.request_with_progress(payload, on_progress=None, timeout=timeout)

    def request_with_progress(
        self,
        payload: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        effective_timeout = timeout if timeout is not None else self.timeout
        with self._lock:
            self.start()
            assert self.process is not None and self.process.stdin is not None

            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except OSError as exc:
                self.close()
                raise RuntimeError(f"Native worker stdin failed: {exc}") from exc

            while True:
                try:
                    line = self._stdout_queue.get(timeout=effective_timeout)
                except Empty as exc:
                    self.close()
                    raise RuntimeError(
                        f"Timed out waiting for the MLX worker after {effective_timeout:.0f}s."
                    ) from exc

                if line is None:
                    stderr = ""
                    if self.process is not None and self.process.stderr is not None:
                        try:
                            stderr = self.process.stderr.read().strip()
                        except OSError:
                            stderr = ""
                    return_code = self.process.poll() if self.process else None
                    self.close()
                    detail = self._exit_detail(stderr, return_code)
                    raise RuntimeError(f"MLX worker exited unexpectedly: {detail}")

                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.close()
                    raise RuntimeError(f"MLX worker returned invalid JSON: {line}") from exc

                if not response.get("ok", False):
                    raise RuntimeError(str(response.get("error") or "MLX worker returned an unknown error."))

                # Intermediate progress message — keep reading.
                if "result" not in response and "progress" in response:
                    if on_progress is not None:
                        try:
                            on_progress(response.get("progress") or {})
                        except Exception:
                            pass
                    continue

                result = response.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("MLX worker returned an invalid result payload.")
                return result

    def stream_request(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        with self._lock:
            self.start()
            assert self.process is not None and self.process.stdin is not None

            try:
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
            except OSError as exc:
                self.close()
                raise RuntimeError(f"Native worker stdin failed: {exc}") from exc

        while True:
            try:
                line = self._stdout_queue.get(timeout=self.timeout)
            except Empty as exc:
                self.close()
                raise RuntimeError("Timed out waiting for the MLX worker.") from exc

            if line is None:
                stderr = ""
                if self.process is not None and self.process.stderr is not None:
                    try:
                        stderr = self.process.stderr.read().strip()
                    except OSError:
                        stderr = ""
                return_code = self.process.poll() if self.process else None
                self.close()
                detail = self._exit_detail(stderr, return_code)
                raise RuntimeError(f"MLX worker exited unexpectedly: {detail}")

            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise RuntimeError(f"MLX worker returned invalid JSON: {line}") from exc

            if not response.get("ok", False):
                raise RuntimeError(str(response.get("error") or "MLX worker returned an unknown error."))

            yield response
            if response.get("done"):
                break
