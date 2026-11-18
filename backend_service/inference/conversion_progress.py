"""Real progress for a running ``mlx_lm convert`` process.

The converter is a subprocess that says little: three ``[INFO]`` lines on
stdout, a file counter on stderr while it downloads a Hub source, and the
output folder growing on disk. ``ConverterWatch`` turns exactly those three
signals into phases on a ``JobProgressTracker`` and counts nothing else, so
the dialog never shows a made-up percentage:

* ``[INFO] Loading`` -> ``loading``
* ``Fetching N files: ... k/N`` (stderr) -> ``downloading`` with k of N files
* ``[INFO] Quantizing`` -> ``quantizing``
* ``[INFO] Quantized model with X bits per weight`` or bytes in the output
  folder -> ``saving`` with the bytes written so far

``run_watched`` drains both pipes while the process runs (a full pipe would
block the converter) and returns their text for the error handling.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from threading import Thread
from typing import IO, Any, Callable

from backend_service.inference.conversion import _path_size_bytes


_FETCH_RE = re.compile(r"Fetching\s+(\d+)\s+files:[^\r\n]*?\|\s*(\d+)/(\d+)")
_BITS_PER_WEIGHT_RE = re.compile(r"Quantized model with ([0-9]+(?:\.[0-9]+)?) bits per weight")


def parse_bits_per_weight(stdout: str) -> float | None:
    """Bits per weight the converter measured for the quantized model, or None when it did not say."""
    match = _BITS_PER_WEIGHT_RE.search(stdout or "")
    return float(match.group(1)) if match else None


class ConverterWatch:
    """Maps the converter's own output and the output folder onto progress phases."""

    def __init__(self, tracker: Any, *, output_dir: str | None, expected_bytes: int | None) -> None:
        self._tracker = tracker
        self._output_dir = output_dir
        self._expected_bytes = expected_bytes
        self._phase = ""
        self._stdout_tail = ""
        self._stderr_tail = ""

    def _enter(self, phase: str, message: str) -> None:
        if self._phase != phase:
            self._phase = phase
            self._tracker.set_phase(phase, message)

    def feed_stdout(self, text: str) -> None:
        self._stdout_tail = (self._stdout_tail + text)[-4096:]
        if self._phase == "saving":
            return
        bits = parse_bits_per_weight(self._stdout_tail)
        if bits is not None:
            self._enter("saving", "Writing the converted model")
            self._tracker.set_detail(bitsPerWeight=bits)
        elif "[INFO] Quantizing" in self._stdout_tail:
            self._enter("quantizing", "Quantizing the weights")
        elif "[INFO] Loading" in self._stdout_tail and self._phase not in {"downloading"}:
            self._enter("loading", "Reading the source model")

    def feed_stderr(self, text: str) -> None:
        self._stderr_tail = (self._stderr_tail + text)[-4096:]
        if self._phase in {"quantizing", "saving"}:
            return
        matches = _FETCH_RE.findall(self._stderr_tail)
        if not matches:
            return
        _count, done, total = (int(value) for value in matches[-1])
        if total <= 0:
            return
        if done >= total:
            # Every file is on disk: the converter is reading the weights now.
            if self._phase == "downloading":
                self._enter("loading", "Reading the source model")
            return
        self._enter("downloading", "Downloading the source model")
        self._tracker.set_step(done, total)

    def tick(self) -> None:
        """Measure the output folder; any byte in it means the model is being written."""
        if not self._output_dir or not Path(self._output_dir).is_dir():
            return
        written = _path_size_bytes(self._output_dir)
        if written <= 0:
            return
        self._enter("saving", "Writing the converted model")
        detail: dict[str, Any] = {"bytesWritten": written}
        if self._expected_bytes:
            detail["expectedBytes"] = int(self._expected_bytes)
        self._tracker.set_detail(**detail)


def _pump(pipe: IO[bytes], sink: list[bytes], feed: Callable[[str], None]) -> None:
    fd = pipe.fileno()
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return
        if not chunk:
            return
        sink.append(chunk)
        try:
            feed(chunk.decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 - progress is best effort, the conversion goes on
            pass


def run_watched(
    process: subprocess.Popen[bytes],
    watch: ConverterWatch,
    *,
    timeout: float,
    poll_seconds: float = 0.5,
) -> tuple[str, str]:
    """Wait for the converter while reporting progress. Returns ``(stdout, stderr)``.

    Raises ``subprocess.TimeoutExpired`` (after ending the process) when it runs past ``timeout``.
    """
    out: list[bytes] = []
    err: list[bytes] = []
    readers = [
        Thread(target=_pump, args=(process.stdout, out, watch.feed_stdout), daemon=True),
        Thread(target=_pump, args=(process.stderr, err, watch.feed_stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(process.args, timeout) from None
                try:
                    watch.tick()
                except Exception:  # noqa: BLE001 - best effort
                    pass
    finally:
        for reader in readers:
            reader.join(timeout=5)
    return (
        b"".join(out).decode("utf-8", errors="replace"),
        b"".join(err).decode("utf-8", errors="replace"),
    )
