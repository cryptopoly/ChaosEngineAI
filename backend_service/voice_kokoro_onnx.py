"""Kokoro voice files for the portable voice lane (FU-083).

``kokoro-onnx`` needs two data files that ship as GitHub release assets, not
on PyPI or the HF hub: the ONNX model (~310 MB) and the voice-style bank
(~27 MB). This module downloads them; ``voice_worker`` loads them. The voice
pack install (``voice_env``) calls the download on Windows, Linux and Intel
Macs; Apple Silicon uses the MLX Kokoro checkpoint instead.
"""

from __future__ import annotations

import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_FILES: tuple[tuple[str, str], ...] = (
    ("kokoro-v1.0.onnx", f"{_RELEASE_BASE}/kokoro-v1.0.onnx"),
    ("voices-v1.0.bin", f"{_RELEASE_BASE}/voices-v1.0.bin"),
)
_TOTAL_BYTES_ESTIMATE = 325_532_387 + 28_214_398


_lock = threading.Lock()
_status: dict[str, Any] = {"state": "idle", "progress": 0.0, "error": None}


def kokoro_dir() -> Path:
    path = Path.home() / ".chaosengine" / "voice" / "kokoro"
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_installed() -> bool:
    return all((kokoro_dir() / name).is_file() for name, _url in _FILES)


def download_status() -> dict[str, Any]:
    with _lock:
        return dict(_status)


def _download_worker() -> None:
    downloaded = 0
    try:
        for name, url in _FILES:
            target = kokoro_dir() / name
            if target.is_file():
                downloaded += target.stat().st_size
                continue
            # Stream to a tempfile in the same directory, then atomically
            # rename — a killed download never leaves a truncated file
            # that ``is_installed`` would mistake for a complete one.
            with tempfile.NamedTemporaryFile(dir=kokoro_dir(), delete=False) as tmp:
                partial = Path(tmp.name)
            try:
                with open(partial, "wb") as out, urllib.request.urlopen(url, timeout=60.0) as response:
                    while True:
                        chunk = response.read(1024 * 512)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        with _lock:
                            _status["progress"] = min(
                                0.99, downloaded / _TOTAL_BYTES_ESTIMATE
                            )
                partial.replace(target)
            except BaseException:
                # Don't strand a multi-hundred-MB partial next to the
                # real files — every retry would leak another one.
                partial.unlink(missing_ok=True)
                raise
        with _lock:
            _status.update(state="completed", progress=1.0, error=None)
    except Exception as exc:
        with _lock:
            _status.update(state="failed", error=str(exc))


def start_download() -> dict[str, Any]:
    with _lock:
        if _status["state"] == "downloading":
            return dict(_status)
        _status.update(state="downloading", progress=0.0, error=None)
    threading.Thread(target=_download_worker, daemon=True).start()
    return download_status()
