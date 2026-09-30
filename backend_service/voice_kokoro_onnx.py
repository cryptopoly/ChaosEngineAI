"""kokoro-onnx TTS lane for non-Apple platforms (FU-083).

The ``kokoro-onnx`` pip package needs two data files that ship as GitHub
release assets, not on PyPI or the HF hub: the ONNX model (~310 MB) and
the voice-style bank (~27 MB). This module owns their download lifecycle
(background thread + polled status, mirroring the shape of the HF
download jobs) and the actual synthesis call.

Apple Silicon never touches this module — the mlx-audio lane in
``voice_runtime.py`` handles TTS there.
"""

from __future__ import annotations

import io
import tempfile
import threading
import urllib.request
import wave
from pathlib import Path
from typing import Any

_RELEASE_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_FILES: tuple[tuple[str, str], ...] = (
    ("kokoro-v1.0.onnx", f"{_RELEASE_BASE}/kokoro-v1.0.onnx"),
    ("voices-v1.0.bin", f"{_RELEASE_BASE}/voices-v1.0.bin"),
)
_TOTAL_BYTES_ESTIMATE = 325_532_387 + 28_214_398

# Kokoro voice-id prefix → espeak language tag kokoro-onnx expects.
_LANG_BY_PREFIX = {"a": "en-us", "b": "en-gb", "j": "ja"}

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


def synthesize(text: str, voice: str, speed: float) -> bytes:
    """Synthesize WAV bytes via kokoro-onnx. Raises RuntimeError when the
    model files are missing (the UI offers the download)."""
    if not is_installed():
        raise RuntimeError(
            "Kokoro ONNX voice files are not downloaded. Use the download "
            "button on the Voice Models tab (~340 MB, one-time)."
        )

    import kokoro_onnx  # type: ignore[import-untyped]
    import numpy as np  # kokoro-onnx dependency

    model_path = kokoro_dir() / _FILES[0][0]
    voices_path = kokoro_dir() / _FILES[1][0]
    kokoro = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
    lang = _LANG_BY_PREFIX.get(voice[:1], "en-us")
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)

    # Encode 16-bit mono PCM with the stdlib ``wave`` module. kokoro-onnx
    # doesn't depend on ``soundfile``, so importing it here failed on a
    # fresh Windows/Linux install that only has what "Install kokoro-onnx"
    # pulled in.
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()
