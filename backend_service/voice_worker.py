"""Voice worker: speech-to-text and text-to-speech in their own process.

Runs from the voice venv (``~/.chaosengine/voice-venv``, built by
``backend_service.voice_env``), never inside the FastAPI backend, so the
speech models can stay loaded between requests without pulling mlx-audio,
parakeet-mlx or faster-whisper into the backend, and a crash or a stuck model
can be killed on its own.

This file imports nothing from ``backend_service``: the venv has only the
voice packages. Protocol: one JSON object per line on stdin, one JSON reply
per line on stdout, matched by ``id``. Library output goes to stderr so it
cannot corrupt the protocol.

Audio is decoded with PyAV (its own FFmpeg libraries), so no ``ffmpeg``
command is needed on PATH.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any

PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr  # anything a library prints goes to stderr

STT_SAMPLE_RATE = 16_000

# Kokoro voice prefix -> language. mlx-audio takes the letter itself;
# kokoro-onnx takes an espeak language tag.
KOKORO_ONNX_LANG = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "p": "pt-br", "j": "ja", "z": "cmn",
}

_stt: dict[str, Any] = {"key": None, "model": None}
_tts: dict[str, Any] = {"key": None, "model": None}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def decode_audio(path: str, sample_rate: int = STT_SAMPLE_RATE):
    """Decode any audio or video file to mono float32 at ``sample_rate``."""
    import av  # type: ignore[import-untyped]
    import numpy as np

    chunks = []
    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError("The file has no audio track.")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("The file contains no audio.")
    return np.concatenate(chunks).astype(np.float32)


def wav_bytes(samples, sample_rate: int) -> bytes:
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Platform fixes
# ---------------------------------------------------------------------------

def _short_espeak_data_path() -> None:
    """espeak-ng drops a data path of about 160+ characters and then exits the
    process; point it at a short copy under ``~/.chaosengine``."""
    try:
        import espeakng_loader  # type: ignore[import-untyped]
    except ImportError:
        return
    source = Path(espeakng_loader.get_data_path())
    if len(str(source)) < 140 or not source.is_dir():
        return
    short = Path.home() / ".chaosengine" / "espeak-ng-data"
    if not (short / "phontab").is_file():
        shutil.copytree(source, short, dirs_exist_ok=True)
    espeakng_loader.get_data_path = lambda: str(short)


def _add_cuda_dll_dirs() -> None:
    """Windows: let ctranslate2 find the CUDA libraries pip installed."""
    if os.name != "nt":
        return
    for base in sys.path:
        nvidia = Path(base) / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------

def _load_stt(backend: str, source: str) -> Any:
    key = (backend, source)
    if _stt["key"] == key:
        return _stt["model"]
    _stt["key"], _stt["model"] = None, None
    if backend == "parakeet-mlx":
        import mlx.core as mx  # type: ignore[import-untyped]
        import parakeet_mlx.parakeet as parakeet_module  # type: ignore[import-untyped]
        from parakeet_mlx import from_pretrained  # type: ignore[import-untyped]

        # parakeet-mlx decodes with the ffmpeg command; use PyAV instead.
        parakeet_module.load_audio = lambda filename, sampling_rate, dtype=None: mx.array(
            decode_audio(str(filename), sampling_rate)
        )
        model = from_pretrained(source)
    elif backend == "mlx-whisper":
        model = source  # mlx-whisper keeps its own loaded-model cache
    elif backend == "faster-whisper":
        _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        model = None
        try:
            import ctranslate2  # type: ignore[import-untyped]

            if ctranslate2.get_cuda_device_count() > 0:
                model = WhisperModel(source, device="cuda", compute_type="float16")
        except Exception as exc:  # noqa: BLE001 - missing CUDA libraries raise many types
            print(f"voice_worker: CUDA unavailable for faster-whisper ({exc}); using CPU", file=sys.stderr)
            model = None
        if model is None:
            model = WhisperModel(source, device="cpu", compute_type="int8")
    else:
        raise ValueError(f"Unknown speech-to-text backend {backend!r}.")
    _stt["key"], _stt["model"] = key, model
    return model


def transcribe(request: dict[str, Any]) -> dict[str, Any]:
    backend = str(request["backend"])
    source = str(request["source"])
    path = str(request["path"])
    language = request.get("language") or None
    model = _load_stt(backend, source)

    if backend == "parakeet-mlx":
        duration = len(decode_audio(path)) / STT_SAMPLE_RATE
        chunk = 120.0 if duration > 150 else None
        result = model.transcribe(path, chunk_duration=chunk, overlap_duration=15.0)
        segments = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
            for s in getattr(result, "sentences", []) or []
        ]
        return {"text": result.text.strip(), "language": None, "segments": segments, "durationS": round(duration, 2)}

    audio = decode_audio(path)
    duration = len(audio) / STT_SAMPLE_RATE
    if backend == "mlx-whisper":
        import mlx_whisper  # type: ignore[import-untyped]

        result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language=language, verbose=None)
        segments = [
            {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]
        return {
            "text": (result.get("text") or "").strip(),
            "language": result.get("language"),
            "segments": segments,
            "durationS": round(duration, 2),
        }

    raw_segments, info = model.transcribe(audio, language=language)
    segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in raw_segments]
    # Segment texts carry their own leading space (none for CJK).
    text = " ".join(s["text"] for s in segments) if info.language not in ("zh", "ja", "ko") else "".join(
        s["text"] for s in segments
    )
    return {"text": text.strip(), "language": info.language, "segments": segments, "durationS": round(duration, 2)}


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------

def _load_tts(engine: str, source: str) -> Any:
    key = (engine, source)
    if _tts["key"] == key:
        return _tts["model"]
    _tts["key"], _tts["model"] = None, None
    _short_espeak_data_path()
    if engine == "mlx-audio":
        from mlx_audio.tts.utils import load_model  # type: ignore[import-untyped]

        model = load_model(source)
    elif engine == "kokoro-onnx":
        from kokoro_onnx import Kokoro  # type: ignore[import-untyped]

        folder = Path(source)
        model = Kokoro(str(folder / "kokoro-v1.0.onnx"), str(folder / "voices-v1.0.bin"))
    else:
        raise ValueError(f"Unknown text-to-speech engine {engine!r}.")
    _tts["key"], _tts["model"] = key, model
    return model


def synthesize(request: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    engine = str(request["engine"])
    model = _load_tts(engine, str(request["source"]))
    text = str(request["text"])
    voice = str(request.get("voice") or "af_heart")
    speed = float(request.get("speed") or 1.0)
    prefix = voice[:1] if voice[:1] in KOKORO_ONNX_LANG else "a"

    if engine == "mlx-audio":
        parts = [
            np.asarray(result.audio, dtype=np.float32).reshape(-1)
            for result in model.generate(text, voice=voice, speed=speed, lang_code=prefix)
        ]
        if not parts:
            raise RuntimeError("The speech model produced no audio.")
        samples = np.concatenate(parts)
        sample_rate = int(getattr(model, "sample_rate", 24_000))
    else:
        samples, sample_rate = model.create(text, voice=voice, speed=speed, lang=KOKORO_ONNX_LANG[prefix])
    return {
        "wavB64": base64.b64encode(wav_bytes(samples, sample_rate)).decode("ascii"),
        "sampleRate": sample_rate,
        "durationS": round(len(samples) / sample_rate, 2),
    }


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def unload(request: dict[str, Any]) -> dict[str, Any]:
    what = request.get("what") or "all"
    if what in ("stt", "all"):
        _stt["key"], _stt["model"] = None, None
    if what in ("tts", "all"):
        _tts["key"], _tts["model"] = None, None
    try:
        import gc

        gc.collect()
        import mlx.core as mx  # type: ignore[import-untyped]

        mx.clear_cache()
    except Exception:  # noqa: BLE001 - mlx absent on Windows / Linux
        pass
    return {}


def status(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "stt": list(_stt["key"]) if _stt["key"] else None,
        "tts": list(_tts["key"]) if _tts["key"] else None,
        "pid": os.getpid(),
    }


_COMMANDS = {
    "ping": lambda _r: {"pong": True},
    "status": status,
    "transcribe": transcribe,
    "synthesize": synthesize,
    "unload": unload,
}


def _reply(payload: dict[str, Any]) -> None:
    PROTOCOL_OUT.write(json.dumps(payload) + "\n")
    PROTOCOL_OUT.flush()


def main() -> None:
    _reply({"id": None, "ok": True, "ready": True, "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            command = _COMMANDS.get(request.get("cmd"))
            if command is None:
                raise ValueError(f"Unknown command {request.get('cmd')!r}.")
            started = time.perf_counter()
            result = command(request)
            result.update({"id": request_id, "ok": True, "seconds": round(time.perf_counter() - started, 3)})
            _reply(result)
        except Exception as exc:  # noqa: BLE001 - every failure goes back to the caller
            traceback.print_exc(file=sys.stderr)
            _reply({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
