"""Voice I/O runtime — STT via mlx-whisper / faster-whisper, TTS via mlx-audio / kokoro-onnx."""

from __future__ import annotations

import importlib.util
import platform
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Availability helpers — checked lazily, never imported at module top
# ---------------------------------------------------------------------------

def _is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _platform() -> str:
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine in ("arm64", "arm"):
        return "apple_silicon"
    if system == "Darwin":
        return "macos_x86"
    if system == "Windows":
        return "windows"
    return "linux"


# ---------------------------------------------------------------------------
# Model / voice catalogs
# ---------------------------------------------------------------------------

_STT_MODELS = [
    {
        "id": "mlx-community/whisper-large-v3-turbo-q4",
        "name": "Whisper Large v3 Turbo (Q4)",
        "sizeGb": 0.8,
        "installed": False,
        "default": True,
    },
    {
        "id": "mlx-community/whisper-small-mlx",
        "name": "Whisper Small",
        "sizeGb": 0.24,
        "installed": False,
        "default": False,
    },
    {
        "id": "mlx-community/whisper-base-mlx",
        "name": "Whisper Base",
        "sizeGb": 0.07,
        "installed": False,
        "default": False,
    },
]

_TTS_VOICES = [
    {"id": "af_heart", "name": "American Female (Heart)", "language": "en-US"},
    {"id": "am_adam", "name": "American Male (Adam)", "language": "en-US"},
    {"id": "bf_emma", "name": "British Female (Emma)", "language": "en-GB"},
    {"id": "bm_george", "name": "British Male (George)", "language": "en-GB"},
    {"id": "jf_alpha", "name": "Japanese Female (Alpha)", "language": "ja-JP"},
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stt_available() -> bool:
    plat = _platform()
    if plat == "apple_silicon":
        return _is_available("mlx_whisper")
    return _is_available("faster_whisper")


def _tts_available() -> bool:
    plat = _platform()
    if plat == "apple_silicon":
        return _is_available("mlx_audio")
    return _is_available("kokoro_onnx")


def _stt_backend() -> str | None:
    plat = _platform()
    if plat == "apple_silicon" and _is_available("mlx_whisper"):
        return "mlx-whisper"
    if _is_available("faster_whisper"):
        return "faster-whisper"
    return None


def _tts_backend() -> str | None:
    plat = _platform()
    if plat == "apple_silicon" and _is_available("mlx_audio"):
        return "mlx-audio"
    if _is_available("kokoro_onnx"):
        return "kokoro-onnx"
    return None


def _hf_snapshot_path(repo_id: str) -> Path | None:
    """Return the newest local HF snapshot dir for repo_id, or None."""
    try:
        from huggingface_hub.file_download import repo_folder_name  # lazy import
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        folder = hf_cache / repo_folder_name(repo_id=repo_id, repo_type="model")
        snapshots = folder / "snapshots"
        if not snapshots.is_dir():
            return None
        children = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        return children[0] if children else None
    except Exception:
        return None


def _mime_to_suffix(mime_type: str) -> str:
    mapping = {
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/wav": ".wav",
        "audio/wave": ".wav",
        "audio/x-wav": ".wav",
        "audio/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/flac": ".flac",
    }
    base = mime_type.split(";")[0].strip().lower()
    return mapping.get(base, ".webm")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_voice_capabilities() -> dict[str, Any]:
    """Return platform/availability summary for the voice runtime."""
    return {
        "sttAvailable": _stt_available(),
        "ttsAvailable": _tts_available(),
        "platform": _platform(),
        "sttBackend": _stt_backend(),
        "ttsBackend": _tts_backend(),
    }


def list_stt_models() -> list[dict[str, Any]]:
    """Return STT model list with installed flags."""
    models = []
    for entry in _STT_MODELS:
        row = dict(entry)
        row["installed"] = _hf_snapshot_path(entry["id"]) is not None
        models.append(row)
    return models


def list_tts_voices() -> list[dict[str, Any]]:
    return list(_TTS_VOICES)


def transcribe_audio(audio_bytes: bytes, mime_type: str, model_id: str) -> str:
    """Run STT on raw audio bytes and return the transcript text.

    Writes audio to a tempfile, transcribes it, then cleans up.
    Heavy deps are lazy-imported inside this function.
    """
    plat = _platform()
    suffix = _mime_to_suffix(mime_type)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        if plat == "apple_silicon" and _is_available("mlx_whisper"):
            import mlx_whisper  # type: ignore[import-untyped]

            result = mlx_whisper.transcribe(str(tmp_path), path_or_hf_repo=model_id)
            return (result.get("text") or "").strip()

        if _is_available("faster_whisper"):
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]

            # Accept either a bare size token (small, base …) or a full repo id.
            model_size = model_id.split("/")[-1] if "/" in model_id else model_id
            model = WhisperModel(model_size, device="auto", compute_type="auto")
            segments, _ = model.transcribe(str(tmp_path))
            return " ".join(seg.text for seg in segments).strip()

        raise RuntimeError(
            "No STT backend available. Install mlx-whisper (Apple Silicon) or "
            "faster-whisper (other platforms)."
        )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def synthesize_speech(text: str, voice: str, speed: float) -> bytes:
    """Run TTS and return WAV bytes.

    Heavy deps are lazy-imported inside this function.
    """
    plat = _platform()

    if plat == "apple_silicon" and _is_available("mlx_audio"):
        import mlx_audio  # type: ignore[import-untyped]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)

        try:
            mlx_audio.tts.generate(
                text=text,
                model="kokoro",
                voice=voice,
                speed=speed,
                output_file=str(out_path),
            )
            return out_path.read_bytes()
        finally:
            try:
                out_path.unlink()
            except Exception:
                pass

    if _is_available("kokoro_onnx"):
        import kokoro_onnx  # type: ignore[import-untyped]
        import soundfile as sf  # type: ignore[import-untyped]

        kokoro = kokoro_onnx.Kokoro()
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang="en-us")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)

        try:
            sf.write(str(out_path), samples, sample_rate)
            return out_path.read_bytes()
        finally:
            try:
                out_path.unlink()
            except Exception:
                pass

    raise RuntimeError(
        "No TTS backend available. Install mlx-audio (Apple Silicon) or "
        "kokoro-onnx (other platforms)."
    )
