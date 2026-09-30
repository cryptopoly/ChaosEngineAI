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


def _stt_install_packages() -> list[str]:
    """Return pip package keys for the platform-appropriate STT backend."""
    return ["mlx-whisper"] if _platform() == "apple_silicon" else ["faster-whisper"]


def _tts_install_packages() -> list[str]:
    """Return pip package keys for the platform-appropriate TTS backend.

    mlx-audio's Kokoro pipeline needs the ``misaki`` G2P package but does
    not declare it as a dependency — and swallows the ImportError at
    generate time (yields zero segments instead of raising). The trailing
    entries are the Japanese G2P chain for the jf_* voices (see the
    allowlist comment in routes/setup for why not ``misaki[ja]``).
    """
    if _platform() == "apple_silicon":
        return [
            "mlx-audio",
            "misaki",
            "fugashi",
            "jaconv",
            "mojimoji",
            "pyopenjtalk",
            "unidic-lite",
        ]
    return ["kokoro-onnx"]


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
    """Return the active local HF snapshot dir for repo_id, or None.

    Delegates to the shared resolver so a relocated cache (Settings →
    ``hfCachePath``, injected as ``HF_HOME``) is honoured — the models
    mlx-audio / mlx-whisper download land there, not in ``~/.cache``.
    """
    from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir  # noqa: PLC0415

    return _hf_repo_snapshot_dir(repo_id)


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
    stt_ok = _stt_available()
    tts_ok = _tts_available()
    plat = _platform()
    # The voice data TTS loads — surfaced so the UI can offer a pre-download
    # instead of a silent multi-hundred-MB fetch on first Generate. Apple
    # Silicon pulls an HF snapshot (repo id set, downloadable through the
    # standard model-download machinery); other platforms use the kokoro-onnx
    # GitHub release files (repo None, download via /api/voice/kokoro/*).
    if plat == "apple_silicon":
        tts_model_repo: str | None = _TTS_MODEL_REPO
        tts_model_installed = _hf_snapshot_path(_TTS_MODEL_REPO) is not None
    else:
        from backend_service import voice_kokoro_onnx  # noqa: PLC0415

        tts_model_repo = None
        tts_model_installed = voice_kokoro_onnx.is_installed()
    return {
        "sttAvailable": stt_ok,
        "ttsAvailable": tts_ok,
        "platform": plat,
        "sttBackend": _stt_backend(),
        "ttsBackend": _tts_backend(),
        # Package keys to pass to /api/setup/install-package when not available.
        "sttInstallPackages": None if stt_ok else _stt_install_packages(),
        "ttsInstallPackages": None if tts_ok else _tts_install_packages(),
        "ttsModelRepo": tts_model_repo,
        "ttsModelInstalled": tts_model_installed,
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
            # Segment texts carry their own leading space (and none for
            # CJK) — concatenate like Whisper's own ``text`` field does.
            return "".join(seg.text for seg in segments).strip()

        raise RuntimeError(
            "No STT backend available. Install mlx-whisper (Apple Silicon) or "
            "faster-whisper (other platforms)."
        )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


# Kokoro checkpoint mlx-audio loads for TTS. The upstream README's
# canonical repo; ~350 MB, snapshot-downloaded to the HF cache on the
# first synthesize call.
_TTS_MODEL_REPO = "prince-canuma/Kokoro-82M"


def synthesize_speech(text: str, voice: str, speed: float) -> bytes:
    """Run TTS and return WAV bytes.

    Heavy deps are lazy-imported inside this function.
    """
    plat = _platform()

    if plat == "apple_silicon" and _is_available("mlx_audio"):
        from mlx_audio.tts.generate import generate_audio  # type: ignore[import-untyped]

        # Kokoro convention: the voice id's first letter is its language
        # code (af_/am_ American, bf_/bm_ British, jf_ Japanese, ...).
        # Without this the pipeline runs English G2P on non-English voices
        # and produces garbled speech.
        # ``voice[:1]`` guard: ``"" in "abc..."`` is True, so an empty id
        # would otherwise reach ``voice[0]`` and raise IndexError.
        lang_code = voice[0] if voice[:1] and voice[0] in "abefhipjz" else "a"

        with tempfile.TemporaryDirectory(prefix="chaosengine-tts-") as tmpdir:
            # join_audio=True collapses multi-segment output into a single
            # ``<prefix>.wav``; the glob fallback covers the per-segment
            # ``<prefix>_000.wav`` naming in case upstream changes the rule.
            generate_audio(
                text=text,
                model=_TTS_MODEL_REPO,
                voice=voice,
                speed=speed,
                lang_code=lang_code,
                output_path=tmpdir,
                file_prefix="tts",
                audio_format="wav",
                join_audio=True,
                verbose=False,
            )
            out_path = Path(tmpdir) / "tts.wav"
            if not out_path.is_file():
                candidates = sorted(Path(tmpdir).glob("tts*.wav"))
                if not candidates:
                    raise RuntimeError(
                        "mlx-audio completed without writing an output file."
                    )
                out_path = candidates[0]
            return out_path.read_bytes()

    if _is_available("kokoro_onnx"):
        from backend_service import voice_kokoro_onnx  # noqa: PLC0415

        return voice_kokoro_onnx.synthesize(text, voice, speed)

    raise RuntimeError(
        "No TTS backend available. Install mlx-audio (Apple Silicon) or "
        "kokoro-onnx (other platforms)."
    )
