"""Voice I/O runtime — STT via parakeet-mlx / mlx-whisper / faster-whisper, TTS via mlx-audio / kokoro-onnx."""

from __future__ import annotations

import importlib.util
import platform
import tempfile
import threading
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

# Speech-to-text catalog. Each entry is one *logical* model with a
# per-backend weight source, because the backends cannot read each
# other's formats: mlx-whisper loads MLX safetensors, faster-whisper
# loads CTranslate2 conversions, parakeet-mlx loads its own MLX port.
# ``list_stt_models`` projects this onto the rows the current platform
# can actually run (FU-092: non-Apple hosts used to be handed MLX ids).
#
# CTranslate2 sources (FU-092 decision): exactly the repos faster-whisper
# itself resolves its size tokens to (``faster_whisper.utils._MODELS`` in
# 1.2.1 — ``large-v3-turbo`` → mobiuslabsgmbh, ``small`` / ``base`` →
# Systran, the faster-whisper maintainers). Pinning the ids explicitly
# instead of passing the size token lets the Models tab download and
# report "installed" through the normal HF-cache machinery, while
# trusting nothing the library doesn't already trust.
_STT_MODELS: list[dict[str, Any]] = [
    {
        "key": "parakeet-tdt-v3",
        "name": "Parakeet TDT 0.6B v3 (25 European languages)",
        "default": False,
        # Apple Silicon only — parakeet-mlx (Apache-2.0). NVIDIA's
        # checkpoint (CC-BY-4.0), MLX port by mlx-community. Roughly 2x
        # Whisper Turbo's speed on M-series; no CJK coverage, so Whisper
        # stays in the list for everything else.
        "sources": {"parakeet-mlx": ("mlx-community/parakeet-tdt-0.6b-v3", 2.5)},
    },
    {
        "key": "whisper-large-v3-turbo",
        "name": "Whisper Large v3 Turbo",
        "default": True,
        "sources": {
            "mlx-whisper": ("mlx-community/whisper-large-v3-turbo-q4", 0.8),
            "faster-whisper": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1.6),
        },
        "aliases": ("large-v3-turbo", "turbo"),
    },
    {
        "key": "whisper-small",
        "name": "Whisper Small",
        "default": False,
        "sources": {
            "mlx-whisper": ("mlx-community/whisper-small-mlx", 0.24),
            "faster-whisper": ("Systran/faster-whisper-small", 0.48),
        },
        "aliases": ("small",),
    },
    {
        "key": "whisper-base",
        "name": "Whisper Base",
        "default": False,
        "sources": {
            "mlx-whisper": ("mlx-community/whisper-base-mlx", 0.07),
            "faster-whisper": ("Systran/faster-whisper-base", 0.15),
        },
        "aliases": ("base",),
    },
]


class UnknownSttModelError(ValueError):
    """The requested STT model id isn't in the catalog (HTTP 400)."""


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
    if _platform() == "apple_silicon":
        # Parakeet rides along: it's the fast default when present, and
        # mlx-whisper covers the languages Parakeet doesn't.
        return ["mlx-whisper", "parakeet-mlx"]
    return ["faster-whisper"]


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
    if plat == "apple_silicon" and _is_available("parakeet_mlx") and _is_available("mlx_whisper"):
        return "parakeet-mlx + mlx-whisper"
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


def _platform_stt_backends() -> tuple[str, ...]:
    """Backends this host can run, in preference order."""
    if _platform() == "apple_silicon":
        return ("parakeet-mlx", "mlx-whisper")
    return ("faster-whisper",)


_BACKEND_MODULES = {
    "parakeet-mlx": "parakeet_mlx",
    "mlx-whisper": "mlx_whisper",
    "faster-whisper": "faster_whisper",
}


def list_stt_models() -> list[dict[str, Any]]:
    """Return the STT rows this platform can run, with installed flags.

    One row per (logical model, backend the platform supports). On Apple
    Silicon Parakeet becomes the default once parakeet-mlx is installed —
    it's the faster lane; otherwise the catalog default (Whisper Turbo)
    stays default.
    """
    backends = _platform_stt_backends()
    parakeet_ready = "parakeet-mlx" in backends and _is_available("parakeet_mlx")
    models = []
    for entry in _STT_MODELS:
        for backend in backends:
            source = entry["sources"].get(backend)
            if source is None:
                continue
            repo, size_gb = source
            if parakeet_ready:
                is_default = backend == "parakeet-mlx"
            else:
                is_default = bool(entry["default"]) and backend != "parakeet-mlx"
            models.append({
                "id": repo,
                "name": entry["name"],
                "sizeGb": size_gb,
                "installed": _hf_snapshot_path(repo) is not None,
                "default": is_default,
                "backend": backend,
                "backendInstalled": _is_available(_BACKEND_MODULES[backend]),
            })
            break
    return models


def resolve_stt_model(model_id: str | None) -> tuple[str, str]:
    """Map a requested model id onto ``(backend, repo)`` for this host.

    Accepts any catalog id (from either platform — a transcript request
    carrying the MLX id on Linux is translated to the CTranslate2
    sibling, and vice versa), a faster-whisper size token, or empty for
    the platform default. Anything else raises ``UnknownSttModelError``:
    the id reaches ``from_pretrained`` / ``WhisperModel``, which also
    accept local filesystem paths, so it must never be free-form.
    """
    rows = list_stt_models()
    requested = (model_id or "").strip()
    if not requested:
        chosen = next((r for r in rows if r["default"] and r["backendInstalled"]), None)
        chosen = chosen or next((r for r in rows if r["backendInstalled"]), None) or (rows[0] if rows else None)
        if chosen is None:
            raise RuntimeError("No speech-to-text model is available on this platform.")
        return chosen["backend"], chosen["id"]

    entry = next(
        (
            e for e in _STT_MODELS
            if requested == e["key"]
            or requested in e.get("aliases", ())
            or any(requested == src[0] for src in e["sources"].values())
        ),
        None,
    )
    if entry is None:
        raise UnknownSttModelError(f"Unknown speech-to-text model '{requested}'.")
    for backend in _platform_stt_backends():
        source = entry["sources"].get(backend)
        if source is not None:
            return backend, source[0]
    raise RuntimeError(f"{entry['name']} is not supported on this platform.")


# Single-slot cache for the loaded STT model. mlx-whisper keeps its own
# ModelHolder; Parakeet (~1.2 GB in bf16) and faster-whisper would
# otherwise reload from disk on every utterance, which costs more than
# the transcription itself. Keyed on the loader too, so swapping the
# backend module (tests, reinstall) never serves a stale object.
_STT_CACHE_LOCK = threading.Lock()
_STT_CACHE: dict[str, Any] = {"key": None, "model": None}


def _cached_stt_model(key: tuple[Any, ...], load: Any) -> Any:
    with _STT_CACHE_LOCK:
        if _STT_CACHE["key"] != key:
            # Drop the old model before loading the next one so two
            # models are never resident at once.
            _STT_CACHE["key"] = None
            _STT_CACHE["model"] = None
            _STT_CACHE["model"] = load()
            _STT_CACHE["key"] = key
        return _STT_CACHE["model"]


def release_stt_model() -> None:
    with _STT_CACHE_LOCK:
        _STT_CACHE["key"] = None
        _STT_CACHE["model"] = None


def list_tts_voices() -> list[dict[str, Any]]:
    return list(_TTS_VOICES)


def transcribe_audio(audio_bytes: bytes, mime_type: str, model_id: str | None) -> str:
    """Run STT on raw audio bytes and return the transcript text.

    Writes audio to a tempfile, transcribes it, then cleans up.
    Heavy deps are lazy-imported inside this function.
    """
    backend, repo = resolve_stt_model(model_id)
    suffix = _mime_to_suffix(mime_type)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        if backend == "parakeet-mlx":
            if not _is_available("parakeet_mlx"):
                raise RuntimeError(
                    "Parakeet needs the parakeet-mlx package. Install it from Voice → Models."
                )
            from parakeet_mlx import from_pretrained  # type: ignore[import-untyped]

            # A local snapshot dir skips the network round-trip
            # from_pretrained otherwise makes through hf_hub_download.
            source = str(_hf_snapshot_path(repo) or repo)
            model = _cached_stt_model(("parakeet", source, from_pretrained), lambda: from_pretrained(source))
            result = model.transcribe(str(tmp_path))
            return (getattr(result, "text", "") or "").strip()

        if backend == "mlx-whisper":
            if not _is_available("mlx_whisper"):
                raise RuntimeError(
                    "No STT backend available. Install mlx-whisper from Voice → Models."
                )
            import mlx_whisper  # type: ignore[import-untyped]

            result = mlx_whisper.transcribe(str(tmp_path), path_or_hf_repo=repo)
            return (result.get("text") or "").strip()

        if not _is_available("faster_whisper"):
            raise RuntimeError(
                "No STT backend available. Install faster-whisper from Voice → Models."
            )
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        source = str(_hf_snapshot_path(repo) or repo)
        model = _cached_stt_model(
            ("faster-whisper", source, WhisperModel),
            lambda: WhisperModel(source, device="auto", compute_type="auto"),
        )
        segments, _ = model.transcribe(str(tmp_path))
        # Segment texts carry their own leading space (and none for
        # CJK) — concatenate like Whisper's own ``text`` field does.
        return "".join(seg.text for seg in segments).strip()
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
