"""Voice I/O: model catalogs, capabilities and settings, plus calls into the worker.

The speech models never load in the backend process. ``voice_env`` builds
the voice venv, ``voice_worker`` runs in it, ``voice_client`` talks to it;
this module decides which model and voice a request uses on this machine.
"""

from __future__ import annotations

import json
import platform
import threading
from pathlib import Path
from typing import Any

from backend_service import voice_env


def _platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "apple_silicon" if platform.machine() in ("arm64", "aarch64") else "macos_x86"
    if system == "Windows":
        return "windows"
    return "linux"


# ---------------------------------------------------------------------------
# Speech-to-text catalog
# ---------------------------------------------------------------------------

# One logical model per row, with a weight source per backend: mlx-whisper
# reads MLX safetensors, faster-whisper CTranslate2 conversions, parakeet-mlx
# its own MLX port. CTranslate2 sources are the repos faster-whisper itself
# maps its size names to (``faster_whisper.utils._MODELS``, 1.2.1).
_STT_MODELS: list[dict[str, Any]] = [
    {
        "key": "parakeet-tdt-v3",
        "name": "Parakeet TDT 0.6B v3",
        "languages": "25 European languages",
        "default": True,
        # NVIDIA weights (CC-BY-4.0), MLX port by mlx-community.
        "sources": {"parakeet-mlx": ("mlx-community/parakeet-tdt-0.6b-v3", 2.5)},
    },
    {
        "key": "whisper-large-v3-turbo",
        "name": "Whisper Large v3 Turbo",
        "languages": "99 languages",
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
        "languages": "99 languages",
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
        "languages": "99 languages",
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


def _lane_backends() -> tuple[str, ...]:
    return ("parakeet-mlx", "mlx-whisper") if voice_env.lane() == "mlx" else ("faster-whisper",)


def _hf_snapshot_path(repo_id: str) -> Path | None:
    from backend_service.helpers.hf_cache_paths import _hf_repo_snapshot_dir  # noqa: PLC0415

    return _hf_repo_snapshot_dir(repo_id)


def list_stt_models() -> list[dict[str, Any]]:
    """STT rows this machine can run: one per logical model, first backend that has it."""
    installed = voice_env.is_installed()
    rows: list[dict[str, Any]] = []
    for entry in _STT_MODELS:
        for backend in _lane_backends():
            source = entry["sources"].get(backend)
            if source is None:
                continue
            repo, size_gb = source
            rows.append({
                "id": repo,
                "key": entry["key"],
                "name": entry["name"],
                "languages": entry["languages"],
                "sizeGb": size_gb,
                "installed": _hf_snapshot_path(repo) is not None,
                "default": False,
                "backend": backend,
                "backendInstalled": installed,
            })
            break
    if rows:
        next(r for r in rows if r["key"] in {e["key"] for e in _STT_MODELS if e["default"]})["default"] = True
    return rows


def resolve_stt_model(model_id: str | None) -> tuple[str, str]:
    """Map a catalog id, key, size alias or empty (default) onto ``(backend, repo)``.

    Anything else is refused: the id reaches loaders that also accept local
    paths, so it must never be free-form.
    """
    rows = list_stt_models()
    requested = (model_id or "").strip()
    if not requested:
        chosen = next((r for r in rows if r["default"]), rows[0] if rows else None)
        if chosen is None:
            raise RuntimeError("No speech-to-text model is available on this machine.")
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
    for backend in _lane_backends():
        source = entry["sources"].get(backend)
        if source is not None:
            return backend, source[0]
    # Parakeet on a machine without MLX: fall back to the default Whisper.
    return resolve_stt_model("")


# ---------------------------------------------------------------------------
# Text-to-speech voices (Kokoro v1.0)
# ---------------------------------------------------------------------------

_KOKORO_VOICES = (
    "af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
    "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa "
    "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis "
    "ef_dora em_alex em_santa ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola "
    "jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo pf_dora pm_alex pm_santa "
    "zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang"
).split()

_LANGUAGES = {
    "a": ("en-US", "English (US)"),
    "b": ("en-GB", "English (UK)"),
    "e": ("es", "Spanish"),
    "f": ("fr", "French"),
    "h": ("hi", "Hindi"),
    "i": ("it", "Italian"),
    "j": ("ja", "Japanese"),
    "p": ("pt-BR", "Portuguese (Brazil)"),
    "z": ("zh", "Chinese (Mandarin)"),
}

DEFAULT_VOICE = "af_heart"


def _voice_language_ready(prefix: str) -> bool:
    """mlx-audio needs extra front-ends for Japanese and Chinese."""
    if voice_env.lane() != "mlx":
        return True
    if prefix == "z":
        return "chinese" in voice_env.installed_extras()
    return prefix != "j"  # Japanese needs a 0.8 GB dictionary; not offered yet


def list_tts_voices() -> list[dict[str, Any]]:
    voices = []
    for voice_id in _KOKORO_VOICES:
        prefix, gender, name = voice_id[0], voice_id[1], voice_id[3:]
        language, language_label = _LANGUAGES[prefix]
        voices.append({
            "id": voice_id,
            "name": name.replace("_", " ").title(),
            "gender": "female" if gender == "f" else "male",
            "language": language,
            "languageLabel": language_label,
            "available": _voice_language_ready(prefix),
        })
    return voices


# ---------------------------------------------------------------------------
# Settings (``~/.chaosengine/voice/settings.json``)
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS: dict[str, Any] = {
    "sttModel": "",
    "language": "",
    "ttsVoice": DEFAULT_VOICE,
    "speed": 1.0,
    "autoSend": False,
    "autoRead": False,
    "idleUnloadMinutes": 10,
}
_SETTINGS_LOCK = threading.Lock()


def _settings_path() -> Path:
    return voice_env.VOICE_DATA_DIR / "settings.json"


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    try:
        stored = json.loads(_settings_path().read_text(encoding="utf-8"))
        if isinstance(stored, dict):
            settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS})
    except (OSError, ValueError):
        pass
    return settings


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    with _SETTINGS_LOCK:
        settings = load_settings()
        settings.update({k: v for k, v in patch.items() if k in DEFAULT_SETTINGS and v is not None})
        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    from backend_service.voice_client import get_worker  # noqa: PLC0415

    get_worker().idle_minutes = float(settings["idleUnloadMinutes"])
    return settings


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

def get_voice_capabilities() -> dict[str, Any]:
    from backend_service.voice_client import worker_pid  # noqa: PLC0415

    installed = voice_env.is_installed()
    tts_data = voice_env.tts_data_installed() if installed else False
    marker = voice_env.read_marker() or {}
    lane = voice_env.lane()
    return {
        "platform": _platform(),
        "lane": lane,
        "installed": installed,
        "sttAvailable": installed,
        "ttsAvailable": installed and tts_data,
        "ttsModelInstalled": tts_data,
        "sttBackend": " + ".join(_lane_backends()) if installed else None,
        "ttsBackend": voice_env.tts_source()[0] if installed else None,
        "cuda": bool(marker.get("cuda")),
        "nvidiaGpu": voice_env.has_nvidia_gpu(),
        "extras": [
            {"id": name, "label": label, "installed": name in voice_env.installed_extras()}
            for name, (lanes, _specs, label) in voice_env.EXTRAS.items()
            if lane in lanes
        ],
        "workerPid": worker_pid(),
        "install": voice_env.job_status(),
    }


# ---------------------------------------------------------------------------
# Work
# ---------------------------------------------------------------------------

def transcribe_file(path: Path, model_id: str | None, language: str | None = None) -> dict[str, Any]:
    from backend_service.voice_client import get_worker  # noqa: PLC0415

    backend, repo = resolve_stt_model(model_id)
    snapshot = _hf_snapshot_path(repo)
    reply = get_worker().request(
        "transcribe",
        timeout=1800.0,
        backend=backend,
        source=str(snapshot or repo),
        path=str(path),
        language=(language or None),
    )
    return {
        "text": reply.get("text", ""),
        "language": reply.get("language"),
        "segments": reply.get("segments", []),
        "durationS": reply.get("durationS"),
        "model": repo,
        "seconds": reply.get("seconds"),
    }


def synthesize(text: str, voice: str, speed: float) -> tuple[bytes, float]:
    """WAV bytes and audio length for one piece of text."""
    import base64

    from backend_service.voice_client import get_worker  # noqa: PLC0415

    known = {v["id"]: v for v in list_tts_voices()}
    if voice not in known:
        raise ValueError(f"Unknown voice '{voice}'.")
    if not known[voice]["available"]:
        raise ValueError(f"{known[voice]['languageLabel']} voices need an extra install on this machine.")
    if not voice_env.tts_data_installed():
        raise RuntimeError("The Kokoro voice model is not downloaded. Reinstall the voice pack from Voice > Models.")
    # mlx-audio works out the model type from the repo name, so it gets the
    # repo id (resolved from the local cache), not the snapshot folder.
    engine, source = voice_env.tts_source()
    reply = get_worker().request("synthesize", timeout=300.0, engine=engine, source=source, text=text, voice=voice, speed=speed)
    return base64.b64decode(reply["wavB64"]), float(reply.get("durationS") or 0.0)
