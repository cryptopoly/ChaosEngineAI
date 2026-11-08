"""Cloned voices: model downloads, making a voice from a voice sample, and speaking with it.

A cloned voice is a Voice Library item of kind ``clone``: the voice sample (an uploaded file or a microphone
recording), its transcript and the permission record (speaker's name, basis, time, sample hash, app version).
Deleting the voice deletes all of it. Speech made with a cloned voice goes through ``apply_watermark``
(AudioSeal, inaudible, always on, D-47), gets the AI tag in the WAV, is saved to the Library as ``speech`` with
``clonedVoice: true``, and is written to the local audit log.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from backend_service import voice_clone_safety as safety
from backend_service import voice_env
from backend_service.catalog.voice_models import CLONE_LICENCE, CLONE_MODELS, VOICE_MODELS, clone_model, voice_model

MIN_REFERENCE_S = 3.0
MAX_REFERENCE_S = 30.0
SAMPLE_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
BASES = ("own", "permission")
MAX_TEXT_CHARS = 5_000
PERMISSION_REQUIRED = (
    "Confirm that this is the person's own voice or that the speaker's permission covers this use "
    "(permission_confirmed: true)."
)

_DOWNLOADS: dict[str, dict[str, Any]] = {}
_DOWNLOAD_LOCK = threading.Lock()


def _app_version() -> str:
    try:
        from backend_service.helpers.system_hardware import _resolve_app_version  # noqa: PLC0415

        return str(_resolve_app_version())
    except Exception:  # noqa: BLE001 - the version is a label, never a blocker
        return "unknown"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def snapshot_dir(row: dict[str, Any]) -> Path | None:
    """The pinned revision's folder in the Hugging Face cache, when its weights are there."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir  # noqa: PLC0415

    path = _hf_repo_cache_dir(row["repo"]) / "snapshots" / row["revision"]
    ready = (path / "config.json").is_file() and (path / "model.safetensors").exists()
    return path if ready else None


def available_here() -> bool:
    return voice_env.lane() == "mlx"


def _ready(row: dict[str, Any]) -> bool:
    """Weights on disk; a model that clones also needs the watermark component."""
    if snapshot_dir(row) is None:
        return False
    return voice_env.watermark_installed() if row["capabilities"]["clone"] else True


def list_models(*, every_model: bool = False) -> list[dict[str, Any]]:
    """The cloning models, or with ``every_model`` all Qwen3-TTS rows (speak, style, clone, design)."""
    lane = voice_env.lane()
    rows = []
    for row in VOICE_MODELS if every_model else CLONE_MODELS:
        job = _DOWNLOADS.get(row["key"], {})
        rows.append({
            "key": row["key"],
            "name": row["name"],
            "summary": row["summary"],
            "repo": row["repo"],
            "revision": row["revision"],
            "sizeGb": row["sizeGb"],
            "languages": row["languages"],
            "default": row["default"],
            "licence": CLONE_LICENCE,
            "kind": row["kind"],
            "capabilities": dict(row["capabilities"]),
            # Speed and memory from the catalog, with its basis and where it was measured; None when not measured.
            "estimates": dict(row["estimates"]) if row.get("estimates") else None,
            "supported": lane in row["lanes"],
            # Ready means the weights, and for a cloning model the watermark component; Download installs both.
            "installed": _ready(row),
            "download": {"state": job.get("state", "idle"), "error": job.get("error"), "phase": job.get("phase")},
        })
    return rows


def start_download(key: str) -> dict[str, Any]:
    row = voice_model(key)
    if not available_here():
        raise ValueError("Voice cloning runs on Apple Silicon Macs for now.")
    with _DOWNLOAD_LOCK:
        if _DOWNLOADS.get(row["key"], {}).get("state") == "downloading":
            return dict(_DOWNLOADS[row["key"]])
        _DOWNLOADS[row["key"]] = {"state": "downloading", "error": None, "startedAt": time.time()}

    def run() -> None:
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415

            _DOWNLOADS[row["key"]]["phase"] = "weights"
            snapshot_download(row["repo"], revision=row["revision"])
            # Cloned output is always watermarked, so the component comes with a cloning model.
            if row["capabilities"]["clone"] and not voice_env.watermark_installed():
                _DOWNLOADS[row["key"]]["phase"] = "watermark"
                voice_env.install_watermark()
            _DOWNLOADS[row["key"]] = {"state": "done", "error": None}
        except Exception as exc:  # noqa: BLE001 - shown on the panel
            step = "the watermark component" if _DOWNLOADS[row["key"]].get("phase") == "watermark" else row["name"]
            _DOWNLOADS[row["key"]] = {"state": "failed", "error": f"Downloading {step} failed: {exc}"}

    threading.Thread(target=run, daemon=True, name=f"clone-download-{row['key']}").start()
    return dict(_DOWNLOADS[row["key"]])


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

def list_voices() -> list[dict[str, Any]]:
    from backend_service import voice_library  # noqa: PLC0415

    return [_public(item) for item in voice_library.list_items(kind="clone")]


def get_voice(voice_id: str) -> dict[str, Any] | None:
    from backend_service import voice_library  # noqa: PLC0415

    if not voice_library.is_valid_id(voice_id):
        return None
    item = voice_library.get(voice_id)
    return item if item and item.get("kind") == "clone" and item.get("consent") else None


def _public(item: dict[str, Any]) -> dict[str, Any]:
    record = item.get("consent") or {}
    return {
        "id": item["id"],
        "name": item.get("title"),
        "createdAt": item.get("createdAt"),
        "durationS": item.get("durationS"),
        "referenceText": item.get("text", item.get("preview", "")),
        "consent": {k: record.get(k) for k in ("speakerName", "basis", "consentedAt", "appVersion")},
    }


def create_voice(
    *,
    speaker_name: str,
    basis: str,
    name: str,
    sample_audio: Path,
    sample_ext: str,
    sample_text: str | None,
    permission_confirmed: bool,
    origin: str,
    transcribe: Any,
) -> dict[str, Any]:
    """Check the permission tick and the voice sample, then save the voice.

    ``transcribe(path) -> {"text", "durationS"}`` gives the sample's length and, when ``sample_text`` is empty,
    its words. Raises ValueError with a plain message when a check fails.
    """
    from backend_service import voice_library  # noqa: PLC0415

    if not permission_confirmed:
        raise ValueError("Confirm that this is your own voice or that the speaker gave permission.")
    speaker = " ".join(str(speaker_name).split())[:60]
    if not speaker:
        raise ValueError("Type the speaker's name.")
    if basis not in BASES:
        raise ValueError("Choose whose voice this is.")
    ext = (sample_ext or "").lower()
    if ext not in SAMPLE_EXTENSIONS:
        raise ValueError("The voice sample must be a WAV, MP3, M4A, FLAC or OGG file.")

    heard = transcribe(sample_audio)
    duration = float(heard.get("durationS") or 0.0)
    if duration < MIN_REFERENCE_S or duration > MAX_REFERENCE_S:
        raise ValueError(f"The voice sample must be 3 to 30 seconds long; this one is {duration:.1f} s.")
    text = " ".join(str(sample_text or "").split()) or " ".join(str(heard.get("text") or "").split())
    if not text:
        raise ValueError("No speech was heard in the voice sample.")

    record = {
        "speakerName": speaker,
        "basis": basis,
        "consentedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "referenceSha256": safety.sha256_file(sample_audio),
        "appVersion": _app_version(),
        "permissionConfirmed": True,
        "origin": origin,
    }
    with tempfile.TemporaryDirectory(prefix="chaosengine-clone-") as folder:
        sample_copy = Path(folder) / f"reference{ext}"
        shutil.copyfile(sample_audio, sample_copy)
        item = voice_library.create(
            "clone",
            {"title": " ".join(name.split())[:80] or speaker, "text": text,
             "durationS": round(duration, 2), "consent": record},
            audio_file=sample_copy,
            audio_ext=ext,
        )
    safety.audit("create", origin=origin, voiceId=item["id"], basis=basis,
                 referenceSha256=record["referenceSha256"])
    return _public(item)


def delete_voice(voice_id: str) -> bool:
    """Delete the voice, its sample and its permission record (the Library logs it)."""
    from backend_service import voice_library  # noqa: PLC0415

    if get_voice(voice_id) is None:
        return False
    return voice_library.delete(voice_id)


# ---------------------------------------------------------------------------
# Speaking
# ---------------------------------------------------------------------------

def pick_clone_model(model_key: str | None) -> dict[str, Any]:
    """The cloning model to run: the one named, else the default if it is downloaded, else any downloaded one.

    With nothing downloaded the default comes back, so the error names the model people would fetch first.
    """
    if (model_key or "").strip():
        return clone_model(model_key)
    default = clone_model("")
    if snapshot_dir(default) is not None:
        return default
    return next((row for row in CLONE_MODELS if snapshot_dir(row) is not None), default)


def speak(
    voice_id: str,
    text: str,
    model_key: str | None,
    *,
    origin: str,
    permission_confirmed: bool = False,
    speed: float = 1.0,
    pitch: float = 0.0,
    seed: int | None = None,
    max_chars: int = MAX_TEXT_CHARS,
) -> dict[str, Any]:
    """Speak ``text`` with a cloned voice; returns the saved Library item (``speech``, ``clonedVoice``).

    Long text is cut into sentences and spoken part by part; ``speed`` and ``pitch`` are applied to the
    joined audio before the watermark and the AI tag, which the finished speech always gets.

    Agents and other computers (``origin`` other than ``ui``) must pass ``permission_confirmed``: their
    confirmation that the speaker's permission covers this use. It is written to the audit log.
    """
    from backend_service import voice_library  # noqa: PLC0415
    from backend_service import voice_speak  # noqa: PLC0415

    voice = get_voice(voice_id)
    if voice is None:
        raise LookupError("Cloned voice not found. Its permission record is missing or it was deleted.")
    if origin != "ui" and not permission_confirmed:
        raise ValueError(PERMISSION_REQUIRED)
    text = text.strip()
    if not text:
        raise ValueError("Text is required.")
    if len(text) > max_chars:
        raise ValueError(f"Cloned speech takes up to {max_chars} characters at a time.")
    if not available_here():
        raise ValueError("Voice cloning runs on Apple Silicon Macs for now.")
    if not voice_env.is_installed():
        raise RuntimeError("The voice pack is not installed. Install it from Voice > Models.")
    row = pick_clone_model(model_key)
    folder = snapshot_dir(row)
    if folder is None:
        raise RuntimeError(f"{row['name']} is not downloaded. Download it in Voice > Models.")
    if not voice_env.watermark_installed():
        raise RuntimeError(safety.WATERMARK_MISSING)
    reference = voice_library.audio_path(voice_id)
    if reference is None:
        raise RuntimeError("This voice's sample is missing. Delete the voice and make it again.")

    plan = voice_speak.Plan(
        "clone", f"clone:{voice_id}", row["key"], row["name"], row["engine"], str(folder),
        {"refAudioPath": str(reference), "refText": str(voice.get("text") or "")},
    )
    rendered = voice_speak.render(plan, text, speed=speed, pitch=pitch, seed=seed)
    wav = rendered.wav
    duration = rendered.duration_s
    version = _app_version()
    with tempfile.TemporaryDirectory(prefix="chaosengine-cloned-") as tmp:
        out = Path(tmp) / "speech.wav"
        out.write_bytes(wav)
        safety.apply_watermark(out)
        tagged = safety.tag_wav(
            out.read_bytes(),
            {"appVersion": version, "model": row["repo"], "engine": row["engine"],
             "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "voiceId": voice_id,
             "watermark": "audioseal-0.2.1:CE5A"},
            f"ChaosEngineAI {version}",
        )
        out.write_bytes(tagged)
        item = voice_library.create(
            "speech",
            {"text": text, "title": f"{voice.get('title')}: {' '.join(text.split())[:40]}",
             "voice": f"clone:{voice_id}", "voiceName": voice.get("title"), "voiceKind": "clone",
             "model": row["repo"], "durationS": round(duration, 2),
             "speed": speed, "clonedVoice": True, "cloneVoiceId": voice_id},
            audio_file=out,
            audio_ext=".wav",
        )
    safety.audit("speak", origin=origin, voiceId=voice_id, libraryId=item["id"], model=row["key"],
                 textLength=len(text), outputSha256=safety.sha256_bytes(tagged),
                 permissionConfirmed=bool(permission_confirmed))
    return {**item, "seconds": rendered.seconds, "chunks": rendered.chunks, "modelName": row["name"]}
