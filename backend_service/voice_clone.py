"""Cloned voices: model downloads, making a voice from a voice sample, and speaking with it.

A cloned voice is a Voice Library item of kind ``clone``: the voice sample (an uploaded file or a microphone
recording), its transcript and the permission record (speaker's name, basis, time, sample hash, app version).
Deleting the voice deletes all of it. Speech made with a cloned voice goes through ``apply_watermark``
(AudioSeal, inaudible, always on, D-47), gets the AI tag in the WAV, is saved to the Library as ``speech`` with
``clonedVoice: true``, and is written to the local audit log.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend_service import voice_clone_safety as safety
from backend_service import voice_env
from backend_service.catalog import voice_models as _catalog
from backend_service.catalog.voice_models import CLONE_MODELS, clone_model, voice_model

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
# Per running download: its cancel event and child process (never sent to the UI).
_CONTROL: dict[str, dict[str, Any]] = {}


def _app_version() -> str:
    try:
        from backend_service.helpers.system_hardware import _resolve_app_version  # noqa: PLC0415

        return str(_resolve_app_version())
    except Exception:  # noqa: BLE001 - the version is a label, never a blocker
        return "unknown"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _revision_dir(repo: str, revision: str) -> Path:
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir  # noqa: PLC0415

    return _hf_repo_cache_dir(repo) / "snapshots" / revision


def weights_ready(path: Path, files: list[str] | None = None) -> bool:
    """``config.json`` and the weights: one ``model.safetensors``, or an index whose every shard is on disk. A row
    that lists its ``files`` (the CUDA rows: PyTorch checkpoints, a GGUF) is ready when each of them is there."""
    if files:
        return all((path / name).is_file() for name in files)
    if not (path / "config.json").is_file():
        return False
    if (path / "model.safetensors").exists():
        return True
    index = path / "model.safetensors.index.json"
    try:
        shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values())
    except (OSError, ValueError, AttributeError):
        return False
    return bool(shards) and all((path / name).exists() for name in shards)


def companion_dir(companion: dict[str, Any]) -> Path | None:
    """A companion repo's pinned folder, when every listed file is there."""
    path = _revision_dir(companion["repo"], companion["revision"])
    return path if all((path / name).exists() for name in companion["files"]) else None


def companion_paths(row: dict[str, Any]) -> dict[str, str]:
    """Repo id -> local folder for the row's loader companions (sent to the worker for its redirect)."""
    paths: dict[str, str] = {}
    for companion in row.get("companions", []):
        folder = companion_dir(companion) if companion.get("role") == "loader" else None
        if folder is not None:
            paths[companion["repo"]] = str(folder)
    return paths


def companion_file(row: dict[str, Any], role: str) -> Path | None:
    """The first listed file of the row's companion with ``role`` (Orpheus GGUF: its SNAC decoder)."""
    for companion in row.get("companions", []):
        folder = companion_dir(companion) if companion.get("role") == role else None
        if folder is not None:
            return folder / companion["files"][0]
    return None


def default_voice_path(row: dict[str, Any]) -> Path | None:
    """The ``conds.safetensors`` a ``defaultVoice`` companion provides (Chatterbox V3 has none of its own)."""
    for companion in row.get("companions", []):
        folder = companion_dir(companion) if companion.get("role") == "defaultVoice" else None
        if folder is not None:
            return folder / "conds.safetensors"
    return None


def snapshot_dir(row: dict[str, Any]) -> Path | None:
    """The pinned revision's folder in the Hugging Face cache, when its weights and every companion are there."""
    path = _revision_dir(row["repo"], row["revision"])
    if not weights_ready(path, row.get("files")):
        return None
    if any(companion_dir(c) is None for c in row.get("companions", [])):
        return None
    return path


NOT_HERE = "Voice cloning needs an Apple Silicon Mac, or the speech GPU add-on on a computer with an NVIDIA GPU."


def available_here() -> bool:
    """Speech models beyond Kokoro run here: Apple Silicon (MLX), or the speech GPU add-on with its NVIDIA GPU."""
    return bool(voice_env.speech_lanes())


def runs_here(row: dict[str, Any]) -> bool:
    return any(name in voice_env.speech_lanes() for name in row["lanes"])


def _ready(row: dict[str, Any]) -> bool:
    """Weights on disk; a model that clones also needs the watermark component."""
    if snapshot_dir(row) is None:
        return False
    return voice_env.watermark_installed() if row["capabilities"]["clone"] else True


def list_models(*, every_model: bool = False) -> list[dict[str, Any]]:
    """The cloning models, or with ``every_model`` every listed speech model (speak, style, clone, design)."""
    rows = []
    for row in _catalog.listed_models() if every_model else CLONE_MODELS:
        job = _DOWNLOADS.get(row["key"], {})
        rows.append({
            "key": row["key"],
            "name": row["name"],
            "summary": row["summary"],
            "repo": row["repo"],
            "revision": row["revision"],
            "upstream": row.get("upstream"),
            "sizeGb": row["sizeGb"],
            # Model plus companion repos: what Download fetches.
            "downloadGb": row["downloadGb"],
            "languages": row["languages"],
            "default": row["default"],
            "licence": dict(row["licence"]),
            "kind": row["kind"],
            "family": row["family"],
            "capabilities": dict(row["capabilities"]),
            "controls": dict(row.get("controls") or {}),
            "companions": [{k: c[k] for k in ("repo", "revision", "files")} for c in row.get("companions", [])],
            "sampleRate": row["sampleRate"],
            "minReferenceS": row.get("minReferenceS"),
            "rank": row.get("rank"),
            "builtinWatermark": row.get("builtinWatermark"),
            # Speed and memory from the catalog, with its basis and where it was measured; None when not measured.
            "estimates": dict(row["estimates"]) if row.get("estimates") else None,
            "supported": runs_here(row),
            # Ready means the weights and companions, and for a cloning model the watermark component; Download
            # installs all of them.
            "installed": _ready(row),
            "download": {
                "state": job.get("state", "idle"), "error": job.get("error"), "phase": job.get("phase"),
                **(_progress(row) if job.get("state") == "downloading" else {}),
            },
        })
    return rows


class _Cancelled(Exception):
    """The person cancelled the download; nothing more is written to the job."""


def _fetch_repo(control: dict[str, Any], repo: str, revision: str, patterns: list[str] | None) -> None:
    """Download one repo in a child process (the shared download helper) so Cancel can stop it mid-file.

    Same child, environment and stop sequence as the Models page downloads: terminate, then kill after 5 s.
    Partial files stay in the cache so a later Download resumes them.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    from backend_service.state._helpers import _spawn_snapshot_download  # noqa: PLC0415

    cancel: threading.Event = control["cancel"]
    if cancel.is_set():
        raise _Cancelled()
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["HF_HUB_DISABLE_XET"] = "1"  # keeps the bytes in the per-repo cache tree the progress count reads
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        process = _spawn_snapshot_download(repo, env, log, allow_patterns=patterns, revision=revision)
        control["proc"] = process
        try:
            while process.poll() is None:
                if cancel.is_set():
                    try:
                        process.terminate()
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise _Cancelled()
                time.sleep(0.3)
        finally:
            control["proc"] = None
        if process.returncode != 0:
            log.flush()
            log.seek(0)
            raise RuntimeError(_read_text_tail_of(log) or f"download exited with status {process.returncode}")


def _read_text_tail_of(handle: Any, limit: int = 400) -> str:
    return handle.read()[-limit:].strip()


def _own(key: str, token: str) -> bool:
    job = _DOWNLOADS.get(key)
    return bool(job) and job.get("token") == token


def start_download(key: str) -> dict[str, Any]:
    row = voice_model(key)
    if row["key"] not in {r["key"] for r in _catalog.listed_models()}:
        raise ValueError(f"{row['name']} is not offered on this version yet.")
    if not available_here():
        raise ValueError(NOT_HERE)
    if not runs_here(row):
        # An Apple-only (MLX) row on Windows/Linux, or a CUDA row on a Mac: its weights cannot load here, so
        # downloading them only fills the disk (Speak offered "Download" for a Mac-only default voice on Windows).
        where = "Apple Silicon Macs" if row["lanes"] == ["mlx"] else "computers with an NVIDIA GPU"
        raise ValueError(f"{row['name']} runs on {where} only, so it cannot be used on this computer.")
    token = uuid.uuid4().hex
    control: dict[str, Any] = {"cancel": threading.Event(), "proc": None}
    with _DOWNLOAD_LOCK:
        if _DOWNLOADS.get(row["key"], {}).get("state") == "downloading":
            return dict(_DOWNLOADS[row["key"]])
        _DOWNLOADS[row["key"]] = {"state": "downloading", "error": None, "startedAt": time.time(), "token": token}
        _CONTROL[row["key"]] = control

    def run() -> None:
        try:
            _DOWNLOADS[row["key"]]["phase"] = "weights"
            _fetch_repo(control, row["repo"], row["revision"], row.get("files"))
            # Companion repos the loaders would otherwise fetch unpinned at load time.
            for companion in row.get("companions", []):
                _fetch_repo(control, companion["repo"], companion["revision"], companion["files"])
            # Cloned output is always watermarked, so the component comes with a cloning model.
            if row["capabilities"]["clone"] and not voice_env.watermark_installed():
                if control["cancel"].is_set():
                    raise _Cancelled()
                _DOWNLOADS[row["key"]]["phase"] = "watermark"
                voice_env.install_watermark(control["cancel"])
            with _DOWNLOAD_LOCK:
                if _own(row["key"], token) and not control["cancel"].is_set():
                    _DOWNLOADS[row["key"]] = {"state": "done", "error": None}
        except _Cancelled:
            pass  # cancel_download already recorded it
        except Exception as exc:  # noqa: BLE001 - shown on the panel
            with _DOWNLOAD_LOCK:
                if _own(row["key"], token) and not control["cancel"].is_set():
                    step = "the watermark component" if _DOWNLOADS[row["key"]].get("phase") == "watermark" else row["name"]
                    _DOWNLOADS[row["key"]] = {"state": "failed", "error": f"Downloading {step} failed: {exc}"}

    threading.Thread(target=run, daemon=True, name=f"clone-download-{row['key']}").start()
    return _public_job(_DOWNLOADS[row["key"]])


def cancel_download(key: str) -> dict[str, Any]:
    """Stop a running speech-model download (weights, companions or the watermark step).

    The job becomes ``cancelled`` at once, so the row offers Download again and Delete is no longer refused.
    Files already fetched stay in the cache for a resume, as for the other model downloads. Raises ValueError for
    an unknown key; cancelling a model that is not downloading changes nothing.
    """
    row = voice_model(key)
    with _DOWNLOAD_LOCK:
        job = _DOWNLOADS.get(row["key"])
        control = _CONTROL.get(row["key"])
        if not job or job.get("state") != "downloading":
            return _public_job(job or {})
        _DOWNLOADS[row["key"]] = {"state": "cancelled", "error": None}
    if control is not None:
        control["cancel"].set()
        process = control.get("proc")
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - the worker thread kills it if this did not stop it
                pass
    return _public_job(_DOWNLOADS[row["key"]])


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if k not in ("token",)}


def _progress(row: dict[str, Any]) -> dict[str, float | None]:
    """Bytes on disk for the model and its companions against the whole download, for the row's progress bar."""
    from backend_service.helpers.hf_cache_paths import _hf_repo_downloaded_bytes  # noqa: PLC0415

    try:
        got = _hf_repo_downloaded_bytes(row["repo"], row.get("files")) + sum(
            _hf_repo_downloaded_bytes(c["repo"], c["files"]) for c in row.get("companions", [])
        )
    except OSError:
        return {"progress": None, "downloadedGb": None, "totalGb": None}
    total = float(row["downloadGb"])
    gb = got / 1e9
    return {"progress": min(0.99, gb / total) if total > 0 else None, "downloadedGb": round(gb, 2), "totalGb": total}


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
    style: str = "",
    language: str = "auto",
    expression: float = 0.5,
    guidance: float = 0.5,
) -> dict[str, Any]:
    """Speak ``text`` with a cloned voice; returns the saved Library item (``speech``, ``clonedVoice``) with
    ``notes``.

    Long text is cut into sentences and spoken part by part; ``speed`` and ``pitch`` are applied to the
    joined audio before the watermark and the AI tag, which the finished speech always gets, whatever the
    model (Qwen3-TTS Base, VoxCPM2, Chatterbox V3, Chatterbox-Turbo). This is the only path that sends a
    person's voice sample to the worker.

    VoxCPM2 picks its mode from ``style``: empty is the exact mode (the clip, the same clip as a prompt and its
    stored transcript); filled is the steered mode (the clip plus the style). Chatterbox V3 takes
    ``language``, ``expression`` and ``guidance``. Other models ignore the style, with a note.

    Agents and other computers (``origin`` other than ``ui``) must pass ``permission_confirmed``: their
    confirmation that the speaker's permission covers this use. It is written to the audit log.
    """
    from backend_service import voice_library  # noqa: PLC0415
    from backend_service import voice_speak  # noqa: PLC0415

    if get_voice(voice_id) is None:
        raise LookupError("Cloned voice not found. Its permission record is missing or it was deleted.")
    if origin != "ui" and not permission_confirmed:
        raise ValueError(PERMISSION_REQUIRED)
    text = text.strip()
    if not text:
        raise ValueError("Text is required.")
    if len(text) > max_chars:
        raise ValueError(f"Cloned speech takes up to {max_chars} characters at a time.")
    plan, row, voice, notes = clone_plan(
        voice_id, model_key, origin=origin, permission_confirmed=permission_confirmed, style=style,
        language=language, expression=expression, guidance=guidance,
    )
    rendered = voice_speak.render(plan, text, speed=speed, pitch=pitch, seed=seed)
    duration = rendered.duration_s
    tagged = mark_cloned(rendered.wav, row, voice_id, altered=speed != 1.0 or pitch != 0)
    with tempfile.TemporaryDirectory(prefix="chaosengine-cloned-") as tmp:
        out = Path(tmp) / "speech.wav"
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
    return {**item, "seconds": rendered.seconds, "chunks": rendered.chunks, "modelName": row["name"],
            "notes": [*notes, *rendered.notes]}


def clone_plan(
    voice_id: str,
    model_key: str | None,
    *,
    origin: str,
    permission_confirmed: bool = False,
    style: str = "",
    language: str = "auto",
    expression: float = 0.5,
    guidance: float = 0.5,
) -> tuple[Any, dict[str, Any], dict[str, Any], list[str]]:
    """Every check a cloned voice passes before it speaks, then its plan: ``(plan, row, voice record, notes)``.

    The permission record must exist; agents and other computers must confirm permission; the voice pack, the
    model and the watermark must be installed; the sample must be long enough for the model. Shared by ``speak``
    (saved to the Library) and Chat read-aloud (``voice_chat``, not saved), so both follow the same rules.
    """
    from backend_service import voice_library  # noqa: PLC0415
    from backend_service import voice_speak  # noqa: PLC0415

    voice = get_voice(voice_id)
    if voice is None:
        raise LookupError("Cloned voice not found. Its permission record is missing or it was deleted.")
    if origin != "ui" and not permission_confirmed:
        raise ValueError(PERMISSION_REQUIRED)
    if not available_here():
        raise ValueError(NOT_HERE)
    if not voice_env.is_installed():
        raise RuntimeError("The voice runtime is not installed. Install it from Voice > Models.")
    row = pick_clone_model(model_key)
    if snapshot_dir(row) is None:
        raise RuntimeError(f"{row['name']} is not downloaded. Download it in Voice > Models.")
    if not voice_env.watermark_installed():
        raise RuntimeError(safety.WATERMARK_MISSING)
    reference = voice_library.audio_path(voice_id)
    if reference is None:
        raise RuntimeError("This voice's sample is missing. Delete the voice and make it again.")
    minimum = float(row.get("minReferenceS") or MIN_REFERENCE_S)
    duration_s = float(voice.get("durationS") or 0.0)
    if duration_s and duration_s < minimum:
        raise ValueError(
            f"{row['name']} needs a voice sample of at least {minimum:g} s; this one is {duration_s:.1f} s. "
            "Pick another cloning model or record a longer sample."
        )
    style = " ".join((style or "").split())
    extra, notes = clone_fields(row, str(reference), str(voice.get("text") or ""), style, language, expression, guidance)
    return voice_speak.plan_for("clone", f"clone:{voice_id}", row, extra), row, voice, notes


def mark_cloned(wav: bytes, row: dict[str, Any], voice_id: str, *, altered: bool = False) -> bytes:
    """The watermark and the AI tag every piece of cloned speech gets, whatever the model.

    ``altered``: the model's output was time-stretched or pitch-shifted afterwards (speed or pitch not neutral).
    A model's own watermark (PerTh on the CUDA Chatterbox rows) does not survive that - measured on the RTX 4090
    run: PerTh 1.0 on a plain clip, 0.0 after speed 1.3 and pitch +2, while our AudioSeal mark read 1.0 - so the
    tag only records it on unaltered audio.
    """
    version = _app_version()
    builtin = None if altered else row.get("builtinWatermark")
    with tempfile.TemporaryDirectory(prefix="chaosengine-cloned-") as tmp:
        out = Path(tmp) / "speech.wav"
        out.write_bytes(wav)
        safety.apply_watermark(out)
        return safety.tag_wav(
            out.read_bytes(),
            {"appVersion": version, "model": row["repo"], "engine": row["engine"],
             "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "voiceId": voice_id,
             "watermark": "audioseal-0.2.1:CE5A",
             # A watermark the model adds itself (PerTh on the CUDA Chatterbox rows), recorded so the tag is honest.
             **({"builtinWatermark": builtin} if builtin else {})},
            f"ChaosEngineAI {version}",
        )


STEERED_NOTE = "VoxCPM2 followed the style; the steered mode trades a little likeness for it."
NO_STYLE_NOTE = "A cloned voice has no style control with this model, so the style text was ignored."


def clone_fields(
    row: dict[str, Any], reference: str, reference_text: str, style: str, language: str, expression: float,
    guidance: float,
) -> tuple[dict[str, Any], list[str]]:
    """Worker fields for speaking with a voice sample on ``row``, and notes for the person."""
    from backend_service import voice_speak  # noqa: PLC0415

    fields: dict[str, Any] = {"refAudioPath": reference, "refText": reference_text}
    notes: list[str] = []
    family = row["family"]
    if family == "voxcpm2":
        fields["mode"] = "steered" if style else "exact"
        if style:
            fields["instruct"] = style
            notes.append(STEERED_NOTE)
        return fields, notes
    if family == "chatterbox":
        fields.update(voice_speak.chatterbox_fields(row, language, expression, guidance))
        fields.pop("defaultVoicePath", None)  # the sample is the voice
    if style:
        notes.append(NO_STYLE_NOTE)
    return fields, notes
