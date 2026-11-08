"""Voice I/O API routes: install, settings, transcription, speech, library.

Routes are plain ``def`` so FastAPI runs them in its thread pool: the work
happens in the voice worker process and must never block the event loop.
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend_service.progress import GenerationCancelled

router = APIRouter()

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_ALLOWED_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".webm",
    ".mp4", ".mov", ".mkv", ".m4v", ".aiff", ".aif", ".caf",
}


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    voice: str = Field(default="af_heart", max_length=32)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class SpeakRequest(BaseModel):
    """One request for every voice: Kokoro, a preset speaker, a designed voice or a cloned voice."""

    text: str = Field(min_length=1, max_length=20_000)
    voice: str = Field(default="af_heart", min_length=1, max_length=64)
    # A style instruction (1.7B CustomVoice), or the written description of the voice for ``design``.
    style: str = Field(default="", max_length=500)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    # Semitones; applied as a post-processing step on every model.
    pitch: float = Field(default=0.0, ge=-12.0, le=12.0)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    # Catalog key of a specific model; empty lets the backend choose from the voice.
    model: str = Field(default="", max_length=64)
    # Required from agents and other computers for cloned voices.
    permissionConfirmed: bool = False


class SettingsRequest(BaseModel):
    sttModel: str | None = Field(default=None, max_length=128)
    language: str | None = Field(default=None, max_length=8)
    ttsVoice: str | None = Field(default=None, max_length=32)
    # The voice last used on the Speak screen (any voice id, ``clone:<id>`` included); "" means none yet.
    speakVoice: str | None = Field(default=None, max_length=64)
    speed: float | None = Field(default=None, ge=0.5, le=2.0)
    autoSend: bool | None = None
    autoRead: bool | None = None
    vadSilenceMs: int | None = Field(default=None, ge=300, le=3000)
    bargeIn: bool | None = None
    idleUnloadMinutes: int | None = Field(default=None, ge=0, le=240)
    allowAgentCloning: bool | None = None


class VadRequest(BaseModel):
    session: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Base64 PCM16 mono at 16 kHz; 2 s at most per call.
    pcm: str = Field(max_length=90_000)
    reset: bool = False


class InstallRequest(BaseModel):
    extras: list[str] = Field(default_factory=list, max_length=8)


class WordModel(BaseModel):
    start: float
    end: float
    text: str = Field(max_length=200)


class SegmentModel(BaseModel):
    start: float
    end: float
    text: str = Field(max_length=20_000)
    speaker: int | None = Field(default=None, ge=0, le=99)
    words: list[WordModel] | None = Field(default=None, max_length=5_000)


class LibraryUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=2_000_000)
    segments: list[SegmentModel] | None = Field(default=None, max_length=50_000)
    # Speaker number (as text) -> name.
    speakers: dict[str, str] | None = Field(default=None, max_length=100)


class LibraryDeleteRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)


def _worker_error(exc: Exception) -> HTTPException:
    from backend_service.voice_client import VoiceWorkerError  # noqa: PLC0415
    from backend_service.voice_runtime import UnknownSttModelError  # noqa: PLC0415

    if isinstance(exc, (UnknownSttModelError, ValueError)) and not isinstance(exc, VoiceWorkerError):
        return HTTPException(status_code=400, detail=str(exc))
    if "invalid data found" in str(exc).lower() or "invaliddataerror" in str(exc).lower():
        return HTTPException(
            status_code=400,
            detail="This file could not be read as audio. Use WAV, MP3, M4A, FLAC, OGG or a video file.",
        )
    # RuntimeError / VoiceWorkerError: not installed, model missing, worker died.
    return HTTPException(status_code=503, detail=str(exc))


# ---------------------------------------------------------------------------
# Runtime, settings, install
# ---------------------------------------------------------------------------

@router.get("/api/voice/runtime")
def voice_runtime(request: Request) -> dict[str, Any]:
    from backend_service.voice_runtime import (  # noqa: PLC0415
        get_voice_capabilities,
        list_stt_models,
        list_tts_voices,
        load_settings,
    )

    caps = get_voice_capabilities()
    caps["sttModels"] = list_stt_models()
    caps["ttsVoices"] = list_tts_voices()
    caps["settings"] = load_settings()
    return caps


@router.get("/api/voice/settings")
def get_settings(request: Request) -> dict[str, Any]:
    from backend_service.voice_runtime import load_settings  # noqa: PLC0415

    return load_settings()


@router.post("/api/voice/settings")
def put_settings(request: Request, body: SettingsRequest) -> dict[str, Any]:
    from backend_service.voice_clone_safety import audit, request_origin  # noqa: PLC0415
    from backend_service.voice_runtime import save_settings  # noqa: PLC0415

    if body.allowAgentCloning is not None:
        # An agent must never be able to grant itself cloning.
        origin = request_origin(request)
        if origin != "ui":
            raise HTTPException(status_code=403, detail="This setting can only be changed in the ChaosEngineAI window.")
        audit("setting", allowAgentCloning=body.allowAgentCloning, origin=origin)
    return save_settings(body.model_dump())


@router.post("/api/voice/install")
def start_install(request: Request, body: InstallRequest) -> dict[str, Any]:
    from backend_service import voice_env  # noqa: PLC0415

    unknown = [name for name in body.extras if name not in voice_env.EXTRAS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown extra: {', '.join(unknown)}.")
    return voice_env.start_install(body.extras)


@router.get("/api/voice/install/status")
def install_status(request: Request) -> dict[str, Any]:
    from backend_service import voice_env  # noqa: PLC0415

    return voice_env.job_status()


@router.post("/api/voice/install/cancel")
def cancel_install(request: Request) -> dict[str, Any]:
    from backend_service import voice_env  # noqa: PLC0415

    return voice_env.cancel_install()


@router.post("/api/voice/worker/stop")
def stop_worker(request: Request) -> dict[str, Any]:
    from backend_service.voice_client import stop_all  # noqa: PLC0415

    stop_all()
    return {"stopped": True}


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------

@router.post("/api/voice/transcribe")
def transcribe(
    request: Request,
    audio: UploadFile,
    # Empty -> the default for this machine (Parakeet on Apple Silicon,
    # Whisper Large v3 Turbo elsewhere).
    model: str = Form(default=""),
    language: str = Form(default=""),
    # "transcribe" or "translate" (to English, Whisper only).
    task: str = Form(default="transcribe", pattern=r"^(transcribe|translate)$"),
    words: bool = Form(default=False),
    speakers: bool = Form(default=False),
    # 0 = work out how many people speak.
    speakerCount: int = Form(default=0, ge=0, le=20),
    # Keep the file and the transcript in the Library.
    save: bool = Form(default=False),
) -> dict[str, Any]:
    from backend_service.voice_runtime import transcribe_file  # noqa: PLC0415

    tmp_path, suffix = _receive_upload(audio)
    try:
        from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415

        try:
            with HEAVY_JOBS.slot("voice", "ui", "transcript"):
                result = transcribe_file(
                    tmp_path,
                    model,
                    language.strip() or None,
                    task=task,
                    words=words,
                    speakers=speakers,
                    speaker_count=speakerCount,
                )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to 400 / 503
            raise _worker_error(exc) from exc
        if save:
            from backend_service import voice_library  # noqa: PLC0415

            title = Path(audio.filename or "").stem or None
            item = voice_library.create(
                "transcript",
                {**result, "title": title, "source": audio.filename or None},
                audio_file=tmp_path,
                audio_ext=suffix,
            )
            result["libraryId"] = item["id"]
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


def _receive_upload(audio: UploadFile) -> tuple[Path, str]:
    """Stream an upload to a temporary file (500 MB cap); returns path and suffix."""
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        suffix = ".wav" if "wav" in (audio.content_type or "") else ".webm"
    with tempfile.NamedTemporaryFile(prefix="chaosengine-stt-", suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        written = 0
        while chunk := audio.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="The file is larger than 500 MB.")
            tmp.write(chunk)
    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Audio file is empty.")
    return tmp_path, suffix


@router.post("/api/voice/vad")
def voice_activity(request: Request, body: VadRequest) -> dict[str, Any]:
    """Speech probability per 32 ms frame, for conversation mode."""
    from backend_service.voice_runtime import detect_speech  # noqa: PLC0415

    try:
        return {"probs": detect_speech(body.session, body.pcm, body.reset)}
    except Exception as exc:  # noqa: BLE001
        raise _worker_error(exc) from exc


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------

@router.post("/api/voice/synthesize")
def synthesize(request: Request, body: SynthesizeRequest) -> Response:
    """Whole text in one WAV with a built-in voice. Kept for older callers; ``/api/voice/speak`` is the general route."""
    from backend_service import voice_speak  # noqa: PLC0415
    from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")
    if body.voice.strip().startswith(voice_speak.CLONE_PREFIX):
        raise HTTPException(status_code=400, detail="Cloned voices speak through /api/voice/speak.")
    try:
        with HEAVY_JOBS.slot("voice", "ui", "speech"):
            result = voice_speak.speak(body.text, body.voice, speed=body.speed, origin="ui")
    except Exception as exc:  # noqa: BLE001
        raise _worker_error(exc) from exc
    return Response(content=result["wav"], media_type="audio/wav")


@router.post("/api/voice/speak/cancel")
def speak_cancel(request: Request) -> dict[str, Any]:
    """The Speak screen's Cancel: stop the screen's own running speech job and drop its waiting ones.

    An agent's job is left alone. Returns ``{cancelled}``: whether anything was stopped or dropped.
    """
    from backend_service import voice_speak  # noqa: PLC0415
    from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415

    def stop_running() -> bool:
        # Only while a screen job runs: the worker is shared with transcription and chat voice.
        return HEAVY_JOBS.running_source("voice") == "ui" and voice_speak.request_cancel()

    return {"cancelled": HEAVY_JOBS.cancel_studio("voice", stop_running)}


@router.post("/api/voice/speak")
def speak(request: Request, body: SpeakRequest) -> dict[str, Any]:
    """Speak with any voice. The backend picks the model from the voice.

    Returns ``{audio (base64 WAV), durationS, seconds, chunks, voice, kind, model, modelName, clonedVoice,
    libraryId, notes}``. A cloned voice is checked like ``/api/voice/clone/speak`` (permission record, agent
    switch, ``permissionConfirmed`` from agents), tagged, watermarked and saved to the Library
    (``libraryId``); other voices are not saved here.
    """
    from backend_service import voice_speak  # noqa: PLC0415
    from backend_service import voice_clone_safety as safety  # noqa: PLC0415
    from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415

    origin = "ui"
    if body.voice.strip().startswith(voice_speak.CLONE_PREFIX):
        origin = safety.request_origin(request)
        if not safety.clone_allowed(request):
            safety.audit("refused", origin=origin, path=str(request.url.path))
            raise HTTPException(status_code=403, detail=safety.REFUSED_MESSAGE)
    try:
        with HEAVY_JOBS.slot("voice", "ui" if origin == "ui" else "agent", "speech"):
            result = voice_speak.speak(
                body.text, body.voice, style=body.style, speed=body.speed, pitch=body.pitch, seed=body.seed,
                model=body.model, origin=origin, permission_confirmed=body.permissionConfirmed,
            )
    except GenerationCancelled as exc:
        raise HTTPException(status_code=499, detail="Speech was cancelled.") from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _worker_error(exc) from exc
    return {"audio": base64.b64encode(result["wav"]).decode("ascii"), **{k: v for k, v in result.items() if k != "wav"}}


@router.get("/api/voice/voices")
def list_voices(request: Request) -> dict[str, Any]:
    """Built-in, preset, designed and cloned voices in one list (cloned ones only where cloning is allowed)."""
    from backend_service import voice_clone_safety as safety  # noqa: PLC0415
    from backend_service import voice_speak  # noqa: PLC0415

    return {"voices": voice_speak.list_voices(include_clones=safety.clone_allowed(request))}


@router.get("/api/voice/models")
def list_voice_models(request: Request) -> dict[str, Any]:
    """Every Qwen3-TTS model (speak, style, clone, design) with install state and capabilities."""
    from backend_service import voice_clone  # noqa: PLC0415

    return {"models": voice_clone.list_models(every_model=True), "available": voice_clone.available_here()}


@router.post("/api/voice/speak/stream")
def speak_stream(request: Request, body: SynthesizeRequest) -> StreamingResponse:
    """Read-aloud: one NDJSON line per sentence, sent as soon as it is spoken.

    Lines: ``{"index", "total", "text", "audio" (base64 WAV), "durationS"}``,
    then ``{"done": true}``; a failure ends the stream with ``{"error"}``.
    """
    from backend_service.voice_runtime import synthesize as run  # noqa: PLC0415
    from backend_service.voice_text import speech_sentences  # noqa: PLC0415

    sentences = speech_sentences(body.text)
    if not sentences:
        raise HTTPException(status_code=400, detail="Nothing to read aloud.")

    def lines() -> Iterator[str]:
        for index, sentence in enumerate(sentences):
            try:
                wav, duration = run(sentence, body.voice, body.speed)
            except Exception as exc:  # noqa: BLE001 - reported in-stream
                yield json.dumps({"error": str(exc), "index": index}) + "\n"
                return
            yield json.dumps({
                "index": index,
                "total": len(sentences),
                "text": sentence,
                "audio": base64.b64encode(wav).decode("ascii"),
                "durationS": duration,
            }) + "\n"
        yield json.dumps({"done": True}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Library: saved transcripts and generated speech
# ---------------------------------------------------------------------------

_MEDIA_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".flac": "audio/flac", ".ogg": "audio/ogg", ".oga": "audio/ogg", ".opus": "audio/ogg",
    ".webm": "audio/webm", ".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
    ".m4v": "video/mp4", ".aiff": "audio/aiff", ".aif": "audio/aiff", ".caf": "audio/x-caf",
}


def _library_id(item_id: str) -> str:
    from backend_service.voice_library import is_valid_id  # noqa: PLC0415

    if not is_valid_id(item_id):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    return item_id


@router.get("/api/voice/library")
def list_library(request: Request, q: str = "", kind: str = "") -> dict[str, Any]:
    from backend_service.voice_library import KINDS, list_items  # noqa: PLC0415

    if kind and kind not in KINDS:
        raise HTTPException(status_code=400, detail="Unknown kind.")
    return {"items": list_items(q[:200], kind)}


@router.post("/api/voice/library/speech")
def save_speech(
    request: Request,
    audio: UploadFile,
    text: str = Form(default="", max_length=200_000),
    voice: str = Form(default="", max_length=64),
    speed: float = Form(default=1.0, ge=0.5, le=2.0),
    title: str = Form(default="", max_length=200),
    # The voice's display name and kind, so the Library card can say "Ryan, Natural" instead of the id.
    voiceName: str = Form(default="", max_length=120),
    voiceKind: str = Form(default="", pattern=r"^(|builtin|preset|design|clone)$"),
) -> dict[str, Any]:
    from backend_service import voice_library  # noqa: PLC0415

    from backend_service.voice_clone_safety import read_wav_tag  # noqa: PLC0415

    tmp_path, suffix = _receive_upload(audio)
    try:
        # Speech from a cloned voice keeps its flag when saved again (the tag is in the file).
        cloned = bool((read_wav_tag(tmp_path) or {}).get("clonedVoice")) if suffix == ".wav" else False
        return voice_library.create(
            "speech",
            {"text": text, "voice": voice or None, "speed": speed, "title": title or None, "clonedVoice": cloned or None,
             "voiceName": " ".join(voiceName.split()) or None, "voiceKind": voiceKind or None},
            audio_file=tmp_path,
            audio_ext=suffix,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/api/voice/library/delete")
def delete_library_items(request: Request, body: LibraryDeleteRequest) -> dict[str, Any]:
    from backend_service import voice_library  # noqa: PLC0415

    ids = [_library_id(item_id) for item_id in body.ids]
    return {"deleted": [item_id for item_id in ids if voice_library.delete(item_id)]}


@router.get("/api/voice/library/{item_id}")
def get_library_item(request: Request, item_id: str) -> dict[str, Any]:
    from backend_service import voice_library  # noqa: PLC0415

    item = voice_library.get(_library_id(item_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found.")
    # Where the audio lives on this machine, so other areas (the storyboard) can import the file.
    audio = voice_library.audio_path(item["id"])
    return {**item, "audioPath": str(audio) if audio is not None else None}


@router.post("/api/voice/library/{item_id}")
def update_library_item(request: Request, item_id: str, body: LibraryUpdateRequest) -> dict[str, Any]:
    from backend_service import voice_library  # noqa: PLC0415

    patch = body.model_dump(exclude_none=True)
    if "speakers" in patch:
        patch["speakers"] = {
            str(k)[:3]: " ".join(str(v).split())[:60] for k, v in patch["speakers"].items() if str(k).isdigit()
        }
    item = voice_library.update(_library_id(item_id), patch)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found.")
    return item


@router.delete("/api/voice/library/{item_id}")
def delete_library_item(request: Request, item_id: str) -> dict[str, Any]:
    from backend_service import voice_library  # noqa: PLC0415

    if not voice_library.delete(_library_id(item_id)):
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"deleted": True}


@router.get("/api/voice/library/{item_id}/audio")
def get_library_audio(request: Request, item_id: str, format: str = "") -> Response:
    """The item's audio; ``format=mp3`` converts it."""
    from backend_service import voice_library  # noqa: PLC0415

    path = voice_library.audio_path(_library_id(item_id))
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found.")
    if format == "mp3":
        return _mp3_response(path)
    if format:
        raise HTTPException(status_code=400, detail="Unknown format.")
    return FileResponse(path, media_type=_MEDIA_TYPES.get(path.suffix, "application/octet-stream"))


@router.post("/api/voice/encode")
def encode_audio(request: Request, audio: UploadFile, format: str = Form(default="mp3")) -> Response:
    """Convert an audio file (Speak studio output) to MP3."""
    if format != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 is supported.")
    tmp_path, _suffix = _receive_upload(audio)
    try:
        return _mp3_response(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _mp3_response(source: Path) -> Response:
    from backend_service.voice_clone_safety import TAG_PREFIX, read_wav_tag  # noqa: PLC0415
    from backend_service.voice_runtime import encode_mp3  # noqa: PLC0415

    # A cloned voice's AI tag goes with it into the MP3.
    tag = read_wav_tag(source) if source.suffix.lower() == ".wav" else None
    with tempfile.TemporaryDirectory(prefix="chaosengine-mp3-") as folder:
        target = Path(folder) / "out.mp3"
        try:
            if tag:
                encode_mp3(source, target, comment=TAG_PREFIX + json.dumps(tag, separators=(",", ":")))
            else:
                encode_mp3(source, target)
        except Exception as exc:  # noqa: BLE001
            raise _worker_error(exc) from exc
        return Response(content=target.read_bytes(), media_type="audio/mpeg")


# Voice cloning lives in its own module; included here so routes/__init__.py stays unchanged.
from backend_service.routes.voice_clone import router as _clone_router  # noqa: E402

router.include_router(_clone_router)
