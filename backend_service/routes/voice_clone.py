"""Voice cloning routes (``/api/voice/clone/*``), included by ``routes/voice.py``.

Every route except reading the model list goes through ``clone_allowed``: the app's own window on
this computer may clone; agents and other machines only after the person turned that on.
Plain ``def`` routes: the work happens in the voice worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

router = APIRouter()


class ModelDownloadRequest(BaseModel):
    key: str = Field(min_length=1, max_length=64)


class CloneSpeakRequest(BaseModel):
    voiceId: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=5_000)
    model: str = Field(default="", max_length=64)
    # Required from agents and other computers; the app window confirmed it when the voice was made.
    permissionConfirmed: bool = False


def _guard(request: Request) -> str:
    from backend_service import voice_clone_safety as safety  # noqa: PLC0415

    origin = safety.request_origin(request)
    if not safety.clone_allowed(request):
        safety.audit("refused", origin=origin, path=str(request.url.path))
        raise HTTPException(status_code=403, detail=safety.REFUSED_MESSAGE)
    return origin


@router.get("/api/voice/clone/models")
def clone_models(request: Request) -> dict[str, Any]:
    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service import voice_clone_safety as safety  # noqa: PLC0415

    return {
        "models": voice_clone.list_models(),
        "available": voice_clone.available_here(),
        "allowedHere": safety.clone_allowed(request),
    }


@router.post("/api/voice/clone/models/download")
def download_clone_model(request: Request, body: ModelDownloadRequest) -> dict[str, Any]:
    from backend_service import voice_clone  # noqa: PLC0415

    _guard(request)
    try:
        return voice_clone.start_download(body.key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/voice/clone/voices")
def list_cloned_voices(request: Request) -> dict[str, Any]:
    from backend_service import voice_clone  # noqa: PLC0415

    _guard(request)
    return {"voices": voice_clone.list_voices()}


@router.post("/api/voice/clone/voices")
def create_cloned_voice(
    request: Request,
    sample: UploadFile,
    speakerName: str = Form(min_length=1, max_length=60),
    basis: str = Form(pattern=r"^(own|permission)$"),
    permissionConfirmed: bool = Form(default=False),
    name: str = Form(default="", max_length=80),
    referenceText: str = Form(default="", max_length=2_000),
) -> dict[str, Any]:
    """Save a voice from its sample (an uploaded file or a microphone recording) with its permission record."""
    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415
    from backend_service.routes.voice import _receive_upload, _worker_error  # noqa: PLC0415
    from backend_service.voice_runtime import transcribe_file  # noqa: PLC0415

    origin = _guard(request)
    if not permissionConfirmed:
        raise HTTPException(status_code=400, detail="Confirm that this is your own voice or that the speaker gave permission.")
    if Path(sample.filename or "").suffix.lower() not in voice_clone.SAMPLE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="The voice sample must be a WAV, MP3, M4A, FLAC or OGG file.")
    sample_path, sample_ext = _receive_upload(sample)
    try:
        def transcribe(path):
            return transcribe_file(path, "", None)

        try:
            with HEAVY_JOBS.slot("voice", "ui" if origin == "ui" else "agent", "voice sample"):
                return voice_clone.create_voice(
                    speaker_name=speakerName,
                    basis=basis,
                    name=name,
                    sample_audio=sample_path,
                    sample_ext=sample_ext,
                    sample_text=referenceText,
                    permission_confirmed=permissionConfirmed,
                    origin=origin,
                    transcribe=transcribe,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _worker_error(exc) from exc
    finally:
        sample_path.unlink(missing_ok=True)


@router.delete("/api/voice/clone/voices/{voice_id}")
def delete_cloned_voice(request: Request, voice_id: str) -> dict[str, Any]:
    from backend_service import voice_clone  # noqa: PLC0415

    _guard(request)
    if not voice_clone.delete_voice(voice_id):
        raise HTTPException(status_code=404, detail="Cloned voice not found.")
    return {"deleted": True}


@router.get("/api/voice/clone/watermark/{item_id}")
def detect_library_watermark(request: Request, item_id: str) -> dict[str, Any]:
    """Whether a Library item's audio carries the cloned-voice watermark (probability, message)."""
    from backend_service import voice_clone_safety as safety  # noqa: PLC0415
    from backend_service import voice_library  # noqa: PLC0415
    from backend_service.routes.voice import _worker_error  # noqa: PLC0415

    if not voice_library.is_valid_id(item_id):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    path = voice_library.audio_path(item_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found.")
    try:
        return safety.detect_watermark(path)
    except Exception as exc:  # noqa: BLE001
        raise _worker_error(exc) from exc


@router.post("/api/voice/clone/speak")
def speak_cloned(request: Request, body: CloneSpeakRequest) -> dict[str, Any]:
    """Speech in a cloned voice, tagged and saved to the Library; returns the Library item."""
    from backend_service import voice_clone  # noqa: PLC0415
    from backend_service.helpers.job_queue import HEAVY_JOBS  # noqa: PLC0415
    from backend_service.routes.voice import _worker_error  # noqa: PLC0415

    origin = _guard(request)
    try:
        with HEAVY_JOBS.slot("voice", "ui" if origin == "ui" else "agent", "cloned speech"):
            return voice_clone.speak(body.voiceId, body.text, body.model, origin=origin,
                                     permission_confirmed=body.permissionConfirmed)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise _worker_error(exc) from exc
