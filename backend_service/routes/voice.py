"""Voice I/O API routes — STT transcription and TTS synthesis endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0


class SaveTranscriptRequest(BaseModel):
    text: str


class SaveAudioRequest(BaseModel):
    text: str
    voice: str
    audioBase64: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/api/voice/runtime")
def voice_runtime(request: Request) -> dict[str, Any]:
    """Return voice backend capabilities + model/voice lists."""
    # Lazy import keeps backend startup fast (CLAUDE.md guideline).
    from backend_service.voice_runtime import (  # noqa: PLC0415
        get_voice_capabilities,
        list_stt_models,
        list_tts_voices,
    )

    caps = get_voice_capabilities()
    caps["sttModels"] = list_stt_models()
    caps["ttsVoices"] = list_tts_voices()
    return caps


@router.get("/api/voice/models")
def voice_models(request: Request) -> dict[str, Any]:
    """Return STT model list and TTS voice list."""
    from backend_service.voice_runtime import list_stt_models, list_tts_voices  # noqa: PLC0415

    return {
        "sttModels": list_stt_models(),
        "ttsVoices": list_tts_voices(),
    }


@router.post("/api/voice/transcribe")
async def transcribe(
    request: Request,
    audio: UploadFile,
    model: str = Form(default="mlx-community/whisper-large-v3-turbo-q4"),
) -> dict[str, Any]:
    """Accept an audio file upload and return the transcript."""
    from backend_service.voice_runtime import transcribe_audio  # noqa: PLC0415

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    mime_type = audio.content_type or "audio/webm"

    start = time.monotonic()
    try:
        text = transcribe_audio(audio_bytes, mime_type, model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    duration_s = round(time.monotonic() - start, 2)
    return {"text": text, "duration_s": duration_s}


@router.post("/api/voice/synthesize")
def synthesize(request: Request, body: SynthesizeRequest) -> Response:
    """Synthesize speech from text and return WAV bytes."""
    from backend_service.voice_runtime import list_tts_voices, synthesize_speech  # noqa: PLC0415

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    known_voices = {v["id"] for v in list_tts_voices()}
    if body.voice not in known_voices:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown voice '{body.voice}'. Available: {', '.join(sorted(known_voices))}.",
        )

    speed = max(0.1, min(4.0, body.speed))

    try:
        wav_bytes = synthesize_speech(body.text, body.voice, speed)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

    return Response(content=wav_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# kokoro-onnx voice-file download (non-Apple TTS lane, FU-083)
# ---------------------------------------------------------------------------

@router.post("/api/voice/kokoro/download")
def start_kokoro_download(request: Request) -> dict[str, Any]:
    from backend_service.voice_kokoro_onnx import start_download  # noqa: PLC0415

    return start_download()


@router.get("/api/voice/kokoro/download/status")
def kokoro_download_status(request: Request) -> dict[str, Any]:
    from backend_service.voice_kokoro_onnx import download_status, is_installed  # noqa: PLC0415

    status = download_status()
    status["installed"] = is_installed()
    return status


# ---------------------------------------------------------------------------
# Gallery — saved transcripts + synthesized audio clips
# ---------------------------------------------------------------------------

@router.get("/api/voice/gallery")
def list_gallery(request: Request) -> dict[str, Any]:
    from backend_service.voice_gallery import list_items  # noqa: PLC0415

    return {"items": list_items()}


@router.post("/api/voice/gallery/transcript")
def save_gallery_transcript(request: Request, body: SaveTranscriptRequest) -> dict[str, Any]:
    from backend_service.voice_gallery import save_transcript  # noqa: PLC0415

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    return save_transcript(text)


@router.post("/api/voice/gallery/audio")
def save_gallery_audio(request: Request, body: SaveAudioRequest) -> dict[str, Any]:
    import base64

    from backend_service.voice_gallery import save_audio  # noqa: PLC0415

    try:
        audio_bytes = base64.b64decode(body.audioBase64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid audio data.") from exc
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio data is empty.")
    return save_audio(body.text, body.voice, audio_bytes)


@router.delete("/api/voice/gallery/{item_id}")
def delete_gallery_item(request: Request, item_id: str) -> dict[str, Any]:
    from backend_service.voice_gallery import delete_item, is_valid_id  # noqa: PLC0415

    if not is_valid_id(item_id):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    if not delete_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"deleted": True}


@router.get("/api/voice/gallery/{item_id}/audio")
def get_gallery_audio(request: Request, item_id: str) -> Response:
    from backend_service.voice_gallery import get_audio_path, is_valid_id  # noqa: PLC0415

    if not is_valid_id(item_id):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    audio_path = get_audio_path(item_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio not found.")
    return Response(content=audio_path.read_bytes(), media_type="audio/wav")
