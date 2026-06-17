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
    from backend_service.voice_runtime import synthesize_speech  # noqa: PLC0415

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="Text is required.")

    speed = max(0.1, min(4.0, body.speed))

    try:
        wav_bytes = synthesize_speech(body.text, body.voice, speed)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

    return Response(content=wav_bytes, media_type="audio/wav")
