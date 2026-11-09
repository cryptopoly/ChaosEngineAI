"""Voice routes that delete files or stop a download (D-73): a speech model, a dictation model, an add-on, the voice runtime.

Included by ``routes/voice.py``. Only the ChaosEngineAI window on this computer may call them: agents and other
machines get 403, whatever their scopes. A delete that would pull files from under a running model is refused with
409 and a plain reason; the person tries again when the job or the download has finished.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class ModelKeyRequest(BaseModel):
    key: str = Field(min_length=1, max_length=128)


class ExtraRequest(BaseModel):
    extra: str = Field(min_length=1, max_length=32)


class UninstallRequest(BaseModel):
    # Also delete the runtime's own downloads (Kokoro quick voices, turn detection, speaker labels).
    alsoRuntimeModels: bool = False


def _window_only(request: Request) -> None:
    from backend_service.voice_clone_safety import request_origin  # noqa: PLC0415

    if request_origin(request) != "ui":
        raise HTTPException(status_code=403, detail="Voice files can only be deleted in the ChaosEngineAI window.")


@router.get("/api/voice/clone/models/delete-plan")
def speech_model_delete_plan(request: Request, key: str) -> dict[str, Any]:
    """What Delete would remove (repos, GB) and keep (companions another downloaded model uses), for the confirm."""
    from backend_service import voice_model_files  # noqa: PLC0415

    _window_only(request)
    try:
        return voice_model_files.removal_plan(key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/voice/clone/models/download/cancel")
def cancel_speech_model_download(request: Request, body: ModelKeyRequest) -> dict[str, Any]:
    """Stop a speech model's download (weights, companions or the watermark step); the row offers Download again."""
    from backend_service import voice_clone  # noqa: PLC0415

    _window_only(request)
    try:
        return {"download": voice_clone.cancel_download(body.key)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/voice/clone/models/delete")
def delete_speech_model(request: Request, body: ModelKeyRequest) -> dict[str, Any]:
    """Delete a speech model's files and the companions only it uses; ``installed`` is false afterwards."""
    from backend_service import voice_model_files  # noqa: PLC0415

    _window_only(request)
    try:
        return voice_model_files.remove_speech_model(body.key)
    except voice_model_files.ModelInUse as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/api/voice/stt/delete")
def delete_dictation_model(request: Request, body: ModelKeyRequest) -> dict[str, Any]:
    """Delete a dictation model (by catalog key or repo) through the generic download delete."""
    from backend_service import voice_model_files  # noqa: PLC0415

    _window_only(request)
    try:
        repo = voice_model_files.prepare_stt_delete(body.key)
    except voice_model_files.ModelInUse as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"repo": repo, "result": request.app.state.chaosengine.delete_download(repo)}


@router.post("/api/voice/extras/remove")
def remove_extra(request: Request, body: ExtraRequest) -> dict[str, Any]:
    """Untick an add-on: uninstall its packages. Ticking one installs it through ``/api/voice/install``."""
    from backend_service import voice_runtime_disk  # noqa: PLC0415

    _window_only(request)
    try:
        return voice_runtime_disk.remove_extra(body.extra)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/voice/uninstall")
def remove_voice_runtime(request: Request, body: UninstallRequest) -> dict[str, Any]:
    """Remove the voice runtime. Downloaded dictation and speech models stay unless the runtime's own are asked for."""
    from backend_service import voice_runtime_disk  # noqa: PLC0415

    _window_only(request)
    try:
        return voice_runtime_disk.remove_runtime(body.alsoRuntimeModels)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
