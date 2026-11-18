"""Fine-tuning and LoRA adapter routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend_service.i18n import localized_detail
from backend_service.helpers.finetuning import (
    list_adapters,
    prepare_dataset,
)

router = APIRouter(prefix="/api", tags=["finetuning"])

# ---------------------------------------------------------------------------
# In-memory training state (placeholder until real training is wired up)
# ---------------------------------------------------------------------------
_training_state: dict[str, Any] = {
    "status": "idle",  # idle | preparing | training | complete | error
    "progress": 0.0,
    "config": None,
    "error": None,
    "run_id": None,
}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartFineTuneRequest(BaseModel):
    modelPath: str = Field(min_length=1)
    datasetPath: str = Field(min_length=1)
    outputPath: str | None = None
    learningRate: float = Field(default=1e-5, gt=0, le=1.0)
    epochs: int = Field(default=1, ge=1, le=100)
    loraRank: int = Field(default=8, ge=1, le=256)
    batchSize: int = Field(default=4, ge=1, le=64)
    loraAlpha: float = Field(default=16.0, ge=1.0, le=256.0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/adapters")
def get_adapters(request: Request) -> dict[str, Any]:
    """List available LoRA adapters found on disk.

    A plain ``def`` on purpose: the scan walks the whole model cache (4 s on a 270 GB cache), and FastAPI runs
    a plain function in a worker thread, so the walk no longer stalls every other request.
    """
    state = request.app.state.chaosengine
    data_dir = Path(state.settings.get("dataDirectory", "")).expanduser()
    if not data_dir.is_dir():
        # Fall back to home Models dir
        data_dir = Path.home() / "Models"

    adapters = list_adapters(data_dir)
    return {"adapters": adapters, "count": len(adapters)}


@router.post("/finetuning/start")
async def start_finetuning(
    body: StartFineTuneRequest,
    request: Request,
) -> dict[str, Any]:
    """Refuse a fine-tuning run: no training backend is wired in yet.

    The request is still validated (422 on a bad body). It used to answer 200 "accepted" for a job that
    never ran, for any path, and left the status at "accepted" for good. Now it says so: 501, nothing
    started, status unchanged. When training is built, launch the subprocess here.
    """
    raise HTTPException(
        status_code=501,
        detail=localized_detail(
            request,
            "Fine-tuning is not available in this version: no training backend is wired in. Nothing was started.",
        ),
    )


@router.get("/finetuning/status")
async def get_finetuning_status() -> dict[str, Any]:
    """Return current training status."""
    return {
        "status": _training_state["status"],
        "progress": _training_state["progress"],
        "run_id": _training_state["run_id"],
        "error": _training_state["error"],
        "config": _training_state["config"],
    }
