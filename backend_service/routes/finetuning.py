"""Fine-tuning and LoRA adapter routes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend_service.helpers.finetuning import (
    list_adapters,
    prepare_dataset,
    FineTuneConfig,
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
async def get_adapters(request: Request) -> dict[str, Any]:
    """List available LoRA adapters found on disk."""
    state = request.app.state.engine
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
    """Start a fine-tuning run (placeholder).

    In a full implementation this would launch an mlx-lm or similar
    training subprocess.  For now it validates the config and returns
    an accepted status.
    """
    global _training_state

    if _training_state["status"] == "training":
        raise HTTPException(
            status_code=409,
            detail="A training run is already in progress.",
        )

    import uuid

    output_path = body.outputPath or str(
        Path(body.modelPath).parent / f"lora-{uuid.uuid4().hex[:8]}"
    )

    config = FineTuneConfig(
        model_path=body.modelPath,
        dataset_path=body.datasetPath,
        output_path=output_path,
        learning_rate=body.learningRate,
        epochs=body.epochs,
        lora_rank=body.loraRank,
        batch_size=body.batchSize,
        lora_alpha=body.loraAlpha,
    )

    _training_state = {
        "status": "preparing",
        "progress": 0.0,
        "config": config.to_dict(),
        "error": None,
        "run_id": uuid.uuid4().hex,
    }

    # TODO: Actually launch training in a background thread/process
    _training_state["status"] = "accepted"

    return {
        "status": _training_state["status"],
        "run_id": _training_state["run_id"],
        "config": _training_state["config"],
        "message": "Fine-tuning request accepted. Training backend not yet wired.",
    }


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
