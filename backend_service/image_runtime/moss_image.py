"""MOSS-Image SFT (OpenMOSS, Apache-2.0) through a vendored copy of its pipeline (CUDA lane).

The repo's ``model_index.json`` names classes from its own ``pipeline_moss_image.py``, which diffusers would only load
with ``trust_remote_code``. The app never does that: ``moss_image_pipeline.py`` is that file copied at a pinned commit
and reviewed, and this module builds the pipeline from the snapshot's folders with those classes (the text encoder is
a stock ``Qwen3_5Model`` in transformers; the VAE is diffusers' ``AutoencoderKLFlux2``). Defaults follow the model
card: 50 steps, guidance 3, the negative branch skips block 22. On a card below ``FULL_BF16_GB`` the 9B text encoder
loads in 4-bit so the whole pipeline stays on the card.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

MOSS_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
# Card size that holds the bf16 pipeline (16.8 GB encoder, 10.2 GB transformer, 0.25 GB VAE) plus activations.
FULL_BF16_GB = 36.0


def _cuda_total_gb(torch: Any) -> float:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3


def is_moss_repo(repo: str | None) -> bool:
    return repo == MOSS_REPO


def load_pipeline(local_path: str, torch: Any, device: str) -> tuple[Any, list[str]]:
    """The MOSS-Image pipeline built from ``local_path`` (no repo code is imported), and its notes."""
    from transformers import AutoModel, AutoTokenizer  # type: ignore
    from diffusers import AutoencoderKLFlux2  # type: ignore

    from backend_service.image_runtime.moss_image_pipeline import (
        MOSSImagePipeline,
        MOSSImageScheduler,
        MOSSImageTransformer,
    )

    root = Path(local_path)
    transformer = MOSSImageTransformer.from_pretrained(root / "transformer", torch_dtype=torch.bfloat16, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    notes: list[str] = []
    encoder_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "local_files_only": True}
    if device == "cuda" and _cuda_total_gb(torch) < FULL_BF16_GB:
        # The bf16 encoder (16.8 GB) and transformer (10.2 GB) do not fit a smaller card together, and moving the
        # encoder off the card after the prompt crashed the process on the 4090 (an access violation in .to("cpu"),
        # host RAM under pressure). In 4-bit the whole pipeline stays on the card.
        from transformers import BitsAndBytesConfig  # type: ignore

        encoder_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        encoder_kwargs["device_map"] = {"": 0}
        notes.append("Text encoder in 4-bit (NF4) so the whole pipeline fits the card.")
    text_encoder = AutoModel.from_pretrained(root / "text_encoder", **encoder_kwargs).eval()
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.float32, local_files_only=True).eval()
    scheduler = MOSSImageScheduler.from_config(MOSSImageScheduler.load_config(root / "scheduler"))
    pipeline = MOSSImagePipeline(transformer.eval(), scheduler, text_encoder, tokenizer, vae)
    pipeline.set_progress_bar_config(disable=True)
    if device == "cuda":
        for part in (transformer, vae):
            part.to("cuda")
        if "quantization_config" not in encoder_kwargs:
            text_encoder.to("cuda")
    elif device != "cpu":
        pipeline.to(device)
    return pipeline, notes


def generate(
    pipeline: Any,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    generator: Any,
    batch_size: int,
    on_step: Any,
) -> tuple[list[Any], float]:
    """``(PIL images, seconds)``. ``on_step(done, total)`` runs before each transformer pass (two a step with guidance)."""
    total_steps = max(1, int(steps))
    passes_per_step = 1 if float(guidance) == 1.0 else 2
    calls = {"n": 0}

    def _pre_forward(_module: Any, _args: Any) -> None:
        on_step(min(total_steps, calls["n"] // passes_per_step), total_steps)
        calls["n"] += 1

    handle = pipeline.transformer.register_forward_pre_hook(_pre_forward)
    started = time.perf_counter()
    try:
        result = pipeline(
            prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=total_steps,
            guidance_scale=float(guidance),
            negative_prompt=negative_prompt or "",
            num_images_per_prompt=max(1, int(batch_size)),
            generator=generator,
        )
    finally:
        handle.remove()
    return list(result.images), max(0.1, time.perf_counter() - started)
