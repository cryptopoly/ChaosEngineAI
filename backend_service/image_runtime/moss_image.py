"""MOSS-Image SFT (OpenMOSS, Apache-2.0) through a vendored copy of its pipeline (CUDA lane).

The repo's ``model_index.json`` names classes from its own ``pipeline_moss_image.py``, which diffusers would only load
with ``trust_remote_code``. The app never does that: ``moss_image_pipeline.py`` is that file copied at a pinned commit
and reviewed, and this module builds the pipeline from the snapshot's folders with those classes (the text encoder is
a stock ``Qwen3_5Model`` in transformers; the VAE is diffusers' ``AutoencoderKLFlux2``). Defaults follow the model
card: 50 steps, guidance 3, the negative branch skips block 22. The 9B text encoder runs once per image, so it moves
to the card for the prompt only (model CPU offload) on cards that cannot hold every part.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

MOSS_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
MOSS_REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
# Room the transformer needs beside every part on the card before CPU offload is skipped.
ACTIVATION_HEADROOM_GB = 8.0


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
    text_encoder = AutoModel.from_pretrained(root / "text_encoder", dtype=torch.bfloat16, local_files_only=True).eval()
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.float32, local_files_only=True).eval()
    scheduler = MOSSImageScheduler.from_config(MOSSImageScheduler.load_config(root / "scheduler"))
    pipeline = MOSSImagePipeline(transformer.eval(), scheduler, text_encoder, tokenizer, vae)
    pipeline.set_progress_bar_config(disable=True)
    notes: list[str] = []
    if device == "cuda":
        parts_gb = sum(
            float(part.get_memory_footprint()) / 1024**3 for part in (transformer, text_encoder, vae)
        )
        total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        if total_gb >= parts_gb + ACTIVATION_HEADROOM_GB:
            pipeline.to("cuda")
        else:
            pipeline.enable_model_cpu_offload()
            notes.append(f"Parts moved to the card in turn ({total_gb:.0f} GB card, about {parts_gb:.0f} GB of weights).")
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
