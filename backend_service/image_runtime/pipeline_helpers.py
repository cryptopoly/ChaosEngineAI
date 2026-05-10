"""Stateless pipeline helpers for ``DiffusersTextToImageEngine``.

Three helpers lifted out of ``image_runtime/__init__.py``:

* ``diffuse_message`` — short status line for the per-step VIDEO
  progress publisher ("Diffusing image" / "Diffusing 4 images").
* ``format_run_label`` — single-line label combining model name +
  resolution for the gallery card.
* ``build_pipeline_kwargs`` — dict of kwargs the diffusers pipeline
  accepts; per-pipeline shaping (FU-020 AYS timesteps stash, Qwen-Image
  ``true_cfg_scale``).

Extracted from ``backend_service/image_runtime/__init__.py`` as part
of the v0.8.0 Phase 1c-16 refactor.
"""

from __future__ import annotations

from typing import Any

from backend_service.image_runtime.types import ImageGenerationConfig


def diffuse_message(config: ImageGenerationConfig) -> str:
    if config.batchSize > 1:
        return f"Diffusing {config.batchSize} images"
    return "Diffusing image"


def format_run_label(config: ImageGenerationConfig) -> str:
    return f"{config.modelName} · {config.width}x{config.height}"


def build_pipeline_kwargs(
    config: ImageGenerationConfig,
    generator: Any,
    pipeline: Any,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": config.prompt,
        "width": config.width,
        "height": config.height,
        "num_inference_steps": config.steps,
        "guidance_scale": config.guidance,
        "num_images_per_prompt": config.batchSize,
        "generator": generator,
    }
    # FU-020: when the user picked an AYS sampler,
    # ``_apply_scheduler`` stashed the precomputed timestep array on
    # the pipeline. Diffusers accepts ``timesteps=`` as an explicit
    # override; when present it takes precedence over
    # ``num_inference_steps`` so we drop the latter to avoid the
    # "got both" warning.
    if pipeline is not None:
        ays_timesteps = getattr(pipeline, "_chaosengine_ays_timesteps", None)
        if ays_timesteps:
            kwargs["timesteps"] = list(ays_timesteps)
            kwargs.pop("num_inference_steps", None)
    lowered_repo = config.repo.lower()
    if "qwen-image" in lowered_repo:
        kwargs.pop("guidance_scale", None)
        kwargs["true_cfg_scale"] = config.guidance
        # Qwen-Image expects a negative prompt value, even if it is intentionally blank.
        kwargs["negative_prompt"] = config.negativePrompt if config.negativePrompt else " "
        return kwargs
    if config.negativePrompt.strip():
        kwargs["negative_prompt"] = config.negativePrompt
    return kwargs
