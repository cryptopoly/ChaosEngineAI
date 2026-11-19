"""FLUX.3 Image Klein through diffusers' modular pipeline (CUDA lane).

BFL ships FLUX.3 for diffusers as a modular pipeline only (``modular_model_index.json``,
``Flux3DistilledModularPipeline`` / ``Flux3ModularPipeline``, in diffusers 0.42), so
``AutoPipelineForText2Image`` cannot open it. This module builds that pipeline from the local
snapshot without its first block, the 9B prompt upsampler (``flux-3-image-klein-pup``, 18.8 GB
more): the user's prompt goes to the image model as typed. The components are loaded from the
snapshot (never from the Hub ids the index names). On a card that cannot hold every part with
room to spare, the text encoder visits the card only to read the prompt (``place_on_cuda``). Progress and cancel ride on a forward hook on the transformer
(one call per step for the distilled checkpoint, two for the guided Base).
"""

from __future__ import annotations

import time
from typing import Any

# Repo -> True for the distilled checkpoint (one transformer pass a step, no guidance). The undistilled Base
# (guided, two passes a step) has no CUDA row yet: it was not run here.
FLUX3_MODULAR_REPOS: dict[str, bool] = {
    "black-forest-labs/flux-3-image-klein": True,
}
# The model takes up to ten reference images; the app's request schema (and the mflux edit lane) stop at four.
MAX_REFERENCE_IMAGES = 4
# Components the modular index lists that this app does not load (the prompt upsampler lives in another repo).
UNUSED_COMPONENTS = frozenset({"prompt_upsampler", "prompt_upsampler_processor"})


# Free GPU memory the transformer and VAE need beside their weights. Measured on an RTX 4090 at 1024 px, bf16: with
# the 8.9 GB text encoder left on the card the VAE decode spilled into shared memory and took 6 s (0.3 s otherwise)
# and a step took 0.53 s instead of 0.39 s.
ACTIVATION_HEADROOM_GB = 8.0


def _component_gb(module: Any) -> float:
    try:
        return float(module.get_memory_footprint()) / 1024**3
    except Exception:  # noqa: BLE001 - a module without the helper counts as small
        return 0.0


def place_on_cuda(pipeline: Any, torch: Any) -> str | None:
    """Put the pipeline on the NVIDIA card; returns a note when the text encoder only visits it.

    A card with room for every part plus ``ACTIVATION_HEADROOM_GB`` keeps everything. Otherwise the transformer and
    VAE stay resident and the text encoder sits in system RAM, moves to the card when it reads the prompt and goes
    back before the first transformer step (accelerate's offload hook, released from a transformer pre-hook).
    """
    parts = [getattr(pipeline, name, None) for name in ("text_encoder", "transformer", "vae")]
    need_gb = sum(_component_gb(part) for part in parts if part is not None) + ACTIVATION_HEADROOM_GB
    total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
    encoder, transformer = parts[0], parts[1]
    if total_gb >= need_gb or encoder is None or transformer is None:
        pipeline.to("cuda")
        return None
    from accelerate import cpu_offload_with_hook  # type: ignore

    for name in ("transformer", "vae"):
        part = getattr(pipeline, name, None)
        if part is not None:
            part.to("cuda")
    _, hook = cpu_offload_with_hook(encoder, execution_device=torch.device("cuda"))
    def _encoder_off(_module: Any, _args: Any) -> None:
        if next(encoder.parameters()).device.type != "cpu":
            hook.offload()
            # Hand the encoder's blocks back, or the cache plus the step's activations spill past the card.
            torch.cuda.empty_cache()

    transformer.register_forward_pre_hook(_encoder_off)
    return (
        f"Text encoder kept in system RAM between prompts ({total_gb:.0f} GB card, about {need_gb:.0f} GB to hold "
        "everything)."
    )


def is_flux3_modular_repo(repo: str | None) -> bool:
    return bool(repo) and repo in FLUX3_MODULAR_REPOS


def load_pipeline(local_path: str, torch: Any, device: str) -> tuple[Any, list[str]]:
    """The FLUX.3 modular pipeline from ``local_path`` on ``device``, and the notes to show with its images."""
    from diffusers import ModularPipeline  # type: ignore

    blocks = ModularPipeline.from_pretrained(local_path, local_files_only=True).blocks
    blocks.sub_blocks.pop("prompt_upsampler", None)
    pipeline = blocks.init_pipeline(local_path)
    names = [name for name in pipeline.pretrained_component_names if name not in UNUSED_COMPONENTS]
    pipeline.load_components(
        names=names,
        pretrained_model_name_or_path=local_path,
        local_files_only=True,
        dtype={"vae": torch.float32, "default": torch.bfloat16},
    )
    notes = ["Prompt sent as typed (the FLUX.3 prompt upsampler is not run: write a full scene description)."]
    vae = getattr(pipeline, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        # The fp32 decode at 1536 px spilled out of a 24 GB card (9.4 s); tiling only starts above the VAE's size.
        vae.enable_tiling()
    if device == "cuda":
        placement_note = place_on_cuda(pipeline, torch)
        if placement_note:
            notes.append(placement_note)
    elif device != "cpu":
        pipeline.to(device)
    return pipeline, notes


def load_references(paths: list[str] | None) -> list[Any]:
    """The reference pictures as RGB PIL images, at most ``MAX_REFERENCE_IMAGES``."""
    from PIL import Image  # type: ignore

    pictures = []
    for path in list(paths or [])[:MAX_REFERENCE_IMAGES]:
        with Image.open(path) as picture:
            pictures.append(picture.convert("RGB"))
    return pictures


def generate(
    pipeline: Any,
    *,
    repo: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    generator: Any,
    batch_size: int,
    reference_paths: list[str] | None,
    on_step: Any,
) -> tuple[list[Any], float]:
    """``(PIL images, seconds)`` for one request. ``on_step(done, total)`` runs before each transformer pass."""
    passes_per_step = 1 if FLUX3_MODULAR_REPOS.get(repo, True) else 2
    total_steps = max(1, int(steps))
    calls = {"n": 0}

    def _pre_forward(_module: Any, _args: Any) -> None:
        step = min(total_steps, calls["n"] // passes_per_step)
        calls["n"] += 1
        on_step(step, total_steps)

    transformer = getattr(pipeline, "transformer", None)
    handle = transformer.register_forward_pre_hook(_pre_forward) if transformer is not None else None
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "width": int(width),
        "height": int(height),
        "num_inference_steps": total_steps,
        "num_images_per_prompt": max(1, int(batch_size)),
        "generator": generator,
    }
    references = load_references(reference_paths)
    if references:
        kwargs["image"] = references if len(references) > 1 else references[0]
    started = time.perf_counter()
    try:
        images = pipeline(output="images", **kwargs)
    finally:
        if handle is not None:
            handle.remove()
    elapsed = max(0.1, time.perf_counter() - started)
    if not isinstance(images, (list, tuple)):
        images = [images]
    return list(images), elapsed
