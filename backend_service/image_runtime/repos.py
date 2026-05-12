"""Repo-name → family classification helpers + sampler registry.

Pure string predicates that decide:

- ``_is_sdxl_repo`` — match SDXL family for fp16-VAE swap
- ``_is_flux_repo`` — FLUX.1 family detection (bf16 + cpu-offload tuning)
- ``_is_flow_matching_repo`` — flow-matching pipelines (FLUX/SD3/
  Qwen-Image/Sana/HiDream) where scheduler swap silently produces noise
- ``_gguf_transformer_class_for_repo`` — pick the right diffusers
  transformer subclass for ``from_single_file`` GGUF loads
- ``_nunchaku_transformer_class_for_repo`` — same shape for SVDQuant
  Nunchaku checkpoints (FU-023)
- ``_locate_sdxl_vae_fix_snapshot`` — find a cached
  ``madebyollin/sdxl-vae-fp16-fix`` snapshot or return None

Plus the sampler registry + ``_apply_scheduler``:

- ``_SAMPLER_REGISTRY`` — UI-facing sampler id → diffusers class +
  optional kwargs. Includes AYS (Align Your Steps) entries that route
  through the private ``_ays_family`` marker for FU-020.
- ``_AYS_TIMESTEPS`` — NVIDIA-published 10-step timestep arrays for
  SD1.5, SDXL, SVD.
- ``_apply_scheduler`` — swap ``pipeline.scheduler`` and stash AYS
  timesteps for the build-kwargs path to pick up.

Extracted from ``image_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib.util
from typing import Any


_SDXL_VAE_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"


def _is_sdxl_repo(repo: str) -> bool:
    """Match SDXL family repos (Stability XL base, refiner, community fine-tunes).

    Matches loosely on substring — a false positive would attempt the
    VAE swap on a non-SDXL repo, but the fp16-fix VAE only loads
    successfully against an SDXL pipeline because the encoder/decoder
    shape has to match. ``AutoencoderKL.from_pretrained`` raises on
    mismatch and the swap silently no-ops, so an over-broad match is
    self-correcting.
    """
    lower = repo.lower()
    return "stable-diffusion-xl" in lower or "sdxl" in lower or "sd_xl" in lower


def _locate_sdxl_vae_fix_snapshot() -> str | None:
    """Return the local path to ``madebyollin/sdxl-vae-fp16-fix`` if cached.

    Uses ``snapshot_download(local_files_only=True)`` so a missing snapshot
    returns ``None`` rather than triggering a download mid-generate. Users
    who want the fp16-fix path opt in by downloading the repo from the
    Setup page (or via ``huggingface-cli download``); until then the
    runtime stays on the existing fp32-on-MPS fallback for SDXL.
    """
    if importlib.util.find_spec("huggingface_hub") is None:
        return None
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        return None
    try:
        return snapshot_download(
            repo_id=_SDXL_VAE_FIX_REPO,
            local_files_only=True,
            resume_download=True,
        )
    except Exception:
        return None


def _is_flux_repo(repo: str) -> bool:
    """Does this HF repo look like a FLUX.1 family model?

    FLUX family checkpoints are published under the
    ``black-forest-labs/FLUX.1-*`` namespace (Dev, Schnell, Kontext, etc.)
    plus a long tail of community fine-tunes that keep "flux" in their
    repo name. We match loosely by lowercased substring — the
    consequence of a false positive (using bf16 + cpu-offload on a non-
    FLUX model) is "slower than optimal on this machine", not incorrect
    output, so erring wide is fine.
    """
    lowered = repo.lower()
    return "flux" in lowered


def _is_flow_matching_repo(repo: str) -> bool:
    """Flow-matching pipelines (FLUX, SD3, Qwen-Image) ship locked
    schedulers — swapping to DDIM/Euler/DPM++ silently produces noise
    because the model was trained against a flow-matching ODE, not
    epsilon/v-prediction. Gate the sampler dropdown on this so the UI
    only shows it for SD1.5 / SDXL / SD2 where scheduler swap is safe.
    """
    lowered = repo.lower()
    return (
        _is_flux_repo(repo)
        or "stable-diffusion-3" in lowered
        or "sd3" in lowered
        or "qwen-image" in lowered
        or "sana" in lowered
        or "hidream" in lowered
    )


def _gguf_transformer_class_for_repo(repo: str) -> str | None:
    """Map a base repo to the diffusers transformer class used for GGUF.

    GGUF ``.from_single_file`` needs the right class — FLUX and SD3 both
    ship their own MMDiT/FluxTransformer variants, and loading a FLUX GGUF
    into ``SD3Transformer2DModel`` produces garbage. Returns ``None`` for
    families we don't ship GGUF variants for (SD1.5/SDXL use UNets, which
    have a different loading path that we don't support yet).
    """
    lowered = repo.lower()
    if _is_flux_repo(repo):
        return "FluxTransformer2DModel"
    if "stable-diffusion-3" in lowered or "sd3" in lowered:
        return "SD3Transformer2DModel"
    if "hidream" in lowered:
        return "HiDreamImageTransformer2DModel"
    return None


def _nunchaku_transformer_class_for_repo(repo: str) -> str | None:
    """FU-023: map a base repo to the Nunchaku transformer subclass.

    Nunchaku exports per-architecture wrappers for SVDQuant 4-bit weights:
        FLUX family       -> NunchakuFluxTransformer2dModel
        Qwen-Image family -> NunchakuQwenImageTransformer2DModel
        SD3 / SD3.5       -> NunchakuSD3Transformer2DModel
        SANA              -> NunchakuSanaTransformer2DModel
        PixArt-Σ          -> NunchakuPixArtSigmaTransformer2DModel

    Returns ``None`` for families Nunchaku hasn't shipped yet (Wan,
    HunyuanVideo, LTX, Z-Image, ERNIE-Image) so the caller falls back
    cleanly. v1.2.1 (2026-01-25) is the pin we ship; new families land
    here when nunchaku adds matching subclasses.
    """
    lowered = repo.lower()
    if _is_flux_repo(repo):
        return "NunchakuFluxTransformer2dModel"
    if "qwen-image" in lowered or "qwen/qwen-image" in lowered:
        return "NunchakuQwenImageTransformer2DModel"
    if "stable-diffusion-3" in lowered or "sd3" in lowered:
        return "NunchakuSD3Transformer2DModel"
    if "sana" in lowered:
        return "NunchakuSanaTransformer2DModel"
    if "pixart-sigma" in lowered:
        return "NunchakuPixArtSigmaTransformer2DModel"
    return None


# FU-020: Align Your Steps (AYS) — NVIDIA's hand-optimised 10-step
# timestep schedules for SD1.5, SDXL and SVD. At 7-10 steps the AYS
# arrays preserve substantially more detail than DPM++ 2M Karras —
# the user study cited in the paper shows a 2× preference at low step
# counts. Numbers are the *timesteps* (not sigmas) the scheduler
# should sample at, not the count itself; passing them via
# ``pipeline(timesteps=...)`` overrides the standard
# ``num_inference_steps`` path.
#
# Reference: NVIDIA AYS project page,
# https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/
_AYS_TIMESTEPS: dict[str, list[int]] = {
    "sd15": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24],
    "sdxl": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13],
    # SVD reserved for the video runtime; not exposed in the image
    # sampler dropdown today but registered here so the same
    # ``_ays_family`` token works if/when we surface it on a video
    # path.
    "svd":  [999, 963, 911, 833, 720, 562, 387, 219, 90, 8],
}


# Maps a stable UI-facing sampler id to (diffusers scheduler class name,
# optional from_config kwargs). The class is imported lazily from
# ``diffusers`` so the runtime doesn't pay the import cost unless a user
# actually picks a non-default sampler. Kwargs let us configure the
# Karras/SDE variants without adding separate classes.
#
# The ``_ays_family`` key is a private marker consumed by
# ``_apply_scheduler`` — when present it pops out of the kwargs (so it
# never reaches diffusers' ``from_config``) and stashes the matching
# AYS timestep array on the pipeline for ``_build_pipeline_kwargs`` to
# pass via the ``timesteps=`` arg.
_SAMPLER_REGISTRY: dict[str, tuple[str, dict[str, Any]]] = {
    "dpmpp_2m": ("DPMSolverMultistepScheduler", {}),
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpmpp_sde": ("DPMSolverSinglestepScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "ays_dpmpp_2m_sd15": ("DPMSolverMultistepScheduler", {"_ays_family": "sd15"}),
    "ays_dpmpp_2m_sdxl": ("DPMSolverMultistepScheduler", {"_ays_family": "sdxl"}),
}


def _apply_scheduler(pipeline: Any, sampler_id: str | None) -> str | None:
    """Swap ``pipeline.scheduler`` to the sampler chosen by the user.

    Returns a short human-readable note on what was applied (or why
    nothing was), to surface in ``GeneratedImage.runtimeNote``. Silent
    failure modes (missing scheduler class on old diffusers, pipeline
    with no ``scheduler`` attribute) fall back to the model default.

    FU-020: when the registry entry includes the ``_ays_family`` private
    marker, the matching AYS timestep array is stashed on
    ``pipeline._chaosengine_ays_timesteps`` so
    ``_build_pipeline_kwargs`` can pass it via the ``timesteps=`` arg
    instead of the usual ``num_inference_steps``.
    """
    if not sampler_id:
        return None
    entry = _SAMPLER_REGISTRY.get(sampler_id)
    if entry is None:
        return f"Unknown sampler '{sampler_id}' — using model default."
    if not hasattr(pipeline, "scheduler") or pipeline.scheduler is None:
        return None
    class_name, registry_kwargs = entry
    try:
        import diffusers  # type: ignore
    except Exception:
        return None
    scheduler_cls = getattr(diffusers, class_name, None)
    if scheduler_cls is None:
        return f"Sampler '{sampler_id}' not available in installed diffusers."
    # Pop private markers (e.g. ``_ays_family``) before passing to
    # ``from_config`` — diffusers rejects unknown kwargs.
    extra_kwargs = dict(registry_kwargs)
    ays_family = extra_kwargs.pop("_ays_family", None)
    try:
        pipeline.scheduler = scheduler_cls.from_config(
            pipeline.scheduler.config, **extra_kwargs,
        )
    except Exception as exc:
        return f"Sampler swap to '{sampler_id}' failed: {type(exc).__name__}. Using model default."
    if ays_family:
        timesteps = _AYS_TIMESTEPS.get(ays_family)
        if timesteps:
            try:
                pipeline._chaosengine_ays_timesteps = list(timesteps)  # type: ignore[attr-defined]
            except Exception:
                # Pipeline objects are usually attribute-friendly, but
                # if a future diffusers version locks slots we swallow
                # and keep the swap-only behaviour rather than failing
                # the run.
                pass
        return f"Sampler: {sampler_id} ({len(timesteps or [])}-step AYS)"
    # Clear any stale stash from a previous AYS-using generate so a
    # later non-AYS run doesn't reuse the timestep array.
    if hasattr(pipeline, "_chaosengine_ays_timesteps"):
        try:
            delattr(pipeline, "_chaosengine_ays_timesteps")
        except Exception:
            pass
    return f"Sampler: {sampler_id}"
