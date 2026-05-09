"""Memory + default-resolution helpers for the video runtime.

Everything pipeline-adjacent that isn't a per-repo lookup or a torch /
diffusers import lives here:

- ``_VIDEO_MODEL_FOOTPRINT_BF16_GB`` + ``_GGUF_QUANT_MULTIPLIERS`` +
  ``_estimate_model_footprint_gb`` (cheap pre-load size estimate)
- ``_should_apply_memory_savers`` (slicing / tiling gate)
- ``_SCHEDULER_CLASSES`` + ``_align_wan_num_frames`` +
  ``_resolve_video_defaults`` (per-model parameter substitution)
- ``_interpolate_frames`` (linear-blend frame interpolator — placeholder
  until RIFE weights ship)
- ``_CORE_DEPS`` / ``_VIDEO_OUTPUT_DEPS`` / ``_VIDEO_MODEL_DEPS`` +
  ``_find_missing`` (probe dependency tuples)

Extracted from ``video_runtime/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

from backend_service.video_runtime.repos import (
    _REQUEST_DEFAULT_GUIDANCE,
    _REQUEST_DEFAULT_STEPS,
    _VIDEO_PIPELINE_DEFAULTS,
)


# Rough bf16 footprint for the full pipeline (transformer + VAE + text
# encoders) keyed on repo. Used by the memory-saver gate — slicing and
# tiling cut quality, so we only enable them when there's actual memory
# pressure. Numbers come from the catalog ``sizeGb`` estimates for the
# stock variants; GGUF Q4/Q6/Q8 variants override at the call site.
_VIDEO_MODEL_FOOTPRINT_BF16_GB: dict[str, float] = {
    "Lightricks/LTX-Video": 10.0,
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": 9.0,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": 28.0,
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": 11.0,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": 28.0,
    "hunyuanvideo-community/HunyuanVideo": 26.0,
    "genmo/mochi-1-preview": 22.0,
    "THUDM/CogVideoX-2b": 19.0,
    "THUDM/CogVideoX-5b": 33.0,
}

# GGUF quant level → multiplier vs the bf16 footprint. Keys are matched as
# substrings in the gguf filename so future quant levels (e.g. ``Q5_K_M``)
# fall through to a sensible default.
_GGUF_QUANT_MULTIPLIERS: tuple[tuple[str, float], ...] = (
    ("Q4", 0.30),
    ("Q5", 0.36),
    ("Q6", 0.42),
    ("Q8", 0.55),
)


def _estimate_model_footprint_gb(
    repo: str, dtype_name: str, gguf_file: str | None = None
) -> float | None:
    """Cheap estimate of a video pipeline's GPU/MPS memory footprint in GB.

    Returns ``None`` if the repo is unrecognised — callers treat that as
    "stay safe" and enable slicing. The dtype name is the str of the
    torch dtype (``"torch.bfloat16"`` etc.); we treat fp16/bf16 as the
    catalog baseline and double for fp32.
    """
    base = _VIDEO_MODEL_FOOTPRINT_BF16_GB.get(repo)
    if base is None:
        return None
    if gguf_file:
        upper = gguf_file.upper()
        for marker, multiplier in _GGUF_QUANT_MULTIPLIERS:
            if marker in upper:
                base = base * multiplier
                break
    if "float32" in dtype_name and "bfloat16" not in dtype_name:
        base = base * 2.0
    return base


def _should_apply_memory_savers(
    device: str, total_memory_gb: float | None, estimated_footprint_gb: float | None
) -> bool:
    """Decide whether to enable attention slicing + VAE slicing/tiling.

    Slicing trades quality for VRAM. The reference workflows don't enable it —
    we used to do it unconditionally, which left a 64 GB Mac running a
    1.3B model in slicing mode for no reason. Heuristic:

    - ``CHAOSENGINE_VIDEO_FORCE_SLICING=1`` always wins (rollback lever).
    - Unknown memory or unknown footprint → stay safe, enable slicing.
    - CPU device → enable, system RAM is shared.
    - Footprint > 70% of device memory → enable.
    - Otherwise → leave the pipeline at full quality.
    """
    if os.getenv("CHAOSENGINE_VIDEO_FORCE_SLICING") == "1":
        return True
    if device == "cpu":
        return True
    if total_memory_gb is None or estimated_footprint_gb is None:
        return True
    if total_memory_gb <= 0:
        return True
    return (estimated_footprint_gb / total_memory_gb) > 0.7


# Diffusers scheduler classes we expose via the ``scheduler`` request field.
# Resolved on the ``diffusers`` module at runtime so an old install that
# lacks one of these classes degrades to a logged warning instead of an
# import-time crash.
_SCHEDULER_CLASSES: dict[str, str] = {
    "unipc": "UniPCMultistepScheduler",
    "euler": "EulerDiscreteScheduler",
    # ``FlowMatchEulerDiscreteScheduler`` is the only scheduler that
    # accepts the ``mu`` kwarg LTXPipeline passes to ``set_timesteps``.
    # Older cached LTX snapshots have plain ``EulerDiscreteScheduler``
    # baked in; we force-swap on LTX to keep the pipeline call valid.
    "flow-euler": "FlowMatchEulerDiscreteScheduler",
    "dpm++": "DPMSolverMultistepScheduler",
    "ddim": "DDIMScheduler",
}


def _align_wan_num_frames(repo: str, requested: int) -> tuple[int, str | None]:
    """Round Wan's ``num_frames`` to the nearest valid ``(4k + 1)`` value.

    Wan models compute ``(n_frames - 1) / 4 + 1`` latent frames internally;
    off-spec counts produce mostly-black/garbled output. We round down to
    the nearest valid count rather than up so we don't silently exceed the
    user's requested clip length and frame budget.

    Returns ``(aligned_count, note_or_None)``. The note is surface-ready
    text — if non-None the caller should publish it to ``VIDEO_PROGRESS``
    and the run log so the UI explains why the count changed.
    """
    if "Wan" not in repo:
        return requested, None
    if requested < 5:
        return 5, "Wan requires num_frames >= 5; clamped."
    aligned = ((requested - 1) // 4) * 4 + 1
    if aligned != requested:
        return aligned, (
            f"Aligned num_frames {requested} → {aligned} (Wan requires 4k+1)."
        )
    return aligned, None


def _resolve_video_defaults(
    repo: str, requested_steps: int, requested_guidance: float
) -> dict[str, Any]:
    """Substitute per-model sweet-spot values when the user kept schema defaults.

    Heuristic: if the request matches the schema defaults exactly (50 steps,
    CFG 3.0) we treat it as "user did not dial this in" and substitute the
    upstream-recommended values. Any explicit deviation is preserved.

    Returns the resolved dict with ``steps``, ``guidance``, ``scheduler``,
    and ``substituted`` (True when at least one value was rewritten).
    """
    overrides = _VIDEO_PIPELINE_DEFAULTS.get(repo, {})
    resolved_steps = requested_steps
    resolved_guidance = requested_guidance
    substituted = False
    if requested_steps == _REQUEST_DEFAULT_STEPS and "steps" in overrides:
        resolved_steps = int(overrides["steps"])
        substituted = substituted or resolved_steps != requested_steps
    if requested_guidance == _REQUEST_DEFAULT_GUIDANCE and "guidance" in overrides:
        resolved_guidance = float(overrides["guidance"])
        substituted = substituted or resolved_guidance != requested_guidance
    return {
        "steps": resolved_steps,
        "guidance": resolved_guidance,
        "scheduler": overrides.get("scheduler"),
        "substituted": substituted,
    }


def _interpolate_frames(frames: list[Any], factor: int) -> list[Any]:
    """Insert ``factor - 1`` blended frames between each source pair.

    This is a linear-blend (numpy-weighted average) frame interpolator —
    simpler and faster than RIFE but gives visibly smoother motion at
    2x/4x. Swap this for a RIFE model call when the weights ship — the
    pipeline shape (``list[np.ndarray]`` in RGB uint8) stays the same.

    A factor of 1 is a no-op. Factors above 1 produce
    ``(len - 1) * factor + 1`` frames so the endpoint timings align
    with the original clip.
    """
    if factor <= 1 or len(frames) < 2:
        return list(frames)
    try:
        import numpy as np  # type: ignore
    except Exception:
        return list(frames)

    def _to_array(frame: Any):
        if hasattr(frame, "shape"):
            return np.asarray(frame)
        return np.asarray(frame, dtype=np.uint8)

    interpolated: list[Any] = []
    total = len(frames)
    for index in range(total - 1):
        current = _to_array(frames[index])
        nxt = _to_array(frames[index + 1])
        if current.shape != nxt.shape:
            # Different shape → skip blending, just duplicate. Robust
            # against frames of mixed dtypes (list of PIL Images).
            interpolated.append(frames[index])
            for _ in range(factor - 1):
                interpolated.append(frames[index])
            continue
        interpolated.append(frames[index])
        for sub_index in range(1, factor):
            alpha = sub_index / factor
            blended = (current.astype(np.float32) * (1.0 - alpha)
                       + nxt.astype(np.float32) * alpha)
            interpolated.append(
                np.clip(blended, 0, 255).astype(current.dtype)
            )
    interpolated.append(frames[-1])
    return interpolated


# Core packages that gate ``realGenerationAvailable``. Without these, the
# runtime can't even preload a model.
_CORE_DEPS: tuple[tuple[str, str], ...] = (
    ("diffusers", "diffusers"),
    ("torch", "torch"),
    ("accelerate", "accelerate"),
    ("huggingface_hub", "huggingface_hub"),
    ("pillow", "PIL"),
)


# Packages required only to write the final mp4. Reported as missing so users
# know what's needed for generation, but we don't block preload on them.
_VIDEO_OUTPUT_DEPS: tuple[tuple[str, str], ...] = (
    ("imageio", "imageio"),
    ("imageio-ffmpeg", "imageio_ffmpeg"),
)


# Packages individual video pipelines pull in lazily — only at preload or
# generate time, depending on the tokenizer / text encoder. Diffusers itself
# imports cleanly without them, so they don't block the runtime, but a user
# who picks LTX-Video without ``tiktoken`` installed sees a runtime error
# mid-generate. Surfacing them in the probe lets the Studio offer a one-
# click install before the user wastes a slow preload.
#
# Coverage at the time of writing:
# - tiktoken: LTX-Video's T5 tokenizer ships in tiktoken format.
# - sentencepiece: Wan (UMT5-XXL), HunyuanVideo, CogVideoX, Mochi (T5).
# - protobuf: required by the SentencePiece-based tokenizers HF loads.
# - ftfy: text-prep utility some pipelines use during prompt encoding.
_VIDEO_MODEL_DEPS: tuple[tuple[str, str], ...] = (
    ("tiktoken", "tiktoken"),
    ("sentencepiece", "sentencepiece"),
    ("protobuf", "google.protobuf"),
    ("ftfy", "ftfy"),
)


def _find_missing(deps: tuple[tuple[str, str], ...]) -> list[str]:
    # ``importlib.util.find_spec`` raises ``ModuleNotFoundError`` (not returns
    # ``None``) when the parent of a dotted name is not importable. Concretely:
    # ``find_spec("google.protobuf")`` blows up with "No module named 'google'"
    # on a machine that never installed protobuf, instead of just reporting
    # that protobuf is missing. Without this guard the probe crashes with a
    # 500 and the Video Studio shows "runtime did not respond" forever.
    missing: list[str] = []
    for package, module_name in deps:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ModuleNotFoundError, ValueError, ImportError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing
