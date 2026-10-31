"""Pre-flight memory gates for chat / image / video generation.

Phase 2.0.5-B: refuses generation requests when free system memory is below
a safety floor, before the runtime gets a chance to OOM and wedge the host.
The gate is intentionally conservative — it doesn't try to predict exact
working-set size (the model is already loaded, KV pressure varies with
context length) — it just bails when the system is already memory-starved.

Decision factors:
  * `available_gb` — `psutil.virtual_memory().available`, the kernel's own
    estimate of memory that can be allocated without forcing major GC or
    swap, which is the right measure on every supported OS.
  * `pressure_percent` — the share of memory that is not available, from the
    same available figure. Swap is left out on purpose: macOS and Linux keep
    pages in swap long after the memory is free again, so counting swap as
    "in use" reported 100% pressure on a Mac the OS itself rated normal.

If either signal trips its limit, refuse with a structured message the UI
can render verbatim. Callers receive `None` on success or a dict with `code`
and `message`.
"""

from __future__ import annotations

from typing import Any


# Minimum free memory required to start a chat generation. Smaller than the
# image/video gates because chat KV growth per turn is typically <1 GB; the
# model itself is already resident.
CHAT_MIN_AVAILABLE_GB = 1.0
# Combined-pressure ceiling. macOS unified memory routinely sits at 90-97%
# pressure during normal use because the kernel aggressively compresses
# pages — the original 92% threshold turned out to be too strict and
# refused generations that would have completed comfortably. We now treat
# `available_gb` as the primary signal and only fall back to the pressure
# ceiling at near-OOM levels (98%+). Raise this only if the available-GB
# floor proves insufficient.
CHAT_MAX_PRESSURE_PERCENT = 98.0

# Phase 2.0.5-H: image generation typically needs 4-12 GB working set on
# top of the already-resident pipeline (latents, attention buffers, VAE
# decode). The gate is a backstop — refuses when the host is already
# strained enough that an OOM during inference would wedge the laptop.
IMAGE_MIN_AVAILABLE_GB = 4.0
# Weights are mapped into memory twice while a pipeline loads, so a model
# needs more free memory than its on-disk size.
IMAGE_MODEL_HEADROOM = 1.1
IMAGE_MAX_PRESSURE_PERCENT = 95.0

# Video gen working set scales with frame count + resolution. Strictest
# of the three gates — a hung video gen on Apple Silicon will typically
# swap-thrash for minutes before recovering.
VIDEO_MIN_AVAILABLE_GB = 6.0
VIDEO_MAX_PRESSURE_PERCENT = 92.0
# Video rows carry a runtime footprint that is already the generation peak.
VIDEO_MODEL_HEADROOM = 1.0

# Larger-than-reference sizes (estimated, not measured; same model as the Studio's own
# warning in src/utils/images.ts). A catalog footprint describes a run at the variant's
# recommended resolution. Attention, latents and the VAE decode grow with the pixel count,
# so every megapixel above that resolution is charged this share of the model's footprint.
# `scripts/measure-image-ceiling.py` measures the real figure on a machine.
EXTRA_SHARE_PER_MEGAPIXEL = 0.15
# Stand-in footprint for a variant that has none, so its size still counts.
UNKNOWN_FOOTPRINT_GB = 10.0
_MEGAPIXEL = 1024 * 1024


def _gb_per_megapixel(footprint_gb: float | None, frames: int | None, reference_frames: int | None) -> float:
    """Extra GB per megapixel above the reference resolution."""
    rate = EXTRA_SHARE_PER_MEGAPIXEL * (footprint_gb or UNKNOWN_FOOTPRINT_GB)
    if frames and reference_frames and frames > reference_frames:
        rate *= frames / reference_frames
    return rate


def size_extra_gb(
    width: int,
    height: int,
    *,
    reference_pixels: int | None,
    footprint_gb: float | None,
    frames: int | None = None,
    reference_frames: int | None = None,
) -> float:
    """Extra memory in GB a size above the reference resolution needs. 0 at or below it.

    Video clips longer than the reference length scale the extra by the length ratio.
    """
    if not reference_pixels or width * height <= reference_pixels:
        return 0.0
    extra_megapixels = (width * height - reference_pixels) / _MEGAPIXEL
    return _gb_per_megapixel(footprint_gb, frames, reference_frames) * extra_megapixels


def _fitting_size(
    width: int,
    height: int,
    *,
    allowed_gb: float,
    reference_pixels: int,
    footprint_gb: float | None,
    frames: int | None,
    reference_frames: int | None,
) -> tuple[int, int]:
    """The requested aspect ratio at the largest area (multiples of 64) that fits ``allowed_gb``."""
    rate = _gb_per_megapixel(footprint_gb, frames, reference_frames)
    pixels = reference_pixels + max(0.0, allowed_gb) / rate * _MEGAPIXEL
    scale = min(1.0, (pixels / (width * height)) ** 0.5)
    return max(256, int(width * scale) // 64 * 64), max(256, int(height * scale) // 64 * 64)


def _size_refusal(
    *,
    code: str,
    noun: str,
    label: str,
    width: int,
    height: int,
    available_gb: float,
    load_gb: float,
    extra_gb: float,
    reference_pixels: int,
    footprint_gb: float | None,
    frames: int | None,
    reference_frames: int | None,
    memory_name: str = "memory",
) -> dict[str, Any] | None:
    """Refusal for a size that does not fit in the free memory, or None when it does."""
    need_gb = load_gb + extra_gb
    if extra_gb <= 0 or available_gb >= need_gb:
        return None
    fit_w, fit_h = _fitting_size(
        width, height,
        allowed_gb=max(0.0, available_gb - load_gb),
        reference_pixels=reference_pixels,
        footprint_gb=footprint_gb,
        frames=frames,
        reference_frames=reference_frames,
    )
    sized_for = f"the {reference_pixels / 1_000_000:.1f} megapixels {label} is sized for"
    need_text = (
        f"about {need_gb:.0f} GB of free {memory_name} ({load_gb:.0f} GB to load the model and {extra_gb:.0f} GB "
        f"because the size is above {sized_for})"
        if load_gb > 0
        else f"about {extra_gb:.0f} GB of free {memory_name} on top of the loaded model, because the size is above {sized_for}"
    )
    return {
        "code": code,
        "message": (
            f"A {width}x{height} {noun} with {label} needs {need_text}, but only {available_gb:.1f} GB "
            f"is available. Try {fit_w}x{fit_h} or smaller, or close other apps and unload models."
        ),
    }


def model_memory_gb(variant: dict[str, Any], *, apple_silicon: bool | None = None) -> float | None:
    """Memory an image / video catalog variant needs once loaded, in GB.

    On Apple Silicon the catalog's MPS runtime footprint (or the generic one)
    is the measured or estimated peak. Elsewhere the weights pass through
    system RAM on their way to the GPU, so the larger of the download size
    and the runtime footprint is used. ``None`` when the row has neither.
    """
    if apple_silicon is None:
        import platform

        apple_silicon = platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")

    def number(key: str) -> float | None:
        value = variant.get(key)
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    if apple_silicon:
        return number("runtimeFootprintMpsGb") or number("runtimeFootprintGb") or number("sizeGb")
    candidates = [v for v in (number("sizeGb"), number("runtimeFootprintGb")) if v]
    return max(candidates) if candidates else None


def gate_chat_generation(
    available_gb: float,
    pressure_percent: float,
    *,
    min_available_gb: float = CHAT_MIN_AVAILABLE_GB,
    max_pressure_percent: float = CHAT_MAX_PRESSURE_PERCENT,
) -> dict[str, Any] | None:
    """Decide whether a chat generation may proceed.

    Returns `None` when the system has enough headroom. Returns a refusal
    dict with `code` and `message` when memory is too tight. The message is
    user-facing — the UI surfaces it directly via the standard chat error
    path.
    """
    if available_gb < min_available_gb:
        return {
            "code": "memory_gate_low_available",
            "message": (
                f"Only {available_gb:.1f} GB of RAM available — at least "
                f"{min_available_gb:.1f} GB free is required to start a "
                "generation safely. Try unloading any warm models or "
                "closing other applications, then retry."
            ),
        }
    if pressure_percent > max_pressure_percent:
        return {
            "code": "memory_gate_high_pressure",
            "message": (
                f"System memory pressure is {pressure_percent:.0f}% — generation "
                "would risk swap thrashing or an OOM kill. Free some memory "
                "(unload warm models, close apps) and retry."
            ),
        }
    return None


def gate_image_generation(
    available_gb: float,
    pressure_percent: float,
    *,
    min_available_gb: float = IMAGE_MIN_AVAILABLE_GB,
    max_pressure_percent: float = IMAGE_MAX_PRESSURE_PERCENT,
    model_gb: float | None = None,
    model_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    reference_pixels: int | None = None,
    footprint_gb: float | None = None,
    headroom: float = IMAGE_MODEL_HEADROOM,
    size_available_gb: float | None = None,
) -> dict[str, Any] | None:
    """Pre-flight check for image generation. Returns refusal or None.

    Image inference can OOM swap-thrash for minutes before recovering, so
    we require materially more headroom than chat. Same shape as
    `gate_chat_generation` so call sites can render the message uniformly.

    ``model_gb`` is the weight size of a model that is not resident yet. A
    model larger than free memory is killed by the OS mid-load, which takes
    the whole backend down with it, so it is refused up front instead.

    ``width`` / ``height`` are the size that will actually render and
    ``reference_pixels`` the area the footprint describes (the variant's
    recommended resolution); a size above it is charged extra memory
    (``size_extra_gb``) and refused when the free memory cannot hold it.
    ``footprint_gb`` is the model's footprint, resident or not (default: ``model_gb``).
    ``headroom`` multiplies the model's size for the load check; a footprint that is already a
    measured process peak passes 1.0, as video does, so it is not counted twice.
    ``size_available_gb`` is the free GPU memory on a CUDA host: the size term is then judged
    against it (the extra memory a large size needs lives on the card, not in system RAM) and the
    model's own load is left to the card's offload modes.
    """
    if model_gb and available_gb < model_gb * headroom:
        label = model_name or "This image model"
        return {
            "code": "memory_gate_image_model_too_large",
            "message": (
                f"{label} needs about {model_gb * headroom:.0f} GB "
                f"of free memory to load, but only {available_gb:.1f} GB is "
                "available. Close other apps, or pick a smaller or quantized "
                "image model."
            ),
        }
    if width and height and reference_pixels:
        refusal = _size_refusal(
            code="memory_gate_image_size_too_large",
            noun="image",
            label=model_name or "this image model",
            width=width,
            height=height,
            available_gb=available_gb if size_available_gb is None else size_available_gb,
            load_gb=0.0 if size_available_gb is not None else (model_gb or 0.0) * headroom,
            extra_gb=size_extra_gb(
                width, height, reference_pixels=reference_pixels, footprint_gb=footprint_gb or model_gb
            ),
            reference_pixels=reference_pixels,
            footprint_gb=footprint_gb or model_gb,
            frames=None,
            reference_frames=None,
            memory_name="GPU memory" if size_available_gb is not None else "memory",
        )
        if refusal is not None:
            return refusal
    if available_gb < min_available_gb:
        return {
            "code": "memory_gate_image_low_available",
            "message": (
                f"Only {available_gb:.1f} GB of RAM available — image "
                f"generation needs at least {min_available_gb:.1f} GB free "
                "to run safely. Unload warm models or close other apps "
                "before retrying."
            ),
        }
    if pressure_percent > max_pressure_percent:
        return {
            "code": "memory_gate_image_high_pressure",
            "message": (
                f"Memory pressure is {pressure_percent:.0f}% — image "
                "generation would risk swap thrashing. Free some memory "
                "before retrying."
            ),
        }
    return None


def gate_video_generation(
    available_gb: float,
    pressure_percent: float,
    *,
    min_available_gb: float = VIDEO_MIN_AVAILABLE_GB,
    max_pressure_percent: float = VIDEO_MAX_PRESSURE_PERCENT,
    model_gb: float | None = None,
    model_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    reference_pixels: int | None = None,
    reference_frames: int | None = None,
    footprint_gb: float | None = None,
    size_available_gb: float | None = None,
) -> dict[str, Any] | None:
    """Pre-flight check for video generation. Returns refusal or None.

    Video working sets scale with frame count + resolution, so the floor
    is the strictest of the three gates. A hung diffusion loop on a memory
    -starved Apple Silicon machine has historically taken the whole host
    down — this gate is the cheapest possible defence.

    ``model_gb`` is the peak memory of a model that is not resident yet; one
    that does not fit is refused before loading, as for images. The size
    arguments work as for images; a clip longer than ``reference_frames``
    scales the extra memory for a size above the reference by the length ratio.
    ``size_available_gb`` works as for images (free GPU memory on a CUDA host).
    """
    if model_gb and available_gb < model_gb * VIDEO_MODEL_HEADROOM:
        label = model_name or "This video model"
        return {
            "code": "memory_gate_video_model_too_large",
            "message": (
                f"{label} needs about {model_gb * VIDEO_MODEL_HEADROOM:.0f} GB "
                f"of free memory to run, but only {available_gb:.1f} GB is "
                "available. Close other apps or unload the chat model, or pick "
                "a smaller or quantized video model."
            ),
        }
    if width and height and reference_pixels:
        refusal = _size_refusal(
            code="memory_gate_video_size_too_large",
            noun="video",
            label=model_name or "this video model",
            width=width,
            height=height,
            available_gb=available_gb if size_available_gb is None else size_available_gb,
            load_gb=0.0 if size_available_gb is not None else (model_gb or 0.0) * VIDEO_MODEL_HEADROOM,
            extra_gb=size_extra_gb(
                width, height, reference_pixels=reference_pixels, footprint_gb=footprint_gb or model_gb,
                frames=frames, reference_frames=reference_frames,
            ),
            reference_pixels=reference_pixels,
            footprint_gb=footprint_gb or model_gb,
            frames=frames,
            reference_frames=reference_frames,
            memory_name="GPU memory" if size_available_gb is not None else "memory",
        )
        if refusal is not None:
            return refusal
    if available_gb < min_available_gb:
        return {
            "code": "memory_gate_video_low_available",
            "message": (
                f"Only {available_gb:.1f} GB of RAM available — video "
                f"generation needs at least {min_available_gb:.1f} GB free "
                "to avoid swap thrashing. Unload warm models or close "
                "other apps before retrying."
            ),
        }
    if pressure_percent > max_pressure_percent:
        return {
            "code": "memory_gate_video_high_pressure",
            "message": (
                f"Memory pressure is {pressure_percent:.0f}% — video "
                "generation would likely OOM. Free some memory before "
                "retrying."
            ),
        }
    return None


def _macos_reclaimable_gb(total_bytes: int) -> float | None:
    """Memory macOS itself counts as available to apps, in GB (None elsewhere).

    ``psutil``'s "available" leaves out file cache that sits in the active list,
    so after a large model download or read it can report 28 GB free on a 64 GB
    Mac that the OS would hand over without a swap. ``kern.memorystatus_level``
    is the percentage the OS uses for its own memory-pressure decisions.
    """
    import subprocess
    import sys

    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True, text=True, timeout=2, check=True,
        )
        level = float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if not 0.0 < level <= 100.0:
        return None
    return total_bytes * level / 100.0 / (1024 ** 3)


def available_memory_gb(total_bytes: int, psutil_available_bytes: int) -> float:
    """Memory a new allocation can get without swapping, in GB.

    The larger of psutil's figure and the level macOS uses for its own
    memory-pressure decisions (see ``_macos_reclaimable_gb``).
    """
    available_gb = psutil_available_bytes / (1024 ** 3)
    return max(available_gb, _macos_reclaimable_gb(total_bytes) or 0.0)


def pressure_percent_from_available(total_gb: float, available_gb: float) -> float:
    """Percent of memory that is not available. 0 when the total is unknown."""
    if total_gb <= 0:
        return 0.0
    return min(100.0, max(0.0, (1.0 - available_gb / total_gb) * 100.0))


def cuda_free_vram_gb() -> float | None:
    """Free GPU memory in GB on a CUDA host, None everywhere else (Mac, CPU-only, unreadable)."""
    try:
        from backend_service.helpers.hardware_class import host_hardware_class

        if host_hardware_class().kind != "cuda":
            return None
        from backend_service.helpers.gpu import get_device_vram_free_gb

        return get_device_vram_free_gb()
    except Exception:
        return None


def snapshot_memory_signals() -> tuple[float, float]:
    """Read current available-RAM + pressure-percent signals.

    Same figures the system snapshot exposes, but cheaper to call
    repeatedly — no model catalog refresh, no GPU probing. Suitable for the
    per-request gate.
    """
    import psutil

    memory = psutil.virtual_memory()
    available_gb = available_memory_gb(memory.total, memory.available)
    total_gb = memory.total / (1024 ** 3)
    return round(available_gb, 1), round(pressure_percent_from_available(total_gb, available_gb), 1)
