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
  * `pressure_percent` — same formula the system snapshot exposes
    (used + compressed + swap), which captures real pressure on macOS where
    `available` underreports compressed pages.

If both signals trip the floor, refuse with a structured message the UI can
render verbatim. Callers receive `None` on success or a dict with `code`
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
IMAGE_MAX_PRESSURE_PERCENT = 95.0

# Video gen working set scales with frame count + resolution. Strictest
# of the three gates — a hung video gen on Apple Silicon will typically
# swap-thrash for minutes before recovering.
VIDEO_MIN_AVAILABLE_GB = 6.0
VIDEO_MAX_PRESSURE_PERCENT = 92.0


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
) -> dict[str, Any] | None:
    """Pre-flight check for image generation. Returns refusal or None.

    Image inference can OOM swap-thrash for minutes before recovering, so
    we require materially more headroom than chat. Same shape as
    `gate_chat_generation` so call sites can render the message uniformly.
    """
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
) -> dict[str, Any] | None:
    """Pre-flight check for video generation. Returns refusal or None.

    Video working sets scale with frame count + resolution, so the floor
    is the strictest of the three gates. A hung diffusion loop on a memory
    -starved Apple Silicon machine has historically taken the whole host
    down — this gate is the cheapest possible defence.
    """
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


def snapshot_memory_signals() -> tuple[float, float]:
    """Read current available-RAM + pressure-percent signals.

    Mirrors the formulas in `helpers/system.system_snapshot` but is cheaper
    to call repeatedly — no model catalog refresh, no GPU probing. Suitable
    for the per-request gate.
    """
    import psutil

    memory = psutil.virtual_memory()
    try:
        swap = psutil.swap_memory()
        swap_used = swap.used
    except OSError:
        swap_used = 0
    total = memory.total
    used = memory.used
    available = memory.available
    available_gb = available / (1024 ** 3)

    # Compressed pages are macOS-specific and not always available; fall
    # back to plain used+swap when the read fails so non-Apple platforms
    # still get a sensible pressure number.
    compressed_used = 0
    try:
        from backend_service.helpers.system import _get_compressed_memory_gb

        compressed_used = _get_compressed_memory_gb() * (1024 ** 3)
    except Exception:
        compressed_used = 0

    swap_used_gb = swap_used / (1024 ** 3)
    used_gb = used / (1024 ** 3)
    compressed_used_gb = compressed_used / (1024 ** 3)
    pressure_numerator = used_gb + compressed_used_gb + swap_used_gb
    total_gb = total / (1024 ** 3)
    pressure_percent = (
        min(100.0, (pressure_numerator / total_gb) * 100)
        if total_gb > 0
        else 0.0
    )
    return round(available_gb, 1), round(pressure_percent, 1)
