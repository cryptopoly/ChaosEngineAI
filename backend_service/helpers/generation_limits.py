"""Make image and video sizes and frame counts valid before an engine sees them.

Engines refuse or garble values their model was not trained for: mflux exits on
a width that is not a multiple of 16, Wan renders mostly black frames from a
frame count that is not ``4k + 1``, LTX needs ``8k + 1``. The Studios offer
presets, but the API, custom aspect ratios and saved settings can still send
anything, so the request is corrected here and each correction is returned as a
note the artifact carries.

Size limits are per model. A catalog variant declares ``maxSide`` (largest
edge), ``maxPixels`` (largest area) and ``nativeSide`` (the resolution it was
trained for); ``size_limits`` reads them, and ``HARD_MAX_SIDE`` is the
safety-net ceiling that no variant, and no request, can exceed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

_IMAGE_MULTIPLE = 16

# No model, and no request, goes above this edge, whatever a catalog row says.
HARD_MAX_SIDE = 8192
# Used for a variant that declares no limit (tracked seeds, repos the catalog
# does not know yet). Unverified: it is the generic value for modern DiT models.
DEFAULT_MAX_SIDE = 4096
DEFAULT_NATIVE_SIDE = 1024

_RESOLUTION_RE = re.compile(r"^\s*(\d+)\s*[xX\u00d7*]\s*(\d+)\s*$")
_REFERENCE_FPS = 24


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def parse_resolution(value: Any) -> tuple[int, int] | None:
    """``"1360x768"`` -> ``(1360, 768)``; None for anything else."""
    match = _RESOLUTION_RE.match(str(value or ""))
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


@dataclass(frozen=True)
class SizeLimits:
    """The size limits of one variant, with defaults filled in and clamped."""

    max_side: int
    max_pixels: int
    native_side: int
    min_side: int | None = None

    def as_kwargs(self) -> dict[str, int]:
        """Keyword arguments for ``normalize_image_size`` / ``normalize_video_size``."""
        return {"max_side": self.max_side, "max_pixels": self.max_pixels}

    def as_fields(self) -> dict[str, int]:
        """The catalog / payload field names."""
        return {"maxSide": self.max_side, "maxPixels": self.max_pixels, "nativeSide": self.native_side}


def size_limits(variant: Mapping[str, Any] | None) -> SizeLimits:
    """Limits for a catalog variant (or any dict carrying the catalog fields).

    A missing field falls back: ``maxSide`` to ``DEFAULT_MAX_SIDE``, ``maxPixels``
    to ``maxSide`` squared, ``nativeSide`` to the long edge of
    ``recommendedResolution``, else ``DEFAULT_NATIVE_SIDE``. ``maxSide`` is held
    at ``HARD_MAX_SIDE`` and ``maxPixels`` at its square.
    """
    variant = variant or {}
    max_side = min(HARD_MAX_SIDE, _positive_int(variant.get("maxSide")) or DEFAULT_MAX_SIDE)
    max_pixels = min(HARD_MAX_SIDE * HARD_MAX_SIDE, _positive_int(variant.get("maxPixels")) or max_side * max_side)
    recommended = parse_resolution(variant.get("recommendedResolution"))
    native_side = (
        _positive_int(variant.get("nativeSide"))
        or (max(recommended) if recommended else None)
        or DEFAULT_NATIVE_SIDE
    )
    return SizeLimits(
        max_side=max_side,
        max_pixels=max_pixels,
        native_side=native_side,
        min_side=_positive_int(variant.get("minSide")),
    )


def size_limit_fields(variant: Mapping[str, Any] | None) -> dict[str, int]:
    """``maxSide`` / ``maxPixels`` / ``nativeSide`` for a variant payload."""
    return size_limits(variant).as_fields()


def reference_pixels(variant: Mapping[str, Any] | None) -> int:
    """The area the variant's memory footprint was measured or estimated at.

    Catalog footprints describe a run at ``recommendedResolution``; the memory
    gate charges extra only for pixels above this. Falls back to the native
    side squared.
    """
    variant = variant or {}
    recommended = parse_resolution(variant.get("recommendedResolution"))
    if recommended:
        return recommended[0] * recommended[1]
    side = size_limits(variant).native_side
    return side * side


def reference_frames(variant: Mapping[str, Any] | None) -> int | None:
    """The frame count the variant's video footprint describes (default clip length at 24 fps)."""
    seconds = (variant or {}).get("defaultDurationSeconds")
    if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds > 0:
        return max(1, round(float(seconds) * _REFERENCE_FPS))
    return None


# (step, offset): a valid frame count is ``step * k + offset``.
_FRAME_RULES: tuple[tuple[str, int, int], ...] = (
    ("ltx", 8, 1),
    ("wan", 4, 1),
    ("hunyuan", 4, 1),
    ("cogvideo", 4, 1),
    ("mochi", 6, 1),
)

# Spatial multiple of the video VAE, patch size included.
_VIDEO_SIZE_MULTIPLES: tuple[tuple[str, int], ...] = (
    ("ltx", 32),
    ("wan", 16),
    ("hunyuan", 16),
    ("cogvideo", 16),
    ("mochi", 16),
)


def snap_to_multiple(value: int, multiple: int, *, lo: int, hi: int) -> int:
    """Nearest multiple of ``multiple`` inside ``[lo, hi]`` (ties round up)."""
    if multiple <= 1:
        return max(lo, min(hi, value))
    snapped = int((value + multiple // 2) // multiple) * multiple
    low = -(-lo // multiple) * multiple
    high = (hi // multiple) * multiple
    return max(low, min(high, snapped))


def _megapixels_text(max_pixels: int) -> str:
    return f"{max_pixels / 1_000_000:.1f} megapixels"


def _fit_pixels(width: int, height: int, multiple: int, lo: int, max_pixels: int) -> tuple[int, int]:
    """Shrink a size to at most ``max_pixels``, keeping the aspect ratio, on multiples of ``multiple``.

    Rounds down so the result never exceeds the limit; each edge stays at or
    above ``lo``.
    """
    if width * height <= max_pixels:
        return width, height
    low = -(-lo // multiple) * multiple
    scale = math.sqrt(max_pixels / (width * height))
    new_width = max(low, int(width * scale) // multiple * multiple)
    new_height = max(low, int(height * scale) // multiple * multiple)
    while new_width * new_height > max_pixels and (new_width > low or new_height > low):
        if new_width >= new_height and new_width > low:
            new_width -= multiple
        else:
            new_height -= multiple
    return new_width, new_height


def _normalize_size(
    width: int,
    height: int,
    *,
    multiple: int,
    floor: int,
    max_side: int | None,
    max_pixels: int | None,
    label: str,
    snap_reason: str,
) -> tuple[int, int, str | None]:
    ceiling = min(HARD_MAX_SIDE, int(max_side)) if max_side else DEFAULT_MAX_SIDE
    # Over a limit, scale both edges together so the aspect ratio survives.
    scaled_width, scaled_height = float(width), float(height)
    side_hit = max(width, height) > ceiling
    if side_hit:
        factor = ceiling / max(width, height)
        scaled_width, scaled_height = width * factor, height * factor
    pixels_hit = bool(max_pixels) and scaled_width * scaled_height > int(max_pixels or 0)
    if pixels_hit:
        factor = math.sqrt(int(max_pixels or 0) / (scaled_width * scaled_height))
        scaled_width, scaled_height = scaled_width * factor, scaled_height * factor
    if side_hit or pixels_hit:
        # Rounding to the nearest multiple could land back over the limit.
        new_width = snap_to_multiple(int(scaled_width) - multiple // 2, multiple, lo=floor, hi=ceiling)
        new_height = snap_to_multiple(int(scaled_height) - multiple // 2, multiple, lo=floor, hi=ceiling)
    else:
        new_width = snap_to_multiple(width, multiple, lo=floor, hi=ceiling)
        new_height = snap_to_multiple(height, multiple, lo=floor, hi=ceiling)
    if max_pixels:
        new_width, new_height = _fit_pixels(new_width, new_height, multiple, floor, int(max_pixels))
    if (new_width, new_height) == (width, height):
        return width, height, None
    reasons: list[str] = []
    if side_hit or pixels_hit:
        bits = []
        if side_hit:
            bits.append(f"{ceiling} px per side")
        if pixels_hit:
            bits.append(_megapixels_text(int(max_pixels or 0)))
        reasons.append(f"{label} allows at most {' and '.join(bits)}")
    else:
        reasons.append(snap_reason)
    return new_width, new_height, f"Size {width}x{height} changed to {new_width}x{new_height}: {'; '.join(reasons)}."


def normalize_image_size(
    width: int,
    height: int,
    *,
    min_side: int | None = None,
    max_side: int | None = None,
    max_pixels: int | None = None,
    model_name: str | None = None,
) -> tuple[int, int, str | None]:
    """Round an image size to multiples of 16 within the model's limits.

    ``max_side`` and ``max_pixels`` are the variant's ``maxSide`` / ``maxPixels``
    (``DEFAULT_MAX_SIDE`` when not given; never above ``HARD_MAX_SIDE``). The note
    names the model whose limit applied.
    """
    return _normalize_size(
        width,
        height,
        multiple=_IMAGE_MULTIPLE,
        floor=max(_IMAGE_MULTIPLE, int(min_side or 0)),
        max_side=max_side,
        max_pixels=max_pixels,
        label=model_name or "This image model",
        snap_reason=f"image models need multiples of {_IMAGE_MULTIPLE}",
    )


def video_size_multiple(repo: str) -> int:
    lowered = repo.lower()
    for tag, multiple in _VIDEO_SIZE_MULTIPLES:
        if tag in lowered:
            return multiple
    return 16


def normalize_video_size(
    width: int,
    height: int,
    repo: str,
    *,
    max_side: int | None = None,
    max_pixels: int | None = None,
    model_name: str | None = None,
) -> tuple[int, int, str | None]:
    """Round a video size to the model's spatial multiple within its limits."""
    multiple = video_size_multiple(repo)
    return _normalize_size(
        width,
        height,
        multiple=multiple,
        floor=multiple * 4,
        max_side=max_side,
        max_pixels=max_pixels,
        label=model_name or "This video model",
        snap_reason=f"this video model needs multiples of {multiple}",
    )


def align_video_frames(repo: str, frames: int) -> tuple[int, str | None]:
    """Round a frame count down to the nearest valid ``step * k + offset``.

    Down, so the clip never runs longer than the user asked for.
    """
    lowered = repo.lower()
    for tag, step, offset in _FRAME_RULES:
        if tag not in lowered:
            continue
        aligned = max(offset + step, ((frames - offset) // step) * step + offset)
        if aligned == frames:
            return frames, None
        return aligned, f"Frame count {frames} changed to {aligned}: this video model needs {step}k+{offset} frames."
    return frames, None


def distilled_steps_note(model_name: str, requested: int, tuned: int | None) -> str | None:
    """A note when a few-step model is asked for many more steps than it was tuned for.

    Distilled and turbo models are trained for a handful of steps (4 to 8). More
    steps cost time in proportion and rarely look better, so the run still does
    what was asked and the note says what it costs.
    """
    if tuned is None or tuned > 8 or requested <= tuned * 2:
        return None
    return (
        f"{model_name} is tuned for {tuned} steps; {requested} steps take about "
        f"{requested / tuned:.0f} times as long and rarely look better."
    )
