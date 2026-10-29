"""Make image and video sizes and frame counts valid before an engine sees them.

Engines refuse or garble values their model was not trained for: mflux exits on
a width that is not a multiple of 16, Wan renders mostly black frames from a
frame count that is not ``4k + 1``, LTX needs ``8k + 1``. The Studios offer
presets, but the API, custom aspect ratios and saved settings can still send
anything, so the request is corrected here and each correction is returned as a
note the artifact carries.
"""

from __future__ import annotations

_IMAGE_MULTIPLE = 16
_IMAGE_MAX_SIDE = 2048

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


def normalize_image_size(
    width: int,
    height: int,
    *,
    min_side: int | None = None,
    max_side: int = _IMAGE_MAX_SIDE,
) -> tuple[int, int, str | None]:
    """Round an image size to multiples of 16 within the model's limits."""
    floor = max(_IMAGE_MULTIPLE, int(min_side or 0))
    new_width = snap_to_multiple(width, _IMAGE_MULTIPLE, lo=floor, hi=max_side)
    new_height = snap_to_multiple(height, _IMAGE_MULTIPLE, lo=floor, hi=max_side)
    if (new_width, new_height) == (width, height):
        return width, height, None
    return new_width, new_height, (
        f"Size {width}x{height} changed to {new_width}x{new_height}: image models need "
        f"multiples of {_IMAGE_MULTIPLE}."
    )


def video_size_multiple(repo: str) -> int:
    lowered = repo.lower()
    for tag, multiple in _VIDEO_SIZE_MULTIPLES:
        if tag in lowered:
            return multiple
    return 16


def normalize_video_size(width: int, height: int, repo: str) -> tuple[int, int, str | None]:
    multiple = video_size_multiple(repo)
    new_width = snap_to_multiple(width, multiple, lo=multiple * 4, hi=2048)
    new_height = snap_to_multiple(height, multiple, lo=multiple * 4, hi=2048)
    if (new_width, new_height) == (width, height):
        return width, height, None
    return new_width, new_height, (
        f"Size {width}x{height} changed to {new_width}x{new_height}: this video model needs "
        f"multiples of {multiple}."
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
