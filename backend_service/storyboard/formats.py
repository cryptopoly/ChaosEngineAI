"""Project format choices (D-64): aspect ratio, resolution, frame rate, and the pixel size each pair gives.

The size is the film's delivered size. Stills and clips are made at the size the video model supports
(the generation route snaps and caps the request) and the assembler fits them to this size with scale and pad.
"""

from __future__ import annotations

from typing import Any

ASPECTS = ("16:9", "9:16", "1:1")
RESOLUTIONS = ("480p", "720p", "1080p", "1440p", "2160p")
FPS_CHOICES = (24, 25, 30, 48, 50, 60)
UPSCALES = ("none", "1440p", "2160p")
DEFAULT_ASPECT = "16:9"
DEFAULT_RESOLUTION = "720p"

# Short side of each resolution. 480p is 854x480 as 16:9 (the usual 480p frame); the rest are exact.
_SHORT_SIDE = {"480p": 480, "720p": 720, "1080p": 1080, "1440p": 1440, "2160p": 2160}

# Projects saved before D-64 had one fixed size per shape and no resolution.
LEGACY_SIZES = {(1280, 720), (720, 1280), (1024, 1024)}


CUSTOM = "custom"  # the label of a size that is no aspect's frame at any short side (D-64 labels, FU-213)


def size_for(aspect: str, resolution: str) -> tuple[int, int]:
    """(width, height) for an aspect ratio at a resolution. Both are even (yuv420p)."""
    if aspect not in ASPECTS:
        raise ValueError(f"aspect must be one of {', '.join(ASPECTS)}")
    if resolution not in _SHORT_SIDE:
        raise ValueError(f"resolution must be one of {', '.join(RESOLUTIONS)}")
    return _frame(aspect, _SHORT_SIDE[resolution])


def _frame(aspect: str, short: int) -> tuple[int, int]:
    if aspect == "1:1":
        return short, short
    long_side = 854 if short == 480 else round(short * 16 / 9)
    return (long_side, short) if aspect == "16:9" else (short, long_side)


def resolution_label(aspect: str, width: int, height: int) -> str:
    """The resolution a real size reads as: a preset when it is that preset's frame, ``"<short side>p"`` when it is
    the aspect's frame at another short side (1024x576 is "576p"), else "custom". So a project set by exact size
    never says 720p for a 1024x576 film (FU-213)."""
    for resolution in RESOLUTIONS:
        if _frame(aspect, _SHORT_SIDE[resolution]) == (width, height):
            return resolution
    short = min(width, height)
    return f"{short}p" if _frame(aspect, short) == (width, height) else CUSTOM


def _short_side(resolution: Any) -> int | None:
    """The short side a resolution names: a preset, or a label like "576p" that ``resolution_label`` gave."""
    if resolution in _SHORT_SIDE:
        return _SHORT_SIDE[resolution]
    if isinstance(resolution, str) and resolution.endswith("p") and resolution[:-1].isdigit():
        return int(resolution[:-1])
    return None


def upscale_size(aspect: str, upscale: str, project: tuple[int, int]) -> tuple[int, int] | None:
    """The final size for an export 'Upscale' choice, or None when it would not be larger than the project."""
    if upscale == "none":
        return None
    width, height = size_for(aspect, upscale)
    return (width, height) if width * height > project[0] * project[1] else None


def apply_aliases(data: dict[str, Any]) -> dict[str, Any]:
    """``aspect`` and the old ``shape`` are names for ``exportPreset``; an alias wins over the stored value."""
    data = dict(data)
    for alias in ("shape", "aspect"):
        if alias in data:
            data["exportPreset"] = data.pop(alias)
    return data


def normalize_defaults(data: Any) -> Any:
    """Accept ``aspect`` and the old ``shape`` as names for ``exportPreset``; fill the size from the format.

    A project saved before D-64 (no ``resolution``, one of the old fixed sizes) reads as the same aspect at 720p.
    Once the size is known, ``resolution`` is the label of that size (``resolution_label``), whatever was stored.
    """
    if not isinstance(data, dict):
        return data
    data = apply_aliases(data)
    aspect = data.get("exportPreset", DEFAULT_ASPECT)
    if aspect not in ASPECTS:
        return data  # the model's own validation reports it
    size = (data.get("width"), data.get("height"))
    if "resolution" not in data:
        if size == (None, None) or size in LEGACY_SIZES:
            data["resolution"] = DEFAULT_RESOLUTION
            data["width"], data["height"] = size_for(aspect, DEFAULT_RESOLUTION)
    elif size[0] is None or size[1] is None:
        short = _short_side(data["resolution"])
        if short is None:
            if data["resolution"] == CUSTOM:
                raise ValueError("resolution 'custom' needs a width and a height")
            return data  # the model's own validation reports it
        data["width"], data["height"] = _frame(aspect, short)
    width, height = data.get("width"), data.get("height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        data["resolution"] = resolution_label(aspect, width, height)
    return data
