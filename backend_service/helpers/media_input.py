"""Input images for generation (video start / end frames, image references).

Inputs are local files the backend user can read. The same checks apply wherever a path comes in
from a request or an agent: resolved, a regular file, an image extension, a size cap.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})
MAX_INPUT_IMAGE_BYTES = 30 * 1024 * 1024


class InputImageError(ValueError):
    """A readable message for the user or agent."""


def resolve_input_image(raw: str) -> Path:
    """The real path of an input image, or ``InputImageError`` saying what is wrong."""
    text = str(raw or "").strip()
    if not text:
        raise InputImageError("The input image path is empty.")
    try:
        path = Path(text).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise InputImageError(f"Input image not found: {text}") from exc
    if not path.is_file():
        raise InputImageError(f"Input image is not a file: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise InputImageError(
            f"Input image must be PNG, JPEG or WebP, not '{path.suffix or 'no extension'}': {path.name}"
        )
    size = path.stat().st_size
    if size == 0:
        raise InputImageError(f"Input image is empty: {path.name}")
    if size > MAX_INPUT_IMAGE_BYTES:
        raise InputImageError(f"Input image is larger than {MAX_INPUT_IMAGE_BYTES // (1024 * 1024)} MB: {path.name}")
    return path


def image_size(path: Path) -> tuple[int, int] | None:
    """Width and height in pixels, or None when Pillow cannot read the file."""
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return None
