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
    if text.startswith("upload:"):
        from backend_service.mcp_bridge.files import resolve_handle

        text = str(resolve_handle(text))
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


def checked_reference_images(raw_paths: list[str] | None, variant: dict) -> list[str]:
    """Resolve reference images for an image model, or raise ``InputImageError`` naming what works.

    Only mflux models and FLUX 3 Klein on diffusers take them; another engine would drop the image silently.
    """
    if not raw_paths:
        return []
    from backend_service.image_runtime.mflux_engine import reference_image_limit

    from backend_service.image_runtime.flux3_modular import FLUX3_MODULAR_REPOS, MAX_REFERENCE_IMAGES

    name = variant.get("name") or variant.get("id")
    engine = (variant.get("engine") or "").lower()
    repo = str(variant.get("repo") or "")
    if engine == "mflux":
        limit = reference_image_limit(repo)
    elif not engine and repo in FLUX3_MODULAR_REPOS:
        limit = MAX_REFERENCE_IMAGES  # FLUX 3 Klein on diffusers (CUDA) edits from its references
    else:
        limit = 0
    if limit == 0:
        raise InputImageError(
            f"{name} cannot take a reference image. Use FLUX.1, FLUX.2 Klein, Qwen Image 2.1 or FLUX 3 Klein "
            "on the mflux (MLX) engine, or FLUX 3 Klein on diffusers (NVIDIA)."
        )
    if len(raw_paths) > limit:
        raise InputImageError(f"{name} takes {limit} reference image{'s' if limit != 1 else ''}; got {len(raw_paths)}.")
    return [str(resolve_input_image(raw)) for raw in raw_paths]
