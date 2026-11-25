"""How a project's style references (S15) reach a still: as reference pictures or as words in the prompt.

A still model that takes reference pictures (mflux models, ``reference_image_limit``) gets the first picture
of each style in the slots the shot's own cast leaves free. Any other model, a shot whose cast fills every
slot, or a project set to ``words``, gets the style written into the prompt instead: the style's own words
and its palette as plain colour names. The take records which (``settings.style``), and the Plan step shows
the same answer before anything renders (``style_mode``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from backend_service.storyboard import media
from backend_service.storyboard.models import Project, Reference, Shot
from backend_service.storyboard.palette import palette_words

StyleMode = Literal["pictures", "words", "none"]


def styles_in_use(project: Project) -> list[Reference]:
    """The style references that say something (a picture, a few words or a palette)."""
    if project.brand.styleUse == "off":
        return []
    return [r for r in project.references.styles if r.images or r.description.strip() or r.palette]


def picture_slots(variant: dict[str, Any] | None) -> int:
    """How many reference pictures the still model takes: the shared check in ``helpers.media_input`` (mflux models and
    the sd.cpp rows that take reference pictures)."""
    if not variant:
        return 0
    from backend_service.helpers.media_input import reference_image_limit

    return int(reference_image_limit(variant))


def style_words(project: Project, styles: list[Reference] | None = None) -> str:
    """"Visual style: warm, hand-drawn shop front; colour palette of deep green, warm cream and amber." or ""."""
    styles = styles_in_use(project) if styles is None else styles
    looks = [r.description.strip().rstrip(".") for r in styles if r.description.strip()]
    colours: list[str] = []
    for ref in styles:
        colours.extend(c for c in ref.palette if c not in colours)
    parts = []
    if looks:
        parts.append(f"Visual style: {'; '.join(looks)}")
    if colours:
        parts.append(f"colour palette of {palette_words(colours[:6])}")
    return ("; ".join(parts) + ".") if parts else ""


def style_mode(project: Project, variant: dict[str, Any] | None, used_slots: int = 0) -> StyleMode:
    styles = styles_in_use(project)
    if not styles:
        return "none"
    with_pictures = [r for r in styles if r.images]
    if project.brand.styleUse == "auto" and with_pictures and picture_slots(variant) - used_slots > 0:
        return "pictures"
    return "words" if style_words(project, styles) else "none"


def apply_to_still(project: Project, shot: Shot, folder: Path, variant: dict[str, Any] | None,
                   paths: list[str], prompt: str) -> tuple[list[str], str, StyleMode]:
    """The reference paths and prompt for one still, with the project's style added one way or the other."""
    styles = [r for r in styles_in_use(project) if r.id not in shot.referenceImages]  # a shot that names a style has it already
    if not styles:
        return paths, prompt, "none"
    mode = style_mode(project, variant, len(paths))
    if mode == "pictures":
        room = picture_slots(variant) - len(paths)
        extra = [str(media.inside(folder, r.images[0].path)) for r in styles if r.images][:room]
        return [*paths, *extra], prompt, "pictures"
    words = style_words(project, styles)
    if not words or words in prompt:
        return paths, prompt, "words" if words else "none"
    return paths, f"{prompt.rstrip()} {words}".strip(), "words"


def apply_to_motion(project: Project, shot: Shot, prompt: str) -> str:
    """A clip made from text alone (``startFrom: text``) has no still to carry the look: add the words."""
    if shot.startFrom != "text":
        return prompt
    words = style_words(project, [r for r in styles_in_use(project) if r.id not in shot.referenceImages])
    return f"{prompt.rstrip()} {words}".strip() if words and words not in prompt else prompt
