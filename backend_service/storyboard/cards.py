"""The brand kit on the exported film (S15): what the title card, the end card and the corner logo look like.

Pure choices from the project, shared in spirit with the rough-cut player (``brandCards.ts`` makes the same
choices so the preview and the export agree): the card text is the Export tab's text or the project name; the
typeface is the brand caption font (else the platform default); the background is the main colour of the first
style reference that has a palette (else black); the text colour is the brand caption colour when it reads on
that background (contrast 3:1 or more), else white or black, whichever reads better. ``plan_entries`` turns
them into the render plan's ``title_card``, ``end_card`` and ``logo`` fields with absolute paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend_service.storyboard.models import Project

MIN_CONTRAST = 3.0


def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def card_colours(project: Project) -> tuple[str, str]:
    """(background, text colour) for the cards, both ``#RRGGBB`` upper case."""
    background = "#000000"
    for ref in project.references.styles:
        if ref.palette:
            background = ref.palette[0].upper()
            break
    wanted = project.brand.captions.colour.upper()
    if contrast(background, wanted) >= MIN_CONTRAST:
        return background, wanted
    white, black = "#FFFFFF", "#000000"
    return background, white if contrast(background, white) >= contrast(background, black) else black


def card_text(project: Project, which: str) -> str:
    raw = project.export.titleText if which == "title" else project.export.endText
    return raw.strip() or project.name.strip()


def card_font_file(project: Project, text: str) -> str | None:
    """The file of the brand caption font, else of the platform's default caption font (None when neither is found)."""
    from backend_service.storyboard import fonts
    from backend_service.storyboard.captions import default_font

    for family in (project.brand.captions.font, default_font(text)):
        if family:
            path = fonts.font_file(family)
            if path and Path(path).suffix.lower() in (".ttf", ".otf", ".ttc"):
                return path
    return None


def logo_path(project: Project, folder: Path) -> str | None:
    from backend_service.storyboard.media import MediaError, inside

    logo = project.brand.logo
    if logo is None:
        return None
    try:
        path = inside(folder, logo.path)
    except MediaError:
        return None
    return str(path) if path.is_file() else None


def plan_entries(project: Project, folder: Path) -> dict[str, Any]:
    """The render plan's ``title_card``, ``end_card`` and ``logo`` (each None when off)."""
    settings = project.export
    logo = logo_path(project, folder)
    background, colour = card_colours(project)
    out: dict[str, Any] = {"title_card": None, "end_card": None, "logo": None}
    for which, on in (("title", settings.titleCard), ("end", settings.endCard)):
        if not on:
            continue
        text = card_text(project, which)
        out[f"{which}_card"] = {"text": text, "seconds": settings.cardSeconds, "background": background, "colour": colour,
                                "font_file": card_font_file(project, text), "logo": logo}
    if logo and settings.showLogo:
        out["logo"] = {"path": logo, "corner": project.brand.logoCorner}
    return out


def text_width(font_file: str | None, text: str) -> Callable[[int], float] | None:
    """A function that measures ``text`` at a font size in the card typeface (None when the font cannot be read)."""
    if not font_file or not text:
        return None
    try:
        from PIL import ImageFont

        ImageFont.truetype(font_file, 12)
    except (ImportError, OSError, ValueError):
        return None

    def measure(size: int) -> float:
        return float(ImageFont.truetype(font_file, size).getlength(text))

    return measure
