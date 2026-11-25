"""Character, place and style ops (the project's cast and look): add one with up to four pictures, edit its
name, look and pictures, remove it. Split out of ``ops`` to keep that module small; ``ops`` re-exports
everything here and registers these ops in ``OPS``. A picture source is a local path, an ``upload:`` handle or
a ``media/`` path of this project (a shot still), resolved by ``ops._source``. A style (S15) keeps the main
colours of its pictures, read again whenever its pictures change; the user may trim that list by hand.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from backend_service.storyboard import media
from backend_service.storyboard.models import (
    MAX_PALETTE,
    MAX_REF_IMAGES,
    HexColour,
    Id,
    Line,
    Project,
    Prompt,
    RefImage,
    Reference,
    new_id,
)
from backend_service.storyboard.ops import OpError, _Args, _import, _set_fields

_Source = Annotated[str, Field(min_length=1, max_length=4096)]


class AddReference(_Args):
    kind: Literal["character", "place", "style"]
    name: Line = ""
    description: Prompt = ""
    image: str | None = Field(None, max_length=4096)
    images: list[_Source] | None = Field(None, max_length=MAX_REF_IMAGES)
    inEveryShot: bool = False  # a character in every shot unless a shot leaves it out (B1309)

    @model_validator(mode="after")
    def _distinct_and_few(self) -> "AddReference":
        """The same source twice is one picture, and more than four are refused before anything is copied in."""
        seen: list[str] = []
        for raw in ([self.image] if self.image else []) + list(self.images or []):
            if raw not in seen:
                seen.append(raw)
        if len(seen) > MAX_REF_IMAGES:
            raise ValueError(f"A character, place or style holds up to {MAX_REF_IMAGES} pictures.")
        self.image, self.images = None, seen
        if self.inEveryShot and self.kind != "character":
            raise ValueError("Only a character can be in every shot.")
        return self


class EditReference(_Args):
    """``image`` replaces every picture with this one (the format 2 meaning); ``addImage`` appends one (up to
    four) and ``removeImage`` drops the picture at that position."""

    refId: Id
    name: Line | None = None
    description: Prompt | None = None
    image: str | None = Field(None, max_length=4096)
    addImage: str | None = Field(None, max_length=4096)
    removeImage: int | None = Field(None, ge=0, lt=MAX_REF_IMAGES)
    palette: list[HexColour] | None = Field(None, max_length=MAX_PALETTE)  # a style's colours, trimmed by hand
    inEveryShot: bool | None = None  # characters only (B1309)


class RemoveReference(_Args):
    refId: Id


def _ref_image(folder: Path, ref: Reference, raw: str) -> RefImage:
    rel, digest, _ = _import(folder, raw, {"image"}, f"media/refs/{ref.id}-{int(time.time() * 1000)}-{secrets.token_hex(3)}")
    return RefImage(path=rel, hash=digest)


def _add_ref_image(folder: Path, ref: Reference, raw: str) -> None:
    if len(ref.images) >= MAX_REF_IMAGES:
        raise OpError(f"A character or place holds up to {MAX_REF_IMAGES} pictures. Remove one first.")
    ref.images = [*ref.images, _ref_image(folder, ref, raw)]


def read_palette(folder: Path, ref: Reference) -> list[str]:
    """The main colours of a style's pictures (an unreadable or missing picture adds none)."""
    from backend_service.storyboard.palette import palette_of

    paths = []
    for image in ref.images:
        try:
            paths.append(media.inside(folder, image.path))
        except media.MediaError:
            continue
    return palette_of(paths) if paths else []


def _is_style(p: Project, ref: Reference) -> bool:
    return any(r.id == ref.id for r in p.references.styles)


def add_reference(p: Project, folder: Path, a: AddReference) -> None:
    ref = Reference(id=new_id("r"), name=a.name, description=a.description, inEveryShot=a.inEveryShot)
    for raw in ([a.image] if a.image else []) + list(a.images or []):
        _add_ref_image(folder, ref, raw)
    if a.kind == "style":
        ref.palette = read_palette(folder, ref)
    {"character": p.references.characters, "place": p.references.places, "style": p.references.styles}[a.kind].append(ref)


def edit_reference(p: Project, folder: Path, a: EditReference) -> None:
    ref = p.reference(a.refId)
    if ref is None:
        raise OpError(f"Unknown reference '{a.refId}'.")
    if a.palette is not None and not _is_style(p, ref):
        raise OpError("Only a style keeps a palette.")
    if a.inEveryShot and not any(r.id == ref.id for r in p.references.characters):
        raise OpError("Only a character can be in every shot.")
    _set_fields(ref, a, {"refId", "image", "addImage", "removeImage"})
    pictures = [img.path for img in ref.images]
    if a.removeImage is not None:
        if a.removeImage >= len(ref.images):
            raise OpError(f"This reference has no picture {a.removeImage + 1}.")
        ref.images = [img for i, img in enumerate(ref.images) if i != a.removeImage]
    if a.image:
        ref.images = [_ref_image(folder, ref, a.image)]
    if a.addImage:
        _add_ref_image(folder, ref, a.addImage)
    if _is_style(p, ref) and [img.path for img in ref.images] != pictures:
        ref.palette = read_palette(folder, ref)  # new pictures, new colours (a hand-trimmed list is read again)


def remove_reference(p: Project, folder: Path, a: RemoveReference) -> None:
    if p.reference(a.refId) is None:
        raise OpError(f"Unknown reference '{a.refId}'.")
    p.references.characters = [r for r in p.references.characters if r.id != a.refId]
    p.references.places = [r for r in p.references.places if r.id != a.refId]
    p.references.styles = [r for r in p.references.styles if r.id != a.refId]
    for shot in p.shots:
        if a.refId in shot.referenceImages:
            shot.referenceImages = [r for r in shot.referenceImages if r != a.refId]
        if a.refId in shot.castOut:
            shot.castOut = [r for r in shot.castOut if r != a.refId]
