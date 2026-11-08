"""Character and place ops (the project's cast): add one with up to four pictures, edit its name, look and
pictures, remove it. Split out of ``ops`` to keep that module small; ``ops`` re-exports everything here and
registers these ops in ``OPS``. A picture source is a local path, an ``upload:`` handle or a ``media/`` path of
this project (a shot still), resolved by ``ops._source``.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from backend_service.storyboard.models import MAX_REF_IMAGES, Id, Line, Project, Prompt, RefImage, Reference, new_id
from backend_service.storyboard.ops import OpError, _Args, _import, _set_fields

_Source = Annotated[str, Field(min_length=1, max_length=4096)]


class AddReference(_Args):
    kind: Literal["character", "place"]
    name: Line = ""
    description: Prompt = ""
    image: str | None = Field(None, max_length=4096)
    images: list[_Source] | None = Field(None, max_length=MAX_REF_IMAGES)

    @model_validator(mode="after")
    def _distinct_and_few(self) -> "AddReference":
        """The same source twice is one picture, and more than four are refused before anything is copied in."""
        seen: list[str] = []
        for raw in ([self.image] if self.image else []) + list(self.images or []):
            if raw not in seen:
                seen.append(raw)
        if len(seen) > MAX_REF_IMAGES:
            raise ValueError(f"A character or place holds up to {MAX_REF_IMAGES} pictures.")
        self.image, self.images = None, seen
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


class RemoveReference(_Args):
    refId: Id


def _ref_image(folder: Path, ref: Reference, raw: str) -> RefImage:
    rel, digest, _ = _import(folder, raw, {"image"}, f"media/refs/{ref.id}-{int(time.time() * 1000)}-{secrets.token_hex(3)}")
    return RefImage(path=rel, hash=digest)


def _add_ref_image(folder: Path, ref: Reference, raw: str) -> None:
    if len(ref.images) >= MAX_REF_IMAGES:
        raise OpError(f"A character or place holds up to {MAX_REF_IMAGES} pictures. Remove one first.")
    ref.images = [*ref.images, _ref_image(folder, ref, raw)]


def add_reference(p: Project, folder: Path, a: AddReference) -> None:
    ref = Reference(id=new_id("r"), name=a.name, description=a.description)
    for raw in ([a.image] if a.image else []) + list(a.images or []):
        _add_ref_image(folder, ref, raw)
    (p.references.characters if a.kind == "character" else p.references.places).append(ref)


def edit_reference(p: Project, folder: Path, a: EditReference) -> None:
    ref = p.reference(a.refId)
    if ref is None:
        raise OpError(f"Unknown reference '{a.refId}'.")
    _set_fields(ref, a, {"refId", "image", "addImage", "removeImage"})
    if a.removeImage is not None:
        if a.removeImage >= len(ref.images):
            raise OpError(f"This reference has no picture {a.removeImage + 1}.")
        ref.images = [img for i, img in enumerate(ref.images) if i != a.removeImage]
    if a.image:
        ref.images = [_ref_image(folder, ref, a.image)]
    if a.addImage:
        _add_ref_image(folder, ref, a.addImage)


def remove_reference(p: Project, folder: Path, a: RemoveReference) -> None:
    if p.reference(a.refId) is None:
        raise OpError(f"Unknown reference '{a.refId}'.")
    p.references.characters = [r for r in p.references.characters if r.id != a.refId]
    p.references.places = [r for r in p.references.places if r.id != a.refId]
    for shot in p.shots:
        if a.refId in shot.referenceImages:
            shot.referenceImages = [r for r in shot.referenceImages if r != a.refId]
