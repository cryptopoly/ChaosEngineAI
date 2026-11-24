"""Plan and brand ops (S8, S15): the brief, a planned set of shots in one step, a batch of changes from the
direction chat, the chat log itself, and the brand kit.

Split out of ``ops`` to keep that module small; ``ops`` imports this module at its end and registers these
ops in ``OPS``. Every change a person can undo is one op, so a whole proposed plan, or every change the
direction chat proposed, goes in and comes out with one Undo. The chat log is kept with the project but is
not part of the undo history (``planChat`` is not a diffed key), so an Undo never deletes what was said.
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from backend_service.storyboard.models import (
    SONG_SHOT_NATIVE_SOUND_DB,
    MAX_PLAN_CHAT,
    MAX_SHOTS,
    Brand,
    CaptionLook,
    HexColour,
    Id,
    Line,
    PlanMessage,
    Project,
    Prompt,
    RefImage,
    Shot,
    new_id,
)
from backend_service.storyboard.ops import OpError, _Args, _import, _shot

MAX_PLANNED_SHOTS = 12


class SetPlan(_Args):
    brief: Prompt | None = None
    filmSeconds: float | None = Field(None, ge=5, le=600)
    clearFilmSeconds: bool = False


class PlannedShot(_Args):
    title: Line = ""
    stillPrompt: Prompt = ""
    motionPrompt: Prompt = ""
    seconds: float = Field(4.0, ge=0.5, le=60)
    referenceImages: list[Id] = Field(default_factory=list, max_length=8)
    trimOut: float | None = Field(None, ge=0, le=3600)  # a cut plan ends the clip on the music (S-M1)


class ApplyPlan(_Args):
    """``replace``: the planned shots take the place of every shot (they stay in Undo); ``append``: they go
    after the shots already there. Up to ``MAX_SHOTS``: a cut plan from a song (S-M1) has one shot per cut."""

    mode: Literal["replace", "append"] = "append"
    shots: list[PlannedShot] = Field(min_length=1, max_length=MAX_SHOTS)


class ShotChange(_Args):
    shotId: Id
    title: Line | None = None
    stillPrompt: Prompt | None = None
    motionPrompt: Prompt | None = None
    seconds: float | None = Field(None, ge=0.5, le=60)


class AddedShot(PlannedShot):
    after: Id | None = None  # the shot it follows; None puts it first


class PlanChanges(_Args):
    edits: list[ShotChange] = Field(default_factory=list, max_length=MAX_SHOTS)
    add: list[AddedShot] = Field(default_factory=list, max_length=MAX_PLANNED_SHOTS)
    remove: list[Id] = Field(default_factory=list, max_length=MAX_SHOTS)

    @model_validator(mode="after")
    def _something(self) -> "PlanChanges":
        if not (self.edits or self.add or self.remove):
            raise ValueError("Give at least one change.")
        return self


class AddPlanMessage(_Args):
    role: Literal["user", "assistant"]
    text: Prompt = Field(min_length=1)


class ClearPlanChat(_Args):
    pass


class CaptionLookArgs(_Args):
    font: Line | None = None
    clearFont: bool = False
    colour: HexColour | None = None
    position: Literal["bottom", "middle", "top"] | None = None


class SetBrand(_Args):
    """``logo`` is a path, an ``upload:`` handle or a project ``media/`` path; ``clearLogo`` removes it."""

    logo: str | None = Field(None, max_length=4096)
    clearLogo: bool = False
    logoCorner: Literal["top-left", "top-right", "bottom-left", "bottom-right"] | None = None
    captions: CaptionLookArgs | None = None
    styleUse: Literal["auto", "words", "off"] | None = None


def _check_refs(p: Project, ids: list[str]) -> None:
    for ref_id in ids:
        if p.reference(ref_id) is None:
            raise OpError(f"Unknown reference '{ref_id}'.")


def _new_shot(p: Project, planned: PlannedShot) -> Shot:
    _check_refs(p, planned.referenceImages)
    return Shot(id=new_id("s"), title=planned.title, stillPrompt=planned.stillPrompt, motionPrompt=planned.motionPrompt,
                seconds=planned.seconds, referenceImages=list(planned.referenceImages), trimOut=planned.trimOut,
                nativeSoundDb=SONG_SHOT_NATIVE_SOUND_DB if p.song is not None else 0.0)


def set_plan(p: Project, folder: Path, a: SetPlan) -> None:
    changes = {}
    if a.brief is not None:
        changes["brief"] = a.brief
    if a.filmSeconds is not None:
        changes["filmSeconds"] = a.filmSeconds
    elif a.clearFilmSeconds:
        changes["filmSeconds"] = None
    if changes:
        p.plan = p.plan.model_copy(update=changes)


def apply_plan(p: Project, folder: Path, a: ApplyPlan) -> None:
    fresh = [_new_shot(p, planned) for planned in a.shots]
    if a.mode == "replace":
        locked = [i + 1 for i, s in enumerate(p.shots) if s.locked]
        if locked:
            raise OpError(f"Shot {', '.join(map(str, locked))} is locked. Unlock it or add the plan after the shots.")
        if any(c.anchor is not None for c in p.audio):
            from backend_service.storyboard.ops import ShotRef, remove_shot

            for shot in list(p.shots):  # anchored sound keeps its place in the film, as a shot removal does
                remove_shot(p, folder, ShotRef(shotId=shot.id))
        p.shots = fresh
        return
    if len(p.shots) + len(fresh) > MAX_SHOTS:
        raise OpError(f"A project holds up to {MAX_SHOTS} shots.")
    p.shots = [*p.shots, *fresh]


def plan_changes(p: Project, folder: Path, a: PlanChanges) -> None:
    """Edits, removals and additions in one step; every shot named must exist and none may be locked."""
    for change in a.edits:
        shot = _shot(p, change.shotId)
        if shot.locked:
            raise OpError(f"Shot '{shot.title or shot.id}' is locked.")
        for key in ("title", "stillPrompt", "motionPrompt", "seconds"):
            value = getattr(change, key)
            if value is not None:
                setattr(shot, key, value)
    for shot_id in a.remove:
        if _shot(p, shot_id).locked:
            raise OpError(f"Shot '{shot_id}' is locked.")
    if a.remove:
        from backend_service.storyboard.ops import ShotRef, remove_shot

        for shot_id in dict.fromkeys(a.remove):
            remove_shot(p, folder, ShotRef(shotId=shot_id))
    for added in a.add:
        if added.after is not None and p.shot(added.after) is None:
            raise OpError(f"Unknown shot '{added.after}'.")
    if len(p.shots) + len(a.add) > MAX_SHOTS:
        raise OpError(f"A project holds up to {MAX_SHOTS} shots.")
    added_ids: set[str] = set()
    for added in a.add:  # in order, so two shots after the same one keep their order
        shot = _new_shot(p, added)
        index = 0 if added.after is None else [s.id for s in p.shots].index(added.after) + 1
        while index < len(p.shots) and p.shots[index].id in added_ids:
            index += 1
        p.shots.insert(index, shot)
        added_ids.add(shot.id)


def add_plan_message(p: Project, folder: Path, a: AddPlanMessage) -> None:
    message = PlanMessage(id=f"m_{secrets.token_hex(4)}", role=a.role, text=a.text, at=time.time())
    p.planChat = [*p.planChat, message][-MAX_PLAN_CHAT:]


def clear_plan_chat(p: Project, folder: Path, a: ClearPlanChat) -> None:
    p.planChat = []


def set_brand(p: Project, folder: Path, a: SetBrand) -> None:
    brand = p.brand.model_dump()
    if a.logo:
        rel, digest, _ = _import(folder, a.logo, {"image"}, f"media/brand/logo-{int(time.time() * 1000)}-{secrets.token_hex(3)}")
        brand["logo"] = RefImage(path=rel, hash=digest).model_dump()
    elif a.clearLogo:
        brand["logo"] = None
    if a.logoCorner is not None:
        brand["logoCorner"] = a.logoCorner
    if a.styleUse is not None:
        brand["styleUse"] = a.styleUse
    if a.captions is not None:
        look = dict(brand["captions"])
        if a.captions.font is not None:
            look["font"] = a.captions.font or None
        elif a.captions.clearFont:
            look["font"] = None
        if a.captions.colour is not None:
            look["colour"] = a.captions.colour
        if a.captions.position is not None:
            look["position"] = a.captions.position
        brand["captions"] = CaptionLook.model_validate(look).model_dump()
    p.brand = Brand.model_validate(brand)


# name -> (argument model, change function, reversible); merged into ops.OPS
PLAN_OPS = {
    "set_plan": (SetPlan, set_plan, True),
    "apply_plan": (ApplyPlan, apply_plan, True),
    "plan_changes": (PlanChanges, plan_changes, True),
    "add_plan_message": (AddPlanMessage, add_plan_message, False),
    "clear_plan_chat": (ClearPlanChat, clear_plan_chat, False),
    "set_brand": (SetBrand, set_brand, True),
}
PLAN_FILE_ARGS = {"set_brand": (("logo", {"image"}),)}
