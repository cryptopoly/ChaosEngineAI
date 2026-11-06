"""Named edit operations on a project, shared by the HTTP routes and the agent tools.

Each op has a strict pydantic argument model and a function that changes the project in place; the store
runs it under the project lock with the caller's ``revision`` and records it in the history. Imports copy
the file into ``media/`` (hard link on the same disk) with a sha256 and a measured duration, so the
project never depends on a Studio output staying where it was.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend_service.storyboard import media
from backend_service.storyboard.models import (
    Anchor,
    AudioClip,
    AudioSource,
    Defaults,
    ExportSettings,
    Id,
    Line,
    Project,
    Prompt,
    Reference,
    Shot,
    Take,
    Transition,
    new_id,
)
from backend_service.storyboard.store import ProjectStore, StoreError


class OpError(StoreError):
    """A readable refusal of one op."""


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- argument models ----
class EditProject(_Args):
    name: Line | None = Field(None, min_length=1)
    defaults: dict[str, Any] | None = None
    export: dict[str, Any] | None = None


class AddReference(_Args):
    kind: Literal["character", "place"]
    name: Line = ""
    description: Prompt = ""
    image: str | None = Field(None, max_length=4096)


class EditReference(_Args):
    refId: Id
    name: Line | None = None
    description: Prompt | None = None
    image: str | None = Field(None, max_length=4096)


class RemoveReference(_Args):
    refId: Id


class ShotFields(_Args):
    title: Line | None = None
    stillPrompt: Prompt | None = None
    motionPrompt: Prompt | None = None
    referenceImages: list[Id] | None = Field(None, max_length=8)
    startFrom: Literal["generate", "upload", "previous-shot"] | None = None
    seconds: float | None = Field(None, ge=0.5, le=60)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    trimIn: float | None = Field(None, ge=0, le=3600)
    trimOut: float | None = Field(None, ge=0, le=3600)
    speed: float | None = Field(None, ge=0.25, le=4.0)
    transition: Transition | None = None
    nativeSoundDb: float | None = Field(None, ge=-60, le=12)


class AddShot(ShotFields):
    index: int | None = Field(None, ge=0)


class EditShot(ShotFields):
    shotId: Id


class ShotRef(_Args):
    shotId: Id


class Reorder(_Args):
    order: list[Id] = Field(max_length=500)


class Lock(_Args):
    shotId: Id
    locked: bool = True


class PickTake(_Args):
    shotId: Id
    takeId: Id


class MarkRedo(_Args):
    shotId: Id
    redo: bool = True


class ImportTake(_Args):
    shotId: Id
    clip: str | None = Field(None, max_length=4096)
    still: str | None = Field(None, max_length=4096)


class Relink(_Args):
    id: Id
    path: str = Field(min_length=1, max_length=4096)


class AudioFields(_Args):
    anchor: Anchor | None = None
    start: float | None = Field(None, ge=0, le=3600)
    trimIn: float | None = Field(None, ge=0, le=3600)
    trimOut: float | None = Field(None, ge=0, le=3600)
    gainDb: float | None = Field(None, ge=-60, le=12)
    fadeIn: float | None = Field(None, ge=0, le=60)
    fadeOut: float | None = Field(None, ge=0, le=60)
    duck: bool | None = None
    loop: bool | None = None
    label: Line | None = None


class AddAudio(AudioFields):
    kind: Literal["music", "sfx", "voice", "import"]
    path: str = Field(min_length=1, max_length=4096)
    clonedVoice: bool = False


class EditAudio(AudioFields):
    clipId: Id


class ClipRef(_Args):
    clipId: Id


# ---- helpers ----
def _shot(project: Project, shot_id: str) -> Shot:
    shot = project.shot(shot_id)
    if shot is None:
        raise OpError(f"Unknown shot '{shot_id}'.")
    return shot


def _clip(project: Project, clip_id: str) -> AudioClip:
    clip = project.clip(clip_id)
    if clip is None:
        raise OpError(f"Unknown audio clip '{clip_id}'.")
    return clip


def _set_fields(target: BaseModel, args: BaseModel, skip: set[str]) -> None:
    for key in args.model_fields_set - skip:
        try:
            setattr(target, key, getattr(args, key))
        except ValidationError as exc:
            raise OpError(f"{key}: {exc.errors()[0]['msg']}") from exc


def _import(folder: Path, raw: str, allowed: set[str], rel_stem: str) -> tuple[str, str, Path]:
    """Copy a source file in; returns (project-relative path, sha256, real source path)."""
    try:
        src = media.resolve_source(raw, allowed)
        rel = f"{rel_stem}{src.suffix.lower()}"
        return rel, media.ingest(folder, src, rel), src
    except media.MediaError as exc:
        raise OpError(str(exc)) from exc


def _seconds(folder: Path, rel: str) -> float | None:
    return media.probe_seconds(media.inside(folder, rel))


# ---- ops ----
def edit_project(p: Project, folder: Path, a: EditProject) -> None:
    if a.name is not None:
        p.name = a.name
    try:
        if a.defaults is not None:
            p.defaults = Defaults.model_validate({**p.defaults.model_dump(), **a.defaults})
        if a.export is not None:
            merged = {**p.export.model_dump(), **a.export}
            if isinstance(a.export.get("aiLabel"), dict):
                merged["aiLabel"] = {**p.export.aiLabel.model_dump(), **a.export["aiLabel"]}
            p.export = ExportSettings.model_validate(merged)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise OpError(f"{'.'.join(str(x) for x in first['loc'])}: {first['msg']}") from exc


def _ref_image(folder: Path, ref: Reference, raw: str) -> None:
    rel, digest, _ = _import(folder, raw, {"image"}, f"media/refs/{ref.id}-{int(time.time() * 1000)}")
    ref.image, ref.hash = rel, digest


def add_reference(p: Project, folder: Path, a: AddReference) -> None:
    ref = Reference(id=new_id("r"), name=a.name, description=a.description)
    if a.image:
        _ref_image(folder, ref, a.image)
    (p.references.characters if a.kind == "character" else p.references.places).append(ref)


def edit_reference(p: Project, folder: Path, a: EditReference) -> None:
    ref = p.reference(a.refId)
    if ref is None:
        raise OpError(f"Unknown reference '{a.refId}'.")
    _set_fields(ref, a, {"refId", "image"})
    if a.image:
        _ref_image(folder, ref, a.image)


def remove_reference(p: Project, folder: Path, a: RemoveReference) -> None:
    if p.reference(a.refId) is None:
        raise OpError(f"Unknown reference '{a.refId}'.")
    p.references.characters = [r for r in p.references.characters if r.id != a.refId]
    p.references.places = [r for r in p.references.places if r.id != a.refId]
    for shot in p.shots:
        if a.refId in shot.referenceImages:
            shot.referenceImages = [r for r in shot.referenceImages if r != a.refId]


def _check_refs(p: Project, ids: list[str] | None) -> None:
    for ref_id in ids or []:
        if p.reference(ref_id) is None:
            raise OpError(f"Unknown reference '{ref_id}'. Add it to the project references first.")


def add_shot(p: Project, folder: Path, a: AddShot) -> None:
    _check_refs(p, a.referenceImages)
    shot = Shot(id=new_id("s"))
    _set_fields(shot, a, {"index"})
    index = len(p.shots) if a.index is None else min(a.index, len(p.shots))
    p.shots.insert(index, shot)


def edit_shot(p: Project, folder: Path, a: EditShot) -> None:
    _check_refs(p, a.referenceImages)
    _set_fields(_shot(p, a.shotId), a, {"shotId"})


def remove_shot(p: Project, folder: Path, a: ShotRef) -> None:
    from backend_service.storyboard.timeline import compute_timeline

    _shot(p, a.shotId)
    span = compute_timeline(p).shot(a.shotId)
    for clip in p.audio:  # anchored sound stays where it was in the film
        if clip.anchor is not None and clip.anchor.shotId == a.shotId:
            start = max(0.0, (span.start if span else 0.0) + clip.anchor.offset)
            clip.anchor = None
            clip.start = start
    p.shots = [s for s in p.shots if s.id != a.shotId]  # media stays on disk for undo


def reorder(p: Project, folder: Path, a: Reorder) -> None:
    if sorted(a.order) != sorted(s.id for s in p.shots):
        raise OpError("order must list every shot id exactly once.")
    by_id = {s.id: s for s in p.shots}
    p.shots = [by_id[i] for i in a.order]


def lock(p: Project, folder: Path, a: Lock) -> None:
    _shot(p, a.shotId).locked = a.locked


def pick_take(p: Project, folder: Path, a: PickTake) -> None:
    shot = _shot(p, a.shotId)
    if shot.take(a.takeId) is None:
        raise OpError(f"Shot {a.shotId} has no take '{a.takeId}'.")
    shot.activeTake = a.takeId
    shot.status = "done"


def mark_redo(p: Project, folder: Path, a: MarkRedo) -> None:
    _shot(p, a.shotId).redo = a.redo


def import_take(p: Project, folder: Path, a: ImportTake) -> None:
    if not a.clip and not a.still:
        raise OpError("Give a clip, a still or both.")
    shot = _shot(p, a.shotId)
    take = Take(id=new_id("t"), createdAt=time.time(), origin="upload")
    stem = f"media/{shot.id}/{take.id}"
    if a.still:
        take.still, digest, _ = _import(folder, a.still, {"image"}, f"{stem}/still")
        take.hashes = {**take.hashes, "still": digest}
    if a.clip:
        take.clip, digest, _ = _import(folder, a.clip, {"video"}, f"{stem}/clip")
        take.hashes = {**take.hashes, "clip": digest}
        take.measuredSeconds = _seconds(folder, take.clip)
    shot.takes.append(take)
    shot.activeTake = take.id
    shot.status = "done"
    shot.redo = False


def relink(p: Project, folder: Path, a: Relink) -> None:
    shot, clip = p.shot(a.id), p.clip(a.id)
    if shot is None and clip is None:
        raise OpError(f"Unknown shot or audio clip '{a.id}'.")
    stamp = int(time.time() * 1000)
    if clip is not None:
        rel, digest, _ = _import(folder, a.path, {"audio", "video"}, f"media/audio/{clip.id}-{stamp}")
        clip.source = AudioSource(**{**clip.source.model_dump(), "path": rel, "hash": digest,
                                     "measuredSeconds": _seconds(folder, rel)})
        clip.status = "ok"
        return
    assert shot is not None
    take = shot.active()
    if take is None:
        raise OpError(f"Shot {a.id} has no take to relink; import one instead.")
    try:
        kind = media.kind_of(media.resolve_source(a.path, {"image", "video"}))
    except media.MediaError as exc:
        raise OpError(str(exc)) from exc
    stem = f"media/{shot.id}/{take.id}"
    if kind == "video":
        take.clip, digest, _ = _import(folder, a.path, {"video"}, f"{stem}/clip-{stamp}")
        take.hashes = {**take.hashes, "clip": digest}
        take.measuredSeconds = _seconds(folder, take.clip)
    else:
        take.still, digest, _ = _import(folder, a.path, {"image"}, f"{stem}/still-{stamp}")
        take.hashes = {**take.hashes, "still": digest}
    shot.status = "done"


def _position(clip_fields: AudioFields, clip: AudioClip | None) -> None:
    if clip_fields.anchor is not None and clip_fields.start is not None:
        raise OpError("Give an anchor or a start, not both.")


def add_audio(p: Project, folder: Path, a: AddAudio) -> None:
    _position(a, None)
    if a.anchor is not None:
        _shot(p, a.anchor.shotId)
    clip_id = new_id("a")
    rel, digest, _ = _import(folder, a.path, {"audio", "video"}, f"media/audio/{clip_id}")
    source = AudioSource(path=rel, hash=digest, measuredSeconds=_seconds(folder, rel),
                         clonedVoice=a.clonedVoice, label=a.label or "")
    clip = AudioClip(id=clip_id, kind=a.kind, source=source, anchor=a.anchor,
                     start=a.start if a.anchor is None else None)
    if clip.anchor is None and clip.start is None:
        clip.start = 0.0
    _set_fields(clip, a, {"kind", "path", "clonedVoice", "label", "anchor", "start"})
    p.audio.append(clip)


def edit_audio(p: Project, folder: Path, a: EditAudio) -> None:
    _position(a, None)
    clip = _clip(p, a.clipId)
    if a.anchor is not None:
        _shot(p, a.anchor.shotId)
        clip.start = None
        clip.anchor = a.anchor
    elif "start" in a.model_fields_set and a.start is not None:
        clip.anchor = None
        clip.start = a.start
    if a.label is not None:
        clip.source = AudioSource(**{**clip.source.model_dump(), "label": a.label})
    _set_fields(clip, a, {"clipId", "anchor", "start", "label"})


def remove_audio(p: Project, folder: Path, a: ClipRef) -> None:
    _clip(p, a.clipId)
    p.audio = [c for c in p.audio if c.id != a.clipId]


# name -> (argument model, change function, reversible)
OPS: dict[str, tuple[type[_Args], Callable[[Project, Path, Any], None], bool]] = {
    "edit_project": (EditProject, edit_project, True),
    "add_reference": (AddReference, add_reference, True),
    "edit_reference": (EditReference, edit_reference, True),
    "remove_reference": (RemoveReference, remove_reference, True),
    "add_shot": (AddShot, add_shot, True),
    "edit_shot": (EditShot, edit_shot, True),
    "remove_shot": (ShotRef, remove_shot, True),
    "reorder": (Reorder, reorder, True),
    "lock": (Lock, lock, True),
    "pick_take": (PickTake, pick_take, True),
    "mark_redo": (MarkRedo, mark_redo, True),
    "import_take": (ImportTake, import_take, True),
    "relink": (Relink, relink, False),
    "add_audio": (AddAudio, add_audio, True),
    "edit_audio": (EditAudio, edit_audio, True),
    "remove_audio": (ClipRef, remove_audio, True),
}


def parse_args(op: str, args: dict[str, Any]) -> Any:
    if op not in OPS:
        raise OpError(f"Unknown op '{op}'. Ops: {', '.join(OPS)}.")
    try:
        return OPS[op][0].model_validate(args or {})
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(x) for x in first.get("loc", ()))
        raise OpError(f"{op}: {where + ': ' if where else ''}{first['msg']}") from exc


def run_op(store: ProjectStore, project_id: str, revision: int, op: str, args: dict[str, Any]) -> Project:
    """Validate the args, then apply the op under the project lock against ``revision``."""
    parsed = parse_args(op, args)
    _, change, reversible = OPS[op]
    return store.mutate(project_id, revision, lambda p, folder: change(p, folder, parsed), op, reversible)
