"""The storyboard project schema (``project.json``) as pydantic models, strict because the file is hostile input.

A project may come from another machine, a bundle or an agent, so every field is bounded, unknown fields
are refused, ids are short safe tokens (they become folder names) and every stored path is project-relative
under ``media/`` with no way to climb out, name a drive, a URL scheme or an option for a command line.
``schemaVersion`` is checked by the store; newer files are refused and older ones are migrated first.
"""

from __future__ import annotations

import re
import secrets
from typing import Annotated, Any, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
MAX_SHOTS = 500
MAX_AUDIO = 500
MAX_TAKES = 50

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PROJECT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_LINE_BREAK_RE = re.compile(r"[\r\n\t]")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def new_project_id() -> str:
    return secrets.token_hex(8)


def check_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError("ids are 1 to 64 letters, digits, '_' or '-' and do not start with '-'")
    return value


def check_rel_path(value: str) -> str:
    """A project-relative path under ``media/``. Raises ValueError for anything else."""
    if not value or len(value) > 255:
        raise ValueError("path must be 1 to 255 characters")
    if _CONTROL_RE.search(value) or _LINE_BREAK_RE.search(value):
        raise ValueError("path must not contain control characters or line breaks")
    if "\\" in value or value.startswith("/") or _SCHEME_RE.match(value):
        raise ValueError("path must be project-relative (no absolute path, drive letter or URL)")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("path must not contain empty, '.' or '..' parts")
    if any(part.startswith("-") for part in parts):
        raise ValueError("path parts must not start with '-'")
    if parts[0] != "media" or len(parts) < 2:
        raise ValueError("media paths must sit under media/")
    return value


def check_line(value: str) -> str:
    if _CONTROL_RE.search(value) or _LINE_BREAK_RE.search(value):
        raise ValueError("text must be one line without control characters")
    return value


def check_text(value: str) -> str:
    if _CONTROL_RE.search(value):
        raise ValueError("text must not contain control characters")
    return value


def check_hash(value: str) -> str:
    if not _HASH_RE.match(value):
        raise ValueError("hash must be a lowercase sha256 hex digest")
    return value


Id = Annotated[str, AfterValidator(check_id)]
RelPath = Annotated[str, AfterValidator(check_rel_path)]
Line = Annotated[str, Field(max_length=200), AfterValidator(check_line)]
Prompt = Annotated[str, Field(max_length=4000), AfterValidator(check_text)]
Sha256 = Annotated[str, AfterValidator(check_hash)]
Scalar = Union[bool, int, float, Annotated[str, Field(max_length=4000), AfterValidator(check_text)], None]
Seconds = Annotated[float, Field(ge=0, le=3600)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Defaults(_Model):
    width: int = Field(1280, ge=64, le=4096)
    height: int = Field(720, ge=64, le=4096)
    fps: int = Field(24, ge=1, le=60)
    imageModel: Line | None = None
    videoModel: Line | None = None
    voice: Line | None = None
    exportPreset: Literal["16:9", "9:16", "1:1"] = "16:9"


class Reference(_Model):
    id: Id
    name: Line = ""
    description: Prompt = ""
    image: RelPath | None = None
    hash: Sha256 | None = None


class References(_Model):
    characters: list[Reference] = Field(default_factory=list, max_length=50)
    places: list[Reference] = Field(default_factory=list, max_length=50)


class Transition(_Model):
    type: Literal["cut", "crossfade"] = "cut"
    seconds: float = Field(0.0, ge=0, le=5)


class Take(_Model):
    id: Id
    still: RelPath | None = None
    clip: RelPath | None = None
    hashes: dict[Literal["still", "clip"], Sha256] = Field(default_factory=dict)
    settings: dict[Annotated[str, Field(max_length=64)], Scalar] = Field(default_factory=dict, max_length=40)
    measuredSeconds: float | None = Field(None, ge=0, le=3600)
    createdAt: float = 0.0
    origin: Literal["generate", "upload", "batch"] = "upload"
    basis: Annotated[str, Field(max_length=128)] | None = None  # fingerprint of the inputs (deps.py)


ShotStatus = Literal["idle", "queued", "running", "done", "failed", "stale", "missing"]


class Shot(_Model):
    id: Id
    order: int = Field(0, ge=0)
    title: Line = ""
    status: ShotStatus = "idle"
    stillPrompt: Prompt = ""
    motionPrompt: Prompt = ""
    referenceImages: list[Id] = Field(default_factory=list, max_length=8)  # ids in project.references
    startFrom: Literal["generate", "upload", "previous-shot"] = "generate"
    endImage: RelPath | None = None
    seconds: float = Field(4.0, ge=0.5, le=60)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    locked: bool = False
    trimIn: Seconds = 0.0
    trimOut: Seconds | None = None  # out point in source seconds; None = the end of the take
    speed: float = Field(1.0, ge=0.25, le=4.0)
    takes: list[Take] = Field(default_factory=list, max_length=MAX_TAKES)
    activeTake: Id | None = None
    transition: Transition = Field(default_factory=Transition)  # how this shot comes in from the previous one
    nativeSoundDb: float = Field(0.0, ge=-60, le=12)
    redo: bool = False  # marked for the next batch

    def take(self, take_id: str | None) -> Take | None:
        return next((t for t in self.takes if t.id == take_id), None) if take_id else None

    def active(self) -> Take | None:
        return self.take(self.activeTake)


class AudioSource(_Model):
    path: RelPath
    hash: Sha256 | None = None
    measuredSeconds: float | None = Field(None, ge=0, le=36000)
    clonedVoice: bool = False
    label: Line = ""


class Anchor(_Model):
    shotId: Id
    offset: float = Field(0.0, ge=-60, le=3600)


class AudioClip(_Model):
    id: Id
    kind: Literal["music", "sfx", "voice", "import"]
    source: AudioSource
    anchor: Anchor | None = None
    start: Seconds | None = None
    trimIn: Seconds = 0.0
    trimOut: Seconds | None = None
    gainDb: float = Field(0.0, ge=-60, le=12)
    fadeIn: float = Field(0.0, ge=0, le=60)
    fadeOut: float = Field(0.0, ge=0, le=60)
    duck: bool = False
    loop: bool = False
    status: Literal["ok", "missing"] = "ok"

    @model_validator(mode="after")
    def _one_position(self) -> "AudioClip":
        if self.anchor is not None and self.start is not None:
            raise ValueError("an audio clip has an anchor or a start, not both")
        return self


class AiLabel(_Model):
    embedTag: bool = True
    endCard: bool = False


class ExportSettings(_Model):
    aiLabel: AiLabel = Field(default_factory=AiLabel)
    loudnessLufs: float | None = Field(-16.0, ge=-40, le=-5)
    endCardText: Line = "Made with AI"
    endCardSeconds: float = Field(2.0, ge=0.5, le=10)


class BatchResult(_Model):
    shotId: Id
    kind: Literal["still", "clip"]
    state: Literal["queued", "running", "done", "failed", "skipped", "cancelled"] = "queued"
    takeId: Id | None = None
    error: Annotated[str, Field(max_length=1000)] | None = None


class BatchRecord(_Model):
    id: Id
    what: Literal["still", "clip", "both"]
    force: bool = False
    state: Literal["queued", "running", "done", "failed", "cancelled", "interrupted"] = "queued"
    createdAt: float = 0.0
    finishedAt: float | None = None
    results: list[BatchResult] = Field(default_factory=list, max_length=2 * MAX_SHOTS)


class Project(_Model):
    schemaVersion: int = SCHEMA_VERSION
    revision: int = Field(1, ge=1)
    id: Annotated[str, Field(pattern=PROJECT_ID_RE.pattern)]
    name: Line
    createdAt: float = 0.0
    defaults: Defaults = Field(default_factory=Defaults)
    references: References = Field(default_factory=References)
    shots: list[Shot] = Field(default_factory=list, max_length=MAX_SHOTS)
    audio: list[AudioClip] = Field(default_factory=list, max_length=MAX_AUDIO)
    export: ExportSettings = Field(default_factory=ExportSettings)
    batches: list[BatchRecord] = Field(default_factory=list, max_length=50)
    undoStack: list[int] = Field(default_factory=list, max_length=200)
    redoStack: list[int] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Project":
        for label, ids in (("shot", [s.id for s in self.shots]), ("audio clip", [a.id for a in self.audio])):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} id")
        for shot in self.shots:
            if len({t.id for t in shot.takes}) != len(shot.takes):
                raise ValueError(f"duplicate take id in shot {shot.id}")
            if shot.activeTake is not None and shot.active() is None:
                raise ValueError(f"shot {shot.id} names an active take it does not have")
        return self

    def shot(self, shot_id: str) -> Shot | None:
        return next((s for s in self.shots if s.id == shot_id), None)

    def clip(self, clip_id: str) -> AudioClip | None:
        return next((a for a in self.audio if a.id == clip_id), None)

    def reference(self, ref_id: str) -> Reference | None:
        return next((r for r in self.references.characters + self.references.places if r.id == ref_id), None)

    def has_cloned_voice(self) -> bool:
        return any(a.source.clonedVoice for a in self.audio)

    def embed_tag(self) -> bool:
        """D-34: the MP4 tag is on by default; any cloned-voice clip forces it on whatever the switch says."""
        return self.export.aiLabel.embedTag or self.has_cloned_voice()


def dump(project: Project) -> dict[str, Any]:
    return project.model_dump(mode="json")
