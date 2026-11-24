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

SCHEMA_VERSION = 3
MAX_SHOTS = 500
MAX_AUDIO = 500
MAX_TAKES = 50
MAX_CAPTIONS = 2000
MAX_REF_IMAGES = 4  # pictures per character or place (front, side, detail...)
MAX_PALETTE = 8  # colours kept per style reference (S15)
MAX_PLAN_CHAT = 60  # direction chat messages kept with the project (S8)
MAX_SONG_SECONDS = 1200.0  # a song a project can be cut to (S-M1): 20 minutes
MAX_BEATS = 8000  # 20 minutes at 400 beats a minute
MAX_SECTIONS = 200
MAX_LYRICS = 500
BOARD_EXTENT = 100000.0  # how far a freestyle board card can sit from the origin, in board pixels (S-M2)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PROJECT_ID_RE = re.compile(r"^[0-9a-f]{16}$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_LINE_BREAK_RE = re.compile(r"[\r\n\t]")
HEX_COLOUR_RE = re.compile(r"^#[0-9a-f]{6}$")


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
HexColour = Annotated[str, Field(pattern=HEX_COLOUR_RE.pattern)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Defaults(_Model):
    width: int = Field(1280, ge=64, le=4096)
    height: int = Field(720, ge=64, le=4096)
    fps: int = Field(24, ge=1, le=60)
    imageModel: Line | None = None  # stills for Make and Make drafts; None picks the fastest installed model (S2)
    finalImageModel: Line | None = None  # stills for the final pass on kept shots; None picks the best-quality one
    videoModel: Line | None = None
    voice: Line | None = None
    exportPreset: Literal["16:9", "9:16", "1:1"] = "16:9"  # the aspect ratio (called "shape" before D-64)
    # A preset ("480p" to "2160p"), another short side ("576p" for 1024x576) or "custom": always the label of
    # width x height (formats.resolution_label), so it never contradicts the real size (FU-213).
    resolution: Annotated[str, Field(pattern=r"^([1-9][0-9]{1,3}p|custom)$")] = "720p"

    @model_validator(mode="before")
    @classmethod
    def _format(cls, data: Any) -> Any:
        from backend_service.storyboard.formats import normalize_defaults

        return normalize_defaults(data)


class RefImage(_Model):
    path: RelPath
    hash: Sha256 | None = None


class Reference(_Model):
    """A character, place or style: a name, a few words and up to four pictures (format 3; format 2 held one
    ``image``). A style also keeps the main colours read from its pictures (``palette``, S15). A character with
    ``inEveryShot`` is in every shot unless a shot leaves it out (``Shot.castOut``, B1309; ``cast.py``)."""

    id: Id
    name: Line = ""
    description: Prompt = ""
    images: list[RefImage] = Field(default_factory=list, max_length=MAX_REF_IMAGES)
    palette: list[HexColour] = Field(default_factory=list, max_length=MAX_PALETTE)
    inEveryShot: bool = False  # characters only (the ops refuse it on a place or style)


class References(_Model):
    characters: list[Reference] = Field(default_factory=list, max_length=50)
    places: list[Reference] = Field(default_factory=list, max_length=50)
    styles: list[Reference] = Field(default_factory=list, max_length=20)  # the look of the film (S15)

    def all(self) -> list[Reference]:
        return self.characters + self.places + self.styles


class CaptionLook(_Model):
    """How captions look when they are drawn into the picture: a font family from this machine (None: the
    default), a colour and a place on the frame."""

    font: Line | None = None
    colour: HexColour = "#ffffff"
    position: Literal["bottom", "middle", "top"] = "bottom"


class Brand(_Model):
    """The project's small brand kit (S15): an optional logo, the caption look, and how the style references
    reach the still model (``auto``: as reference pictures when the model takes them, else as words)."""

    logo: RefImage | None = None
    logoCorner: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-right"
    captions: CaptionLook = Field(default_factory=CaptionLook)
    styleUse: Literal["auto", "words", "off"] = "auto"


class Plan(_Model):
    """The idea behind the film (S8): a free-text brief and the length the user aims for."""

    brief: Prompt = ""
    filmSeconds: float | None = Field(None, ge=5, le=600)


class PlanMessage(_Model):
    """One line of the project's direction chat with the chat model."""

    id: Id
    role: Literal["user", "assistant"]
    text: Prompt
    at: float = 0.0


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
    renderPass: Literal["draft", "final"] = "final"  # a quick draft or a full take; takes saved before S6 are final


ShotStatus = Literal["idle", "queued", "running", "done", "failed", "stale", "missing"]


# A film cut to a song plays the song, not each clip's own generated sound: at 0 dB that sound buries the song and
# changes with every cut (measured on a 30 s music video: the film's level followed the song with a correlation of 0.05,
# and 1.0 with the clips' sound muted). A shot made under a song starts muted; ``nativeSoundDb`` still sets it.
SONG_SHOT_NATIVE_SOUND_DB = -60.0


class Shot(_Model):
    """One shot. Its cast (B1309): the references it ticks (``referenceImages``) plus every ``inEveryShot`` character
    not in ``castOut``; the ops keep the two lists apart, and ``cast.effective_references`` is the only resolver."""

    id: Id
    order: int = Field(0, ge=0)
    title: Line = ""
    status: ShotStatus = "idle"
    stillPrompt: Prompt = ""
    motionPrompt: Prompt = ""
    referenceImages: list[Id] = Field(default_factory=list, max_length=8)  # ids in project.references
    castOut: list[Id] = Field(default_factory=list, max_length=50)  # inEveryShot characters this shot leaves out
    # "text": no picture at all, a text-to-video clip made from the motion prompt, on any video model.
    startFrom: Literal["generate", "upload", "previous-shot", "text"] = "generate"
    endImage: RelPath | None = None
    # Lip sync (D-103): a vocal clip (WAV or FLAC) the clip's mouth follows; the clip is made by an LTX-2.3 lip-sync run
    # from the shot's still, 3 to 8 s, and carries this audio as its sound.
    vocalClip: RelPath | None = None
    # FU-373: where in the vocal file the clip starts (``audioStartSeconds``); FU-380 / D-110: a sung vocal gets the
    # singing wording and, by default, 3 takes kept by the sync score (``vocalTakes`` None = that default).
    vocalStart: float = Field(0.0, ge=0, le=3600)
    vocalSinging: bool = False
    vocalTakes: int | None = Field(None, ge=1, le=4)
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
    kept: bool = False  # chosen for the final pass (S6)

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


def _drop_legacy(data: Any, keys: tuple[str, ...]) -> Any:
    """Projects saved before the end card was removed still carry its fields; ignore them."""
    if isinstance(data, dict) and any(k in data for k in keys):
        return {k: v for k, v in data.items() if k not in keys}
    return data


class AiLabel(_Model):
    embedTag: bool = True

    @model_validator(mode="before")
    @classmethod
    def _drop_end_card(cls, data: Any) -> Any:
        return _drop_legacy(data, ("endCard",))


class ExportSettings(_Model):
    aiLabel: AiLabel = Field(default_factory=AiLabel)
    loudnessLufs: float | None = Field(-16.0, ge=-40, le=-5)
    captionsSrt: bool = False  # write the captions as an .srt next to the film
    captionsBurn: bool = False  # draw the captions into the picture
    stems: bool = False  # also write each audio lane as its own .wav
    upscale: Literal["none", "1440p", "2160p"] = "none"  # scale the final film up (lanczos); never below the project size
    # The brand kit on the film (S15): the logo in its corner, a title card first and an end card last.
    showLogo: bool = True  # draw brand.logo in brand.logoCorner when there is one
    titleCard: bool = False
    endCard: bool = False
    titleText: Line = ""  # empty: the project name
    endText: Line = ""  # empty: the project name
    cardSeconds: float = Field(2.0, ge=1.0, le=5.0)

    @model_validator(mode="before")
    @classmethod
    def _drop_end_card(cls, data: Any) -> Any:
        return _drop_legacy(data, ("endCardText", "endCardSeconds"))


class Caption(_Model):
    """One caption. With ``clipId`` the times are seconds in that audio clip's source (so the caption follows
    the clip when it moves or is trimmed); without it they are seconds on the film's timeline."""

    id: Id
    clipId: Id | None = None
    start: Annotated[float, Field(ge=0, le=36000)]
    end: Annotated[float, Field(ge=0, le=36000)]
    text: Annotated[str, Field(min_length=1, max_length=500), AfterValidator(check_text)]

    @model_validator(mode="after")
    def _ordered(self) -> "Caption":
        if self.end <= self.start:
            raise ValueError("a caption must end after it starts")
        return self


class BatchResult(_Model):
    shotId: Id
    kind: Literal["still", "clip"]
    state: Literal["queued", "running", "done", "failed", "skipped", "cancelled"] = "queued"
    takeId: Id | None = None
    error: Annotated[str, Field(max_length=1000)] | None = None


class SongSection(_Model):
    """One part of the song (S-M1): seconds in the song, a short label and how loud it is within the song (0 to 1)."""

    start: Annotated[float, Field(ge=0, le=MAX_SONG_SECONDS)]
    end: Annotated[float, Field(ge=0, le=MAX_SONG_SECONDS)]
    label: Annotated[str, Field(max_length=40), AfterValidator(check_line)] = ""
    energy: float | None = Field(None, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> "SongSection":
        if self.end <= self.start:
            raise ValueError("a section must end after it starts")
        return self


class Song(_Model):
    """The song a music video is cut to (S-M1): its file in the project, what the analysis found (tempo, beats,
    sections, all estimates: ``rough`` and ``note`` say so), the lyrics, and the mixer clip it was put on."""

    path: RelPath
    hash: Sha256 | None = None
    title: Line = ""
    seconds: float = Field(gt=0, le=MAX_SONG_SECONDS)
    bpm: float | None = Field(None, ge=30, le=400)
    beats: list[Annotated[float, Field(ge=0, le=MAX_SONG_SECONDS)]] = Field(default_factory=list, max_length=MAX_BEATS)
    sections: list[SongSection] = Field(default_factory=list, max_length=MAX_SECTIONS)
    lyrics: list[Line] = Field(default_factory=list, max_length=MAX_LYRICS)
    clipId: Id | None = None
    libraryId: Annotated[str, Field(pattern=r"^\d+_[0-9a-f]{8}$")] | None = None
    rough: bool = True
    note: Annotated[str, Field(max_length=600), AfterValidator(check_text)] = ""
    analysedAt: float = 0.0

    @model_validator(mode="after")
    def _ordered(self) -> "Song":
        if any(b < a for a, b in zip(self.beats, self.beats[1:])):
            raise ValueError("beats must be in time order")
        if any(b.start < a.start for a, b in zip(self.sections, self.sections[1:])):
            raise ValueError("sections must be in time order")
        if any(s.end > self.seconds + 0.5 for s in self.sections):
            raise ValueError("a section ends after the song")
        return self


BatchState = Literal["queued", "running", "paused", "done", "failed", "cancelled", "interrupted"]


class BatchRecord(_Model):
    id: Id
    # final: kept shots again with the quality model, then clips; redo-clip / redo-still: one half again, new seed (S11)
    what: Literal["still", "clip", "both", "final", "redo-clip", "redo-still"]
    force: bool = False
    draft: bool = False  # a "Make drafts" run: the fast model at a smaller size
    # paused: stopped between two steps (Pause, or the app stopped while it ran); its queued results are the
    # steps a resume runs. interrupted: stopped with nothing left it can resume (S3).
    state: BatchState = "queued"
    createdAt: float = 0.0
    finishedAt: float | None = None
    results: list[BatchResult] = Field(default_factory=list, max_length=2 * MAX_SHOTS)
    pausedAt: float | None = None
    pauseReason: Literal["user", "restart"] | None = None
    resumedAt: float | None = None
    resumes: int = Field(0, ge=0, le=10000)


class BoardPos(_Model):
    """Where one card sits on the freestyle board (S-M2), in board pixels at zoom 1."""

    x: float = Field(0.0, ge=-BOARD_EXTENT, le=BOARD_EXTENT)
    y: float = Field(0.0, ge=-BOARD_EXTENT, le=BOARD_EXTENT)


class Board(_Model):
    """The freestyle board (S-M2): card positions by shot or reference id. A card with no position is laid out
    by the app; positions of shots or references that are gone are ignored (and dropped on the next move)."""

    pos: dict[Id, BoardPos] = Field(default_factory=dict, max_length=2 * MAX_SHOTS)


class Project(_Model):
    schemaVersion: int = SCHEMA_VERSION
    revision: int = Field(1, ge=1)
    id: Annotated[str, Field(pattern=PROJECT_ID_RE.pattern)]
    name: Line
    createdAt: float = 0.0
    defaults: Defaults = Field(default_factory=Defaults)
    references: References = Field(default_factory=References)
    plan: Plan = Field(default_factory=Plan)
    brand: Brand = Field(default_factory=Brand)
    planChat: list[PlanMessage] = Field(default_factory=list, max_length=MAX_PLAN_CHAT)  # not undoable
    song: Song | None = None  # the song a music video is cut to (S-M1)
    board: Board = Field(default_factory=Board)  # freestyle board card positions (S-M2)
    shots: list[Shot] = Field(default_factory=list, max_length=MAX_SHOTS)
    audio: list[AudioClip] = Field(default_factory=list, max_length=MAX_AUDIO)
    export: ExportSettings = Field(default_factory=ExportSettings)
    captions: list[Caption] = Field(default_factory=list, max_length=MAX_CAPTIONS)
    batches: list[BatchRecord] = Field(default_factory=list, max_length=50)
    undoStack: list[int] = Field(default_factory=list, max_length=200)
    redoStack: list[int] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Project":
        for label, ids in (("shot", [s.id for s in self.shots]), ("audio clip", [a.id for a in self.audio])):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} id")
        if len({c.id for c in self.captions}) != len(self.captions):
            raise ValueError("duplicate caption id")
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
        return next((r for r in self.references.all() if r.id == ref_id), None)

    def has_cloned_voice(self) -> bool:
        return any(a.source.clonedVoice for a in self.audio)

    def embed_tag(self) -> bool:
        """D-34: the MP4 tag is on by default; any cloned-voice clip forces it on whatever the switch says."""
        return self.export.aiLabel.embedTag or self.has_cloned_voice()


def dump(project: Project) -> dict[str, Any]:
    return project.model_dump(mode="json")
