"""The render plan: everything the assemble engine needs to make one film, already resolved.

The project store turns a project folder into a ``RenderPlan`` (absolute paths, absolute audio start
times, measured nothing yet); the engine in ``assemble.py`` turns the plan into an MP4. The plan is
plain dataclasses with ``from_dict`` / ``to_dict`` so it can cross a process or a JSON file. Every
value is checked here, at the boundary, because a plan can be built from a ``project.json`` that came
from another machine or an agent: input paths must be absolute local regular files (never URLs, never
starting with ``-``, never with a line break), numbers must be finite and in range, and free text
that later reaches an ffmpeg filter is limited to printable characters.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUDIO_KINDS = ("music", "sfx", "voice", "import")
TRANSITION_TYPES = ("cut", "crossfade")

_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
# A URL or protocol prefix ("http:", "concat:", "file:"). One letter is a Windows drive ("C:\\").
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+:")

MAX_END_CARD_CHARS = 200
MAX_METADATA_BYTES = 8192


class PlanError(ValueError):
    """A plan field is missing, out of range or unsafe. The message names the field."""


def _fail(where: str, message: str) -> PlanError:
    return PlanError(f"{where}: {message}")


def _check_path_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(where, "must be a non-empty path string")
    if any(ch in value for ch in "\n\r\x00"):
        raise _fail(where, "path must not contain a line break or NUL")
    if value.startswith("-"):
        raise _fail(where, "path must not start with '-'")
    if _SCHEME.match(value) or "://" in value:
        raise _fail(where, "must be a local file path, not a URL")
    if not Path(value).is_absolute():
        raise _fail(where, "path must be absolute")
    return value


def check_input_file(value: Any, where: str) -> str:
    """An absolute path to an existing regular file; returns it resolved."""
    text = _check_path_text(value, where)
    path = Path(text)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise _fail(where, f"file not found: {path.name}") from exc
    if not resolved.is_file():
        raise _fail(where, f"not a regular file: {path.name}")
    return str(resolved)


def check_output_path(value: Any, where: str, suffix: str | None = None) -> str:
    text = _check_path_text(value, where)
    path = Path(text)
    if suffix and path.suffix.lower() != suffix:
        raise _fail(where, f"must end in {suffix}")
    if not path.parent.is_dir():
        raise _fail(where, "the folder for this path does not exist")
    if path.exists() and not path.is_file():
        raise _fail(where, "exists and is not a regular file")
    return text


def check_dir_path(value: Any, where: str) -> str:
    text = _check_path_text(value, where)
    path = Path(text)
    if path.exists() and not path.is_dir():
        raise _fail(where, "exists and is not a folder")
    return text


def _num(value: Any, where: str, *, lo: float | None = None, hi: float | None = None,
         lo_open: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(where, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise _fail(where, "must be finite")
    if lo is not None and (number < lo or (lo_open and number == lo)):
        raise _fail(where, f"must be {'>' if lo_open else '>='} {lo}")
    if hi is not None and number > hi:
        raise _fail(where, f"must be <= {hi}")
    return number


def _int(value: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(where, "must be an integer")
    if not lo <= value <= hi:
        raise _fail(where, f"must be between {lo} and {hi}")
    return value


def _ident(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise _fail(where, "must be 1 to 128 of A-Z a-z 0-9 _ . : -")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(where, "must be true or false")
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(where, "must be an object")
    return value


def check_card_text(value: Any, where: str) -> str:
    """Printable text, up to 200 characters; line breaks allowed, other control characters not."""
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, "must be non-empty text")
    if len(value) > MAX_END_CARD_CHARS:
        raise _fail(where, f"must be at most {MAX_END_CARD_CHARS} characters")
    for ch in value:
        if ch != "\n" and (ord(ch) < 32 or 0x7F <= ord(ch) < 0xA0 or ch in "\u2028\u2029"):
            raise _fail(where, "must not contain control characters")
    return value


def _gain_db(value: Any, where: str) -> float:
    """A gain in dB; ``-inf`` (float or the string "-inf") means mute."""
    if value == "-inf" or (isinstance(value, float) and value == -math.inf):
        return -math.inf
    return _num(value, where, lo=-96.0, hi=24.0)


@dataclass
class Transition:
    type: str = "cut"
    seconds: float = 0.0

    @classmethod
    def from_dict(cls, data: Any, where: str) -> "Transition":
        if data is None:
            return cls()
        data = _mapping(data, where)
        kind = data.get("type", "cut")
        if kind not in TRANSITION_TYPES:
            raise _fail(f"{where}.type", f"must be one of {', '.join(TRANSITION_TYPES)}")
        seconds = _num(data.get("seconds", 0.0), f"{where}.seconds", lo=0.0, hi=10.0)
        if kind == "crossfade" and seconds <= 0:
            raise _fail(f"{where}.seconds", "a crossfade needs seconds > 0")
        return cls(type=kind, seconds=seconds if kind == "crossfade" else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "seconds": self.seconds}


@dataclass
class ShotInput:
    id: str
    video: str
    hash: str
    trim_in: float = 0.0
    trim_out: float | None = None
    speed: float = 1.0
    native_sound_db: float = 0.0
    transition_in: Transition = field(default_factory=Transition)

    @classmethod
    def from_dict(cls, data: Any, where: str = "shot") -> "ShotInput":
        data = _mapping(data, where)
        trim_in = _num(data.get("trim_in", 0.0), f"{where}.trim_in", lo=0.0)
        trim_out = data.get("trim_out")
        if trim_out is not None:
            trim_out = _num(trim_out, f"{where}.trim_out", lo=trim_in, lo_open=True)
        return cls(
            id=_ident(data.get("id"), f"{where}.id"),
            video=check_input_file(data.get("video"), f"{where}.video"),
            hash=_ident(data.get("hash"), f"{where}.hash"),
            trim_in=trim_in,
            trim_out=trim_out,
            speed=_num(data.get("speed", 1.0), f"{where}.speed", lo=0.25, hi=4.0),
            native_sound_db=_gain_db(data.get("native_sound_db", 0.0), f"{where}.native_sound_db"),
            transition_in=Transition.from_dict(data.get("transition_in"), f"{where}.transition_in"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "video": self.video, "hash": self.hash, "trim_in": self.trim_in,
            "trim_out": self.trim_out, "speed": self.speed,
            "native_sound_db": "-inf" if self.native_sound_db == -math.inf else self.native_sound_db,
            "transition_in": self.transition_in.to_dict(),
        }


@dataclass
class AudioInput:
    id: str
    kind: str
    path: str
    start: float = 0.0
    duration: float | None = None
    trim_in: float = 0.0
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    duck: bool = False
    loop: bool = False

    @classmethod
    def from_dict(cls, data: Any, where: str = "audio") -> "AudioInput":
        data = _mapping(data, where)
        kind = data.get("kind")
        if kind not in AUDIO_KINDS:
            raise _fail(f"{where}.kind", f"must be one of {', '.join(AUDIO_KINDS)}")
        duration = data.get("duration")
        if duration is not None:
            duration = _num(duration, f"{where}.duration", lo=0.0, lo_open=True, hi=24 * 3600)
        return cls(
            id=_ident(data.get("id"), f"{where}.id"),
            kind=kind,
            path=check_input_file(data.get("path"), f"{where}.path"),
            start=_num(data.get("start", 0.0), f"{where}.start", lo=0.0, hi=24 * 3600),
            duration=duration,
            trim_in=_num(data.get("trim_in", 0.0), f"{where}.trim_in", lo=0.0),
            gain_db=_num(data.get("gain_db", 0.0), f"{where}.gain_db", lo=-96.0, hi=24.0),
            fade_in=_num(data.get("fade_in", 0.0), f"{where}.fade_in", lo=0.0, hi=600.0),
            fade_out=_num(data.get("fade_out", 0.0), f"{where}.fade_out", lo=0.0, hi=600.0),
            duck=_bool(data.get("duck", kind == "music"), f"{where}.duck"),
            loop=_bool(data.get("loop", False), f"{where}.loop"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "path": self.path, "start": self.start,
            "duration": self.duration, "trim_in": self.trim_in, "gain_db": self.gain_db,
            "fade_in": self.fade_in, "fade_out": self.fade_out, "duck": self.duck, "loop": self.loop,
        }


@dataclass
class EndCard:
    text: str
    seconds: float = 3.0

    @classmethod
    def from_dict(cls, data: Any, where: str = "end_card") -> "EndCard":
        data = _mapping(data, where)
        return cls(text=check_card_text(data.get("text"), f"{where}.text"),
                   seconds=_num(data.get("seconds", 3.0), f"{where}.seconds", lo=0.5, hi=30.0))

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "seconds": self.seconds}


@dataclass
class RenderPlan:
    width: int
    height: int
    fps: int
    out_path: str
    work_dir: str
    shots: list[ShotInput]
    preview: bool = False
    loudness_lufs: float | None = -16.0
    audio: list[AudioInput] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    end_card: EndCard | None = None
    captions_srt: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> "RenderPlan":
        data = _mapping(data, "plan")
        width = _int(data.get("width"), "width", 16, 7680)
        height = _int(data.get("height"), "height", 16, 4320)
        if width % 2 or height % 2:
            raise _fail("width/height", "must be even (yuv420p)")
        shots_raw = data.get("shots")
        if not isinstance(shots_raw, list) or not shots_raw:
            raise _fail("shots", "must be a non-empty list")
        if len(shots_raw) > 500:
            raise _fail("shots", "at most 500 shots")
        shots = [ShotInput.from_dict(item, f"shots[{i}]") for i, item in enumerate(shots_raw)]
        audio_raw = data.get("audio", [])
        if not isinstance(audio_raw, list) or len(audio_raw) > 200:
            raise _fail("audio", "must be a list of at most 200 clips")
        audio = [AudioInput.from_dict(item, f"audio[{i}]") for i, item in enumerate(audio_raw)]
        for label, items in (("shots", shots), ("audio", audio)):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise _fail(label, "ids must be unique")
        loudness = data.get("loudness_lufs", -16.0)
        if loudness is not None:
            loudness = _num(loudness, "loudness_lufs", lo=-70.0, hi=-5.0)
        metadata = _mapping(data.get("metadata", {}), "metadata")
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise _fail("metadata", "keys and values must be strings")
        if sum(len(k.encode()) + len(v.encode()) for k, v in metadata.items()) > MAX_METADATA_BYTES:
            raise _fail("metadata", f"must be at most {MAX_METADATA_BYTES} bytes")
        end_card = data.get("end_card")
        captions = data.get("captions_srt")
        if captions is not None:
            captions = check_input_file(captions, "captions_srt")
            if Path(captions).suffix.lower() != ".srt":
                raise _fail("captions_srt", "must be an .srt file")
        return cls(
            width=width,
            height=height,
            fps=_int(data.get("fps"), "fps", 1, 120),
            preview=_bool(data.get("preview", False), "preview"),
            loudness_lufs=loudness,
            out_path=check_output_path(data.get("out_path"), "out_path", ".mp4"),
            work_dir=check_dir_path(data.get("work_dir"), "work_dir"),
            shots=shots,
            audio=audio,
            metadata=dict(metadata),
            end_card=EndCard.from_dict(end_card) if end_card is not None else None,
            captions_srt=captions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width, "height": self.height, "fps": self.fps, "preview": self.preview,
            "loudness_lufs": self.loudness_lufs, "out_path": self.out_path, "work_dir": self.work_dir,
            "shots": [shot.to_dict() for shot in self.shots],
            "audio": [clip.to_dict() for clip in self.audio],
            "metadata": dict(self.metadata),
            "end_card": self.end_card.to_dict() if self.end_card else None,
            "captions_srt": self.captions_srt,
        }

    def validate(self) -> "RenderPlan":
        """Re-run every check on a plan built in code rather than via ``from_dict``."""
        return RenderPlan.from_dict(self.to_dict())
