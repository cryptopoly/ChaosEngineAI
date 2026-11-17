"""Captions for a Storyboard film: generate them from the voice lane, place them on the timeline, write SRT.

Generation runs the existing speech-to-text (``voice_runtime.transcribe_file``, the function behind
``POST /api/voice/transcribe``, under the same heavy-job slot) on each voice clip. The result is stored as
``Caption`` rows tied to the clip, with times in the clip's own source seconds, so a caption follows its
clip when the clip is moved, trimmed or re-anchored to a shot. ``cues`` turns the rows into timeline
seconds, ``to_srt`` writes them. Nothing here touches ffmpeg: the render burns in the SRT file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend_service.storyboard.models import Project
from backend_service.storyboard.timeline import Timeline

MAX_CUE_CHARS = 84  # two lines of 42, the usual broadcast limit
MIN_CUE_SECONDS = 0.3
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


@dataclass(frozen=True)
class Cue:
    id: str
    start: float
    end: float
    text: str


def cues(project: Project, timeline: Timeline) -> list[Cue]:
    """Every caption in timeline seconds, in time order. Captions of a missing, looped or off-film clip are left out."""
    spans = {span.id: span for span in timeline.audio}
    out: list[Cue] = []
    for caption in project.captions:
        if caption.clipId is None:
            start, end = caption.start, caption.end
        else:
            span = spans.get(caption.clipId)
            clip = project.clip(caption.clipId)
            if span is None or clip is None or clip.loop:
                continue
            lo, hi = span.trim_in, span.trim_in + span.duration
            if caption.end <= lo or caption.start >= hi:
                continue  # trimmed away
            start = span.start + max(caption.start, lo) - span.trim_in
            end = span.start + min(caption.end, hi) - span.trim_in
        end = min(end, timeline.total) if timeline.total else end
        if end - start < 0.05:
            continue
        out.append(Cue(caption.id, round(start, 3), round(end, 3), caption.text))
    return sorted(out, key=lambda c: (c.start, c.end))


def shifted(items: list[Cue], seconds: float) -> list[Cue]:
    """The cues moved later by ``seconds`` (the title card in front of the exported film)."""
    if seconds <= 0:
        return items
    return [Cue(c.id, round(c.start + seconds, 3), round(c.end + seconds, 3), c.text) for c in items]


def _stamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    return f"{millis // 3600000:02d}:{millis // 60000 % 60:02d}:{millis // 1000 % 60:02d},{millis % 1000:03d}"


def clean_text(text: str) -> str:
    """One cue's text made safe for an SRT file that libass will draw: no control characters, no blank
    line (it ends a cue), no ``-->``, and no ``{...}`` (libass reads those as style overrides)."""
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("-->", "->").replace("{", "(").replace("}", ")")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def to_srt(items: list[Cue]) -> str:
    blocks = []
    number = 0
    for cue in items:
        text = clean_text(cue.text)
        if not text:
            continue
        number += 1
        blocks.append(f"{number}\n{_stamp(cue.start)} --> {_stamp(cue.end)}\n{text}\n")
    return "\n".join(blocks)


def split_segment(start: float, end: float, text: str) -> list[dict[str, Any]]:
    """Cut a long sentence into cues of at most ``MAX_CUE_CHARS``, timed in proportion to their length."""
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= MAX_CUE_CHARS:
        return [{"start": start, "end": end, "text": text}]
    pieces: list[str] = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > MAX_CUE_CHARS:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    total = sum(len(p) for p in pieces)
    out, cursor = [], start
    for index, piece in enumerate(pieces):
        stop = end if index == len(pieces) - 1 else cursor + (end - start) * len(piece) / total
        out.append({"start": round(cursor, 3), "end": round(stop, 3), "text": piece})
        cursor = stop
    return out


def cues_from_segments(segments: list[dict[str, Any]], limit: float | None) -> list[dict[str, Any]]:
    """Cue dicts from speech-to-text segments: times rounded, empty text dropped, tiny cues stretched, long ones split."""
    out: list[dict[str, Any]] = []
    for segment in segments:
        try:
            start, end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if start < 0 or start != start or end != end:
            continue
        if limit is not None:
            end = min(end, limit)
        if end - start < MIN_CUE_SECONDS:
            end = start + MIN_CUE_SECONDS
            if limit is not None:
                end = min(end, limit)
        if end <= start:
            continue
        out.extend(split_segment(round(start, 3), round(end, 3), str(segment.get("text", ""))))
    return out


Transcribe = Callable[[Path, str], dict[str, Any]]


def transcribe_with_studio(path: Path, language: str) -> dict[str, Any]:
    """The Voice Studio's speech-to-text on one file, queued behind other heavy jobs like the transcribe route.

    Inside an agent job (which already holds the heavy-job slot) the slot is re-entrant: it runs at once."""
    from backend_service.helpers.job_queue import HEAVY_JOBS
    from backend_service.voice_runtime import transcribe_file

    with HEAVY_JOBS.slot("voice", "ui", "captions"):
        return transcribe_file(path, "", language.strip() or None)


def write_srt(project: Project, folder: Path, items: list[Cue]) -> Path:
    """Write the cues as ``exports/<project name>-captions.srt`` (replacing the last one)."""
    from backend_service.storyboard.timeline import _slug

    target = folder / "exports" / f"{_slug(project.name)}-captions.srt"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(to_srt(items), encoding="utf-8")
    tmp.replace(target)
    return target


CAPTION_POSITIONS = ("bottom", "top", "middle")
CAPTION_SIZES = ("small", "medium", "large")
_COLOUR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_FONT = re.compile(r"^[A-Za-z0-9 ]{1,64}$")
# Hangul, CJK punctuation, kana, Han, compatibility forms: text a Latin-only font draws as empty boxes.
_CJK = re.compile(r"[ᄀ-ᇿ⺀-鿿가-힯豈-﫿＀-￯]")
_FONTS = {  # (Latin, CJK) per platform; each is a system font there, and the CJK one falls back for Hangul
    "darwin": ("Helvetica Neue", "Hiragino Sans"),
    "win32": ("Segoe UI", "Microsoft YaHei"),
    "linux": ("DejaVu Sans", "Noto Sans CJK SC"),
}


@dataclass(frozen=True)
class CaptionLook:
    """How burnt-in captions look. Every field is checked, so it can reach an ffmpeg filter as is."""

    position: str = "bottom"
    size: str = "medium"
    colour: str = "#FFFFFF"
    font: str = ""

    def as_dict(self) -> dict[str, str]:
        out = {"position": self.position, "size": self.size, "colour": self.colour}
        if self.font:
            out["font"] = self.font
        return out


def default_font(text: str, platform: str | None = None) -> str:
    """A system font that has the glyphs of ``text`` (a CJK one when the captions use CJK script)."""
    import sys

    key = platform or sys.platform
    latin, cjk = _FONTS.get(key, _FONTS["linux"] if key.startswith("linux") else ("", ""))
    return cjk if _CJK.search(text) else latin


def _field(look: Any, *names: str) -> Any:
    for name in names:
        value = look.get(name) if isinstance(look, dict) else getattr(look, name, None)
        if value is not None:
            return value
    return None


def _project_look(project: Any) -> Any:
    """The project's own caption look (the brand kit, S15), or None while a project has none."""
    brand = getattr(project, "brand", None)
    for holder, names in ((brand, ("captionLook", "captions")), (getattr(project, "export", None), ("captionLook",))):
        look = _field(holder, *names) if holder is not None else None
        if look is not None and not isinstance(look, list):  # project.captions is the cue list, not a look
            return look
    return None


def look_for(project: Any, text: str, platform: str | None = None) -> CaptionLook:
    """The look burnt-in captions get: the project's caption look where it has one, else white, bottom,
    medium, in a font that can draw ``text``. A value that is not one of ours falls back to the default."""
    look = _project_look(project)
    position = size = colour = font = None
    if look is not None:
        position, size = _field(look, "position"), _field(look, "size")
        colour, font = _field(look, "colour", "color"), _field(look, "font", "fontFamily")
    return CaptionLook(
        position=position if position in CAPTION_POSITIONS else "bottom",
        size=size if size in CAPTION_SIZES else "medium",
        colour=colour.upper() if isinstance(colour, str) and _COLOUR.match(colour) else "#FFFFFF",
        font=font if isinstance(font, str) and _FONT.match(font) else default_font(text, platform),
    )


def check_look(value: Any) -> dict[str, str]:
    """A caption look dict from a render plan, checked; raises ValueError naming the bad field."""
    if not isinstance(value, dict):
        raise ValueError("must be an object")
    unknown = set(value) - {"position", "size", "colour", "font"}
    if unknown:
        raise ValueError(f"unknown field {sorted(unknown)[0]!r}")
    checks = {"position": lambda v: v in CAPTION_POSITIONS, "size": lambda v: v in CAPTION_SIZES,
              "colour": lambda v: isinstance(v, str) and bool(_COLOUR.match(v)),
              "font": lambda v: isinstance(v, str) and bool(_FONT.match(v))}
    for key, item in value.items():
        if not checks[key](item):
            raise ValueError(f"{key} is not allowed")
    return {str(k): str(v) for k, v in value.items()}


def voice_clip_ids(project: Project) -> list[str]:
    return [c.id for c in project.audio if c.kind == "voice" and c.status != "missing" and not c.loop]


def check_clip_ids(project: Project, clip_ids: list[str]) -> None:
    """Every id must be an audio clip of the project; the error lists the valid ones (a model sent shot ids)."""
    from backend_service.storyboard.ops import OpError

    known = {c.id for c in project.audio}
    unknown = [str(c) for c in clip_ids if str(c) not in known]
    if not unknown:
        return
    voice = voice_clip_ids(project)
    valid = ", ".join(c.id for c in project.audio) or "none"
    raise OpError(
        f"Unknown audio clip '{unknown[0]}'. clip_ids takes audio clip ids, not shot ids. The project's audio clips: "
        f"{valid} (voice: {', '.join(voice) or 'none'}). Omit clip_ids to caption every voice lane."
    )


def generate(store: Any, project_id: str, revision: int, clip_ids: list[str] | None = None,
             language: str = "", transcribe: Transcribe | None = None) -> tuple[Any, list[str]]:
    """Transcribe voice clips and store the result as captions. Returns (project, notes).

    The slow transcription runs before any project lock; each clip's captions are then replaced in one
    undoable ``set_captions`` op. Captions you added by hand (no clip) and those of other clips stay.
    """
    from backend_service.storyboard import media
    from backend_service.storyboard.ops import OpError, run_op
    from backend_service.storyboard.store import RevisionConflict

    run = transcribe or transcribe_with_studio
    project = store.load(project_id)
    if project.revision != revision:
        raise RevisionConflict(project)
    folder = store.folder(project_id)
    wanted = clip_ids if clip_ids else voice_clip_ids(project)
    check_clip_ids(project, wanted)
    if not wanted:
        raise OpError("There is no voice clip to caption. Add speech to the voice lane first.")
    notes: list[str] = []
    results: list[tuple[str, list[dict[str, Any]]]] = []
    for clip_id in wanted:
        clip = project.clip(clip_id)
        if clip is None:
            raise OpError(f"Unknown audio clip '{clip_id}'.")
        name = clip.source.label or clip.id
        if clip.status == "missing":
            notes.append(f"{name}: the file is missing, skipped.")
            continue
        reply = run(media.inside(folder, clip.source.path), language)
        found = cues_from_segments(list(reply.get("segments") or []), clip.source.measuredSeconds)
        if not found:
            notes.append(f"{name}: no speech was recognised.")
        if reply.get("note"):
            notes.append(str(reply["note"]))
        results.append((clip_id, found))
    current = project
    for clip_id, found in results:
        current = run_op(store, project_id, current.revision, "set_captions", {"clipId": clip_id, "cues": found})
    return current, notes
