"""Lyric captions for the attached song (S-M1): one caption per lyric line, timed from the song.

Two timings. ``sections``: the lines are shared out over the sung parts (every section but ``intro`` / ``outro``) in
proportion to their length, spaced evenly inside each part and started on the nearest beat. ``speech``: the song is
run through the installed speech-to-text (``captions.transcribe_with_studio``, the same queue slot as voice captions),
the recognised words get times spread over their segment, and the lyric words are matched to them in order
(``difflib``); a line takes the times of its matched words, and lines with no match are spaced between their matched
neighbours. Sung words are hard for speech-to-text, so this is rough too and the notes count what was placed.

The captions go on the song's mixer clip through the ordinary ``set_captions`` op (they follow the clip, one Undo
removes them); burn-in and ``.srt`` are the existing export switches and caption look.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Callable

from backend_service.storyboard.models import Project, Song

MAX_LINE_SECONDS = 7.0
MIN_LINE_SECONDS = 0.8
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

Transcribe = Callable[[Any, str], dict[str, Any]]


class LyricsError(ValueError):
    """A readable refusal. ``code`` is "no-stt" when speech timing was asked for and cannot run here."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


def speech_status() -> tuple[bool, str | None]:
    """Whether speech-to-text alignment can run here, and why not when it cannot."""
    try:
        from backend_service import voice_env, voice_runtime

        if not voice_env.is_installed():
            return False, "Speech-to-text is not installed. Install the voice runtime under Voice > Models to time lyrics by ear."
        rows = voice_runtime.list_stt_models()
        default = next((r for r in rows if r.get("default")), None)
        if default is None:
            return False, "No speech-to-text model is available on this machine."
        if not default.get("installed"):
            return False, f"The speech-to-text model ({default.get('name')}) is not downloaded. Get it under Voice > Models."
        return True, None
    except Exception as exc:  # noqa: BLE001 - a status check never fails the request
        return False, f"Speech-to-text is not available: {str(exc)[:200]}"


def _snap(t: float, beats: list[float]) -> float:
    if not beats:
        return t
    near = min(beats, key=lambda b: abs(b - t))
    return near if abs(near - t) <= 0.5 else t


def by_sections(song: Song, lines: list[str]) -> list[dict[str, Any]]:
    """Cues for ``lines`` shared over the sung sections by length, evenly spaced, starting on beats."""
    parts = [(s.start, s.end) for s in song.sections if s.label not in ("intro", "outro")]
    if not parts:
        parts = [(s.start, s.end) for s in song.sections] or [(0.0, song.seconds)]
    total = sum(b - a for a, b in parts)
    if not lines or total <= 0:
        return []
    shares = [len(lines) * (b - a) / total for a, b in parts]
    counts = [int(x) for x in shares]
    for i in sorted(range(len(parts)), key=lambda i: shares[i] - counts[i], reverse=True)[: len(lines) - sum(counts)]:
        counts[i] += 1
    cues, index = [], 0
    for (a, b), count in zip(parts, counts):
        if not count:
            continue
        step = (b - a) / count
        for k in range(count):
            start = _snap(a + k * step, song.beats)
            end = min(a + (k + 1) * step, start + MAX_LINE_SECONDS, song.seconds)
            if end - start < MIN_LINE_SECONDS:
                end = min(start + MIN_LINE_SECONDS, song.seconds)
            cues.append({"start": round(start, 3), "end": round(end, 3), "text": lines[index]})
            index += 1
    return cues


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def by_speech(song: Song, lines: list[str], segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Cues for ``lines`` from speech-to-text segments; returns (cues, lines placed by a match)."""
    heard: list[tuple[str, float, float]] = []  # (word, start, end)
    for segment in segments:
        try:
            a, b = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        words = _words(str(segment.get("text", "")))
        if not words or b <= a:
            continue
        size = sum(len(w) for w in words)
        cursor = a
        for word in words:
            span = (b - a) * len(word) / size
            heard.append((word, cursor, cursor + span))
            cursor += span
    sung = [(i, w) for i, line in enumerate(lines) for w in _words(line)]
    matched: dict[int, list[tuple[float, float]]] = {}
    matcher = difflib.SequenceMatcher(None, [w for _, w in sung], [w for w, _, _ in heard], autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            line = sung[block.a + k][0]
            _, start, end = heard[block.b + k]
            matched.setdefault(line, []).append((start, end))
    times: list[tuple[float, float] | None] = [
        (min(s for s, _ in spans), max(e for _, e in spans)) if (spans := matched.get(i)) else None for i in range(len(lines))]
    placed = sum(1 for t in times if t is not None)
    if not placed:
        return [], 0
    index = 0
    while index < len(lines):  # each run of unmatched lines is spaced between the matched lines around it
        if times[index] is not None:
            index += 1
            continue
        stop = index
        while stop < len(lines) and times[stop] is None:
            stop += 1
        run = stop - index
        after = times[stop][0] if stop < len(lines) else None  # type: ignore[index]
        lo = times[index - 1][1] if index > 0 else max(0.0, after - MAX_LINE_SECONDS * run)  # type: ignore[index,operator]
        hi = after if after is not None else min(song.seconds, lo + MAX_LINE_SECONDS * run)
        slot = max(MIN_LINE_SECONDS, (hi - lo) / run)
        for k in range(index, stop):
            times[k] = (lo + slot * (k - index), lo + slot * (k - index + 1))
        index = stop
    cues = []
    for line, span in zip(lines, times):
        start, end = span  # type: ignore[misc]
        end = min(max(end, start + MIN_LINE_SECONDS), start + MAX_LINE_SECONDS, song.seconds)
        if end > start:
            cues.append({"start": round(start, 3), "end": round(end, 3), "text": line})
    return cues, placed


def make(store: Any, project_id: str, revision: int, timing: str = "auto", language: str = "",
         transcribe: Transcribe | None = None, status: Callable[[], tuple[bool, str | None]] | None = None,
         ) -> tuple[Project, str, list[str]]:
    """Time the song's lyrics and store them as captions on its mixer clip. Returns (project, timing used, notes).
    The slow speech-to-text runs before any project lock."""
    from backend_service.storyboard import media
    from backend_service.storyboard.ops import run_op
    from backend_service.storyboard.store import RevisionConflict

    project = store.load(project_id)
    if project.revision != revision:
        raise RevisionConflict(project)
    song = project.song
    if song is None:
        raise LyricsError("This project has no song. Attach one first.")
    if not song.lyrics:
        raise LyricsError("The song has no lyrics. Add the lines first (edit_song lyrics).")
    if not song.clipId or project.clip(song.clipId) is None:
        raise LyricsError("The song is not on the mixer. Attach it with addToMixer so the captions follow it.")
    notes: list[str] = []
    ready, why = (status or speech_status)()
    used = "sections"
    cues: list[dict[str, Any]] = []
    if timing == "speech" and not ready:
        raise LyricsError(why or "Speech-to-text is not available.", code="no-stt")
    if timing in ("auto", "speech") and ready:
        from backend_service.storyboard.captions import transcribe_with_studio

        reply = (transcribe or transcribe_with_studio)(media.inside(store.folder(project_id), song.path), language)
        cues, placed = by_speech(song, list(song.lyrics), list(reply.get("segments") or []))
        if cues:
            used = "speech"
            notes.append(f"Speech-to-text placed {placed} of {len(song.lyrics)} lines; the others are spaced between them. "
                         "Sung words are hard to recognise, so check the timing.")
        else:
            notes.append("Speech-to-text recognised none of the lyrics; the lines are timed from the sections instead.")
    elif timing == "auto":
        notes.append(f"Timed from the song's sections. {why}" if why else "Timed from the song's sections.")
    if not cues:
        cues = by_sections(song, list(song.lyrics))
        notes.append("Each sung section shares the lines evenly; adjust single captions if a line comes early or late.")
    saved = run_op(store, project_id, project.revision, "set_captions", {"clipId": song.clipId, "cues": cues})
    return saved, used, notes
