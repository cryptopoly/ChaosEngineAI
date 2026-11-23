"""Song to lip-sync shots (FU-371): give each shot its slice of the song's vocal.

The shots are where the cut plan put them on the film (``timeline.compute_timeline``), with the song's mixer clip
giving the song time of each shot. Each slice is the vocal from 0.4 s before the shot (the lead-in that lets the face
start the phrase, FU-380; less at the song's start) to the shot's end, made into the shot's own WAV (stereo 48 kHz,
``lipsync_runtime.prepare_audio``). The shot then lasts the lead-in longer, with ``trimIn`` at the lead-in, so the
film still shows the shot's own span and the mouth stays on the music. Every slice is marked sung (singing wording,
3 takes by default, D-110).

Measured recipe (``docs/research/lip-sync-singing-experiments-2026-11.md``): the vocal stem follows best (0.44 vs 0.37
for the mix), so a stem is used when one is given; without it the song itself is sliced. With a stem, a shot whose
slice is mostly silent is left without a vocal (it would be a shot of a singer not singing). Shots outside the 3 to
8 s lip-sync range, at another speed, or past the song's end are left alone and named. Nothing here writes the
project: ``set_shot_vocals`` (``song_ops``) does, in one op and one Undo.

With a stem, two more steps give each shot the window the research measured (phrase, then a little silence):

- FU-607: a cut between two sung shots moves to the nearest phrase start (a voiced run after at least 0.3 s of
  silence) within 1.5 s, while both shots stay inside 3 to 5 s (or their own length, when they were outside it). The
  film keeps its total length; only where the cut falls changes.
- FU-609: a slice gets room after its last voiced frame: when the shot ends less than 0.5 s after its phrase, the slice
  runs on to 0.5 s after it. The room is the stem itself when it is silent there, else silence. Like the lead-in, the
  shot grows by the room and ``trimOut`` cuts it off again, so the film keeps the shot's length. Measured: a 4 s cut
  ending on the last syllable had 0 of 11 takes beat chance; the same line with 0.7 s of silence after it was usable
  (``docs/research/lip-sync-singing-experiments-2026-11.md``, follow-up of 2026-11-23). Without a stem the phrase end
  is unknown, so every slice gets the room as silence.
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project

LEAD_IN = 0.4
TAIL = 0.5  # FU-609: room after the phrase (the research windows had 0.6-0.7 s)
SUNG_MIN_SECONDS, SUNG_MAX_SECONDS = 3.0, 5.0  # the cut plan's bounds for sung shots (FU-380: 3-5 s sing best)
VOICED_SHARE_MIN = 0.2  # a stem slice with less of its time voiced than this has no singing to follow
PHRASE_GAP = 0.3  # FU-607: silence before a voiced run that makes it a phrase start
SNAP_RANGE = 1.5  # FU-607: how far a cut may move to reach a phrase start
_FRAME_SECONDS = 0.05
_VOICED_FRACTION = 0.15  # a frame is voiced above 15 % of the stem's 95th-percentile frame level (as lip_sync_score)


def song_offset(project: Project) -> float:
    """Song seconds at film time 0: the song's mixer clip start and in point (0 when it is not on the mixer)."""
    song = project.song
    clip = project.clip(song.clipId) if song is not None and song.clipId else None
    if clip is None or clip.anchor is not None:
        return 0.0
    return float(clip.trimIn) - float(clip.start or 0.0)


def voiced_frames(samples: Any, rate: int) -> Any:
    """True per 50 ms frame where the vocal stem sounds."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float64)
    count = int(audio.size / (rate * _FRAME_SECONDS) + 1e-6)
    if count == 0:
        return np.zeros(0, dtype=bool)
    # Frame edges on exact 50 ms marks: a whole-sample hop (551 at 11025 Hz) drifts 0.05 s every 100 s of song.
    edges = np.round(np.arange(count + 1) * rate * _FRAME_SECONDS).astype(np.int64)
    rms = np.sqrt(np.add.reduceat(audio[: edges[-1]] ** 2, edges[:-1]) / np.diff(edges))
    ref = float(np.percentile(rms, 95))
    return rms > _VOICED_FRACTION * ref if ref > 0 else np.zeros(count, dtype=bool)


def _frame(seconds: float) -> int:
    return int(seconds / _FRAME_SECONDS + 1e-6)  # 8.0 / 0.05 is 159.99999999999997


def voiced_share(voiced: Any, start: float, end: float) -> float:
    lo, hi = _frame(start), _frame(end)
    window = voiced[max(0, lo): max(0, hi)]
    return float(window.mean()) if window.size else 0.0


def phrase_starts(voiced: Any) -> list[float]:
    """Song seconds where a phrase starts: a voiced frame after at least ``PHRASE_GAP`` of silence (or the start)."""
    gap = round(PHRASE_GAP / _FRAME_SECONDS)
    starts: list[float] = []
    silent = gap
    for index, sounding in enumerate(voiced):
        if sounding:
            if silent >= gap:
                starts.append(round(index * _FRAME_SECONDS, 3))
            silent = 0
        else:
            silent += 1
    return starts


def silence_after(voiced: Any, start: float, end: float) -> float:
    """Seconds from the last voiced frame in ``[start, end)`` to ``end`` (the whole window when none is voiced)."""
    lo, hi = max(0, _frame(start)), max(0, _frame(end))
    window = voiced[lo:hi]
    sounding = [index for index, value in enumerate(window) if value]
    if not sounding:
        return end - start
    return max(0.0, end - (lo + sounding[-1] + 1) * _FRAME_SECONDS)


def room_after(voiced: Any, begins: float, length: float, budget: float, tail: float = TAIL) -> tuple[float, bool]:
    """(seconds to add after the shot so ``tail`` follows its phrase, whether the stem may fill them). Without a stem
    the phrase end is unknown: the whole ``tail``, as silence. Never more than ``budget``."""
    if voiced is None:
        extra, natural = tail, False
    else:
        end = begins + length
        extra = max(0.0, tail - silence_after(voiced, begins, end))
        natural = voiced_share(voiced, end, end + extra) == 0.0
    extra = round(min(extra, max(0.0, budget)), 3)
    return (extra, natural) if extra >= _FRAME_SECONDS else (0.0, False)


def _snap(windows: list[dict[str, Any]], voiced: Any) -> int:
    """FU-607: move each cut between two adjacent sung shots to the nearest phrase start; returns the cuts moved."""
    starts = phrase_starts(voiced)
    moved = 0
    for left, right in zip(windows, windows[1:]):
        if not (left["sung"] and right["sung"] and right["adjacent"]):
            continue
        cut = right["begins"]
        bounds = [(min(SUNG_MIN_SECONDS, w["length"]), max(SUNG_MAX_SECONDS, w["length"])) for w in (left, right)]
        for start in sorted((s for s in starts if abs(s - cut) <= SNAP_RANGE), key=lambda s: (abs(s - cut), s)):
            delta = round(start - cut, 3)
            if abs(delta) < _FRAME_SECONDS:
                break  # already on a phrase start
            a, b = round(left["length"] + delta, 3), round(right["length"] - delta, 3)
            if not (bounds[0][0] - 1e-6 <= a <= bounds[0][1] + 1e-6 and bounds[1][0] - 1e-6 <= b <= bounds[1][1] + 1e-6):
                continue
            if min(voiced_share(voiced, left["begins"], left["begins"] + a),
                   voiced_share(voiced, start, start + b)) < VOICED_SHARE_MIN:
                continue
            left["length"], right["begins"], right["length"] = a, round(start, 3), b
            moved += 1
            break
    return moved


def plan_vocals(project: Project, voiced: Any = None, lead_in: float = LEAD_IN, tail: float = TAIL) -> dict[str, Any]:
    """``{"items": [...], "skipped": [...], "snapped": n}``: per shot the song window of its slice (``lead`` before
    it, ``tail`` of room after it, ``natural`` when the stem fills that room), or why it gets none."""
    from backend_service.lipsync_runtime import MAX_SECONDS, MIN_SECONDS
    from backend_service.storyboard.timeline import compute_timeline

    song = project.song
    if song is None:
        raise ValueError("This project has no song. Attach one first (Song > Attach).")
    offset = song_offset(project)
    spans = {span.id: span for span in compute_timeline(project).shots}
    windows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for number, shot in enumerate(project.shots, start=1):
        span = spans.get(shot.id)
        if span is None:
            continue
        begins = span.start + offset
        length = round(span.duration, 3)
        lead = round(max(0.0, min(lead_in, begins)), 3)
        clip = length + lead
        reason = None
        if shot.speed != 1.0:
            reason = "plays at another speed; lip sync needs real time"
        elif shot.locked:
            reason = "is locked"
        elif begins < -1e-6 or begins + length > song.seconds + 0.05:
            reason = "is outside the song"
        elif clip < MIN_SECONDS - 1e-6 or clip > MAX_SECONDS + 1e-6:
            reason = f"lasts {length:.1f} s; a lip-sync shot with its lead-in lasts {MIN_SECONDS:g} to {MAX_SECONDS:g} s"
        elif voiced is not None and voiced_share(voiced, begins, begins + length) < VOICED_SHARE_MIN:
            reason = "has no singing in the vocal stem"
        if reason:
            skipped.append({"shotId": shot.id, "number": number, "reason": f"Shot {number} {reason}."})
        # A cut moves only between two sung shots that follow each other with a plain cut (FU-607).
        adjacent = bool(windows) and windows[-1]["number"] == number - 1 and span.transition_type == "cut"
        windows.append({"shotId": shot.id, "number": number, "begins": begins, "length": length, "sung": not reason,
                        "adjacent": adjacent})
    snapped = _snap(windows, voiced) if voiced is not None else 0
    items: list[dict[str, Any]] = []
    for window in windows:
        if not window["sung"]:
            continue
        begins, length = window["begins"], window["length"]
        lead = round(max(0.0, min(lead_in, begins)), 3)
        extra, natural = room_after(voiced, begins, length, MAX_SECONDS - lead - length, tail)
        items.append({"shotId": window["shotId"], "number": window["number"], "songStart": round(begins - lead, 3),
                      "lead": lead, "length": length, "tail": extra, "natural": natural})
    return {"items": items, "skipped": skipped, "snapped": snapped}
