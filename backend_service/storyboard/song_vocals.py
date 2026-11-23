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
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project

LEAD_IN = 0.4
SUNG_MIN_SECONDS, SUNG_MAX_SECONDS = 3.0, 5.0  # the cut plan's bounds for sung shots (FU-380: 3-5 s sing best)
VOICED_SHARE_MIN = 0.2  # a stem slice with less of its time voiced than this has no singing to follow
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
    hop = max(1, int(rate * _FRAME_SECONDS))
    count = audio.size // hop
    if count == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((audio[: count * hop].reshape(count, hop) ** 2).mean(axis=1))
    ref = float(np.percentile(rms, 95))
    return rms > _VOICED_FRACTION * ref if ref > 0 else np.zeros(count, dtype=bool)


def voiced_share(voiced: Any, start: float, end: float) -> float:
    lo, hi = int(start / _FRAME_SECONDS), int(end / _FRAME_SECONDS)
    window = voiced[max(0, lo): max(0, hi)]
    return float(window.mean()) if window.size else 0.0


def plan_vocals(project: Project, voiced: Any = None, lead_in: float = LEAD_IN) -> dict[str, Any]:
    """``{"items": [...], "skipped": [...]}``: per shot the song window of its slice, or why it gets none."""
    from backend_service.lipsync_runtime import MAX_SECONDS, MIN_SECONDS
    from backend_service.storyboard.timeline import compute_timeline

    song = project.song
    if song is None:
        raise ValueError("This project has no song. Attach one first (Song > Attach).")
    offset = song_offset(project)
    spans = {span.id: span for span in compute_timeline(project).shots}
    items: list[dict[str, Any]] = []
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
            continue
        items.append({"shotId": shot.id, "number": number, "songStart": round(begins - lead, 3),
                      "lead": lead, "length": length})
    return {"items": items, "skipped": skipped}
