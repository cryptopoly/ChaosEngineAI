"""Lip-sync coverage (audit change 3): which shots sing, which are synced, and why the others are not, with a fix.

``song_vocals.plan_vocals`` names a skipped shot with a reason code from ``skip_code`` (and ``no-singing`` from the
stem), and ``routes/storyboard_long.song_vocals`` stores the run's record on ``Song.coverage``. ``coverage`` reads
that record and the shots as they are now into the table the app shows: per shot its song time, a state (synced,
waiting, skipped, unassigned), a reason code with a plain sentence, and the fix code the app maps to a button.

Rules match the run: a lip-sync shot lasts at least ``MIN_SHOT_SECONDS`` (2 s; a 2 to 3 s shot gets silent room up
to the 3 s a clip needs), and its clip, lead-in included, at most ``lipsync_runtime.MAX_SECONDS`` (8 s). A reason
the shots themselves show (length, speed, lock, place in the song) is read live, so a fixed shot stops being named;
``no-singing`` comes from the stem and only the stored record knows it.
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project, Shot

LEAD_IN = 0.4  # the vocal before a shot that lets the face start the phrase (FU-380; ``song_vocals`` uses it)
MIN_SHOT_SECONDS = 2.0  # the cut plan's own shortest shot: a shorter one is left alone, a longer one can lip-sync
CLIP_MAX_SECONDS = 8.0  # as ``lipsync_runtime.MAX_SECONDS`` (a test keeps the two equal)

# reason code -> fix code (None: nothing to fix, the shot plays as an ordinary clip)
SKIP_CODES: dict[str, str | None] = {
    "other-speed": "real-speed",
    "locked": "unlock",
    "outside-song": None,
    "too-short": "lengthen",
    "too-long": "shorten",
    "no-singing": None,
    "not-assigned": "assign-vocals",
}
STATES = ("synced", "waiting", "skipped", "unassigned")
FIXES = ("real-speed", "unlock", "lengthen", "shorten", "assign-vocals", "make-lipsync")


def skip_code(shot: Shot, begins: float, length: float, song_seconds: float, lead: float) -> str | None:
    """Why a shot cannot lip-sync as it stands (None when it can), checked in the run's order."""
    if shot.speed != 1.0:
        return "other-speed"
    if shot.locked:
        return "locked"
    if begins < -1e-6 or begins + length > song_seconds + 0.05:
        return "outside-song"
    if length < MIN_SHOT_SECONDS - 1e-6:
        return "too-short"
    if length + lead > CLIP_MAX_SECONDS + 1e-6:
        return "too-long"
    return None


def skip_message(code: str, number: int, length: float, lead: float) -> str:
    """One plain sentence for a reason code."""
    if code == "other-speed":
        return f"Shot {number} plays at another speed; lip sync needs real time."
    if code == "locked":
        return f"Shot {number} is locked."
    if code == "outside-song":
        return f"Shot {number} is outside the song."
    if code == "too-short":
        return f"Shot {number} lasts {length:.1f} s; a lip-sync shot needs at least {MIN_SHOT_SECONDS:g} s."
    if code == "too-long":
        return (f"Shot {number} lasts {length:.1f} s; a lip-sync shot lasts at most {CLIP_MAX_SECONDS - lead:g} s "
                "(its lead-in comes on top).")
    if code == "no-singing":
        return f"Shot {number} has no singing in the vocal stem."
    return f"Shot {number} has no vocal yet."


def record(plan: dict[str, Any], source: str, at: float) -> dict[str, Any]:
    """The ``Song.coverage`` record of one ``plan_vocals`` run (shots that got a vocal and the skipped ones)."""
    shots = [{"shotId": i["shotId"], "number": i["number"], "code": None, "message": "",
              "songStart": round(i["songStart"] + i["lead"], 3), "length": i["length"]} for i in plan["items"]]
    shots += [{"shotId": s["shotId"], "number": s["number"], "code": s["code"], "message": s["reason"],
               "songStart": s["songStart"], "length": s["length"]} for s in plan["skipped"]]
    shots.sort(key=lambda s: s.pop("number"))
    return {"source": source, "at": at, "shots": shots}


def _synced(shot: Shot) -> bool:
    """The shot's clip is made and current (a vocal change marks it stale through the fingerprint)."""
    take = shot.active()
    return take is not None and bool(take.clip) and shot.status == "done"


def coverage(project: Project, timeline: Any = None) -> dict[str, Any] | None:
    """The coverage table, or None when the project has no song. ``timeline`` is ``compute_timeline(project)``."""
    from backend_service.storyboard.song_vocals import song_offset
    from backend_service.storyboard.timeline import compute_timeline

    song = project.song
    if song is None:
        return None
    timeline = timeline or compute_timeline(project)
    spans = {span.id: span for span in timeline.shots}
    stored = {s.shotId: s for s in song.coverage.shots} if song.coverage else {}
    offset = song_offset(project)
    rows: list[dict[str, Any]] = []
    counts = dict.fromkeys(STATES, 0)
    for number, shot in enumerate(project.shots, start=1):
        span = spans.get(shot.id)
        if span is None:
            continue
        begins, length = round(span.start + offset, 3), round(span.duration, 3)
        lead = round(max(0.0, min(LEAD_IN, begins)), 3)
        code: str | None = None
        if shot.vocalClip:
            state = "synced" if _synced(shot) else "waiting"
        else:
            code = skip_code(shot, begins, length, song.seconds, lead)
            kept = stored.get(shot.id)
            if code is None and kept is not None and kept.code == "no-singing":
                code = "no-singing"  # only the stem knew this
            state = "skipped" if code else "unassigned"
            code = code or "not-assigned"
        counts[state] += 1
        rows.append({
            "shotId": shot.id, "number": number, "songStart": begins, "songEnd": round(begins + length, 3),
            "seconds": length, "state": state, "code": code,
            "message": skip_message(code, number, length, lead) if code else "",
            "fix": SKIP_CODES.get(code) if code else ("make-lipsync" if state == "waiting" else None),
            "minSeconds": MIN_SHOT_SECONDS, "maxSeconds": round(CLIP_MAX_SECONDS - lead, 3),
        })
    return {
        "source": song.coverage.source if song.coverage else None,
        "at": song.coverage.at if song.coverage else None,
        "counts": {**counts, "total": len(rows), "sings": counts["synced"] + counts["waiting"]},
        "shots": rows,
    }
