"""Ready to export? (audit change 6): the project's open points in three groups, each with a code and a sentence.

``stops`` would make the export fail or come out empty; ``worthFixing`` makes a film the user probably does not want
(failed or missing shots, drafts in a final export, lip-sync gaps, a film shorter or longer than its song, clip sound
over the song); ``forInfo`` states what the film will be. Every item has a stable ``code``, a plain ``message``, the
``shotIds`` it is about and an ``action`` code the app maps to a button (None when there is nothing to press). The
app's own preflight list keeps working beside this; nothing here blocks a render.
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project

SONG_SLACK = 1.0  # seconds the film and the song may differ before it is worth saying
AUDIBLE_DB = -40.0  # a clip's own sound at or above this is heard over the song


def _item(code: str, message: str, shot_ids: list[str] | None = None, action: str | None = None,
          **data: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "shotIds": shot_ids or [], "action": action, "data": data}


def _clock(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def _shots(n: int) -> str:
    return "1 shot" if n == 1 else f"{n} shots"


def _song_items(project: Project, timeline: Any, worth: list, info: list) -> None:
    song = project.song
    if song is None or not timeline.shots:
        return
    clip = project.clip(song.clipId) if song.clipId else None
    span = next((a for a in timeline.audio if a.id == clip.id), None) if clip is not None else None
    if clip is None:
        song_end = song.seconds
    elif span is None or clip.loop:
        song_end = None  # off the film, or looping to its end: no length to compare
    else:  # the timeline cuts audio at the film's end, so the song's own end comes from the clip
        length = clip.source.measuredSeconds or song.seconds
        out = min(clip.trimOut, length) if clip.trimOut is not None else length
        song_end = span.start + max(0.0, out - span.trim_in)
    film = timeline.total
    if song_end is not None and abs(film - song_end) > SONG_SLACK:
        worth.append(_item("song-length", f"The film is {_clock(film)}, the song {_clock(song_end)}.", action="open-mixer",
                           filmSeconds=round(film, 3), songSeconds=round(song_end, 3)))
    made = [s for s in project.shots if s.active() is not None and s.active().clip]  # type: ignore[union-attr]
    loud = [s.id for s in made if s.nativeSoundDb > AUDIBLE_DB]
    muted = [s.id for s in made if s.nativeSoundDb <= AUDIBLE_DB]
    if loud:
        worth.append(_item("clip-sound-audible", f"{_shots(len(loud))} play their own sound over the song.", loud,
                           "open-mixer"))
    if muted:
        info.append(_item("clip-sound-muted", f"Clip sound is muted under the song in {_shots(len(muted))}.", muted,
                          "open-mixer"))


def _coverage_items(cov: dict[str, Any] | None, worth: list, info: list) -> None:
    if not cov or (not cov["counts"]["sings"] and cov["source"] is None):
        return  # lip sync never used on this project
    rows = cov["shots"]
    waiting = [r["shotId"] for r in rows if r["state"] == "waiting"]
    if waiting:
        worth.append(_item("lipsync-waiting", f"{_shots(len(waiting))} with a vocal have no lip-synced clip yet.",
                           waiting, "make-lipsync"))
    for row in rows:
        if row["state"] == "skipped" and row["fix"]:
            worth.append(_item("lipsync-gap", f"{row['message']} It is not lip-synced.", [row["shotId"]], row["fix"],
                               reason=row["code"]))
    plain = [r["shotId"] for r in rows if r["state"] == "skipped" and not r["fix"]]
    if plain:
        info.append(_item("lipsync-no-singing", f"{_shots(len(plain))} have no singing and play as ordinary clips.",
                          plain))
    counts = cov["counts"]
    info.append(_item("lipsync-coverage", f"{counts['synced']} of {counts['total']} shots are lip-synced.",
                      synced=counts["synced"], sings=counts["sings"], total=counts["total"]))


def readiness(project: Project, timeline: Any, cov: dict[str, Any] | None, qual: dict[str, Any],
              final: bool = True) -> dict[str, Any]:
    """``{"ready", "stops", "worthFixing", "forInfo"}`` for a final export (``final`` False: for a preview, where
    drafts are only information). ``ready`` is True when nothing stops the export."""
    stops: list[dict[str, Any]] = []
    worth: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []
    shots = project.shots
    ready = [s.id for s in timeline.shots if s.ready]
    if not shots:
        stops.append(_item("no-shots", "The project has no shots yet.", action="plan-shots"))
    elif not ready:
        stops.append(_item("no-clips", "No shot has a clip yet, so there is nothing to export.", action="make-clips"))
    failed = [s.id for s in shots if s.status == "failed"]
    if failed:
        worth.append(_item("failed-shots", f"{_shots(len(failed))} did not render.", failed, "open-storyboard"))
    missing = [s.id for s in shots if s.id not in set(ready) and s.id not in failed]
    if shots and ready and missing:
        worth.append(_item("missing-clips", f"{len(missing)} of {len(shots)} shots have no clip and are left out.",
                           missing, "make-clips"))
    drafts = list(qual["draftShots"])
    if drafts:
        size = qual["draftSize"]
        text = f"{_shots(len(drafts))} {'is' if len(drafts) == 1 else 'are'} still a draft ({size['width']} x {size['height']})."
        (worth if final else info).append(_item("drafts", text, drafts, "choose-final", **size))
    stale = sorted({s.id for s in shots if s.status == "stale"} | set(qual["staleShots"]),
                   key=[s.id for s in shots].index)
    if stale:
        worth.append(_item("stale-shots", f"{_shots(len(stale))} changed since {'it was' if len(stale) == 1 else 'they were'} made.",
                           stale, "remake-stale"))
    _coverage_items(cov, worth, info)
    _song_items(project, timeline, worth, info)
    target = qual["target"]
    info.insert(0, _item("format", f"{target['width']} x {target['height']} · {target['fps']} fps", **target))
    return {"ready": not stops, "stops": stops, "worthFixing": worth, "forInfo": info}
