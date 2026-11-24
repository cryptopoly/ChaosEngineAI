"""A cut plan from the attached song (S-M1): shots as long as the music asks, never longer than one clip.

Each section of the song is split into the fewest shots that keep every shot at or under ``clip_seconds`` (the
longest clip the project's video model makes, from the catalogue) and at or over ``min_seconds``; each cut lands on
the beat nearest an even split when one is close enough, else on the even split. Sections shorter than
``min_seconds`` join their neighbour. The shots end exactly where the song ends, so with the song at the film's start
on the mixer the cuts fall on the music. Each shot carries ``trimOut`` equal to its length, so a clip a few frames
longer than asked (frame counts are rounded per model) is cut back and the film does not drift off the beat.

The result has the shape of the Plan tab's proposal (``planner.propose``): apply it with ``apply_plan`` in
``replace`` mode, one op and one Undo. Nothing is written here.
"""

from __future__ import annotations

import math
from typing import Any

from backend_service.storyboard.models import MAX_SHOTS, Project

DEFAULT_CLIP_SECONDS = 4.0
DEFAULT_MIN_SECONDS = 2.0
MAX_CLIP_FRAMES = 257  # the frame cap generators.make_clip asks a video model for
SNAP_FRACTION = 0.35  # a beat this far (as a share of the even piece) from the even split is still taken


class CutPlanError(ValueError):
    """A readable refusal (no song, or a song that needs more shots than a project holds)."""


def clip_seconds_for(variant: dict[str, Any] | None, fps: int, requested: float | None) -> tuple[float, str]:
    """(longest shot, where that came from): the model's clip length from the catalogue, lowered by a request; 4 s
    when no model is known."""
    model = None
    if variant is not None:
        value = variant.get("defaultDurationSeconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            model = min(float(value), MAX_CLIP_FRAMES / max(1, fps))
    if requested is not None and (model is None or requested < model):
        return float(requested), "request"
    if model is not None:
        return model, "model"
    return DEFAULT_CLIP_SECONDS, "default"


def _merge_short(sections: list[dict[str, Any]], min_seconds: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for section in sections:
        if out and section["end"] - section["start"] < min_seconds:
            out[-1] = {**out[-1], "end": section["end"]}
        elif out and out[-1]["end"] - out[-1]["start"] < min_seconds:
            out[-1] = {**section, "start": out[-1]["start"]}
        else:
            out.append(dict(section))
    return out


def split_section(start: float, end: float, beats: list[float], cap: float, low: float) -> list[float]:
    """Cut points inside [start, end] (end included, start not) so every piece is in [low, cap]; cuts snap to beats."""
    length = end - start
    count = max(1, math.ceil(length / cap - 1e-9))
    while True:
        piece = length / count
        cuts: list[float] = []
        cursor = start
        ok = True
        for k in range(1, count):
            ideal = start + k * piece
            room = [b for b in beats if cursor + low <= b <= min(cursor + cap, end - low)]
            near = min(room, key=lambda b: abs(b - ideal)) if room else None
            cut = near if near is not None and abs(near - ideal) <= SNAP_FRACTION * piece else ideal
            cut = min(max(cut, cursor + low), cursor + cap)
            cuts.append(round(cut, 3))
            cursor = cut
        if end - cursor > cap + 1e-6:
            ok = False
        if ok or count * low > length:
            return cuts + [round(end, 3)]
        count += 1


def plan_cuts(project: Project, clip_seconds: float, min_seconds: float = DEFAULT_MIN_SECONDS,
              keep_prompts: bool = True, length_seconds: float | None = None) -> dict[str, Any]:
    """Planned shots (``StoryboardPlannedShot`` with ``trimOut``) and the sections they came from. With
    ``length_seconds`` only the song's first that many seconds are cut (a 30 s ad from a long song)."""
    song = project.song
    if song is None:
        raise CutPlanError("This project has no song. Attach one first (Song > Attach).")
    cap = max(0.5, min(60.0, float(clip_seconds)))
    low = max(0.5, min(float(min_seconds), cap))
    total = song.seconds if length_seconds is None else max(low, min(song.seconds, float(length_seconds)))
    sections = [{"start": s.start, "end": min(s.end, total), "label": s.label} for s in song.sections if s.start < total]
    if not sections or sections[0]["start"] > 0.01:
        sections.insert(0, {"start": 0.0, "end": sections[0]["start"] if sections else total, "label": "intro"})
    if sections[-1]["end"] < total - 0.01:
        sections.append({"start": sections[-1]["end"], "end": total, "label": "outro"})
    sections = _merge_short([s for s in sections if s["end"] > s["start"]], low)
    old = list(project.shots)
    shots: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for section in sections:
        cuts = split_section(section["start"], section["end"], list(song.beats), cap, low)
        cursor = section["start"]
        label = section["label"] or "part"
        for index, cut in enumerate(cuts, start=1):
            seconds = round(cut - cursor, 3)
            if seconds < 0.5:  # a sliver left by rounding joins the shot before it
                if shots:
                    shots[-1]["seconds"] = shots[-1]["trimOut"] = round(shots[-1]["seconds"] + seconds, 3)
                cursor = cut
                continue
            source = old[len(shots)] if keep_prompts and len(shots) < len(old) else None
            shots.append({
                "title": label if len(cuts) == 1 else f"{label} {index}",
                "stillPrompt": source.stillPrompt if source else "",
                "motionPrompt": source.motionPrompt if source else "",
                "seconds": min(60.0, seconds),
                "trimOut": min(60.0, seconds),
                "referenceImages": list(source.referenceImages) if source else [],
                # B1309: every inEveryShot character is in each new shot; a kept shot's opt-outs stay with it.
                "castOut": list(source.castOut) if source else [],
            })
            cursor = cut
        summary.append({"start": section["start"], "end": section["end"], "label": label, "shots": len(cuts)})
    if len(shots) > MAX_SHOTS:
        raise CutPlanError(f"The song needs {len(shots)} shots at up to {cap:g} s each; a project holds {MAX_SHOTS}. "
                           "Use a video model with longer clips or a shorter song.")
    note = (f"Cuts fall on the nearest beat where one is close; every shot is at most {cap:g} s, the longest clip the "
            "video model makes." if song.beats else
            f"No beats were found, so each section is split evenly; every shot is at most {cap:g} s.")
    note += " The song's sections and beats are estimates: edit them and make the plan again if a cut feels off."
    if keep_prompts and old:
        note += f" The first {min(len(old), len(shots))} shots keep the prompts of the shots already in the project."
    if total < song.seconds - 0.01:
        note += f" Only the song's first {total:g} s of {song.seconds:g} s are cut."
    return {"shots": shots, "mode": "replace", "totalSeconds": round(total, 3), "sections": summary, "note": note,
            "rough": song.rough}
