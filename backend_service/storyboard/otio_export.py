"""Write the Storyboard cut as an OpenTimelineIO (``.otio``) JSON file, so it can be finished in an editor.

OpenTimelineIO is Apache-2.0 licensed and its file format is plain JSON, so this module writes the few
schema objects it needs (Timeline, Stack, Track, Clip, Gap, Transition, ExternalReference) itself and adds
no dependency. Layout: one video track of the shots; an audio track holding the shots' own sound; and one
or more audio tracks per lane (voice, music, sfx, import; a lane gets extra tracks only where its clips
overlap). Media are absolute ``file://`` links to the files in the project folder, so the file opens on
this machine; copy the project's ``media/`` folder along and relink to open it elsewhere.

A crossfade becomes a ``SMPTE_Dissolve`` transition centred on the cut: each neighbour gives up half the
fade at the joint (the editor reads those frames from the media beyond the clip's edge), so the track is
as long as the rendered film. Gain, fades and ducking have no OTIO equivalent and ride along in each
clip's ``metadata.chaosengine``. A shot with a speed change carries a ``LinearTimeWarp`` effect; editors
differ in how they read it, so it is reported as a warning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend_service.storyboard.media import inside
from backend_service.storyboard.models import Project
from backend_service.storyboard.timeline import compute_timeline

LANES = (("voice", "Voice"), ("music", "Music"), ("sfx", "Sound effects"), ("import", "Imported audio"))


def _rt(seconds: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": round(seconds * rate, 4)}


def _range(start: float, duration: float, rate: float) -> dict[str, Any]:
    return {"OTIO_SCHEMA": "TimeRange.1", "start_time": _rt(start, rate), "duration": _rt(duration, rate)}


def _item(schema: str, name: str, **fields: Any) -> dict[str, Any]:
    return {"OTIO_SCHEMA": schema, "metadata": fields.pop("metadata", {}), "name": name, **fields}


def _clip(name: str, path: Path, available: float | None, start: float, duration: float, rate: float,
          meta: dict[str, Any], effects: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    reference = _item("ExternalReference.1", path.name, available_range=_range(0.0, available, rate) if available else None,
                      available_image_bounds=None, target_url=path.as_uri())
    return _item("Clip.1", name, source_range=_range(start, duration, rate), effects=effects or [], markers=[],
                 enabled=True, media_reference=reference, metadata={"chaosengine": meta})


def _gap(duration: float, rate: float) -> dict[str, Any]:
    return _item("Gap.1", "", source_range=_range(0.0, duration, rate), effects=[], markers=[], enabled=True)


def _track(name: str, kind: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return _item("Track.1", name, source_range=None, effects=[], markers=[], enabled=True, children=children, kind=kind)


def build_otio(project: Project, folder: Path) -> tuple[dict[str, Any], list[str]]:
    """The OTIO document (a plain dict) and warnings. Raises ValueError when no shot has a clip."""
    timeline = compute_timeline(project, ready_only=True)
    if not timeline.shots:
        raise ValueError("Nothing to export: no shot has a clip yet.")
    rate = float(project.defaults.fps)
    warnings: list[str] = list(timeline.problems)
    spans = timeline.shots
    video: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        shot = project.shot(span.id)
        take = shot.active() if shot else None
        assert shot is not None and take is not None and take.clip
        path = inside(folder, take.clip)
        head = span.transition_seconds / 2 if span.transition_type == "crossfade" else 0.0
        nxt = spans[index + 1] if index + 1 < len(spans) else None
        tail = nxt.transition_seconds / 2 if nxt is not None and nxt.transition_type == "crossfade" else 0.0
        if head:
            video.append(_item("Transition.1", "Crossfade", transition_type="SMPTE_Dissolve", parameters={},
                               in_offset=_rt(span.transition_seconds / 2, rate),
                               out_offset=_rt(span.transition_seconds / 2, rate)))
        duration = span.duration - head - tail
        effects = []
        if shot.speed != 1.0:
            effects.append(_item("LinearTimeWarp.1", "Speed", effect_name="LinearTimeWarp", time_scalar=shot.speed))
            warnings.append(f"shot {shot.title or shot.id}: speed {shot.speed:g}x is written as a LinearTimeWarp; "
                            "check it in your editor")
        start = span.source_in + head * shot.speed
        label = shot.title or shot.id
        meta = {"shotId": shot.id, "trimIn": shot.trimIn, "speed": shot.speed}
        video.append(_clip(label, path, take.measuredSeconds, start, duration, rate, meta, effects))
        native.append(_clip(f"{label} (sound)", path, take.measuredSeconds, start, duration, rate,
                            {**meta, "nativeSoundDb": shot.nativeSoundDb}, list(effects)))
    tracks = [_track("Picture", "Video", video)]
    # The shots' own sound has the same cut points; its clips are butt-joined (the fade is the editor's to add).
    tracks.append(_track("Shot sound", "Audio", native))

    for kind, label in LANES:
        placed = []
        for span in timeline.audio:
            clip = project.clip(span.id)
            if clip is not None and clip.kind == kind:
                placed.append((span, clip))
        placed.sort(key=lambda pair: pair[0].start)
        lanes: list[list[tuple[Any, Any]]] = []
        for pair in placed:
            for lane in lanes:
                if lane[-1][0].end <= pair[0].start + 1e-6:
                    lane.append(pair)
                    break
            else:
                lanes.append([pair])
        for number, lane in enumerate(lanes, 1):
            children: list[dict[str, Any]] = []
            cursor = 0.0
            for span, clip in lane:
                if span.start > cursor + 1e-6:
                    children.append(_gap(span.start - cursor, rate))
                meta = {"audioId": clip.id, "gainDb": clip.gainDb, "fadeIn": clip.fadeIn, "fadeOut": clip.fadeOut,
                        "duck": clip.duck, "loop": clip.loop, "clonedVoice": clip.source.clonedVoice}
                if clip.loop:
                    warnings.append(f"audio {clip.source.label or clip.id}: a looped clip is written once, "
                                    "not repeated")
                children.append(_clip(clip.source.label or clip.id, inside(folder, clip.source.path),
                                      clip.source.measuredSeconds, span.trim_in, span.duration, rate, meta))
                cursor = span.end
            name = label if len(lanes) == 1 else f"{label} {number}"
            tracks.append(_track(name, "Audio", children))

    stack = _item("Stack.1", "tracks", source_range=None, effects=[], markers=[], enabled=True, children=tracks)
    document = _item("Timeline.1", project.name, global_start_time=None, tracks=stack,
                     metadata={"chaosengine": {"app": "ChaosEngineAI", "fps": rate, "seconds": round(timeline.total, 4)}})
    return document, warnings


def write_otio(project: Project, folder: Path) -> tuple[Path, list[str]]:
    from backend_service.storyboard.timeline import _slug

    document, warnings = build_otio(project, folder)
    target = folder / "exports" / f"{_slug(project.name)}.otio"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)
    return target, warnings
