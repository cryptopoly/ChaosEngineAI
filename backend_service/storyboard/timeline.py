"""The film's timeline from measured take durations, and the render plan the assembler consumes.

Pure functions. Each shot's length is its active take's measured seconds (planned seconds only when a
take was never measured), cut to its in and out points (``trimIn`` / ``trimOut`` are positions in the
source, seconds) and divided by ``speed``. A crossfade into a shot overlaps the previous shot, so it
shortens the film by its length; it is clamped to half the shorter of the two shots. Audio anchored to
a shot starts at that shot's start plus the offset, so it follows the shot when earlier shots change
length. ``build_render_plan`` turns the timeline into the plain dict ``RenderPlan.from_dict`` reads,
with absolute paths and prompt-free provenance metadata (D-34 rules).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend_service.storyboard.models import Project, Shot

MIN_SHOT_SECONDS = 0.04
PREVIEW_SHORT_SIDE = 360


@dataclass
class ShotSpan:
    id: str
    start: float
    end: float
    duration: float
    source_in: float
    source_out: float
    transition_type: str
    transition_seconds: float
    measured: bool
    ready: bool


@dataclass
class AudioSpan:
    id: str
    start: float
    duration: float
    trim_in: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class Timeline:
    total: float
    shots: list[ShotSpan] = field(default_factory=list)
    audio: list[AudioSpan] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def shot(self, shot_id: str) -> ShotSpan | None:
        return next((s for s in self.shots if s.id == shot_id), None)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for span in out["audio"]:
            span["end"] = round(span["start"] + span["duration"], 4)
        return out


def shot_ready(shot: Shot) -> bool:
    take = shot.active()
    return take is not None and bool(take.clip) and shot.status != "missing"


def _source_seconds(shot: Shot) -> tuple[float, bool]:
    take = shot.active()
    if take is not None and take.measuredSeconds:
        return float(take.measuredSeconds), True
    return float(shot.seconds), False


def compute_timeline(project: Project, ready_only: bool = False) -> Timeline:
    """Shot and audio positions in seconds. ``ready_only`` keeps only shots with a playable clip (the render)."""
    timeline = Timeline(total=0.0)
    prev: ShotSpan | None = None
    for shot in project.shots:
        ready = shot_ready(shot)
        if ready_only and not ready:
            timeline.problems.append(f"shot {shot.id} has no playable clip and is left out")
            continue
        length, measured = _source_seconds(shot)
        source_in = min(shot.trimIn, length)
        source_out = min(shot.trimOut, length) if shot.trimOut is not None else length
        if source_out - source_in < MIN_SHOT_SECONDS:
            timeline.problems.append(f"shot {shot.id}: trims leave nothing; trims ignored")
            source_in, source_out = 0.0, length
        duration = (source_out - source_in) / shot.speed
        kind, seconds = "cut", 0.0
        if prev is not None and shot.transition.type == "crossfade" and shot.transition.seconds > 0:
            kind = "crossfade"
            limit = min(prev.duration, duration) / 2
            seconds = min(shot.transition.seconds, limit)
            if seconds < shot.transition.seconds:
                timeline.problems.append(f"shot {shot.id}: crossfade shortened to {seconds:.3f} s (half the shorter shot)")
        start = (prev.end - seconds) if prev is not None else 0.0
        span = ShotSpan(shot.id, start, start + duration, duration, source_in, source_out, kind, seconds, measured, ready)
        timeline.shots.append(span)
        prev = span
    timeline.total = prev.end if prev is not None else 0.0
    for clip in project.audio:
        if clip.status == "missing":
            timeline.problems.append(f"audio {clip.id} is missing its file and is left out")
            continue
        if clip.anchor is not None:
            anchor = timeline.shot(clip.anchor.shotId)
            if anchor is None:
                timeline.problems.append(f"audio {clip.id} is anchored to shot {clip.anchor.shotId}, which is not in the film")
                continue
            start = anchor.start + clip.anchor.offset
        else:
            start = clip.start or 0.0
        trim_in = clip.trimIn
        if start < 0:
            trim_in, start = trim_in - start, 0.0
        length = clip.source.measuredSeconds
        if clip.loop or length is None:
            if length is None and not clip.loop:
                timeline.problems.append(f"audio {clip.id} was never measured; it plays to the end of the film")
            duration = timeline.total - start
        else:
            out = min(clip.trimOut, length) if clip.trimOut is not None else length
            duration = min(out - trim_in, timeline.total - start)
        if duration <= 0:
            timeline.problems.append(f"audio {clip.id} falls outside the film")
            continue
        timeline.audio.append(AudioSpan(clip.id, start, duration, trim_in))
    return timeline


def output_size(project: Project, preview: bool) -> tuple[int, int]:
    """Frame size for the export preset (a preset change reframes, it does not regenerate)."""
    w, h = project.defaults.width, project.defaults.height
    long_side, short_side = max(w, h), min(w, h)
    preset = project.defaults.exportPreset
    if preset == "16:9":
        width, height = long_side, long_side * 9 / 16
    elif preset == "9:16":
        width, height = long_side * 9 / 16, long_side
    else:
        width = height = short_side
    if preview:
        scale = PREVIEW_SHORT_SIDE / min(width, height)
        if scale < 1:
            width, height = width * scale, height * scale
    return int(round(width / 2) * 2), int(round(height / 2) * 2)


def provenance(project: Project) -> dict[str, Any]:
    """Prompt-free provenance for the MP4 comment tag, or {} when the tag is off (D-34)."""
    if not project.embed_tag():
        return {}
    models = sorted({str(t.settings["model"]) for s in project.shots for t in [s.active()]
                     if t is not None and t.settings.get("model")})
    meta: dict[str, Any] = {
        "app": "ChaosEngineAI",
        "machineGenerated": True,
        "models": models,
        "project": hashlib.sha256(f"{project.id}:{project.name}".encode("utf-8")).hexdigest()[:16],
    }
    if project.has_cloned_voice():
        meta["clonedVoice"] = True
    try:
        from backend_service.helpers.system_hardware import _resolve_app_version

        meta["appVersion"] = _resolve_app_version()
    except Exception:
        pass
    return meta


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")[:40] or "film"


def build_render_plan(project: Project, preview: bool, folder: Path | None = None) -> dict[str, Any]:
    """The plain-dict render plan. Paths are absolute and resolved inside ``folder`` (the project folder)."""
    from backend_service.storyboard.media import inside

    if folder is None:
        from backend_service.storyboard.store import default_store

        folder = default_store().folder(project.id)
    folder = folder.resolve()
    timeline = compute_timeline(project, ready_only=True)
    if not timeline.shots:
        raise ValueError("Nothing to render: no shot has a clip yet.")
    width, height = output_size(project, preview)
    shots: list[dict[str, Any]] = []
    for span in timeline.shots:
        take = project.shot(span.id).active()  # type: ignore[union-attr]
        shots.append({
            "id": span.id,
            "video": str(inside(folder, take.clip)),  # type: ignore[union-attr, arg-type]
            "hash": take.hashes.get("clip", ""),  # type: ignore[union-attr]
            "trim_in": round(span.source_in, 4),
            "trim_out": round(span.source_out, 4),
            "speed": project.shot(span.id).speed,  # type: ignore[union-attr]
            "native_sound_db": project.shot(span.id).nativeSoundDb,  # type: ignore[union-attr]
            "transition_in": {"type": span.transition_type, "seconds": round(span.transition_seconds, 4)},
        })
    audio: list[dict[str, Any]] = []
    for span in timeline.audio:
        clip = project.clip(span.id)
        assert clip is not None
        audio.append({
            "id": clip.id, "kind": clip.kind, "path": str(inside(folder, clip.source.path)),
            "start": round(span.start, 4), "duration": round(span.duration, 4), "trim_in": round(span.trim_in, 4),
            "gain_db": clip.gainDb, "fade_in": clip.fadeIn, "fade_out": clip.fadeOut, "duck": clip.duck, "loop": clip.loop,
        })
    name = "preview.mp4" if preview else f"{_slug(project.name)}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    label = project.export.aiLabel
    return {
        "width": width,
        "height": height,
        "fps": project.defaults.fps,
        "preview": preview,
        "loudness_lufs": project.export.loudnessLufs,
        "out_path": str(folder / "exports" / name),
        "work_dir": str(folder / "cache" / "render"),
        "shots": shots,
        "audio": audio,
        "metadata": provenance(project),
        "end_card": {"text": project.export.endCardText, "seconds": project.export.endCardSeconds} if label.endCard else None,
        "captions_srt": None,
    }
