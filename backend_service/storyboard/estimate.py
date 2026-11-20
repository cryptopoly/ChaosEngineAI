"""How long, how much memory and how much disk a batch of shots would take, before it runs (D-65 phase 3).

Nothing here is a number of its own. Time per still and per clip comes from this project's own takes made with
the same model (``stillSeconds`` / ``clipSeconds`` that the batch records; a clip's time is taken per second of
clip), else from the catalog's ``estimatedGenerationSeconds`` for that model scaled to the project's size and the
shot's frames from the row's ``estimateRef`` (FU-165, ``helpers.time_estimate``); a row without one is scaled from
its default clip length to the shot's length. Peak memory is the model's catalog footprint for this machine
(``helpers.fit``), the larger of the image and the video model, since a batch unloads one before the other. Disk
is the average size of this project's own stills and clips times the count; without any yet it is unknown.
``basis`` is ``measured`` only when every time came from this machine; a missing model is named, never guessed.
Measured times count only for takes of the same pass (S6): a small draft never prices a full still. ``passes``
prices both passes at once, drafts (fast model, draft size) and the final pass on the kept shots (quality model).
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from backend_service.helpers.time_estimate import variant_seconds
from backend_service.storyboard.batch import clip_outdated, needs_final, needs_work, still_only
from backend_service.storyboard.models import Project, Shot


def plan_steps(project: Project, what: str, shot_ids: list[str] | None) -> list[tuple[Shot, str]]:
    """(shot, still|clip) for what a batch would make, as ``Batch.plan`` picks it (skipped steps left out)."""
    if shot_ids is not None:
        wanted = set(shot_ids)
        shots = [s for s in project.shots if s.id in wanted]
    else:
        shots = [s for s in project.shots if needs_work(s, what) and not s.locked]
    steps: list[tuple[Shot, str]] = []
    if what == "final":
        steps += [(shot, "still") for shot in shots if shot.startFrom == "generate"]
        return steps + [(shot, "clip") for shot in shots if shot.startFrom != "upload"]
    if what == "redo-still":
        return [(shot, "still") for shot in shots if shot.startFrom == "generate" and shot.active() is not None]
    if what == "redo-clip":
        return [(shot, "clip") for shot in shots if shot.startFrom != "upload"
                and not (shot.startFrom == "generate" and not (shot.active() and shot.active().still))]  # type: ignore[union-attr]
    for shot in shots:
        behind = still_only(shot) or clip_outdated(shot)
        if what in ("still", "both") and shot.startFrom == "generate" and not (what == "both" and behind and not shot.redo):
            steps.append((shot, "still"))
    for shot in shots:
        if what in ("clip", "both") and shot.startFrom != "upload":
            steps.append((shot, "clip"))
    return steps


def _clip_frames(variant: dict[str, Any], shot: Shot, fps: int) -> int:
    """Frames the batch would ask for, as ``generators.make_clip`` picks them."""
    from backend_service.helpers.generation_limits import shot_clip_frames

    return shot_clip_frames(str(variant.get("repo") or ""), shot.seconds, fps)


def _clip_seconds(variant: dict[str, Any], shot: Shot, width: int, height: int, fps: int) -> float | None:
    """A shot with a vocal clip is a lip-sync run, priced from the row's measured lip-sync figures."""
    frames = _clip_frames(variant, shot, fps)
    if shot.vocalClip:
        from backend_service.lipsync_runtime import estimate_seconds

        return estimate_seconds(variant, width, height, frames)
    return variant_seconds(variant, "video", width, height, frames)


def _measured(project: Project, kind: str, model_name: str | None, render_pass: str = "final") -> list[float]:
    """Seconds per still, or per second of clip, from takes of this pass this project made with ``model_name``."""
    rates: list[float] = []
    for shot in project.shots:
        for take in shot.takes:
            if take.renderPass != render_pass:
                continue
            settings = take.settings or {}
            took = settings.get(f"{kind}Seconds")
            if not isinstance(took, (int, float)) or took <= 0 or settings.get(f"{kind}Model") != model_name:
                continue
            if kind == "clip":
                if take.measuredSeconds:
                    rates.append(float(took) / float(take.measuredSeconds))
            else:
                rates.append(float(took))
    return rates


def _average_size(folder: Path, project: Project, kind: str) -> float | None:
    sizes: list[int] = []
    for shot in project.shots:
        for take in shot.takes:
            rel = take.still if kind == "still" else take.clip
            if not rel:
                continue
            try:
                sizes.append((folder / rel).stat().st_size)
            except OSError:
                continue
    return statistics.mean(sizes) if sizes else None


def estimate(
    project: Project,
    folder: Path,
    *,
    what: str = "both",
    shot_ids: list[str] | None = None,
    image_variant: dict[str, Any] | None,
    video_variant: dict[str, Any] | None,
    missing: list[str],
    hardware: Any = None,
    render_pass: str = "final",
    sizes: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """The estimate for one batch. ``missing`` holds plain sentences about models that are not installed.

    ``render_pass`` picks which measured takes count; ``sizes`` ({"still"|"clip": (w, h)}) prices a draft size."""
    from backend_service.helpers.fit import variant_fit

    steps = plan_steps(project, what, shot_ids)
    stills = [shot for shot, kind in steps if kind == "still"]
    clips = [shot for shot, kind in steps if kind == "clip"]

    seconds: float | None = 0.0
    parts: dict[str, float | None] = {}
    measured_all = bool(steps)
    for kind, shots, variant in (("still", stills, image_variant), ("clip", clips, video_variant)):
        if not shots:
            continue
        if variant is None:
            seconds, measured_all, parts[kind] = None, False, None
            continue
        rates = _measured(project, kind, variant.get("name"), render_pass)
        if rates:
            rate = statistics.median(rates)
            part = rate * len(shots) if kind == "still" else sum(rate * shot.seconds for shot in shots)
        else:
            measured_all = False
            catalog = variant.get("estimatedGenerationSeconds")
            if not isinstance(catalog, (int, float)) or catalog <= 0:
                seconds, parts[kind] = None, None
                continue
            width, height = (sizes or {}).get(kind) or (project.defaults.width, project.defaults.height)
            if kind == "still":
                part = (variant_seconds(variant, "image", width, height) or float(catalog)) * len(shots)
            elif variant.get("estimateRef"):
                part = sum(_clip_seconds(variant, shot, width, height, project.defaults.fps) or float(catalog) for shot in shots)
            else:
                base = variant.get("defaultDurationSeconds")
                part = sum(float(catalog) * (shot.seconds / float(base) if base else 1.0) for shot in shots)
        parts[kind] = part
        if seconds is not None:
            seconds += part

    peak: float | None = None
    total: float | None = None
    try:
        from backend_service.helpers.hardware_class import host_hardware_class

        hw = hardware or host_hardware_class()
        total = round(hw.usable_gb, 1) if hw.usable_gb > 0 else None
        for kind, shots, variant in (("image", stills, image_variant), ("video", clips, video_variant)):
            if shots and variant is not None:
                need = variant_fit(dict(variant), kind, hw).get("needGb")
                if isinstance(need, (int, float)):
                    peak = max(peak or 0.0, float(need))
    except Exception:  # noqa: BLE001 - an estimate never fails the screen
        peak = None

    disk: float | None = 0.0
    for kind, shots in (("still", stills), ("clip", clips)):
        if not shots:
            continue
        size = _average_size(folder, project, kind)
        if size is None:
            disk = None
            break
        disk += size * len(shots) / 1024**2

    return {
        "what": what,
        "shots": len({shot.id for shot, _ in steps}),
        "stills": len(stills),
        "clips": len(clips),
        "seconds": round(seconds, 1) if seconds is not None and steps else None,
        "peakGb": round(peak, 1) if peak is not None else None,
        "totalGb": total,
        "diskMb": round(disk, 2) if disk is not None and steps else None,
        "basis": ("measured" if measured_all else "estimated") if steps else None,
        "missing": missing,
        "imageModel": image_variant.get("name") if image_variant else None,
        "videoModel": video_variant.get("name") if video_variant else None,
        "stillSeconds": _round(parts.get("still")) if stills else None,
        "clipSeconds": _round(parts.get("clip")) if clips else None,
    }


def _round(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


def passes(
    project: Project,
    folder: Path,
    *,
    shot_ids: list[str] | None,
    draft_variant: dict[str, Any] | None,
    final_variant: dict[str, Any] | None,
    video_variant: dict[str, Any] | None,
    draft_sizes: dict[str, tuple[int, int]],
    hardware: Any = None,
) -> dict[str, Any]:
    """Both passes (S6). ``draft``: stills of the shots a Make would work on (or ``shot_ids``) with the fast model at
    the draft size, and their clips at that size, priced apart (``clipSeconds``) because drafts make clips only when
    asked; its ``seconds`` is the stills. ``final``: the kept shots again with the quality model at the project size
    plus their clips; with no shot kept yet it is priced for the same shots as the drafts (``assumedAllKept``)."""
    draft = estimate(project, folder, what="both", shot_ids=shot_ids, image_variant=draft_variant,
                     video_variant=video_variant, missing=[], hardware=hardware, render_pass="draft", sizes=draft_sizes)
    # Make drafts without clips is a stills batch: a shot that has a still but no clip gets a new draft still too.
    stills = estimate(project, folder, what="still", shot_ids=shot_ids, image_variant=draft_variant, video_variant=None,
                      missing=[], hardware=hardware, render_pass="draft", sizes=draft_sizes)
    draft["stills"], draft["stillSeconds"] = stills["stills"], stills["stillSeconds"]
    draft["seconds"] = draft["stillSeconds"] if draft["stills"] else None
    still = draft_sizes.get("still") or (project.defaults.width, project.defaults.height)
    draft["width"], draft["height"] = still
    kept = [s.id for s in project.shots if s.kept]
    assumed = not kept
    if assumed:  # every shot the drafts cover, or every unlocked one that is not an upload
        kept = shot_ids if shot_ids is not None else [s.id for s in project.shots if not s.locked and s.startFrom != "upload"]
    final = estimate(project, folder, what="final", shot_ids=kept if assumed else None, image_variant=final_variant,
                     video_variant=video_variant, missing=[], hardware=hardware)
    final["assumedAllKept"] = assumed
    final["width"], final["height"] = project.defaults.width, project.defaults.height
    for out in (draft, final):
        out.pop("missing", None)
    return {"draft": draft, "final": final}
