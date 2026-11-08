"""How long, how much memory and how much disk a batch of shots would take, before it runs (D-65 phase 3).

Nothing here is a number of its own. Time per still and per clip comes from this project's own takes made with
the same model (``stillSeconds`` / ``clipSeconds`` that the batch records; a clip's time is taken per second of
clip), else from the catalog's ``estimatedGenerationSeconds`` for that model (a clip scaled from the catalog's
default clip length to the shot's length). Peak memory is the model's catalog footprint for this machine
(``helpers.fit``), the larger of the image and the video model, since a batch unloads one before the other. Disk
is the average size of this project's own stills and clips times the count; without any yet it is unknown.
``basis`` is ``measured`` only when every time came from this machine; a missing model is named, never guessed.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from backend_service.storyboard.batch import needs_work, still_only
from backend_service.storyboard.models import Project, Shot


def plan_steps(project: Project, what: str, shot_ids: list[str] | None) -> list[tuple[Shot, str]]:
    """(shot, still|clip) for what a batch would make, as ``Batch.plan`` picks it (skipped steps left out)."""
    if shot_ids is not None:
        wanted = set(shot_ids)
        shots = [s for s in project.shots if s.id in wanted]
    else:
        shots = [s for s in project.shots if needs_work(s, what) and not s.locked]
    steps: list[tuple[Shot, str]] = []
    for shot in shots:
        if what in ("still", "both") and shot.startFrom == "generate" and not (what == "both" and still_only(shot) and not shot.redo):
            steps.append((shot, "still"))
    for shot in shots:
        if what in ("clip", "both") and shot.startFrom != "upload":
            steps.append((shot, "clip"))
    return steps


def _measured(project: Project, kind: str, model_name: str | None) -> list[float]:
    """Seconds per still, or per second of clip, from takes this project made with ``model_name``."""
    rates: list[float] = []
    for shot in project.shots:
        for take in shot.takes:
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
) -> dict[str, Any]:
    """The estimate for one batch. ``missing`` holds plain sentences about models that are not installed."""
    from backend_service.helpers.fit import variant_fit

    steps = plan_steps(project, what, shot_ids)
    stills = [shot for shot, kind in steps if kind == "still"]
    clips = [shot for shot, kind in steps if kind == "clip"]

    seconds: float | None = 0.0
    measured_all = bool(steps)
    for kind, shots, variant in (("still", stills, image_variant), ("clip", clips, video_variant)):
        if not shots:
            continue
        if variant is None:
            seconds, measured_all = None, False
            continue
        rates = _measured(project, kind, variant.get("name"))
        if rates:
            rate = statistics.median(rates)
            part = rate * len(shots) if kind == "still" else sum(rate * shot.seconds for shot in shots)
        else:
            measured_all = False
            catalog = variant.get("estimatedGenerationSeconds")
            if not isinstance(catalog, (int, float)) or catalog <= 0:
                seconds = None
                continue
            if kind == "still":
                part = float(catalog) * len(shots)
            else:
                base = variant.get("defaultDurationSeconds")
                part = sum(float(catalog) * (shot.seconds / float(base) if base else 1.0) for shot in shots)
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
    }
