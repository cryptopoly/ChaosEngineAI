"""The guide block every project payload carries (``routes/storyboard._payload``, ``GET .../guide``).

One place that builds, from the project and its timeline: ``quality`` (draft or final, sizes, stale for the format;
``quality.py``), ``coverage`` (lip sync per shot, None without a song; ``coverage.py``), ``failures`` (each failed
shot's latest failure as a code, sentence and action; ``failure.py``), ``readiness`` (the three export groups;
``readiness.py``), ``stages`` (the stage rail; ``stages.py``) and ``cast`` (each shot's effective reference ids,
``cast.effective_references``). All read-only and model-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend_service.storyboard.models import Project


def failures(project: Project) -> list[dict[str, Any]]:
    """For every shot that is failed now, its latest failed step (newest batch first), in shot order."""
    from backend_service.storyboard.failure import result_failure

    failed = {s.id for s in project.shots if s.status == "failed"}
    found: dict[str, dict[str, Any]] = {}
    for record in reversed(project.batches):
        for result in record.results:
            if result.state == "failed" and result.shotId in failed and result.shotId not in found:
                found[result.shotId] = {**result_failure(result), "batchId": record.id}
    return [found[s.id] for s in project.shots if s.id in found]


def exported(folder: Path | None) -> bool:
    """A film was exported (any file in ``exports/`` besides the Mixer's preview)."""
    if folder is None:
        return False
    exports = folder / "exports"
    try:
        return any(p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".webm") and p.name != "preview.mp4"
                   for p in exports.iterdir())
    except OSError:
        return False


def build(project: Project, timeline: Any = None, folder: Path | None = None) -> dict[str, Any]:
    from backend_service.storyboard.coverage import coverage
    from backend_service.storyboard.quality import quality
    from backend_service.storyboard.readiness import readiness
    from backend_service.storyboard.stages import stages
    from backend_service.storyboard.timeline import compute_timeline

    from backend_service.storyboard.cast import effective_references

    timeline = timeline or compute_timeline(project)
    cov = coverage(project, timeline)
    qual = quality(project)
    ready = readiness(project, timeline, cov, qual)
    return {"quality": qual, "coverage": cov, "failures": failures(project), "readiness": ready,
            "stages": stages(project, timeline, cov, qual, ready, exported(folder)),
            # B1309: each shot's effective cast (ticked references, then the in-every-shot characters it keeps)
            "cast": {shot.id: effective_references(project, shot) for shot in project.shots}}
