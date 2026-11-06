"""Which shots are stale: a pure function of the project, never a trigger for regeneration.

A generated take records a fingerprint (``basis``) of what it was made from: the shot's own prompts and
timing, the references it uses, the picture defaults and, for ``startFrom: previous-shot``, the previous
shot's active take and clip hash. A shot is stale when its active take's basis no longer matches, or when
it chains from a shot that is stale (so redoing C marks every chained shot after it). Uploaded takes have
no basis and are never stale. The user decides what to re-run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from backend_service.storyboard.models import Project, Shot

_PICTURE_DEFAULTS = ("width", "height", "fps", "imageModel", "videoModel")


def previous_shot(project: Project, shot: Shot) -> Shot | None:
    index = next((i for i, s in enumerate(project.shots) if s.id == shot.id), 0)
    return project.shots[index - 1] if index > 0 else None


def fingerprint(project: Project, shot: Shot) -> str:
    """What a take of ``shot`` made now would be made from, as a short hash."""
    refs: list[Any] = []
    for ref_id in shot.referenceImages:
        ref = project.reference(ref_id)
        refs.append([ref_id, ref.name, ref.description, ref.hash] if ref else [ref_id, None])
    basis: dict[str, Any] = {
        "still": shot.stillPrompt,
        "motion": shot.motionPrompt,
        "seed": shot.seed,
        "seconds": shot.seconds,
        "startFrom": shot.startFrom,
        "endImage": shot.endImage,
        "refs": refs,
        "defaults": [getattr(project.defaults, key) for key in _PICTURE_DEFAULTS],
    }
    if shot.startFrom == "previous-shot":
        prev = previous_shot(project, shot)
        take = prev.active() if prev else None
        basis["previous"] = [prev.id, take.id, take.hashes.get("clip")] if prev and take else None
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def stale_shots(project: Project) -> set[str]:
    stale: set[str] = set()
    prev: Shot | None = None
    for shot in project.shots:
        take = shot.active()
        if take is not None and take.basis is not None:
            if take.basis != fingerprint(project, shot):
                stale.add(shot.id)
            elif shot.startFrom == "previous-shot" and prev is not None and prev.id in stale:
                stale.add(shot.id)
        prev = shot
    return stale


def apply_stale(project: Project) -> None:
    """Flip finished shots between ``done`` and ``stale``; other states are left alone."""
    stale = stale_shots(project)
    for shot in project.shots:
        if shot.status in ("done", "stale"):
            shot.status = "stale" if shot.id in stale else "done"
        elif shot.status == "idle" and shot.active() is not None and shot.id in stale:
            shot.status = "stale"
