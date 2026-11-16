"""Which shots are stale: a pure function of the project, never a trigger for regeneration.

A generated take records a fingerprint (``basis``) of what it was made from: the shot's own prompts and
timing, the references it uses, the picture defaults and, for ``startFrom: previous-shot``, the previous
shot's active take and clip hash, and, for a take that used the project's style (``settings.style``), the style as
it went in (S15). A shot is stale when its active take's basis no longer matches, or when
it chains from a shot that is stale (so redoing C marks every chained shot after it), or when its clip was made
from an older still than the take now holds (``clip_outdated``, after Redo still only). Uploaded takes have
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


def style_basis(project: Project, shot: Shot, used: str) -> list[Any]:
    """What of the project's style a take used (S15): the first picture of each style when it went in as pictures,
    the style's words and palette when it went in as words, and the brand kit's choice of how styles are used.
    The logo and caption look are left out: they are drawn on the export, not into a shot."""
    from backend_service.storyboard.style import styles_in_use

    styles = [r for r in styles_in_use(project) if r.id not in shot.referenceImages]
    if used == "pictures":
        parts: list[Any] = [[r.id, r.images[0].hash if r.images else None] for r in styles]
    else:
        parts = [[r.id, r.description.strip(), list(r.palette)] for r in styles]
    return [project.brand.styleUse, parts]


def fingerprint(project: Project, shot: Shot) -> str:
    """What a take of ``shot`` made now would be made from, as a short hash."""
    refs: list[Any] = []
    for ref_id in shot.referenceImages:
        ref = project.reference(ref_id)
        if ref is None:
            refs.append([ref_id, None])
            continue
        hashes = [img.hash for img in ref.images]
        # One picture keeps the format 2 fingerprint (a bare hash), so old takes do not turn stale.
        refs.append([ref_id, ref.name, ref.description, hashes[0] if len(hashes) == 1 else (hashes or None)])
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
    take = shot.active()
    used = (take.settings or {}).get("style") if take is not None else None
    if used in ("pictures", "words"):  # only a take that used the style turns stale when the style changes
        basis["style"] = style_basis(project, shot, str(used))
    if shot.startFrom == "previous-shot":
        prev = previous_shot(project, shot)
        take = prev.active() if prev else None
        basis["previous"] = [prev.id, take.id, take.hashes.get("clip")] if prev and take else None
    raw = json.dumps(basis, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def clip_outdated(shot: Shot) -> bool:
    """The active take's clip was made from another still than the one the take now holds (after Redo still only).

    The clip records the hash of the still it started from (``settings.clipStill``); takes made before that have
    none and are never outdated by this rule."""
    take = shot.active()
    if take is None or not take.clip:
        return False
    made_from = (take.settings or {}).get("clipStill")
    still = take.hashes.get("still")
    return isinstance(made_from, str) and bool(still) and made_from != still


def stale_shots(project: Project) -> set[str]:
    stale: set[str] = set()
    prev: Shot | None = None
    for shot in project.shots:
        take = shot.active()
        if take is not None and take.basis is not None:
            if take.basis != fingerprint(project, shot) or clip_outdated(shot):
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
