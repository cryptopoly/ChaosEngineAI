"""Quality labels (audit change 2): what each take was made at, so a draft is never mistaken for the film.

Read from what the takes already store: ``renderPass`` (draft or final) and the settings a batch records per half
(``stillWidth`` / ``stillHeight``, ``clipWidth`` / ``clipHeight`` / ``clipFps``; takes made before those keys have
only ``width`` / ``height`` / ``fps`` of the half made last). A half is ``stale`` for the current quality choice when
it no longer matches the project's format: a final half at another size (or a clip at another frame rate), a draft
half at another shape. An uploaded take has no recorded size and is never called stale here.

A model makes a project's size on its own grid (LTX-2.3 makes 1280 x 720 as 1248 x 704), so a final half counts as
the right size when it matches either the project's size or the size the project's model makes for it (the helper
the generators use, ``normalize_video_size`` / ``normalize_image_size``). With no known model, a half within one
grid step (``GRID``) of the project's size on each side counts as right.
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project, Take

SHAPE_TOLERANCE = 0.1  # a draft is half size in multiples of 64: 1280 x 720 gives 640 x 384, 6 % off in shape


GRID = 64  # the coarsest size grid of a video model; with no known model, this far off a side is still the right size


def _num(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None


def _half(settings: dict[str, Any], kind: str, last: bool) -> dict[str, int] | None:
    """The recorded size of one half; the plain keys belong to it only when it was the half made last."""
    width = _num(settings.get(f"{kind}Width")) or (_num(settings.get("width")) if last else None)
    height = _num(settings.get(f"{kind}Height")) or (_num(settings.get("height")) if last else None)
    if width is None or height is None:
        return None
    out = {"width": width, "height": height}
    if kind == "clip":
        fps = _num(settings.get("clipFps")) or (_num(settings.get("fps")) if last else None)
        if fps is not None:
            out["fps"] = fps
    return out


def model_sizes(project: Project) -> dict[str, tuple[int, int] | None]:
    """The size the project's still and video models make for the project's size (``None`` for a model that is not
    set or not in the catalog), by the same rounding the generators apply."""
    from backend_service.helpers.generation_limits import (
        normalize_image_size, normalize_video_size, size_limits)

    d = project.defaults
    out: dict[str, tuple[int, int] | None] = {"still": None, "clip": None}
    if d.imageModel:
        from backend_service.helpers.images import _find_image_variant

        variant = _find_image_variant(d.imageModel)
        if variant is not None:
            limits = size_limits(variant).as_kwargs()
            out["still"] = normalize_image_size(d.width, d.height, min_side=size_limits(variant).min_side,
                                                model_name=str(variant.get("name") or ""), **limits)[:2]
    if d.videoModel:
        from backend_service.helpers.video import _find_video_variant

        variant = _find_video_variant(d.videoModel)
        if variant is not None:
            out["clip"] = normalize_video_size(d.width, d.height, str(variant.get("repo") or ""),
                                               model_name=str(variant.get("name") or ""),
                                               **size_limits(variant).as_kwargs())[:2]
    return out


def _stale(size: dict[str, int] | None, draft: bool, project: Project,
           aligned: tuple[int, int] | None = None, known: bool = False) -> bool:
    """``aligned`` is the size the half's model makes for the project's size; ``known`` says it is the model's own."""
    if size is None:
        return False
    want = project.defaults
    if draft:
        return abs(size["width"] / size["height"] - want.width / want.height) > SHAPE_TOLERANCE * (want.width / want.height)
    here = (size["width"], size["height"])
    if here != (want.width, want.height) and here != aligned:
        near = not known and abs(here[0] - want.width) < GRID and abs(here[1] - want.height) < GRID
        if not near:
            return True
    return "fps" in size and size["fps"] != want.fps


def take_quality(project: Project, take: Take,
                 aligned: dict[str, tuple[int, int] | None] | None = None) -> dict[str, Any]:
    """``{"pass", "draft", "still", "clip", "stillStale", "clipStale"}`` for one take (sizes None when unknown)."""
    settings = take.settings or {}
    aligned = model_sizes(project) if aligned is None else aligned
    draft = take.renderPass == "draft"
    still = _half(settings, "still", last=not take.clip) if take.still else None
    clip = _half(settings, "clip", last=True) if take.clip else None
    return {"pass": take.renderPass, "draft": draft, "still": still, "clip": clip,
            "stillStale": _stale(still, draft, project, aligned["still"], aligned["still"] is not None),
            "clipStale": _stale(clip, draft, project, aligned["clip"], aligned["clip"] is not None)}


def quality(project: Project) -> dict[str, Any]:
    """The project's quality block: the film's size and rate, every take's label (``takes[shotId][takeId]``), the
    shots whose active take is a draft, and those whose active take no longer matches the format."""
    takes: dict[str, dict[str, Any]] = {}
    drafts, stale = [], []
    aligned = model_sizes(project)
    for shot in project.shots:
        labels = {take.id: take_quality(project, take, aligned) for take in shot.takes}
        if labels:
            takes[shot.id] = labels
        active = labels.get(shot.activeTake or "")
        if active is None:
            continue
        if active["draft"]:
            drafts.append(shot.id)
        if active["stillStale"] or active["clipStale"]:
            stale.append(shot.id)
    from backend_service.storyboard.batch import draft_size

    d = project.defaults
    draft_w, draft_h = draft_size(d.width, d.height)  # before a model's minimum side; a take records its real size
    return {"target": {"width": d.width, "height": d.height, "fps": d.fps},
            "draftSize": {"width": draft_w, "height": draft_h}, "draftShots": drafts, "staleShots": stale,
            "takes": takes}
