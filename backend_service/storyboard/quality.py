"""Quality labels (audit change 2): what each take was made at, so a draft is never mistaken for the film.

Read from what the takes already store: ``renderPass`` (draft or final) and the settings a batch records per half
(``stillWidth`` / ``stillHeight``, ``clipWidth`` / ``clipHeight`` / ``clipFps``; takes made before those keys have
only ``width`` / ``height`` / ``fps`` of the half made last). A half is ``stale`` for the current quality choice when
it no longer matches the project's format: a final half at another size (or a clip at another frame rate), a draft
half at another shape. An uploaded take has no recorded size and is never called stale here.
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project, Take

SHAPE_TOLERANCE = 0.1  # a draft is half size in multiples of 64: 1280 x 720 gives 640 x 384, 6 % off in shape


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


def _stale(size: dict[str, int] | None, draft: bool, project: Project) -> bool:
    if size is None:
        return False
    want = project.defaults
    if draft:
        return abs(size["width"] / size["height"] - want.width / want.height) > SHAPE_TOLERANCE * (want.width / want.height)
    if (size["width"], size["height"]) != (want.width, want.height):
        return True
    return "fps" in size and size["fps"] != want.fps


def take_quality(project: Project, take: Take) -> dict[str, Any]:
    """``{"pass", "draft", "still", "clip", "stillStale", "clipStale"}`` for one take (sizes None when unknown)."""
    settings = take.settings or {}
    draft = take.renderPass == "draft"
    still = _half(settings, "still", last=not take.clip) if take.still else None
    clip = _half(settings, "clip", last=True) if take.clip else None
    return {"pass": take.renderPass, "draft": draft, "still": still, "clip": clip,
            "stillStale": _stale(still, draft, project), "clipStale": _stale(clip, draft, project)}


def quality(project: Project) -> dict[str, Any]:
    """The project's quality block: the film's size and rate, every take's label (``takes[shotId][takeId]``), the
    shots whose active take is a draft, and those whose active take no longer matches the format."""
    takes: dict[str, dict[str, Any]] = {}
    drafts, stale = [], []
    for shot in project.shots:
        labels = {take.id: take_quality(project, take) for take in shot.takes}
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
