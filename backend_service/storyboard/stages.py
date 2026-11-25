"""The guided flow (audit changes 1 and 9): the project's stages, each with a status, one sentence and one action.

Stages, in order: song, look, shots, drafts, final, lipsync, sound, check, export (the audit's stage rail). Each is
``{"id", "applies", "optional", "status", "message", "action"}``: ``status`` is ``not-started``, ``in-progress``,
``done`` or ``needs-attention``; ``action`` is a code the app maps to its one primary button (None when the stage
has nothing to press). Song and lip sync apply only with a song attached (a Music video start shows the Song stage
anyway; that template is the app's). ``current`` is the first stage that applies and is not done, passing over an
optional stage nobody started. Built from the coverage, quality and readiness blocks, so the rail never disagrees
with them.
"""

from __future__ import annotations

from typing import Any

from backend_service.storyboard.models import Project

STAGE_IDS = ("song", "look", "shots", "drafts", "final", "lipsync", "sound", "check", "export")
STATUSES = ("not-started", "in-progress", "done", "needs-attention")


def _stage(stage_id: str, status: str, message: str, action: str | None, applies: bool = True,
           optional: bool = False) -> dict[str, Any]:
    return {"id": stage_id, "applies": applies, "optional": optional, "status": status, "message": message,
            "action": action}


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _batch_state(project: Project) -> str | None:
    live = [b.state for b in project.batches if b.state in ("queued", "running", "paused")]
    return live[-1] if live else None


def _making(project: Project, qual: dict[str, Any], timeline: Any) -> list[dict[str, Any]]:
    """The drafts and final stages."""
    from backend_service.storyboard.batch import needs_final

    shots = project.shots
    batch = _batch_state(project)
    with_take = [s for s in shots if s.active() is not None]
    failed = [s for s in shots if s.status == "failed"]
    ready = {s.id for s in timeline.shots if s.ready}
    if not shots:
        drafts = _stage("drafts", "not-started", "Write the shots first.", None)
    elif batch in ("queued", "running"):
        drafts = _stage("drafts", "in-progress", "The shots are being made.", "watch-batch")
    elif batch == "paused":
        drafts = _stage("drafts", "in-progress", "Making the shots is paused.", "resume-batch")
    elif failed:
        drafts = _stage("drafts", "needs-attention", f"{_n(len(failed), 'shot', 'shots')} did not render.", "retry-failed")
    elif not with_take:
        drafts = _stage("drafts", "not-started", "Make quick drafts to choose the shots you like.", "make-drafts")
    elif len(with_take) < len(shots):
        left = len(shots) - len(with_take)
        drafts = _stage("drafts", "in-progress", f"{_n(left, 'shot has', 'shots have')} no picture yet.", "make-missing")
    else:
        drafts = _stage("drafts", "done", "Every shot has a picture.", None)

    target = qual["target"]
    size = f"{target['width']} x {target['height']}"
    draft_ids = qual["draftShots"]
    chosen = [s for s in shots if s.id in draft_ids and needs_final(s)]
    missing = [s for s in with_take if s.id not in ready]
    stale = {s.id for s in shots if s.status == "stale"} | set(qual["staleShots"])
    if not with_take:
        final = _stage("final", "not-started", f"Make the shots at {size}.", "make-final")
    elif chosen:
        final = _stage("final", "in-progress", f"Make {_n(len(chosen), 'chosen shot', 'chosen shots')} at {size}.",
                       "make-final")
    elif draft_ids:
        final = _stage("final", "not-started", f"Choose the shots to make at {size} "
                       f"({_n(len(draft_ids), 'draft', 'drafts')}).", "choose-final")
    elif missing:
        final = _stage("final", "in-progress", f"{_n(len(missing), 'shot has', 'shots have')} no clip yet.", "make-clips")
    elif stale:
        final = _stage("final", "needs-attention", _n(len(stale), "shot changed since it was made.",
                                                      "shots changed since they were made."), "remake-stale")
    elif len(with_take) < len(shots):
        final = _stage("final", "in-progress", "Some shots have no picture yet.", "make-missing")
    else:
        final = _stage("final", "done", f"Every shot is made at {size}.", None)
    return [drafts, final]


def _lipsync(project: Project, cov: dict[str, Any] | None) -> dict[str, Any]:
    if project.song is None or cov is None:
        return _stage("lipsync", "not-started", "Attach a song to lip-sync its singing.", None, applies=False,
                      optional=True)
    counts = cov["counts"]
    if not counts["sings"] and cov["source"] is None:
        return _stage("lipsync", "not-started", "Lip-sync the singing (optional): give each shot its vocal.",
                      "assign-vocals", optional=True)
    gaps = [r for r in cov["shots"] if r["state"] == "skipped" and r["fix"]]
    summary = f"{counts['synced']} of {counts['total']} shots are lip-synced."
    if counts["waiting"]:
        return _stage("lipsync", "in-progress", f"{summary} {_n(counts['waiting'], 'clip', 'clips')} to make.",
                      "make-lipsync", optional=True)
    if gaps:
        return _stage("lipsync", "needs-attention", f"{summary} {_n(len(gaps), 'shot', 'shots')} can be fixed.",
                      "open-coverage", optional=True)
    return _stage("lipsync", "done", summary, None, optional=True)


def stages(project: Project, timeline: Any, cov: dict[str, Any] | None, qual: dict[str, Any],
           ready: dict[str, Any], exported: bool = False) -> dict[str, Any]:
    """``{"current": stage id or None, "stages": [...]}`` in ``STAGE_IDS`` order."""
    song = project.song
    out = [
        _stage("song", "done", "The song is attached.", None) if song is not None
        else _stage("song", "not-started", "Add the song this film is cut to.", "attach-song", applies=False,
                    optional=True),
        _stage("look", "done", "The cast and look are set.", None, optional=True) if project.references.all()
        else _stage("look", "not-started", "Add the characters, places and look of the film (optional).",
                    "open-cast", optional=True),
    ]
    if not project.shots:
        out.append(_stage("shots", "not-started", "Cut the song into shots." if song else "Write the shots.",
                          "cut-plan" if song else "plan-shots"))
    else:
        blank = [s for s in project.shots if s.startFrom == "generate" and not (s.stillPrompt or s.title).strip()]
        out.append(_stage("shots", "needs-attention", f"{_n(len(blank), 'shot has', 'shots have')} no prompt.",
                          "open-storyboard") if blank else _stage("shots", "done", "The shots are written.", None))
    out += _making(project, qual, timeline)
    out.append(_lipsync(project, cov))
    loud = next((i for i in ready["worthFixing"] if i["code"] == "clip-sound-audible"), None)
    if loud is not None:
        out.append(_stage("sound", "needs-attention", loud["message"], "open-mixer", optional=True))
    elif project.audio:
        out.append(_stage("sound", "done", "The sound is in place.", None, optional=True))
    else:
        out.append(_stage("sound", "not-started", "Add voice, music or effects (optional).", "open-mixer",
                          optional=True))
    final_done = next(s for s in out if s["id"] == "final")["status"] == "done"
    if ready["stops"] or ready["worthFixing"]:
        count = len(ready["stops"]) + len(ready["worthFixing"])
        status = "needs-attention" if final_done else "not-started"
        out.append(_stage("check", status, f"{_n(count, 'point is', 'points are')} worth a look before you export.",
                          "open-check"))
    else:
        out.append(_stage("check", "done" if final_done else "not-started", "Nothing stops the export.", "open-check"))
    out.append(_stage("export", "done" if exported else "not-started",
                      "The film is exported." if exported else "Export the film.", "export"))
    current = next((s["id"] for s in out if s["applies"] and s["status"] != "done"
                    and not (s["optional"] and s["status"] == "not-started")), None)
    return {"current": current, "stages": out}
