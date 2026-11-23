"""Long projects (S-M1): pause and resume a batch, attach a song, a cut plan from it, and lyric captions.

Kept apart from ``routes/storyboard.py`` (over its size cap); the store, ops, error mapping and batch wiring are
shared with it. A paused batch keeps its remaining steps across a restart and resumes under its own id. A song is
analysed before the project lock is taken and stored with the ordinary ``set_song`` op, so Undo takes it off; the
cut plan only proposes shots (``apply_plan`` applies them); lyric captions are ``set_captions`` on the song's clip.
The agent tools call the helpers here (``attach_song``, ``cut_plan``, ``resume_batch``).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

router = APIRouter()


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResumeRequest(_Body):
    revision: int


class SongRequest(_Body):
    revision: int
    libraryId: str | None = Field(None, max_length=40)
    path: str | None = Field(None, min_length=1, max_length=4096)
    title: str | None = Field(None, max_length=200)
    addToMixer: bool = True

    @model_validator(mode="after")
    def _one_source(self) -> "SongRequest":
        if (self.libraryId is None) == (self.path is None):
            raise ValueError("Give libraryId or path (one of them).")
        return self


class CutPlanRequest(_Body):
    maxShotSeconds: float | None = Field(None, ge=0.5, le=60)
    minShotSeconds: float = Field(2.0, ge=0.5, le=30)
    keepPrompts: bool = True
    lengthSeconds: float | None = Field(None, ge=1, le=3600)  # cut only the song's first seconds (a short ad)


class SongVocalsRequest(_Body):
    revision: int
    stemPath: str | None = Field(None, min_length=1, max_length=4096)  # the song's vocal stem; None slices the song
    singing: bool = True
    leadIn: float = Field(0.4, ge=0, le=1)
    tail: float = Field(0.5, ge=0, le=1)  # room after each phrase (FU-609), cut off again by trimOut


class LyricCaptionsRequest(_Body):
    revision: int
    timing: Literal["auto", "sections", "speech"] = "auto"
    language: str = Field("", max_length=16)


def _shared() -> Any:
    from backend_service.routes import storyboard as shared

    return shared


# ---- shared with the agent tools ----
def resume_batch(request: Request, project_id: str, batch_id: str, revision: int | None) -> Any:
    """Continue the paused batch ``batch_id`` from its remaining steps, with the same wiring as a new batch."""
    from backend_service.storyboard.store import StoreError

    shared = _shared()
    project = shared.get_store().load(project_id)
    record = next((b for b in project.batches if b.id == batch_id), None)
    if record is None:
        raise StoreError(f"Unknown batch '{batch_id}'.")
    batch = shared.build_batch(request, project_id, record.what, None, record.force, record.draft)
    batch.resume(batch_id, revision)
    batch.start_thread()
    return batch


def attach_song(project_id: str, revision: int, library_id: str | None, path: str | None, title: str | None,
                add_to_mixer: bool = True) -> tuple[Any, dict[str, Any]]:
    """Check and analyse the song (no lock held), then store it with ``set_song``. Returns (project, analysis)."""
    from backend_service.storyboard import song
    from backend_service.storyboard.ops import run_op

    store = _shared().get_store()
    store.load(project_id)  # an unknown project is refused before the slow part
    lyrics: list[str] = []
    hint = None
    if library_id is not None:
        source, meta = song.library_source(library_id)
        name = title or str(meta.get("title") or "")
        lyrics = song.lyric_lines(str(meta.get("lyrics") or ""))
        bpm = meta.get("bpm")
        hint = float(bpm) if isinstance(bpm, (int, float)) and not isinstance(bpm, bool) and 30 <= bpm <= 400 else None
    else:
        source = song.check_source(str(path))
        name = title or source.stem
    found = song.analyse(source, hint)
    args = {"path": str(source), "title": " ".join(name.split())[:200], "seconds": found["seconds"], "bpm": found["bpm"],
            "beats": found["beats"], "sections": found["sections"], "lyrics": lyrics, "libraryId": library_id,
            "note": found["note"], "addToMixer": add_to_mixer}
    project = run_op(store, project_id, revision, "set_song", args)
    analysis = {"seconds": found["seconds"], "bpm": found["bpm"], "beats": len(found["beats"]),
                "sections": found["sections"], "rough": found["rough"], "note": found["note"]}
    return project, analysis


def cut_plan(request: Request, project_id: str, max_shot_seconds: float | None = None, min_shot_seconds: float = 2.0,
             keep_prompts: bool = True, length_seconds: float | None = None) -> dict[str, Any]:
    """Shots cut to the song, capped by the project's video model clip length; nothing is written."""
    from backend_service.storyboard import cutplan

    shared = _shared()
    project = shared.get_store().load(project_id)
    variant, _ = shared._project_variant(request, "video", project.defaults.videoModel)
    cap, source = cutplan.clip_seconds_for(variant, project.defaults.fps, max_shot_seconds)
    # No length asked: the film's length from the Plan tab (its Length chips), so "30 s" and a long song give a 30 s cut.
    length = length_seconds if length_seconds is not None else project.plan.filmSeconds
    plan = cutplan.plan_cuts(project, cap, min_shot_seconds, keep_prompts, length)
    return {**plan, "clipSeconds": round(cap, 3), "clipSecondsSource": source,
            "videoModel": (variant or {}).get("name")}


def song_vocals(project_id: str, revision: int, stem_path: str | None, singing: bool = True,
                lead_in: float = 0.4, tail: float = 0.5) -> tuple[Any, dict[str, Any]]:
    """Song to lip-sync shots (FU-371): slice the vocal stem (or the song) at the shots and attach each slice in one
    op. With a stem the cuts between sung shots move to phrase starts (FU-607) and each slice runs on past its phrase
    (FU-609). Returns (project, summary). The slices are cut before the project lock is taken."""
    import tempfile
    from pathlib import Path

    from backend_service import lipsync_runtime
    from backend_service.storyboard import media, song
    from backend_service.storyboard import song_vocals as sv
    from backend_service.storyboard.ops import run_op

    store = _shared().get_store()
    project = store.load(project_id)
    if project.song is None:
        raise ValueError("This project has no song. Attach one first (Song > Attach).")
    if stem_path:
        source = song.check_source(stem_path)
        voiced = sv.voiced_frames(song.decode(source), song.SAMPLE_RATE)
    else:
        source, voiced = media.inside(store.folder(project_id), project.song.path), None
    plan = sv.plan_vocals(project, voiced, lead_in, tail)
    summary = {"source": "stem" if stem_path else "song", "attached": len(plan["items"]), "skipped": plan["skipped"],
               "snapped": plan["snapped"]}
    if not plan["items"]:
        return project, summary
    with tempfile.TemporaryDirectory(prefix="chaosengine-vocals-") as tmp:
        items = []
        for item in plan["items"]:
            dst = Path(tmp) / f"{item['shotId']}.wav"
            try:
                shown = item["lead"] + item["length"]
                lipsync_runtime.prepare_audio(source, dst, item["songStart"], shown + item["tail"],
                                              keep=None if item["natural"] else shown)
            except lipsync_runtime.LipSyncError as exc:
                raise ValueError(str(exc)) from exc
            items.append({"shotId": item["shotId"], "audio": str(dst), "lead": item["lead"], "length": item["length"],
                          "tail": item["tail"], "singing": singing})
        project = run_op(store, project_id, revision, "set_shot_vocals", {"items": items})
    return project, summary


# ---- routes ----
@router.post("/api/storyboard/projects/{project_id}/batches/{batch_id}/pause")
def pause_batch(project_id: str, batch_id: str) -> dict[str, Any]:
    """Stop the run after the step in progress; the batch becomes ``paused`` with its remaining steps kept."""
    from backend_service.storyboard.batch import get_batch

    batch = get_batch(batch_id)
    if batch is None or batch.project_id != project_id or batch.done.is_set():
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' is not running.")
    return {"pausing": batch.pause()}


@router.post("/api/storyboard/projects/{project_id}/batches/{batch_id}/resume")
def resume(request: Request, project_id: str, batch_id: str, body: ResumeRequest) -> dict[str, Any]:
    shared = _shared()
    batch = shared._guard(resume_batch, request, project_id, batch_id, body.revision)
    return {"batchId": batch.id, **shared._batch_info(project_id, batch.id)}


@router.post("/api/storyboard/projects/{project_id}/song")
def attach(project_id: str, body: SongRequest) -> dict[str, Any]:
    """Attach a song from the Music library or a file, analyse it and (by default) put it on the mixer."""
    shared = _shared()
    project, analysis = shared._guard(attach_song, project_id, body.revision, body.libraryId, body.path, body.title,
                                      body.addToMixer)
    return {**shared._payload(project), "analysis": analysis}


@router.post("/api/storyboard/projects/{project_id}/cut-plan")
def propose_cut_plan(request: Request, project_id: str, body: CutPlanRequest) -> dict[str, Any]:
    """Shots from the song's sections and beats, in the Plan tab's shape; apply with ``apply_plan`` (replace)."""
    shared = _shared()
    return shared._guard(cut_plan, request, project_id, body.maxShotSeconds, body.minShotSeconds, body.keepPrompts,
                         body.lengthSeconds)


@router.post("/api/storyboard/projects/{project_id}/song/vocals")
def attach_song_vocals(project_id: str, body: SongVocalsRequest) -> dict[str, Any]:
    """Give each shot its slice of the song's vocal for lip sync (a stem if given), sung, with a 0.4 s lead-in."""
    shared = _shared()
    project, summary = shared._guard(song_vocals, project_id, body.revision, body.stemPath, body.singing, body.leadIn,
                                     body.tail)
    return {**shared._payload(project), "vocals": summary}


@router.get("/api/storyboard/lyrics/speech")
def lyrics_speech() -> dict[str, Any]:
    """Whether lyric captions can be timed by speech-to-text on this machine, and why not."""
    from backend_service.storyboard import lyrics

    ready, reason = lyrics.speech_status()
    return {"ready": ready, "reason": reason}


@router.post("/api/storyboard/projects/{project_id}/lyrics/captions")
def lyric_captions(project_id: str, body: LyricCaptionsRequest) -> dict[str, Any]:
    """Time the song's lyrics as captions on its mixer clip (sections, or speech-to-text when installed)."""
    from backend_service.storyboard import lyrics

    shared = _shared()
    try:
        project, used, notes = shared._guard(lyrics.make, shared.get_store(), project_id, body.revision, body.timing,
                                             body.language)
    except HTTPException as exc:
        cause = exc.__cause__
        if isinstance(cause, lyrics.LyricsError) and cause.code:
            raise HTTPException(status_code=409, detail={"code": cause.code, "message": str(cause)}) from cause
        raise
    return {**shared._payload(project), "timing": used, "notes": notes}
