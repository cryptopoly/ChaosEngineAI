"""Storyboard agent tools, grouped by verb (D-30): project, shot, audio, generate, render.

They run the same store, ops, batch and render code as ``/api/storyboard``. Edits carry the project's
``revision``; a stale one is a readable error with the current number. ``storyboard_generate`` and
``storyboard_render`` return a job id for ``wait_for_job``; a finished render carries ``download``. Export
and render answers name the file and its ``download``, never a local path (the UI routes keep paths).
Descriptions are short on purpose: each one costs an agent tokens on every call.
"""

from __future__ import annotations

from typing import Any, Callable

from backend_service.mcp_bridge.jobs import STORE
from backend_service.mcp_bridge.tools import ToolError, _obj, _submitted

_PID = {"type": "string", "description": "Project id."}
_REV = {"type": "integer", "minimum": 1, "description": "Project revision you last saw (from get)."}
_FIELDS = {"type": "object", "description": "Fields to set, as in the project JSON."}

STORYBOARD_TOOLS: list[dict[str, Any]] = [
    {
        "name": "storyboard_project",
        "description": "Edit: film projects. create(name), list, get (shots, takes, timeline, revision), edit(fields: name, "
        "defaults, export{captionsSrt, captionsBurn, stems, ...}), undo, redo, delete, export (.chaosfilm), import.",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["create", "list", "get", "edit", "undo", "redo", "delete", "export", "import"]},
            "project_id": _PID, "revision": _REV,
            "name": {"type": "string", "minLength": 1, "maxLength": 200},
            "fields": _FIELDS,
            "path": {"type": "string", "description": "Path of a .chaosfilm file in a storyboard exports folder (import)."},
            "file": {"type": "string", "description": "Bundle file name an export returned, with its project_id (import)."},
        }, ["action"]),
    },
    {
        "name": "storyboard_shot",
        "description": "Edit: shots. add(fields), edit(shot_id, fields), remove, reorder(order), lock, unlock, "
        "pick_take(take_id), redo (marks for the next generate), import(path of a clip or still).",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["add", "edit", "remove", "reorder", "lock", "unlock", "pick_take", "redo", "import"]},
            "project_id": _PID, "revision": _REV,
            "shot_id": {"type": "string"}, "take_id": {"type": "string"},
            "fields": {**_FIELDS, "description": "title, stillPrompt, motionPrompt, startFrom (generate|upload|previous-shot), "
                       "seconds, seed, trimIn, trimOut, speed, transition{type,seconds}, nativeSoundDb, index."},
            "order": {"type": "array", "items": {"type": "string"}, "maxItems": 500},
            "path": {"type": "string", "description": "upload: handle, or a file in the app's output folders."},
        }, ["action", "project_id", "revision"]),
    },
    {
        "name": "storyboard_audio",
        "description": "Edit: audio lanes. add(kind, path, fields), edit(clip_id, fields), remove(clip_id). Anchor to a shot "
        "with fields.anchor {shotId, offset} so the sound follows it.",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["add", "edit", "remove"]},
            "project_id": _PID, "revision": _REV, "clip_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["music", "sfx", "voice", "import"]},
            "path": {"type": "string", "description": "upload: handle, or a file in the app's output folders."},
            "fields": {**_FIELDS, "description": "anchor{shotId,offset} or start, trimIn, trimOut, gainDb, fadeIn, fadeOut, duck, loop, label."},
        }, ["action", "project_id", "revision"]),
    },
    {
        "name": "storyboard_generate",
        "description": "Make: new takes: all stills, then all clips. Default: shots that need work, skipping locked. "
        "Returns a job id for wait_for_job.",
        "inputSchema": _obj({
            "project_id": _PID, "revision": _REV,
            "shot_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 500},
            "what": {"type": "string", "enum": ["still", "clip", "both"]},
            "force": {"type": "boolean", "description": "Also run locked shots."},
        }, ["project_id", "revision"]),
    },
    {
        "name": "storyboard_render",
        "description": "Make: render the film to MP4 (preview: small and fast). Returns a job id; the output has the file, download, "
        "warnings and extras (captions, stems). A failed render can be resumed by calling again.",
        "inputSchema": _obj({"project_id": _PID, "revision": _REV, "preview": {"type": "boolean"},
                             "notify": {"type": "boolean", "description": "Show a system banner on this computer when it ends."}},
                            ["project_id", "revision"]),
    },
    {
        "name": "storyboard_captions",
        "description": "Edit: captions. generate(clip_ids, language; transcribes voice clips, a job), export_srt (download), "
        "set(clip_id, cues), add(start, end, text), edit(caption_id), remove(caption_id).",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["generate", "export_srt", "set", "add", "edit", "remove"]},
            "project_id": _PID, "revision": _REV,
            "clip_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 200, "description": "generate: voice clips (default all)."},
            "language": {"type": "string", "maxLength": 16},
            "clip_id": {"type": "string", "description": "set, add: the audio clip the captions belong to (none: the timeline's own)."},
            "caption_id": {"type": "string", "maxLength": 64},
            "cues": {"type": "array", "maxItems": 500, "items": {"type": "object"}, "description": "set: [{start, end, text}] seconds."},
            "start": {"type": "number", "minimum": 0}, "end": {"type": "number", "minimum": 0},
            "text": {"type": "string", "minLength": 1, "maxLength": 500},
        }, ["action", "project_id"]),
    },
    {
        "name": "storyboard_export_otio",
        "description": "Make: write the cut as an OpenTimelineIO file for Resolve or Premiere. Returns the file, download and warnings.",
        "inputSchema": _obj({"project_id": _PID}, ["project_id"]),
    },
]


def _store() -> Any:
    from backend_service.routes.storyboard import get_store

    return get_store()


def _need(args: dict[str, Any], *names: str) -> None:
    missing = [n for n in names if n not in args]
    if missing:
        raise ToolError(f"This action needs {', '.join(missing)}.")


def _summary(project: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend_service.storyboard.timeline import compute_timeline

    timeline = compute_timeline(project)
    spans = {s.id: s for s in timeline.shots}
    starts = {a.id: round(a.start, 3) for a in timeline.audio}
    return {
        "projectId": project.id,
        "revision": project.revision,
        "seconds": round(timeline.total, 3),
        "shots": [{"id": s.id, "title": s.title, "status": s.status, "locked": s.locked, "activeTake": s.activeTake,
                   "takes": len(s.takes), "seconds": round(spans[s.id].duration, 3) if s.id in spans else None}
                  for s in project.shots],
        "audio": [{"id": a.id, "kind": a.kind, "status": a.status, "start": starts.get(a.id)} for a in project.audio],
        **(extra or {}),
    }


def _run(fn: Callable[[], Any]) -> Any:
    from backend_service.storyboard.media import MediaError
    from backend_service.storyboard.render_adapter import RenderBusy
    from backend_service.storyboard.store import RevisionConflict, StoreError

    try:
        return fn()
    except RevisionConflict as exc:
        raise ToolError(f"The project is at revision {exc.current.revision} now. Call storyboard_project get and retry.") from exc
    except (StoreError, MediaError, RenderBusy, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def _op(args: dict[str, Any], op: str, op_args: dict[str, Any]) -> Any:
    from backend_service.storyboard.ops import run_op

    return _run(lambda: run_op(_store(), str(args["project_id"]), int(args["revision"]), op, op_args))


_OUTSIDE = ("Agents can import only an upload: handle or a file inside the app's own output folders (images, videos, "
            "voice and music libraries, storyboard exports). Send other files with upload_file or POST /api/agent/uploads "
            "and pass the handle.")


def _agent_source(raw: Any) -> str:
    """An agent-named input: an ``upload:`` handle, or a file inside ``output_roots()``; anything else is refused.

    The HTTP routes (the user at the UI) may import from any path; an agent may not, or a later export would
    hand it a copy of any media file on the machine.
    """
    from backend_service.mcp_bridge.files import HANDLE_PREFIX, fetchable

    text = str(raw or "").strip()
    if text.startswith(HANDLE_PREFIX):
        return text
    real = fetchable(text) if text else None
    if real is None:
        raise ToolError(_OUTSIDE)
    return str(real)


def _added(before: Any, after: Any, attr: str) -> str | None:
    old = {x.id for x in getattr(before, attr)}
    return next((x.id for x in getattr(after, attr) if x.id not in old), None)


def storyboard_project(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.mcp_bridge.files import download_info
    from backend_service.storyboard import bundle
    from backend_service.storyboard.models import dump
    from backend_service.storyboard.timeline import compute_timeline

    action, store = args["action"], _store()
    if action == "list":
        return {"projects": store.list()}
    if action == "create":
        _need(args, "name")
        return _summary(_run(lambda: store.create(str(args["name"]), (args.get("fields") or {}).get("defaults"))))
    if action == "import":
        from pathlib import Path

        if "file" in args:  # a bundle an earlier export made: that project's exports/<file>
            _need(args, "project_id")
            name = str(args["file"])
            exports = _run(lambda: store.folder(str(args["project_id"]))) / "exports"
            if Path(name).name != name or not name.endswith(bundle.BUNDLE_SUFFIX) or not (exports / name).is_file():
                raise ToolError(f"'{name[:80]}' is not an export of that project. Use the file name storyboard_project export returned.")
            src = exports / name
        else:
            _need(args, "path")
            if str(args["path"]).startswith("upload:"):
                raise ToolError("A .chaosfilm bundle cannot be uploaded; import one from a storyboard exports folder.")
            src = Path(_agent_source(args["path"]))
        return _summary(_run(lambda: bundle.import_bundle(store, src)))
    _need(args, "project_id")
    pid = str(args["project_id"])
    if action == "get":
        project = _run(lambda: store.load(pid))
        return {"project": dump(project), "timeline": compute_timeline(project).as_dict()}
    if action == "export":
        path = _run(lambda: bundle.export_bundle(store, pid))
        return {"file": path.name, "download": download_info(path)}  # no local path for the agent
    _need(args, "revision")
    rev = int(args["revision"])
    if action == "edit":
        return _summary(_op(args, "edit_project", dict(args.get("fields") or {})))
    if action == "undo":
        return _summary(_run(lambda: store.undo(pid, rev)))
    if action == "redo":
        return _summary(_run(lambda: store.redo(pid, rev)))
    return _run(lambda: store.delete(pid, rev))


def storyboard_shot(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    action = args["action"]
    fields = dict(args.get("fields") or {})
    if action == "add":
        before = _run(lambda: _store().load(str(args["project_id"])))
        after = _op(args, "add_shot", fields)
        return _summary(after, {"shotId": _added(before, after, "shots")})
    if action == "reorder":
        _need(args, "order")
        return _summary(_op(args, "reorder", {"order": list(args["order"])}))
    _need(args, "shot_id")
    shot = {"shotId": args["shot_id"]}
    if action == "edit":
        return _summary(_op(args, "edit_shot", {**fields, **shot}))
    if action == "remove":
        return _summary(_op(args, "remove_shot", shot))
    if action in ("lock", "unlock"):
        return _summary(_op(args, "lock", {**shot, "locked": action == "lock"}))
    if action == "pick_take":
        _need(args, "take_id")
        return _summary(_op(args, "pick_take", {**shot, "takeId": args["take_id"]}))
    if action == "redo":
        return _summary(_op(args, "mark_redo", {**shot, "redo": True}))
    _need(args, "path")
    from backend_service.storyboard.media import IMAGE_EXTS

    source = _agent_source(args["path"])
    if source.startswith("upload:"):
        from backend_service.helpers.media_input import InputImageError
        from backend_service.mcp_bridge.files import UPLOAD_EXTENSIONS, resolve_handle

        try:
            is_image = resolve_handle(source, UPLOAD_EXTENSIONS).suffix in IMAGE_EXTS
        except InputImageError as exc:
            raise ToolError(str(exc)) from exc
    else:
        is_image = any(source.lower().endswith(e) for e in IMAGE_EXTS)
    return _summary(_op(args, "import_take", {**shot, ("still" if is_image else "clip"): source}))


def storyboard_audio(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    action = args["action"]
    fields = dict(args.get("fields") or {})
    if action == "add":
        _need(args, "kind", "path")
        before = _run(lambda: _store().load(str(args["project_id"])))
        after = _op(args, "add_audio", {**fields, "kind": args["kind"], "path": _agent_source(args["path"])})
        return _summary(after, {"clipId": _added(before, after, "audio")})
    _need(args, "clip_id")
    if action == "edit":
        return _summary(_op(args, "edit_audio", {**fields, "clipId": args["clip_id"]}))
    return _summary(_op(args, "remove_audio", {"clipId": args["clip_id"]}))


def storyboard_generate(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes.storyboard import start_batch

    pid = str(args["project_id"])
    batch = _run(lambda: start_batch(ctx.request, pid, int(args["revision"]), str(args.get("what", "both")),
                                     list(args["shot_ids"]) if "shot_ids" in args else None, bool(args.get("force"))))

    def run() -> tuple[dict[str, Any], list[str]]:
        batch.done.wait()
        record = batch.record()
        notes = [f"{r.shotId} {r.kind}: {r.state}{' - ' + r.error if r.error else ''}" for r in record.results]
        return {"batchId": batch.id, "state": record.state, "projectId": pid}, notes

    return _submitted(STORE.submit("storyboard", f"storyboard {batch.id}", run, cancel=batch.cancel))


def _extras(pid: str, files: list[str]) -> list[dict[str, Any]]:
    """Files a render wrote next to the film (captions, stems): name plus download, never a local path."""
    from backend_service.mcp_bridge.files import download_info

    folder = _store().folder(pid)
    return [{"file": rel, "download": download_info(folder / rel)} for rel in files if (folder / rel).is_file()]


def storyboard_render(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes.storyboard import start_render
    from backend_service.storyboard.render_adapter import RENDERS

    pid = str(args["project_id"])
    job = _run(lambda: start_render(pid, int(args["revision"]), bool(args.get("preview", True)), bool(args.get("notify", False))))

    def run() -> tuple[dict[str, Any], list[str]]:
        job.done.wait()
        if job.state == "cancelled":
            from backend_service.progress import GenerationCancelled

            raise GenerationCancelled("render cancelled")
        if job.state != "done":
            info = job.describe()
            resume = (f" {info['shotsReady']} of {info['shotsTotal']} shots are rendered; call storyboard_render again to resume."
                      if info.get("resumable") else "")
            raise RuntimeError((job.error or f"render {job.state}") + resume)
        from backend_service.mcp_bridge.files import download_info

        # The file name and the download, not the local path (the job store adds download only for "path").
        output: dict[str, Any] = {"file": job.out_path.name, "preview": job.preview, "projectId": pid,
                                  "download": download_info(job.out_path), "extras": _extras(pid, job.extras)}
        if job.warnings:
            output["warnings"] = list(job.warnings)
        return output, list(job.warnings)

    return _submitted(STORE.submit("storyboard", f"render {pid}", run, cancel=lambda: RENDERS.cancel(pid)))


def storyboard_captions(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.mcp_bridge.files import download_info
    from backend_service.storyboard import captions
    from backend_service.storyboard.timeline import compute_timeline

    action, pid, store = args["action"], str(args["project_id"]), _store()
    if action == "export_srt":
        project = _run(lambda: store.load(pid))
        items = captions.cues(project, compute_timeline(project, ready_only=True))
        if not items:
            raise ToolError("There are no captions to export. Call storyboard_captions generate first.")
        path = _run(lambda: captions.write_srt(project, store.folder(pid), items))
        return {"file": path.name, "cues": len(items), "download": download_info(path)}
    _need(args, "revision")
    rev = int(args["revision"])
    clip = {"clipId": args["clip_id"]} if "clip_id" in args else {}
    if action == "generate":
        language, clip_ids = str(args.get("language") or ""), list(args["clip_ids"]) if "clip_ids" in args else None

        def run() -> tuple[dict[str, Any], list[str]]:
            project, notes = captions.generate(store, pid, rev, clip_ids, language)
            return {"projectId": pid, "revision": project.revision, "captions": len(project.captions)}, list(notes)

        return _submitted(STORE.submit("voice", f"captions {pid}", run))
    if action == "set":
        _need(args, "cues")
        return _summary(_op(args, "set_captions", {"cues": list(args["cues"]), **clip}))
    if action == "add":
        _need(args, "start", "end", "text")
        return _summary(_op(args, "add_caption", {k: args[k] for k in ("start", "end", "text")} | clip))
    _need(args, "caption_id")
    if action == "edit":
        changes = {k: args[k] for k in ("start", "end", "text") if k in args}
        return _summary(_op(args, "edit_caption", {"captionId": args["caption_id"], **changes}))
    return _summary(_op(args, "remove_caption", {"captionId": args["caption_id"]}))


def storyboard_export_otio(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.mcp_bridge.files import download_info
    from backend_service.storyboard import otio_export

    pid, store = str(args["project_id"]), _store()
    project = _run(lambda: store.load(pid))
    path, warnings = _run(lambda: otio_export.write_otio(project, store.folder(pid)))
    return {"file": path.name, "download": download_info(path), "warnings": warnings}


STORYBOARD_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "storyboard_project": storyboard_project,
    "storyboard_shot": storyboard_shot,
    "storyboard_audio": storyboard_audio,
    "storyboard_generate": storyboard_generate,
    "storyboard_render": storyboard_render,
    "storyboard_captions": storyboard_captions,
    "storyboard_export_otio": storyboard_export_otio,
}
