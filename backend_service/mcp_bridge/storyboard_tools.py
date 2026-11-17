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
from backend_service.mcp_bridge.storyboard_guidance import lanes, parse_size, render_warnings, settings_info

_PID = {"type": "string", "description": "Project id."}
_REV = {"type": "integer", "minimum": 1, "description": "Project revision you last saw (from get)."}
_FIELDS = {"type": "object", "description": "Fields to set, as in the project JSON."}

STORYBOARD_TOOLS: list[dict[str, Any]] = [
    {
        "name": "storyboard_project",
        "description": "Edit: film projects. create(name, size or aspect+resolution, fps, image_model, video_model, voice), "
        "list, get, edit(same, or fields), undo, redo, delete, export, import, add_ref/edit_ref/remove_ref (cast), plan, brand.",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["create", "list", "get", "edit", "undo", "redo", "delete", "export", "import",
                                                  "add_ref", "edit_ref", "remove_ref", "plan", "brand"]},
            "kind": {"type": "string", "enum": ["character", "place", "style"],
                     "description": "add_ref. character, place: keeps a face or place the same across shots. style: the look of "
                     "the whole film (pictures, words, palette read from them)."},
            "ref_id": {"type": "string", "maxLength": 64, "description": "edit_ref, remove_ref. Shots use it in fields.referenceImages."},
            "description": {"type": "string", "maxLength": 4000, "description": "add_ref, edit_ref: what it looks like."},
            "images": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 4096},
                       "description": "add_ref: up to 4 pictures (upload: handles, output files, or a take's media/ still). Make one "
                       "hero picture first with generate_image and pass its output path. Shots then list the ref id in "
                       "fields.referenceImages and leave seed unset."},
            "add_image": {"type": "string", "maxLength": 4096, "description": "edit_ref: one more picture (4 at most)."},
            "remove_image": {"type": "integer", "minimum": 0, "maximum": 3, "description": "edit_ref: position of a picture to drop."},
            "palette": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 7},
                        "description": "edit_ref on a style: its colours, trimmed."},
            "brief": {"type": "string", "maxLength": 4000, "description": "plan: the idea of the film."},
            "film_seconds": {"type": "number", "minimum": 0, "maximum": 600, "description": "plan: target length (5 to 600); 0 clears."},
            "logo": {"type": "string", "maxLength": 4096, "description": "brand: logo picture (upload: handle or output file); \"\" removes."},
            "logo_corner": {"type": "string", "enum": ["top-left", "top-right", "bottom-left", "bottom-right"]},
            "style_use": {"type": "string", "enum": ["auto", "words", "off"], "description": "brand: how shots take the style."},
            "caption": {"type": "object", "description": "brand: caption look {font (installed family, \"\" for default), colour #RRGGBB, "
                        "position bottom|middle|top}; also the card typeface and colour."},
            "project_id": _PID, "revision": _REV,
            "name": {"type": "string", "minLength": 1, "maxLength": 200,
                     "description": "create: the project's name. add_ref (required), edit_ref: the cast entry's name."},
            "fields": {**_FIELDS, "description": "edit: fields as in the project JSON. defaults{width, height, fps, imageModel, "
                       "finalImageModel, videoModel, voice, exportPreset, resolution}; export{captionsBurn (draw captions into "
                       "the picture), captionsSrt, stems, upscale, loudnessLufs}."},
            "size": {"type": "string", "maxLength": 16, "description": "create, edit: exact film size WIDTHxHEIGHT, e.g. 1024x576 "
                     "(even, 64 to 4096; sets the aspect too). Multiples of 64 need no resize for LTX clips."},
            "aspect": {"type": "string", "enum": ["16:9", "9:16", "1:1"], "description": "Aspect ratio (default 16:9)."},
            "shape": {"type": "string", "enum": ["16:9", "9:16", "1:1"], "description": "Old name for aspect."},
            "resolution": {"type": "string", "enum": ["480p", "720p", "1080p", "1440p", "2160p"],
                           "description": "Size from the aspect: 720p (default) at 16:9 is 1280x720. size overrides it."},
            "fps": {"type": "integer", "minimum": 24, "maximum": 60, "description": "24 (default), 25, 30, 48, 50 or 60."},
            "image_model": {"type": "string", "maxLength": 200,
                            "description": "create, edit: image model id for stills (list_models image). Unset: the fastest installed."},
            "video_model": {"type": "string", "maxLength": 200,
                            "description": "create, edit: video model id for clips (list_models video). Unset: the best installed fit."},
            "voice": {"type": "string", "maxLength": 200,
                      "description": "create, edit: the project's default narration voice id (list_voices)."},
            "upscale": {"type": "string", "enum": ["none", "1440p", "2160p"], "description": "Export upscale; none keeps the project size."},
            "path": {"type": "string", "description": "Path of a .chaosfilm file in a storyboard exports folder (import)."},
            "file": {"type": "string", "description": "Bundle file name an export returned, with its project_id (import)."},
        }, ["action"]),
    },
    {
        "name": "storyboard_shot",
        "description": "Edit: shots. add(fields), edit(shot_id, fields), remove, reorder(order), lock, unlock, "
        "pick_take(take_id), redo (marks for the next generate), import(path of a clip or still), "
        "end_image(path; none clears; LTX-2).",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["add", "edit", "remove", "reorder", "lock", "unlock", "pick_take", "redo", "import", "end_image"]},
            "project_id": _PID, "revision": _REV,
            "shot_id": {"type": "string"}, "take_id": {"type": "string"},
            "fields": {**_FIELDS, "description": "title, stillPrompt, motionPrompt, startFrom (generate|upload|previous-shot|text), "
                       "seconds, seed (leave unset on shots with referenceImages: the reference's seed returns a copy of it), trimIn, trimOut, "
                       "speed, transition{type,seconds}, nativeSoundDb, index."},
            "order": {"type": "array", "items": {"type": "string"}, "maxItems": 500},
            "path": {"type": "string", "description": "upload: handle, or a file in the app's output folders (import, end_image)."},
        }, ["action", "project_id", "revision"]),
    },
    {
        "name": "storyboard_audio",
        "description": "Edit: audio lanes. add(kind, path or library_id, fields), edit(clip_id, fields), remove(clip_id). "
        "A made sound plays only once added. Anchor to a shot with fields.anchor {shotId, offset}.",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["add", "edit", "remove"]},
            "project_id": _PID, "revision": _REV, "clip_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["music", "sfx", "voice", "import"]},
            "path": {"type": "string", "description": "add: the output path speak, make_music (tracks[0].path) or make_sound "
                     "returned, an upload: handle, or a file in the app's output folders."},
            "library_id": {"type": "string", "maxLength": 40, "description": "add: a Music library track id (make_music tracks[].id) "
                           "instead of path."},
            "fields": {**_FIELDS, "description": "anchor{shotId,offset} or start, trimIn, trimOut, gainDb, fadeIn, fadeOut, duck (music ducks under voice "
                       "unless duck=false), loop, label."},
        }, ["action", "project_id", "revision"]),
    },
    {
        "name": "storyboard_generate",
        "description": "Make: new takes: all stills, then all clips. Default: shots that need work, skipping locked. "
        "Returns a job id for wait_for_job.",
        "inputSchema": _obj({
            "project_id": _PID, "revision": _REV,
            "shot_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 500},
            "what": {"type": "string", "enum": ["still", "clip", "both", "final", "redo-clip", "redo-still"],
                     "description": "final: redo only the shots marked Keep for final, with the quality image model. "
                     "redo-clip: new clip from the current still; redo-still: new still, old clip kept until remade."},
            "draft": {"type": "boolean", "description": "Make drafts: the fast image model at a smaller size."},
            "force": {"type": "boolean", "description": "Also run locked shots."},
        }, ["project_id", "revision"]),
    },
    {
        "name": "storyboard_batch",
        "description": "Make: a generate batch. status, pause (stops after the step in progress; a restart also pauses), "
        "resume(revision; a job id like generate), cancel.",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["status", "pause", "resume", "cancel"]},
            "project_id": _PID, "revision": _REV,
            "batch_id": {"type": "string", "maxLength": 64, "description": "From generate's output or the project's batches."},
        }, ["action", "project_id", "batch_id"]),
    },
    {
        "name": "storyboard_song",
        "description": "Edit: the song a music video is cut to. attach(library_id or path), get, edit(lyrics, sections, bpm, "
        "title), remove, cut_plan(seconds; shots on its beats; apply: true replaces shots, one undo), lyric_captions(timing).",
        "inputSchema": _obj({
            "action": {"type": "string", "enum": ["attach", "get", "edit", "remove", "cut_plan", "lyric_captions"]},
            "project_id": _PID, "revision": _REV,
            "library_id": {"type": "string", "maxLength": 40, "description": "attach: a Music library track id."},
            "path": {"type": "string", "description": "attach: upload: handle or a file in the app's output folders."},
            "title": {"type": "string", "maxLength": 200},
            "lyrics": {"type": "array", "maxItems": 500, "items": {"type": "string", "maxLength": 200},
                       "description": "edit: one caption line each."},
            "sections": {"type": "array", "maxItems": 200, "items": {"type": "object"},
                         "description": "edit: [{start, end, label}] seconds."},
            "bpm": {"type": "number", "minimum": 0, "maximum": 400, "description": "edit: 0 clears."},
            "max_shot_seconds": {"type": "number", "minimum": 0.5, "maximum": 60, "description": "cut_plan: lower cap."},
            "seconds": {"type": "number", "minimum": 1, "maximum": 3600,
                        "description": "cut_plan: the film's length (a 30 s ad: 30); only the song's first seconds are cut."},
            "apply": {"type": "boolean", "description": "cut_plan: replace the shots with the plan (one undo)."},
            "timing": {"type": "string", "enum": ["auto", "sections", "speech"],
                       "description": "lyric_captions: speech needs speech-to-text installed."},
            "language": {"type": "string", "maxLength": 16},
        }, ["action", "project_id"]),
    },
    {
        "name": "storyboard_render",
        "description": "Make: render the film to MP4 (preview by default). Returns a job id and warnings; the output has the file, "
        "download, warnings and extras (captions, stems). A failed render resumes when called again.",
        "inputSchema": _obj({"project_id": _PID, "revision": _REV,
                             "preview": {"type": "boolean", "description": "true (default) makes a small fast draft, not the finished "
                                         "film; pass preview: false for the final MP4."},
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
        "cast": [{"id": r.id, "kind": kind, "name": r.name, "pictures": len(r.images)}
                 for kind, refs in (("character", project.references.characters), ("place", project.references.places),
                                    ("style", project.references.styles))
                 for r in refs],
        **(extra or {}),
    }


def _run(fn: Callable[[], Any]) -> Any:
    from backend_service.storyboard.media import MediaError
    from backend_service.storyboard.render_adapter import RenderBusy
    from backend_service.storyboard.store import ProjectNotFound, RevisionConflict, StoreError

    try:
        return fn()
    except RevisionConflict as exc:
        raise ToolError(f"The project is at revision {exc.current.revision} now. Call storyboard_project get and retry.") from exc
    except ProjectNotFound as exc:
        raise ToolError(f"{exc} {_project_id_hint(str(exc))}") from exc
    except (StoreError, MediaError, RenderBusy, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def _project_id_hint(message: str) -> str:
    """How to fix a project name passed as project_id (S-P: chat models of every size sent the name first, and a
    1.7B model never recovered): the id of the project with that name, or where ids come from."""
    import re

    asked = re.search(r"'(.*)'", message)
    name = asked.group(1).strip().casefold() if asked else ""
    rows = [r for r in _store().list() if str(r.get("name", "")).strip().casefold() == name] if name else []
    if len(rows) == 1:
        return f"project_id is the project's id, not its name: '{rows[0]['name']}' has the id {rows[0]['id']}."
    return "project_id is the id from storyboard_project list, not the project's name."


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


def _cast_source(raw: Any) -> str:
    """A cast picture: what ``_agent_source`` allows, or a still of this project by its ``media/`` path (from get),
    which the op keeps inside the project folder."""
    text = str(raw or "").strip()
    if text.startswith("media/"):
        return text
    return _agent_source(text)


def _added(before: Any, after: Any, attr: str) -> str | None:
    old = {x.id for x in getattr(before, attr)}
    return next((x.id for x in getattr(after, attr) if x.id not in old), None)


# Top-level arguments that set a project default, and the project field each one sets.
_DEFAULT_ARGS = {"aspect": "aspect", "shape": "shape", "resolution": "resolution", "fps": "fps",
                 "image_model": "imageModel", "video_model": "videoModel", "voice": "voice"}


def _format_fields(args: dict[str, Any]) -> dict[str, Any]:
    """``fields`` with the top-level format and model arguments (size, aspect, shape, resolution, fps, image_model,
    video_model, voice, upscale) folded in. An exact ``size`` sets width and height, and the aspect when none is given."""
    fields = dict(args.get("fields") or {})
    defaults = dict(fields.get("defaults") or {})
    for key, field in _DEFAULT_ARGS.items():
        if args.get(key) is not None:
            defaults[field] = args[key]
    if args.get("size") is not None:
        width, height = parse_size(args["size"])
        shape = "16:9" if width > height else "9:16" if width < height else "1:1"
        asked = defaults.get("aspect", defaults.get("shape"))
        if asked is not None and asked != shape:
            raise ToolError(f"size {width}x{height} does not fit aspect {asked}; leave aspect out and size sets it.")
        defaults.pop("shape", None)
        defaults.update(width=width, height=height, aspect=shape)
    if defaults:
        fields["defaults"] = defaults
    if args.get("upscale") is not None:
        fields["export"] = {**(fields.get("export") or {}), "upscale": args["upscale"]}
    return fields


def _plan_args(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "brief" in args:
        out["brief"] = str(args["brief"] or "")
    if "film_seconds" in args:
        if not args["film_seconds"]:
            out["clearFilmSeconds"] = True
        else:
            out["filmSeconds"] = args["film_seconds"]
    if not out:
        raise ToolError("plan needs brief or film_seconds.")
    return out


def _brand_args(args: dict[str, Any]) -> dict[str, Any]:
    """set_brand arguments from the tool's: the logo is an agent input (upload handle or output file), the caption font
    must be installed on this computer."""
    out: dict[str, Any] = {}
    if "logo" in args:
        if args["logo"] is None or str(args["logo"]).strip() == "":
            out["clearLogo"] = True
        else:
            out["logo"] = _cast_source(args["logo"])
    if args.get("logo_corner") is not None:
        out["logoCorner"] = args["logo_corner"]
    if args.get("style_use") is not None:
        out["styleUse"] = args["style_use"]
    caption = args.get("caption")
    if caption is not None:
        if not isinstance(caption, dict) or set(caption) - {"font", "colour", "position"}:
            raise ToolError("caption takes font, colour and position.")
        look: dict[str, Any] = {k: caption[k] for k in ("colour", "position") if caption.get(k) is not None}
        if isinstance(look.get("colour"), str):
            look["colour"] = look["colour"].lower()
        if "font" in caption:
            font = str(caption["font"] or "").strip()
            if not font:
                look["clearFont"] = True
            else:
                from backend_service.storyboard.fonts import installed_fonts

                if font not in installed_fonts():
                    raise ToolError(f"The font '{font[:64]}' is not installed on this computer. Use an installed family "
                                    "or \"\" for the default.")
                look["font"] = font
        out["captions"] = look
    if not out:
        raise ToolError("brand needs logo, logo_corner, style_use or caption.")
    return out


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
        fields = _format_fields(args)
        created = _run(lambda: store.create(str(args["name"]), fields.get("defaults")))
        if fields.get("export"):
            created = _op({"project_id": created.id, "revision": created.revision}, "edit_project", {"export": fields["export"]})
        return _summary(created, settings_info(ctx, created, True))
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
        fields = _format_fields(args)
        edited = _op(args, "edit_project", fields)
        return _summary(edited, settings_info(ctx, edited, False) if {"defaults", "export"} & fields.keys() else None)
    if action == "add_ref":
        _need(args, "kind")
        name = " ".join(str(args.get("name") or "").split())
        if not name:
            raise ToolError("add_ref needs a name (the character, place or style as shots refer to it).")
        op_args = {"kind": args["kind"], "name": name, "description": str(args.get("description") or ""),
                   "images": [_cast_source(x) for x in args.get("images") or []]}
        before = _run(lambda: store.load(pid))
        after = _op(args, "add_reference", op_args)
        added = next((r.id for r in after.references.all() if before.reference(r.id) is None), None)
        extra: dict[str, Any] = {"refId": added, "next": f"Shots keep it when fields.referenceImages lists '{added}'; "
                                 "leave seed unset on them."}
        if not op_args["images"] and args["kind"] != "style":
            extra["warning"] = (f"This {args['kind']} has no picture, so shots cannot keep the likeness. Make one with "
                                f"generate_image, then edit_ref add_image with its output path.")
        return _summary(after, extra)
    if action in ("edit_ref", "remove_ref"):
        _need(args, "ref_id")
        if action == "remove_ref":
            return _summary(_op(args, "remove_reference", {"refId": args["ref_id"]}))
        changes: dict[str, Any] = {k: args[k] for k in ("name", "description") if k in args}
        if args.get("add_image"):
            changes["addImage"] = _cast_source(args["add_image"])
        if "remove_image" in args:
            changes["removeImage"] = args["remove_image"]
        if "palette" in args:
            changes["palette"] = list(args["palette"] or [])
        return _summary(_op(args, "edit_reference", {"refId": args["ref_id"], **changes}))
    if action == "plan":
        return _summary(_op(args, "set_plan", _plan_args(args)))
    if action == "brand":
        return _summary(_op(args, "set_brand", _brand_args(args)))
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
    if action == "end_image":
        # No path clears it; a path is an upload handle or a file in the output folders, like any agent input.
        path = str(args.get("path") or "").strip()
        return _summary(_op(args, "set_end_image", {**shot, "image": _cast_source(path) if path else None}))
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


def _library_audio(item_id: str) -> str:
    """The audio file of a Music library track, checked like any agent input."""
    from backend_service import music_library

    path = music_library.audio_path(item_id) if music_library.is_valid_id(item_id) else None
    if path is None:
        raise ToolError(f"No Music library track '{item_id[:40]}'. Use the id make_music or music_library list returned.")
    return _agent_source(str(path))


def storyboard_audio(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    action = args["action"]
    fields = dict(args.get("fields") or {})
    if action == "add":
        _need(args, "kind")
        if ("path" in args) == ("library_id" in args):
            raise ToolError("add needs path (the output path speak, make_music or make_sound returned) or library_id "
                            "(a Music library track id), one of them.")
        source = _library_audio(str(args["library_id"])) if "library_id" in args else _agent_source(args["path"])
        before = _run(lambda: _store().load(str(args["project_id"])))
        after = _op(args, "add_audio", {**fields, "kind": args["kind"], "path": source})
        return _summary(after, {"clipId": _added(before, after, "audio"), "lanes": lanes(after)})
    _need(args, "clip_id")
    if action == "edit":
        return _summary(_op(args, "edit_audio", {**fields, "clipId": args["clip_id"]}))
    return _summary(_op(args, "remove_audio", {"clipId": args["clip_id"]}))


def storyboard_generate(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes.storyboard import start_batch

    pid = str(args["project_id"])
    batch = _run(lambda: start_batch(ctx.request, pid, int(args["revision"]), str(args.get("what", "both")),
                                     list(args["shot_ids"]) if "shot_ids" in args else None, bool(args.get("force")),
                                     bool(args.get("draft"))))

    return _batch_job(batch, pid)


def _batch_job(batch: Any, pid: str) -> dict[str, Any]:
    """A job for ``wait_for_job`` that ends when the batch run ends (done, failed, cancelled or paused)."""

    def run() -> tuple[dict[str, Any], list[str]]:
        batch.done.wait()
        record = batch.record()
        notes = [f"{r.shotId} {r.kind}: {r.state}{' - ' + r.error if r.error else ''}" for r in record.results]
        return {"batchId": batch.id, "state": record.state, "projectId": pid}, notes

    return _submitted(STORE.submit("storyboard", f"storyboard {batch.id}", run, cancel=batch.cancel,
                                 progress=batch.progress))


def storyboard_batch(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes.storyboard import _batch_info
    from backend_service.routes.storyboard_long import resume_batch
    from backend_service.storyboard.batch import get_batch
    from backend_service.storyboard.batch_queue import cancel_paused

    action, pid, bid = args["action"], str(args["project_id"]), str(args["batch_id"])

    def info() -> dict[str, Any]:
        try:
            return _batch_info(pid, bid)
        except Exception as exc:  # noqa: BLE001 - the route's 404 as a tool error
            raise ToolError(getattr(exc, "detail", None) or str(exc)) from exc

    if action == "status":
        return info()
    live = get_batch(bid)
    running = live is not None and live.project_id == pid and not live.done.is_set()
    if action == "pause":
        if not running:
            raise ToolError("That batch is not running.")
        return {"pausing": live.pause(), "batchId": bid}  # type: ignore[union-attr]
    if action == "cancel":
        if running:
            return {"cancelled": live.cancel(), "batchId": bid}  # type: ignore[union-attr]
        return {"cancelled": _run(lambda: cancel_paused(_store(), pid, bid)), "batchId": bid}
    _need(args, "revision")
    batch = _run(lambda: resume_batch(ctx.request, pid, bid, int(args["revision"])))
    return _batch_job(batch, pid)


def _song_edit(args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "title" in args:
        out["title"] = " ".join(str(args["title"] or "").split())[:200]
    if "lyrics" in args:
        out["lyrics"] = [" ".join(str(x).split())[:200] for x in args["lyrics"] or [] if str(x).strip()]
    if "sections" in args:
        out["sections"] = list(args["sections"] or [])
    if "bpm" in args:
        if args["bpm"]:
            out["bpm"] = args["bpm"]
        else:
            out["clearBpm"] = True
    if not out:
        raise ToolError("edit needs lyrics, sections, bpm or title.")
    return out


def storyboard_song(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes import storyboard_long
    from backend_service.storyboard import lyrics

    action, pid, store = args["action"], str(args["project_id"]), _store()
    if action == "get":
        project = _run(lambda: store.load(pid))
        song = project.song.model_dump(mode="json") if project.song else None
        if song is not None:
            song["beats"] = len(song["beats"])  # the count; the list is long and an agent rarely needs it
        return {"projectId": pid, "revision": project.revision, "song": song}
    if action == "cut_plan":
        plan = _run(lambda: storyboard_long.cut_plan(ctx.request, pid, args.get("max_shot_seconds"),
                                                    length_seconds=args.get("seconds")))
        if not args.get("apply"):
            return plan
        _need(args, "revision")
        applied = _op(args, "apply_plan", {"mode": "replace", "shots": plan["shots"]})
        return _summary(applied, {"note": plan["note"], "clipSeconds": plan["clipSeconds"]})
    _need(args, "revision")
    rev = int(args["revision"])
    if action == "attach":
        library_id = args.get("library_id")
        if (library_id is None) == ("path" not in args):
            raise ToolError("attach needs library_id or path (one of them).")
        path = None if library_id is not None else _agent_source(args["path"])
        project, analysis = _run(lambda: storyboard_long.attach_song(pid, rev, library_id, path, args.get("title")))
        return _summary(project, {"song": analysis})
    if action == "edit":
        return _summary(_op(args, "edit_song", _song_edit(args)))
    if action == "remove":
        return _summary(_op(args, "clear_song", {}))
    timing, language = str(args.get("timing") or "auto"), str(args.get("language") or "")

    def run() -> tuple[dict[str, Any], list[str]]:
        project, used, notes = lyrics.make(store, pid, rev, timing, language)
        return {"projectId": pid, "revision": project.revision, "timing": used, "captions": len(project.captions)}, notes

    try:
        project = store.load(pid)
        if timing == "speech" and not lyrics.speech_status()[0]:
            raise ToolError(lyrics.speech_status()[1] or "Speech-to-text is not available.")
        if project.song is None or not project.song.lyrics:
            raise ToolError("The project needs a song with lyrics first (storyboard_song attach, then edit lyrics).")
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - a missing project as a tool error
        raise ToolError(str(exc)) from exc
    return _submitted(STORE.submit("voice", f"lyrics {pid}", run))


def _extras(pid: str, files: list[str]) -> list[dict[str, Any]]:
    """Files a render wrote next to the film (captions, stems): name plus download, never a local path."""
    from backend_service.mcp_bridge.files import download_info

    folder = _store().folder(pid)
    return [{"file": rel, "download": download_info(folder / rel)} for rel in files if (folder / rel).is_file()]


def storyboard_render(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.routes.storyboard import start_render
    from backend_service.storyboard.render_adapter import RENDERS

    from backend_service.mcp_bridge.jobs import CURRENT_OWNER
    from backend_service.tools.studio import CHAT_OWNER

    pid = str(args["project_id"])
    preview = bool(args.get("preview", True))
    # Chat's model talks to the user at this computer: it gets the absolute path to say word for word (it once
    # made one up, FU-209). An external agent gets the file name and download only, as every bridge answer.
    local = CURRENT_OWNER.get() == CHAT_OWNER
    before = render_warnings(_run(lambda: _store().load(pid)), preview)
    job = _run(lambda: start_render(pid, int(args["revision"]), preview, bool(args.get("notify", False))))

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
        warnings = [*before, *job.warnings]
        if warnings:
            output["warnings"] = warnings
        if not local:
            return output, warnings
        path = str(job.out_path.resolve())
        return {"path": path, **output}, [f"The film is saved at {path}. Tell the user this exact path.", *warnings]

    def progress() -> dict[str, Any]:
        import time

        fraction = float(job.fraction or 0.0)
        info: dict[str, Any] = {"phase": job.message or "rendering", "step": round(fraction * 100), "totalSteps": 100}
        elapsed = time.time() - float(job.started_at or time.time())
        if fraction >= 0.15 and elapsed > 0:  # the pace so far; earlier than that the guess is noise
            info["etaSeconds"], info["etaBasis"] = round(elapsed / fraction * (1 - fraction)), "steps"
        return info

    submitted = _submitted(STORE.submit("storyboard", f"render {pid}", run, cancel=lambda: RENDERS.cancel(pid), progress=progress))
    return {**submitted, "warnings": before} if before else submitted


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
        if clip_ids:  # an unknown id fails now, with the valid ones, not later as a failed job (FU-209)
            project = _run(lambda: store.load(pid))
            _run(lambda: captions.check_clip_ids(project, clip_ids))

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
    "storyboard_batch": storyboard_batch,
    "storyboard_song": storyboard_song,
    "storyboard_render": storyboard_render,
    "storyboard_captions": storyboard_captions,
    "storyboard_export_otio": storyboard_export_otio,
}
