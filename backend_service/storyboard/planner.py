"""Film from an idea (S8): the loaded chat model proposes shots, reworks one shot, or answers the direction chat.

It calls the model through ``state.runtime.generate``, the same path Chat and the Chat Studio tools use (no
second client), with a JSON schema where the engine takes one. Nothing here writes the project: the routes
return proposals, and the user applies them as ordinary ops (``apply_plan``, ``edit_shot``,
``plan_changes``), so Undo, Redo and the revision check hold. The model sees the brief, the format, the cast
and the style (words and palette; the pictures themselves only when the loaded model reads images) and the
shots so far, by number. Its reply is cleaned before anyone sees it: lengths cut to the project limits, one
line titles, seconds in range, shot numbers that do not exist dropped.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

from backend_service.storyboard import media
from backend_service.storyboard.models import Project, Shot
from backend_service.storyboard.palette import palette_words

MIN_SHOTS, MAX_SHOTS = 4, 8
MAX_IMAGES = 4
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_BUSY = threading.Lock()


class PlanError(Exception):
    """A readable refusal; ``code`` lets the UI offer the right next step (``no-model``, ``busy``, ``bad-reply``)."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


SYSTEM = (
    "You plan short films for a local video app. A film is a list of shots. Each shot becomes one still picture, "
    "made from its stillPrompt by an image model, and then a short clip, made from that still and its motionPrompt "
    "by a video model.\n"
    "Rules for every shot:\n"
    "- title: 2 to 5 words.\n"
    "- stillPrompt: one frame written for an image model: the subject, the setting, the light, the framing and the "
    "colours, in 1 to 3 concrete sentences. No camera moves, no on-screen text, no logos.\n"
    "- motionPrompt: one sentence: what moves in the shot and how the camera moves.\n"
    "- seconds: between 2 and 10 (a clip is at most about 10 seconds; a longer film needs more shots).\n"
    "- Describe a character or place the same way in every shot it appears in, so it keeps its look.\n"
    "- Follow the visual style and palette when one is given.\n"
    "Answer with JSON only, no other text."
)

_SHOT_FIELDS = {
    "title": {"type": "string"},
    "stillPrompt": {"type": "string"},
    "motionPrompt": {"type": "string"},
    "seconds": {"type": "number"},
}

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {"type": "array", "minItems": MIN_SHOTS, "maxItems": MAX_SHOTS,
                  "items": {"type": "object", "properties": _SHOT_FIELDS, "required": list(_SHOT_FIELDS)}},
        "note": {"type": "string"},
    },
    "required": ["shots"],
}

REWORK_SCHEMA = {
    "type": "object",
    "properties": {**_SHOT_FIELDS, "why": {"type": "string"}},
    "required": ["stillPrompt", "why"],
}

DIRECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "edits": {"type": "array", "items": {"type": "object", "properties": {"shot": {"type": "integer"}, **_SHOT_FIELDS},
                                              "required": ["shot"]}},
        "add": {"type": "array", "items": {"type": "object", "properties": {"after": {"type": "integer"}, **_SHOT_FIELDS},
                                            "required": ["after", "stillPrompt"]}},
        "remove": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["reply"],
}


# ---- cleaning what the model wrote ----
def _text(value: Any, limit: int, one_line: bool = False) -> str:
    text = _CONTROL.sub("", str(value or "")).strip()
    if one_line:
        text = re.sub(r"[\r\n\t]+", " ", text)
    return text[:limit].strip()


def _seconds(value: Any, default: float = 4.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):  # NaN or infinity (a model can write 1e999)
        return default if number != number else (10.0 if number > 0 else 1.0)
    return max(1.0, min(10.0, round(number * 2) / 2))  # a clip model makes about 10 s at most


def _items(value: Any) -> list[Any]:
    """A list from the model's answer, or an empty one when it wrote something else there (a number, a word)."""
    return value if isinstance(value, list) else []


def clean_shot(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    shot = {
        "title": _text(raw.get("title"), 200, one_line=True),
        "stillPrompt": _text(raw.get("stillPrompt"), 4000),
        "motionPrompt": _text(raw.get("motionPrompt"), 4000),
        "seconds": _seconds(raw.get("seconds")),
    }
    return shot if shot["stillPrompt"] or shot["title"] else None


def parse_reply(text: str) -> dict[str, Any]:
    """The JSON object in a model reply (thinking, a code fence or a sentence around it are ignored)."""
    from backend_service.helpers.request_guards import strip_json_fence
    from backend_service.reasoning_split import strip_thinking_tokens

    body = strip_json_fence(strip_thinking_tokens(text or "")).strip()
    try:
        value = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise PlanError("The model did not answer with a plan. Try again, or load a larger chat model.", "bad-reply") from None
        try:
            value = json.loads(body[start:end + 1])
        except ValueError:
            raise PlanError("The model's answer was not readable. Try again, or load a larger chat model.", "bad-reply") from None
    if not isinstance(value, dict):
        raise PlanError("The model did not answer with a plan. Try again.", "bad-reply")
    return value


# ---- what the model is told ----
def _format_line(project: Project) -> str:
    d = project.defaults
    return f"Format: {d.exportPreset} at {d.resolution} ({d.width}x{d.height})."


def _refs_lines(project: Project) -> list[str]:
    lines: list[str] = []
    for label, refs in (("Character", project.references.characters), ("Place", project.references.places)):
        for ref in refs:
            if ref.name or ref.description:
                lines.append(f"{label}: {ref.name or 'unnamed'}{': ' + ref.description if ref.description else ''}")
    from backend_service.storyboard.style import styles_in_use

    for ref in styles_in_use(project):
        bits = [ref.description.strip()] if ref.description.strip() else []
        if ref.palette:
            bits.append(f"palette {palette_words(ref.palette)}")
        lines.append(f"Visual style{' ' + repr(ref.name) if ref.name else ''}: {'; '.join(bits) or 'see the pictures'}")
    return lines


def context(project: Project, brief: str | None = None, film_seconds: float | None = None) -> str:
    brief = project.plan.brief if brief is None else brief
    film_seconds = project.plan.filmSeconds if film_seconds is None else film_seconds
    lines = [f"Project: {project.name}", _format_line(project)]
    if film_seconds:
        lines.append(f"Target length: about {round(film_seconds)} seconds.")
    if brief.strip():
        lines.append(f"Brief: {brief.strip()}")
    lines.extend(_refs_lines(project))
    return "\n".join(lines)


def shots_text(project: Project) -> str:
    if not project.shots:
        return "There are no shots yet."
    rows = []
    for number, shot in enumerate(project.shots, 1):
        rows.append(f"Shot {number}: {shot.title or 'untitled'} ({shot.seconds:g} s){' [locked]' if shot.locked else ''}\n"
                    f"  still: {shot.stillPrompt or '-'}\n  motion: {shot.motionPrompt or '-'}")
    total = sum(shot.seconds for shot in project.shots)
    return f"Current shots ({len(project.shots)}, {total:g} s in all):\n" + "\n".join(rows)


def reference_pictures(project: Project, folder: Path, limit: int = MAX_IMAGES) -> list[str]:
    """Base64 JPEGs (512 px) of the style pictures first, then the cast, one per reference."""
    from PIL import Image

    out: list[str] = []
    refs = project.references.styles + project.references.characters + project.references.places
    for ref in refs:
        if len(out) >= limit or not ref.images:
            continue
        try:
            with Image.open(media.inside(folder, ref.images[0].path)) as image:
                rgb = image.convert("RGB")
                rgb.thumbnail((512, 512))
                buffer = io.BytesIO()
                rgb.save(buffer, format="JPEG", quality=85)
        except (OSError, ValueError, media.MediaError):
            continue
        out.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return out


# ---- calling the model ----
def chat_model(state: Any) -> dict[str, Any] | None:
    loaded = getattr(getattr(state, "runtime", None), "loaded_model", None)
    if loaded is None:
        return None
    return {"name": loaded.name, "ref": loaded.ref, "vision": bool(getattr(loaded, "visionEnabled", False))}


def ask(state: Any, project: Project, folder: Path, prompt: str, schema: dict[str, Any],
        history: list[dict[str, str]] | None = None, max_tokens: int = 6000) -> tuple[dict[str, Any], dict[str, Any]]:
    """One model call; returns (the parsed JSON, facts for the UI: model name and seconds taken)."""
    model = chat_model(state)
    if model is None:
        raise PlanError("No chat model is loaded. Load one to plan with it.", "no-model")
    if not _BUSY.acquire(blocking=False):
        raise PlanError("The model is still answering the last request.", "busy")
    started = time.perf_counter()
    try:
        images = reference_pictures(project, folder) if model["vision"] else None
        state.add_log("storyboard", "info", f"Planning with {model['name']} for {project.name}")
        try:
            result = state.runtime.generate(prompt=prompt, history=history or [], system_prompt=SYSTEM,
                                            max_tokens=max_tokens, temperature=0.7, images=images or None,
                                            json_schema=schema)
        except RuntimeError as exc:
            raise PlanError(f"The chat model could not answer: {exc}", "model-error") from exc
    finally:
        _BUSY.release()
    facts = {"model": model["name"], "seconds": round(time.perf_counter() - started, 1),
             "sawPictures": bool(images)}
    return parse_reply(result.text), facts


def propose(state: Any, project: Project, folder: Path, brief: str, film_seconds: float | None,
            count: int | None) -> dict[str, Any]:
    if not brief.strip() and not project.plan.brief.strip():
        raise PlanError("Write a few words about the film first.", "no-brief")
    want = f"exactly {count}" if count else f"{MIN_SHOTS} to {MAX_SHOTS}"
    length = f" whose seconds add up to about {round(film_seconds)}" if film_seconds else ""
    prompt = (f"{context(project, brief, film_seconds)}\n\nWrite {want} shots{length} that tell this film from start to "
              'finish. Answer as {"shots": [{"title": ..., "stillPrompt": ..., "motionPrompt": ..., "seconds": ...}], '
              '"note": "one sentence on the idea"}.')
    reply, facts = ask(state, project, folder, prompt, PROPOSE_SCHEMA)
    shots = [s for s in (clean_shot(raw) for raw in _items(reply.get("shots"))) if s][:MAX_SHOTS + 4]
    if not shots:
        raise PlanError("The model answered without any shots. Try again, or say more in the brief.", "bad-reply")
    return {"shots": shots, "note": _text(reply.get("note"), 400), **facts}


def _history(raw: list[dict[str, str]], limit: int = 12) -> list[dict[str, str]]:
    return [{"role": m["role"], "text": _text(m.get("text"), 4000)} for m in raw[-limit:]
            if m.get("role") in ("user", "assistant") and m.get("text")]


def rework(state: Any, project: Project, folder: Path, shot: Shot, message: str,
           history: list[dict[str, str]]) -> dict[str, Any]:
    number = next(i for i, s in enumerate(project.shots, 1) if s.id == shot.id)
    prompt = (f"{context(project)}\n\n{shots_text(project)}\n\nRework shot {number} only. The person asks: "
              f"{message.strip() or 'make it better'}\n"
              'Answer as {"title": ..., "stillPrompt": ..., "motionPrompt": ..., "seconds": ..., '
              '"why": "one or two sentences on what you changed and why"}.')
    reply, facts = ask(state, project, folder, prompt, REWORK_SCHEMA, _history(history), max_tokens=4000)
    cleaned = clean_shot({"title": shot.title, "motionPrompt": shot.motionPrompt, "seconds": shot.seconds, **reply})
    if cleaned is None or not cleaned["stillPrompt"]:
        raise PlanError("The model did not write a new prompt. Try again with a clearer ask.", "bad-reply")
    return {"shot": cleaned, "why": _text(reply.get("why"), 600) or "No reason given.", **facts}


def _number_to_id(project: Project, value: Any) -> str | None:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return project.shots[number - 1].id if 1 <= number <= len(project.shots) else None


def _kept_before(project: Project, shot_id: str | None, removed: list[str]) -> str | None:
    """``shot_id``, or the nearest shot before it that stays, when the same answer removes it."""
    if shot_id is None or shot_id not in removed:
        return shot_id
    ids = [s.id for s in project.shots]
    for candidate in reversed(ids[:ids.index(shot_id)]):
        if candidate not in removed:
            return candidate
    return None


def direct(state: Any, project: Project, folder: Path, message: str) -> dict[str, Any]:
    """The project-level direction chat: a reply, and the changes it proposes as ``plan_changes`` arguments."""
    chat = [{"role": m.role, "text": m.text} for m in project.planChat]
    prompt = (f"{context(project)}\n\n{shots_text(project)}\n\nThe person directing the film says: {message.strip()}\n"
              "Change only what they ask for. You may edit, add or remove shots, or re-plan the whole film. "
              "Never change a locked shot. "
              'Answer as {"reply": "a short answer to the person", '
              '"edits": [{"shot": number, "title"?, "stillPrompt"?, "motionPrompt"?, "seconds"?}], '
              '"add": [{"after": number of the shot it follows, 0 for the start, "title", "stillPrompt", "motionPrompt", "seconds"}], '
              '"remove": [shot numbers]}. Leave a list empty when nothing changes there.')
    reply, facts = ask(state, project, folder, prompt, DIRECT_SCHEMA, _history(chat))
    locked = {s.id for s in project.shots if s.locked}
    edits = []
    for raw in _items(reply.get("edits")):
        shot_id = _number_to_id(project, raw.get("shot")) if isinstance(raw, dict) else None
        if not shot_id or shot_id in locked:
            continue
        change: dict[str, Any] = {"shotId": shot_id}
        for key, limit in (("title", 200), ("stillPrompt", 4000), ("motionPrompt", 4000)):
            if raw.get(key):
                change[key] = _text(raw[key], limit, one_line=key == "title")
        if raw.get("seconds") is not None:
            change["seconds"] = _seconds(raw["seconds"])
        if len(change) > 1:
            edits.append(change)
    remove = [sid for sid in dict.fromkeys(_number_to_id(project, n) for n in _items(reply.get("remove")))
              if sid and sid not in locked]
    add = []
    for raw in _items(reply.get("add"))[:12]:
        shot = clean_shot(raw)
        if shot is None:
            continue
        after = raw.get("after")
        after_id = None if after in (0, "0", None) else _number_to_id(project, after)
        if after not in (0, "0", None) and after_id is None:
            after_id = project.shots[-1].id if project.shots else None  # a number past the end: at the end
        add.append({**shot, "after": _kept_before(project, after_id, remove)})
    return {"reply": _text(reply.get("reply"), 2000) or "Done.", "changes": {"edits": edits, "add": add, "remove": remove},
            **facts}
