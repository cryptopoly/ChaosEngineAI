"""Studio tools for Chat (D-60): the agent bridge's studio tools, offered to the chat model in process.

When the Chat composer's Studio switch is on, ``studio_registry`` returns a registry with the usual chat
tools plus a chosen subset of the bridge's tools (images, video, speech, music, sounds, Storyboard, the
job tools). Each call runs the bridge's own handler with a chat context: the scopes are fixed to
``read`` and ``generate`` (no ``files``, no ``download``, no ``clone``), jobs go on the same job store and
heavy-job queue with owner ``chat``, and results are trimmed (paths and short summaries, no base64, no
download links) so small models keep their context. Renders need the user's approval first; that gate
lives in ``studio_gate`` and runs in the agent loop, not here.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from backend_service.tools import BaseTool, StructuredToolOutput, ToolRegistry

# What the chat model may do. Never "files" (download links, uploads, exports), "download" or "clone".
CHAT_SCOPES = frozenset({"read", "generate"})
# Job owner for the shared job store; the per-owner unfinished-job cap applies to Chat as one owner.
CHAT_OWNER = "chat"

# The bridge tools Chat gets, in the order the model sees them. Left out on purpose: recommend_models and
# upload_file / transcribe (paths and uploads), storyboard_export_otio (a download), the cloned-voice tools.
STUDIO_TOOL_NAMES = (
    "system_status",
    "list_models",
    "list_voices",
    "generate_image",
    "generate_video",
    "speak",
    "make_music",
    "make_sound",
    "wait_for_job",
    "job_status",
    "list_jobs",
    "cancel_job",
    "storyboard_project",
    "storyboard_shot",
    "storyboard_audio",
    "storyboard_generate",
    "storyboard_render",
    "storyboard_captions",
    "music_library",
)

# Actions taken out of a grouped tool's enum for Chat: file moves need the files scope, and deletes are
# permanent, so the user does those in the app.
_DROPPED_ACTIONS = {
    "storyboard_project": {"export", "import", "delete"},
    "storyboard_captions": {"export_srt"},
    "music_library": {"delete"},
}
# Arguments taken out of a schema for Chat (base64 previews cost a small model its whole context).
_DROPPED_ARGS = {"generate_image": {"include_preview"}}
# Input-image arguments: in Chat they must be app outputs or upload: handles, never any file on the disk.
_INPUT_IMAGE_ARGS = {"generate_image": ("reference_images",), "generate_video": ("start_image", "end_image")}
_INPUT_NOTE = " Input images: only files the app made (output folders) or upload: handles."
# Job tools that name a job: Chat only sees and acts on the jobs it started.
_OWN_JOB_TOOLS = {"job_status", "wait_for_job", "cancel_job"}
# Keys removed from every result before the model sees it.
_STRIPPED_KEYS = {"previewBase64", "download", "data_base64"}
_MAX_RESULT_CHARS = 1800
_MAX_LIST_ITEMS = 12

STUDIO_HINT = (
    "Studio tools are on. You can make images (generate_image), video clips (generate_video), speech "
    "(speak, with voice, style and pitch), music (make_music), sound effects (make_sound) and short films "
    "(storyboard_* tools) with the models installed on this computer. Work like this: 1) say the plan in a "
    "short list (what, prompt, model, size); 2) call list_models first when you need a model id; 3) put "
    "every render call of the plan in one reply, the user approves the plan in the app before anything "
    "renders; 4) call wait_for_job for each job id until it is done; jobs run one at a time. If a call "
    "comes back 'declined', do not call it again: ask what to change. Never claim a file exists before "
    "wait_for_job says done."
)


def _bridge() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from backend_service.mcp_bridge.tools import HANDLERS, TOOLS

    return TOOLS, HANDLERS


def chat_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """The bridge schema with Chat's dropped actions and arguments removed."""
    schema = copy.deepcopy(tool["inputSchema"])
    props = schema.get("properties", {})
    for arg in _DROPPED_ARGS.get(tool["name"], ()):
        props.pop(arg, None)
    dropped = _DROPPED_ACTIONS.get(tool["name"])
    if dropped and "action" in props:
        props["action"]["enum"] = [a for a in props["action"]["enum"] if a not in dropped]
    return schema


def needs_approval(name: str, args: dict[str, Any] | None = None) -> bool:
    """True for a call that renders (starts a job): image, video, speech, music, sound, storyboard generate/render."""
    from backend_service.mcp_bridge.scopes import starts_job

    return starts_job(name, args or {})


def _trim(value: Any, depth: int = 0) -> Any:
    if isinstance(value, dict):
        return {k: _trim(v, depth + 1) for k, v in value.items() if k not in _STRIPPED_KEYS and v is not None}
    if isinstance(value, list):
        items = [_trim(v, depth + 1) for v in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"... {len(value) - _MAX_LIST_ITEMS} more")
        return items
    if isinstance(value, str) and len(value) > 400:
        return value[:400] + "..."
    return value


def trim_result(payload: Any) -> dict[str, Any]:
    """A short copy of a tool answer: no base64, no download links, lists cut to 12, long strings cut."""
    trimmed = _trim(payload) if isinstance(payload, dict) else {"result": _trim(payload)}
    if isinstance(trimmed, dict) and isinstance(trimmed.get("notes"), list):
        trimmed["notes"] = trimmed["notes"][:3]
    return trimmed


def _text(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, separators=(",", ":"), default=str)
    return text if len(text) <= _MAX_RESULT_CHARS else text[:_MAX_RESULT_CHARS] + "...(cut)"


class StudioTool(BaseTool):
    """One bridge tool, run in process with Chat's scopes and job owner."""

    def __init__(self, tool: dict[str, Any], handler: Any, app_request: Any) -> None:
        self._tool = tool
        self._handler = handler
        self._request = app_request
        self._schema = chat_schema(tool)

    @property
    def name(self) -> str:
        return str(self._tool["name"])

    @property
    def description(self) -> str:
        note = _INPUT_NOTE if self.name in _INPUT_IMAGE_ARGS else ""
        return str(self._tool["description"]) + note

    @property
    def provenance(self) -> str:
        return "studio"

    def parameters_schema(self) -> dict[str, Any]:
        return self._schema

    def execute(self, **kwargs: Any) -> str:
        return self.execute_structured(**kwargs).text

    def execute_structured(self, **kwargs: Any) -> StructuredToolOutput:
        from backend_service.mcp_bridge.jobs import CURRENT_OWNER, STORE, JobLimitError
        from backend_service.mcp_bridge.scopes import required_scope
        from backend_service.mcp_bridge.tools import ToolError, make_context, validate

        args = dict(kwargs)
        if self.name == "wait_for_job" and "timeout_seconds" not in args:
            args["timeout_seconds"] = 60  # fewer loop turns for the chat model
        problem = validate(self._schema, args)
        if problem:
            return self._error(f"Invalid input for {self.name}: {problem}.")
        scope = required_scope(self.name, args)
        if scope not in CHAT_SCOPES:
            return self._error(f"Chat cannot use {self.name} with these arguments. Ask the user to do it in the app.")
        refused = _chat_input_problem(self.name, args) or _foreign_job_problem(self.name, args)
        if refused:
            return self._error(refused)
        token = CURRENT_OWNER.set(CHAT_OWNER)
        try:
            if needs_approval(self.name, args):
                STORE.check_capacity()
            if self.name == "list_jobs":
                payload = _own_jobs(args)
            else:
                payload = self._handler(make_context(self._request), args)
        except (ToolError, JobLimitError) as exc:
            return self._error(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool bug is a readable answer, never a dead chat
            return self._error(f"{self.name} failed: {type(exc).__name__}: {exc}")
        finally:
            CURRENT_OWNER.reset(token)
        trimmed = trim_result(payload)
        return StructuredToolOutput(text=_text(trimmed), render_as="studio", data={"tool": self.name, **trimmed})

    def _error(self, message: str) -> StructuredToolOutput:
        return StructuredToolOutput(text=f"Error: {message}", render_as="studio", data={"tool": self.name, "error": message})


def chat_input_allowed(value: Any) -> bool:
    """An upload: handle, or a file inside the app's output folders (resolved, then a parent-prefix check)."""
    from backend_service.mcp_bridge.files import HANDLE_PREFIX, fetchable

    text = str(value or "").strip()
    return bool(text) and (text.startswith(HANDLE_PREFIX) or fetchable(text) is not None)


def _chat_input_problem(name: str, args: dict[str, Any]) -> str | None:
    for arg in _INPUT_IMAGE_ARGS.get(name, ()):
        raw = args.get(arg)
        values = raw if isinstance(raw, list) else ([raw] if raw else [])
        for value in values:
            if not chat_input_allowed(value):
                return (f"{arg} '{value}' is not allowed in Chat: use an image the app made (its path from a "
                        "finished job) or an upload: handle.")
    return None


def _foreign_job_problem(name: str, args: dict[str, Any]) -> str | None:
    if name not in _OWN_JOB_TOOLS:
        return None
    from backend_service.mcp_bridge.jobs import STORE

    job = STORE.get(str(args.get("job_id") or ""))
    if job is None or job.owner != CHAT_OWNER:
        return f"Unknown job '{args.get('job_id')}'. Chat only sees the jobs it started; call list_jobs."
    return None


def _own_jobs(args: dict[str, Any]) -> dict[str, Any]:
    from backend_service.mcp_bridge.jobs import MAX_KEPT, STORE

    jobs = [j for j in STORE.list(args.get("state"), MAX_KEPT) if j.owner == CHAT_OWNER]
    return {"jobs": [j.describe() for j in jobs[: int(args.get("limit", 20))]]}


def studio_tools(app_request: Any) -> list[StudioTool]:
    tools, handlers = _bridge()
    by_name = {t["name"]: t for t in tools}
    return [StudioTool(by_name[n], handlers[n], app_request) for n in STUDIO_TOOL_NAMES if n in by_name and n in handlers]


def studio_registry(app_request: Any, base: ToolRegistry, available: list[str] | None) -> ToolRegistry:
    """The chat tools (``available`` of ``base``, all when None) plus the studio tools."""
    registry = ToolRegistry()
    for tool in base.list_tools():
        if available is None or tool.name in available:
            registry.register(tool)
    for tool in studio_tools(app_request):
        registry.register(tool)
    return registry
