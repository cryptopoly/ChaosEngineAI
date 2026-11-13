"""Agent tools for cloned voices: ``list_cloned_voices`` and ``speak_cloned`` (bodies only).

Registered in ``tools.py`` behind the ``clone`` token scope (``scopes.required_scope``); the scope check runs first,
then ``clone_allowed`` here. Over ``/mcp`` the origin is always "agent", so even the app's own token needs
Voice > Settings > Allow agents to use cloned voices (off by default). Agents cannot make a cloned voice: a
person makes it in the app with the permission tick, the speaker's name and the basis. An agent can only speak
with a voice that has that stored permission record (its id is the reference, D-28), and each ``speak_cloned``
call must carry ``permission_confirmed: true``, which goes to the audit log. Output is tagged, watermarked,
saved to the Voice Library and logged exactly like speech made in the app.
"""

from __future__ import annotations

from typing import Any, Callable

from backend_service.mcp_bridge.jobs import STORE
from backend_service.mcp_bridge.tools import ToolError, _PROJECT_PROPS, _obj, _submitted, project_scope

CLONE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_cloned_voices",
        "description": "Read: cloned voices the person made in the app with a stored permission record, with ids. "
        "Refused unless they allowed agents to use cloned voices.",
        "inputSchema": _obj({}),
    },
    {
        "name": "speak_cloned",
        "description": "Make: start a speech job in a cloned voice (returns a job id). voice_id comes from list_cloned_voices and is "
        "tied to its permission record. permission_confirmed must be true. Output is tagged and watermarked as AI-made.",
        "inputSchema": _obj(
            {
                "voice_id": {"type": "string", "minLength": 1, "maxLength": 40, "description": "Id from list_cloned_voices."},
                "text": {"type": "string", "minLength": 1, "maxLength": 5000},
                "permission_confirmed": {
                    "type": "boolean",
                    "description": "true: this is the person's own voice, or the speaker's permission covers this use. "
                    "Recorded in the audit log.",
                },
                "model": {"type": "string", "enum": ["qwen3-tts-0.6b", "qwen3-tts-1.7b", "qwen3-tts-0.6b-cuda", "qwen3-tts-1.7b-cuda"]},
                **_PROJECT_PROPS,
            },
            ["voice_id", "text", "permission_confirmed"],
        ),
    },
]


def _require_allowed(ctx: Any) -> str:
    from backend_service import voice_clone_safety as safety  # noqa: PLC0415

    request = getattr(ctx, "request", None)
    origin = safety.request_origin(request) if request is not None else "agent"
    allowed = safety.clone_allowed(request) if request is not None else safety.agent_cloning_enabled()
    if not allowed:
        safety.audit("refused", origin=origin, tool=True)
        raise ToolError(safety.REFUSED_MESSAGE)
    return origin


def list_cloned_voices(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service import voice_clone  # noqa: PLC0415

    _require_allowed(ctx)
    return {"voices": [{k: v[k] for k in ("id", "name", "durationS")} for v in voice_clone.list_voices()]}


def speak_cloned(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    from backend_service import voice_clone, voice_library  # noqa: PLC0415

    origin = _require_allowed(ctx)
    if args.get("permission_confirmed") is not True:
        raise ToolError(voice_clone.PERMISSION_REQUIRED)
    voice_id = str(args["voice_id"])
    if voice_clone.get_voice(voice_id) is None:
        raise ToolError(f"Cloned voice '{voice_id}' not found. Call list_cloned_voices.")
    text, model = str(args["text"]), str(args.get("model") or "")

    scope = project_scope(args)

    def run() -> tuple[dict[str, Any], list[str]]:
        with scope:
            item = voice_clone.speak(voice_id, text, model, origin=origin,
                                     permission_confirmed=True)
        path = voice_library.audio_path(item["id"])
        return {"path": str(path) if path else None, "durationSeconds": item.get("durationS"),
                "libraryId": item["id"], "clonedVoice": True}, ["Tagged as AI-made with a cloned voice."]

    return _submitted(STORE.submit("voice", "cloned speech", run))


CLONE_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "list_cloned_voices": list_cloned_voices,
    "speak_cloned": speak_cloned,
}
