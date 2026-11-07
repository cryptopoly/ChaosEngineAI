"""Which scope each tool needs, and the MCP annotations every tool carries.

``required_scope`` looks at the action too: ``storyboard_project`` list and get only read.
"""

from __future__ import annotations

from typing import Any

_READ_TOOLS = {"system_status", "recommend_models", "list_models", "list_voices", "job_status", "wait_for_job", "list_jobs"}
_FILE_TOOLS = {"upload_file", "upload_image"}
_READ_ACTIONS = {"list", "get"}

# Tools that start a job and so count against the per-token cap.
JOB_TOOLS = {"generate_image", "generate_video", "speak", "transcribe", "make_music", "storyboard_generate", "storyboard_render"}

ALIASES = {"upload_image": "upload_file"}


def required_scope(name: str, args: dict[str, Any] | None = None) -> str:
    name = ALIASES.get(name, name)
    if name in _READ_TOOLS:
        return "read"
    if name == "upload_file":
        return "files"
    if name == "storyboard_project":
        action = (args or {}).get("action")
        if action in _READ_ACTIONS:
            return "read"
        if action == "export":
            return "files"  # the answer is a download link
    return "generate"


def lowest_scope(name: str) -> str:
    """The scope that shows a tool in tools/list (storyboard_project has read actions)."""
    return "read" if name == "storyboard_project" else required_scope(name)


def scope_error(name: str, scope: str) -> str:
    return (
        f"This token cannot use {name}: it does not have the '{scope}' scope. Tell the user to change the "
        "token's permissions in ChaosEngineAI: Settings > Integrations > Agent access."
    )


def _ann(title: str, read_only: bool = False, destructive: bool = False, idempotent: bool = False) -> dict[str, Any]:
    # openWorldHint is false for all of them: the bridge only talks to this computer.
    return {"title": title, "readOnlyHint": read_only, "destructiveHint": destructive,
            "idempotentHint": idempotent, "openWorldHint": False}


ANNOTATIONS: dict[str, dict[str, Any]] = {
    "system_status": _ann("System status", read_only=True, idempotent=True),
    "recommend_models": _ann("Recommend models", read_only=True, idempotent=True),
    "list_models": _ann("List models", read_only=True, idempotent=True),
    "list_voices": _ann("List voices", read_only=True, idempotent=True),
    "list_jobs": _ann("List jobs", read_only=True, idempotent=True),
    "job_status": _ann("Job status", read_only=True, idempotent=True),
    "wait_for_job": _ann("Wait for job", read_only=True, idempotent=True),
    "cancel_job": _ann("Cancel job", destructive=True, idempotent=True),
    "generate_image": _ann("Generate image"),
    "generate_video": _ann("Generate video"),
    "speak": _ann("Speak text"),
    "transcribe": _ann("Transcribe audio"),
    "make_music": _ann("Make music"),
    "upload_file": _ann("Upload file"),
    "storyboard_project": _ann("Storyboard project", destructive=True),
    "storyboard_shot": _ann("Storyboard shot", destructive=True),
    "storyboard_audio": _ann("Storyboard audio", destructive=True),
    "storyboard_generate": _ann("Storyboard generate"),
    "storyboard_render": _ann("Storyboard render"),
}


def apply_annotations(tools: list[dict[str, Any]]) -> None:
    """Add the annotations to each tool dict; a tool without an entry is a bug the contract test catches."""
    for tool in tools:
        if tool["name"] in ANNOTATIONS:
            tool["annotations"] = dict(ANNOTATIONS[tool["name"]])
