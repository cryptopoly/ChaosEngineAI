"""A Keep going run's record, its stop reasons and the small rules around it (FU-191).

``Run`` is what is saved per chat (``record``) plus what lives only in memory (the render grant, the turn in
flight, the request the studio tools run with). ``STOP_CODES`` holds the English reason for each way a run
stops; the app shows its own translation of the code and falls back to this text.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol


DEFAULT_CAP = 20
MIN_CAP, MAX_CAP = 1, 100
NO_PROGRESS_LIMIT = 3
REPEATED_ERROR_LIMIT = 3
TURN_FAILURE_LIMIT = 2
DEFAULT_GRANT_HOURS = 6
MAX_GRANT_HOURS = 12
CONTEXT_FULL_SHARE = 0.85
FILE_VERSION = 1

ACTIVE = ("armed", "running", "waiting")

# Stop code -> English reason (the log line and the app's fallback text; the app shows its own translation).
STOP_CODES: dict[str, str] = {
    "complete": "The film looks finished: every shot has a clip, the audio is in and the final film is rendered.",
    "complete_no_project": "The model says it is finished and no job is running.",
    "cap_reached": "Reached the limit of {cap} nudges.",
    "no_progress": "Nothing changed after {limit} nudges in a row (no project edit, finished job or new file).",
    "repeated_error": "The same error came back {limit} times: {detail}",
    "context_full": "The chat is nearly out of context ({detail}). Start a new chat to go on; the project is saved.",
    "plan_declined": "You declined the plan.",
    "plan_edit": "You asked to change the plan.",
    "plan_unanswered": "A plan waited without an answer, so nothing was made.",
    "approval_expired": "The render approval for this run expired. Resume to approve again.",
    "user_stopped": "You stopped it.",
    "user_message": "You sent a message, so Keep going stopped.",
    "model_unavailable": "The chat model is not loaded and could not be loaded again: {detail}",
    "turn_failed": "The chat model failed {limit} times in a row: {detail}",
    "app_restart": "The app restarted while Keep going ran.",
    "studio_off": "Studio tools were turned off.",
    "gpu_reload_failed": "The chat model could not be loaded again after a render: {detail}",
}
# Stops a Resume can pick up from (the others need a new brief or a new chat).
RESUMABLE = frozenset(STOP_CODES) - {"complete", "complete_no_project", "user_message", "context_full", "studio_off"}


class TurnStartError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class Hooks(Protocol):
    """What the controller needs from the app; tests pass fakes."""

    def start_turn(self, run: "Run", prompt: str, auto_until: float | None) -> None: ...
    def turn_running(self, session_id: str) -> bool: ...
    def pending_plans(self, session_id: str) -> int: ...
    def chat_jobs(self) -> list[Any]: ...
    def goal_in_history(self, run: "Run") -> bool: ...
    def last_user_text(self, session_id: str) -> str | None: ...
    def message_count(self, session_id: str) -> int: ...
    def cancel_turn(self, session_id: str) -> None: ...
    def log(self, level: str, text: str) -> None: ...


@dataclass
class Grant:
    """How renders are approved in this run: 'auto' until ``until`` (epoch seconds), or 'manual' (plan cards)."""

    mode: str
    until: float | None = None

    def live(self, now: float) -> bool:
        return self.mode == "auto" and self.until is not None and now < self.until


@dataclass
class Run:
    id: str
    session_id: str
    state: str = "armed"
    cap: int = DEFAULT_CAP
    goal: str = ""
    project_id: str | None = None
    nudges: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    stop_code: str | None = None
    stop_reason: str | None = None
    stop_detail: str | None = None
    stopped_at: float | None = None
    template: dict[str, Any] = field(default_factory=dict)
    snapshot: dict[str, Any] | None = None
    no_progress: int = 0
    error_streaks: dict[str, int] = field(default_factory=dict)
    turn_failures: int = 0
    last_error: list[str] | None = None  # [tool, text]
    seen_jobs: list[str] = field(default_factory=list)  # chat jobs already reported as finished
    resumes: int = 0
    gpu_mode: str = "auto"  # phase 2: auto | always | never
    adapter: str = "storyboard"  # what counts as progress and done (adapters.py)
    last_reply: str = ""  # the model's last reply, for an adapter's hint after a Resume
    # In memory only (never saved): the grant, the turn in flight, the request the tools run with.
    grant: Grant | None = None
    turn_pending: bool = False
    dispatching: bool = False
    waiting_thread: bool = False
    nudge_why: str = "stopped"
    abort: threading.Event = field(default_factory=threading.Event)
    app_request: Any = None

    _SAVED = ("id", "session_id", "state", "cap", "goal", "project_id", "nudges", "started_at", "updated_at",
              "stop_code", "stop_reason", "stop_detail", "stopped_at", "template", "snapshot", "no_progress",
              "error_streaks", "turn_failures", "last_error", "seen_jobs", "resumes", "gpu_mode",
              "adapter", "last_reply")

    def record(self) -> dict[str, Any]:
        return {k: copy.deepcopy(getattr(self, k)) for k in self._SAVED}

    @classmethod
    def from_record(cls, data: dict[str, Any]) -> "Run":
        known = {k: data[k] for k in cls._SAVED if k in data}
        run = cls(**known)
        run.cap = clamp_cap(run.cap)
        return run


def clamp_cap(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CAP
    return max(MIN_CAP, min(MAX_CAP, number))


def clamp_hours(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_GRANT_HOURS)
    return max(0.25, min(float(MAX_GRANT_HOURS), number))


def reason_text(code: str, *, cap: int = DEFAULT_CAP, detail: str | None = None) -> str:
    template = STOP_CODES.get(code, code)
    limit = {"no_progress": NO_PROGRESS_LIMIT, "repeated_error": REPEATED_ERROR_LIMIT,
             "turn_failed": TURN_FAILURE_LIMIT}.get(code, 0)
    return template.format(cap=cap, limit=limit, detail=(detail or "").strip() or "no details")


def _finished_ids(jobs: list[Any]) -> list[str]:
    return sorted(str(j.id) for j in jobs if getattr(j, "state", "") in _FINISHED)[-200:]


_FINISHED = ("done", "failed", "cancelled", "interrupted")


def _grant(mode: str, hours: Any, now: float) -> Grant:
    if mode == "auto":
        return Grant("auto", now + clamp_hours(hours) * 3600.0)
    return Grant("manual", None)


def _gpu_mode(value: str) -> str:
    return value if value in ("auto", "always", "never") else "auto"


# Request fields a nudge does not copy from the brief's turn: the text, images, one-off overrides and the
# approval flag (the run's grant decides that).
_NOT_COPIED = {"prompt", "images", "studioAutoApprove", "oneTurnOverride", "voiceTurn", "title"}


def _template(fields: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if k not in _NOT_COPIED and v is not None}


def _context_error(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in ("context length", "context window", "exceeds the context",
                                           "context size", "too many tokens", "n_ctx", "out of context"))
