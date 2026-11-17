"""What a Keep going run counts as progress, what its nudge says and when its goal is done (FU-191).

Everything else a run does (the nudge cap, the no-progress guard, the same-error stop, the grant, Stop,
cancel on a user message, persistence and Resume) lives in the controller and does not depend on the kind of
work. The work-specific part sits behind this small interface so another kind of long tool-using chat can get
its own adapter later. Two ship: ``StoryboardAdapter`` (Studio chats, the one the app uses today) and
``GenericAdapter`` (any tool chat: tool results and output files as progress, the model's own next step as
the hint). Only the Storyboard one is reachable from the app for now.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from backend_service.keep_going.nudge import TurnOutcome, claims_done
from backend_service.keep_going.project_state import FilmState, read_film_state


@dataclass
class View:
    """What an adapter read about the work after a turn (or before a nudge)."""

    signature: Any = None  # compared between turns: equal means no progress
    summary: str = ""  # one or two lines for the nudge
    next_step: str = ""  # one concrete step for the nudge
    complete: str | None = None  # a stop code when the goal looks done ("complete", "complete_no_project")
    project_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    name: str

    def read(self, *, goal: str, project_id: str | None, outcome: TurnOutcome | None,
             last_reply: str) -> View: ...


class StoryboardAdapter:
    """Studio chats: progress, the missing pieces and "done" come from the saved Storyboard project."""

    name = "storyboard"

    def __init__(self, reader: Callable[[str | None], FilmState | None] | None = None) -> None:
        self.reader = reader

    def read(self, *, goal: str, project_id: str | None, outcome: TurnOutcome | None, last_reply: str) -> View:
        film = (self.reader or read_film_state)(project_id)
        if film is None:
            done = None
            if outcome is not None and claims_done(outcome.text) and not outcome.max_iterations:
                done = "complete_no_project"
            return View(signature=None, summary="No storyboard project yet for this brief.",
                        next_step="Create the storyboard project with storyboard_project create, then add the "
                                  "shots from the brief.",
                        complete=done, project_id=project_id)
        return View(signature=film.snapshot(), summary=film.summary(goal), next_step=film.next_step(goal),
                    complete="complete" if film.complete(goal) else None, project_id=film.project_id,
                    extra={"film": film})


_NEXT = re.compile(r"[^.!?\n]*\b(next|then I will|I will now|I'll now|remaining|still to do|to do)\b[^.!?\n]*[.!?]?", re.I)


def stated_next_step(text: str) -> str | None:
    """The model's own last sentence about what comes next, if it wrote one."""
    found = [m.group(0).strip() for m in _NEXT.finditer(text or "")]
    return found[-1][:300] if found else None


class GenericAdapter:
    """Any tool chat (not offered in the app yet): progress is a tool call with a new result or a new output
    file since the last turn; the nudge restates the goal plus the model's own next step; the goal is done
    when the model says so and made no tool call in that turn."""

    name = "generic"

    def read(self, *, goal: str, project_id: str | None, outcome: TurnOutcome | None, last_reply: str) -> View:
        calls = outcome.tool_calls if outcome is not None else []
        results = sorted(
            f"{c.get('name')}:{hashlib.sha256(str(c.get('result') or '').encode('utf-8')).hexdigest()[:16]}"
            for c in calls)
        files = sorted({str((c.get("data") or {}).get("path")) for c in calls
                        if isinstance(c.get("data"), dict) and (c.get("data") or {}).get("path")})
        text = outcome.text if outcome is not None else last_reply
        done = None
        if outcome is not None and not calls and claims_done(outcome.text) and not outcome.max_iterations:
            done = "complete_no_project"
        step = stated_next_step(text) or "Do the next step of the goal with the tools."
        return View(signature={"results": results, "files": files} if outcome is not None else None,
                    summary="", next_step=step, complete=done, project_id=project_id)


ADAPTERS: dict[str, Callable[[], Any]] = {"storyboard": StoryboardAdapter, "generic": GenericAdapter}


def adapter_for(name: str | None) -> Any:
    return ADAPTERS.get(name or "storyboard", StoryboardAdapter)()
