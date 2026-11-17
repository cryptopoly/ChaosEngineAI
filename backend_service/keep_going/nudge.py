"""Reading a finished Studio turn and writing the next Keep going message (FU-191).

``TurnOutcome`` is what the chat turn hands the controller when it ends. The helpers here pull out what the
controller decides on (tool errors the model left unfixed, a declined plan, the project the run works on, a
"finished" claim), and ``nudge_text`` writes the app's message for the next turn: short and concrete, never a
bare "continue". It names what the project still lacks, the last error word for word and one next step; the
brief is repeated only when the chat history no longer holds it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NUDGE_PREFIX = "Keep going"
MAX_GOAL_CHARS = 4000
MAX_ERROR_CHARS = 400
# The model's own "it is done" wording; only ever believed when the project agrees.
_DONE_WORDS = re.compile(
    r"\b(all done|(is|are) (now )?(finished|complete|completed|done|ready)|film is ready|"
    r"(everything|it) is (finished|done|complete)|task complete)\b",
    re.I,
)


@dataclass
class TurnOutcome:
    """How one chat turn ended. ``assistant`` is the saved reply (None when the turn failed before one)."""

    session_id: str
    assistant: dict[str, Any] | None = None
    cancelled: bool = False
    error: str | None = None
    prompt_tokens: int = 0
    context_tokens: int = 0
    max_iterations: bool = False
    nudge: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str((self.assistant or {}).get("text") or "")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls = (self.assistant or {}).get("toolCalls")
        return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def _call_error(call: dict[str, Any]) -> str | None:
    data = call.get("data")
    if isinstance(data, dict):
        if data.get("declined"):
            return None
        if isinstance(data.get("error"), str) and data["error"].strip():
            return data["error"].strip()
    result = str(call.get("result") or "")
    if result.startswith("Error:"):
        return result[len("Error:"):].strip() or result
    return None


def unfixed_errors(outcome: TurnOutcome) -> list[tuple[str, str]]:
    """(tool, error) for failed calls the model did not fix later in the turn (the same tool ran fine after)."""
    calls = outcome.tool_calls
    out: list[tuple[str, str]] = []
    for index, call in enumerate(calls):
        error = _call_error(call)
        if error is None:
            continue
        name = str(call.get("name") or "")
        fixed = any(str(c.get("name") or "") == name and _call_error(c) is None and not _declined(c)
                    for c in calls[index + 1:])
        if not fixed:
            out.append((name, error))
    return out


def _declined(call: dict[str, Any]) -> bool:
    data = call.get("data")
    return isinstance(data, dict) and bool(data.get("declined"))


def error_signature(tool: str, error: str) -> str:
    """Same tool and same text, numbers kept (a revision number that moves is a different error)."""
    return f"{tool}:{' '.join(error.split())[:300]}"


def plan_outcome(outcome: TurnOutcome) -> str | None:
    """'unanswered', 'edit' or 'declined' when a plan card of the turn ended that way; None otherwise."""
    plans = (outcome.assistant or {}).get("studioPlans")
    for plan in plans if isinstance(plans, list) else []:
        if not isinstance(plan, dict):
            continue
        status = plan.get("status")
        if status == "unanswered":
            return "unanswered"
        if status == "edit":
            return "edit"
        decisions = plan.get("decisions") or {}
        if status == "decided" and decisions and "approve" not in decisions.values():
            return "declined"
    return None


def project_id_from(outcome: TurnOutcome) -> str | None:
    """The storyboard project the turn worked on last (from the tools' answers, else their arguments)."""
    found: str | None = None
    for call in outcome.tool_calls:
        if not str(call.get("name") or "").startswith("storyboard_"):
            continue
        data = call.get("data")
        args = call.get("arguments")
        pid = data.get("projectId") if isinstance(data, dict) else None
        if not pid and isinstance(args, dict):
            pid = args.get("project_id")
        if isinstance(pid, str) and pid.strip() and _call_error(call) is None:
            found = pid.strip()
    return found


def claims_done(text: str) -> bool:
    return bool(_DONE_WORDS.search(text or ""))


def _short(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def nudge_text(
    *,
    number: int,
    cap: int,
    goal: str,
    include_goal: bool,
    summary: str,
    next_step: str,
    last_error: tuple[str, str] | None,
    finished_jobs: list[str],
    why: str,
) -> str:
    """The app's message for the next turn. ``summary`` and ``next_step`` come from the run's adapter. ``why`` is the reason this nudge is sent ('cap' when the turn hit
    the step limit, 'stopped' when the model ended its turn, 'claimed' when it said done but the project
    disagrees, 'error' after a failed call, 'resume' after Resume)."""
    lines = [f"{NUDGE_PREFIX} (automatic message {number} of {cap} from the app, not from the user)."]
    if include_goal:
        lines.append(f"The brief, as the user gave it:\n\"\"\"\n{_short_block(goal, MAX_GOAL_CHARS)}\n\"\"\"")
    else:
        lines.append(f"Goal: the brief in the user's first message (\"{_short(goal, 90)}\").")
    intro = {
        "cap": "Your last turn reached the tool-step limit before the work was done.",
        "claimed": "You said the work is finished, but the project disagrees.",
        "error": "Your last turn ended after a tool error.",
        "resume": "Resuming after a pause. Rebuild your picture from the project state below, not from memory.",
        "swap_done": "The renders you queued ran while the chat model was unloaded; their results are below.",
    }.get(why)
    if intro:
        lines.append(intro)
    if summary:
        lines.append(summary)
    if finished_jobs:
        lines.append("Since then: " + "; ".join(_short(j, 160) for j in finished_jobs[:4]) + ".")
    if last_error is not None:
        tool, error = last_error
        lines.append(f"Last error ({tool}): {_short(error, MAX_ERROR_CHARS)}")
    step = next_step or "Do the next step of the brief with the tools."
    if last_error is not None:
        step = f"Fix that error first, then: {step[0].lower()}{step[1:]}"
    lines.append(f"Next step: {step}")
    lines.append("Do it now with the tools. When everything in the brief is done, say so in one line.")
    return "\n".join(lines)


def _short_block(text: str, limit: int) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def finished_job_lines(jobs: list[Any]) -> list[str]:
    """'image job j-1 done', 'video job j-2 failed: <error>' for jobs that ended since the last nudge."""
    out = []
    for job in jobs:
        state = str(getattr(job, "state", ""))
        line = f"{getattr(job, 'kind', 'job')} job {getattr(job, 'id', '?')} {state}"
        error = getattr(job, "error", None)
        if state != "done" and error:
            line += f": {_short(str(error), 160)}"
        out.append(line)
    return out
