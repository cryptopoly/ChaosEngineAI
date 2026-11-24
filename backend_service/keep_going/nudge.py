"""Reading a finished Studio turn and writing the next Keep going message (FU-191).

``TurnOutcome`` is what the chat turn hands the controller when it ends. The helpers here pull out what the
controller decides on (tool errors the model left unfixed, a declined plan, the project the run works on, a
"finished" claim), and ``nudge_text`` writes the app's message for the next turn: short and concrete, never a
bare "continue". It names what the project still lacks, the last error word for word and one next step; the
brief is repeated only when the chat history no longer holds it.
"""

from __future__ import annotations

import json
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
# Words that say the model means to go on ("Music is done. Now let me add it" is not a claim that all is done).
_CONTINUE_WORDS = re.compile(
    r"\b(let me|let's|i'll|i will|i'm going to|i am going to|going to|about to|next,? i|then i|now i|"
    r"while (that|it) runs|after that|once (that|it)|still needs?|still to|remaining|left to do)\b",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
UNREADABLE_PREFIX = "your tool call could not be read"


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
    """The reply ends the turn saying the work is finished: a done phrase in its last two sentences, no word
    about going on anywhere in it, and no tool call it tried to make (FU-197)."""
    body = (text or "").strip()
    if not body or _CONTINUE_WORDS.search(body) or unreadable_text(body) is not None:
        return False
    return bool(_DONE_WORDS.search(" ".join(_SENTENCE_END.split(body)[-2:])))


def unreadable_text(text: str) -> tuple[str | None, str] | None:
    """(tool, reason) for a tool call in ``text`` that could not be read (the agent loop's detector)."""
    from backend_service.agent_call_repair import unreadable_call
    from backend_service.tools.studio import STUDIO_TOOL_NAMES

    bad = unreadable_call(text or "", STUDIO_TOOL_NAMES)
    return (bad.tool, bad.reason) if bad is not None else None


def unreadable(outcome: TurnOutcome) -> tuple[str | None, str] | None:
    """(tool, reason) when the turn ended on a tool call that could not be read: the final reply holds the
    fragment, or the turn's last tool result is the agent loop's "could not be read" error."""
    found = unreadable_text(outcome.text)
    if found is not None:
        return found
    calls = outcome.tool_calls
    error = _call_error(calls[-1]) if calls else None
    if error and error.startswith(UNREADABLE_PREFIX):
        reason = error.split(": ", 1)[1].split("; send it again", 1)[0] if ": " in error else error
        return str(calls[-1].get("name") or "") or None, reason
    return None


def last_tool_error(outcome: TurnOutcome) -> tuple[str, str] | None:
    """(tool, error) when the turn's last tool result was an error, else None: an earlier error the model moved
    past is not repeated in the next nudge (FU-197)."""
    calls = outcome.tool_calls
    if not calls or _declined(calls[-1]):
        return None
    error = _call_error(calls[-1])
    return (str(calls[-1].get("name") or ""), error) if error else None


def stall_cause(outcome: TurnOutcome, bad: tuple[str | None, str] | None, last: tuple[str, str] | None) -> str:
    """Why a turn made no progress, as far as the turn shows (FU-198): its call could not be read, a call was
    refused (with the call, to see the same one again), or something else."""
    if bad is not None:
        return "unreadable"
    if last is not None:
        return "refused:" + call_signature(outcome.tool_calls[-1])
    return "other"


def stall_stop(causes: list[str], limit: int, bad: tuple[str | None, str] | None,
               last: tuple[str, str] | None) -> tuple[str, str | None]:
    """The no-progress stop, named for its cause when every stalled turn had the same one (FU-198)."""
    if len(causes) == limit and len(set(causes)) == 1:
        if causes[0] == "unreadable" and bad is not None:
            return "calls_unreadable", f"{bad[0]}: {bad[1]}" if bad[0] else bad[1]
        if causes[0].startswith("refused:") and last is not None:
            return "call_refused", f"{last[0]}: {last[1]}"
    return "no_progress", None


def call_signature(call: dict[str, Any]) -> str:
    """Tool and arguments: the same call made again."""
    try:
        args = json.dumps(call.get("arguments"), sort_keys=True, default=str)
    except (TypeError, ValueError):
        args = str(call.get("arguments"))
    return f"{call.get('name')}:{args[:500]}"


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
    assets: str = "",
) -> str:
    """The app's message for the next turn. ``summary`` and ``next_step`` come from the run's adapter. ``why`` is the reason this nudge is sent ('cap' when the turn hit
    the step limit, 'stopped' when the model ended its turn, 'claimed' when it said done but the project
    disagrees, 'error' after a failed call, 'unreadable' when its last tool call could not be read, with
    ``last_error`` as (tool, reason), 'resume' after Resume)."""
    lines = [f"{NUDGE_PREFIX} (automatic message {number} of {cap} from the app, not from the user)."]
    if include_goal:
        lines.append(f"The brief, as the user gave it:\n\"\"\"\n{_short_block(goal, MAX_GOAL_CHARS)}\n\"\"\"")
    else:
        lines.append(f"Goal: the brief in the user's first message (\"{_short(goal, 90)}\").")
    intro = {
        "cap": "Your last turn stopped at the step limit.",
        "claimed": "The project is not finished yet.",
        "error": "Your last turn ended after a tool error.",
        "resume": "Resuming after a pause. Rebuild your picture from the project state below, not from memory.",
        "swap_done": "The renders you queued ran while the chat model was unloaded; their results are below.",
        "no_call": "Your last reply made no tool call, so nothing was queued, written or started, whatever it said. "
                   "Only a tool call does work; a sentence does not.",
    }.get(why)
    if why == "unreadable":
        tool, reason = last_error or (None, "")
        named = f" ({tool})" if tool else ""
        lines.append(f"Your last tool call{named} could not be read, so it did not run"
                     f"{': ' + _short(reason, 200) if reason else ''}. Send it again as a tool call"
                     f"{': ' + tool if tool else ''}.")
        last_error = None
    if intro:
        lines.append(intro)
    if summary:
        lines.append(summary)
    if assets:
        lines.append(assets)
    if finished_jobs:
        lines.append("Since then: " + "; ".join(_short(j, 160) for j in finished_jobs[:4]) + ".")
    if last_error is not None:
        tool, error = last_error
        lines.append(f"Last error ({tool}): {_short(error, MAX_ERROR_CHARS)}")
    step = next_step or "Do the next step of the brief with the tools."
    if last_error is not None:
        step = f"Fix that error first, then: {step[0].lower()}{step[1:]}"
    lines.append(f"Next step: {step}")
    lines.append("Do it now with a tool call, not with a text reply. When everything in the brief is done, say so in one line.")
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
