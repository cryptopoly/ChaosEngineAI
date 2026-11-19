"""Tool calls the model wrote slightly wrong (FU-196).

``repair_call`` mends one narrow fault: a call object whose JSON lacks only its last closing braces or brackets,
next to a ``<tool_call>`` opener or a lone ``</tool_call>`` (Qwen3.8 did this on every ``storyboard_audio`` call of
a Keep going rehearsal, 2026-11-17). It appends the missing closers and nothing else, and keeps the result only
when it parses, names an available tool and has an object for ``arguments``. It never changes or adds a value,
and never mends a reply that was cut off at the token limit (that call is incomplete, not misformatted).

``unreadable_call`` finds what is left: a call-looking fragment that still cannot be read, with a plain reason
and the call format the model was using, so the agent loop can tell the model instead of dropping the call.
Prose is left alone, and so is a complete call object inside a fenced code block (one the user asked to see).
In a reply where other calls were read, both run on the text left once those are blanked out (FU-205,
``agent_parse.read_calls``), and the fragment carries its place among the calls for the message.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Collection

from backend_service import agent_lfm_calls

_OPEN = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_CLOSE = re.compile(r"\s*</tool_call>", re.IGNORECASE)
_CALL_START = re.compile(r'\{\s*"name"\s*:\s*"([^"\\]{1,128})"')
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_XML_NAME = re.compile(r"<function=([^>\n]{1,128})>")
_LFM_NAME = re.compile(r"\[?\s*([A-Za-z_]\w{0,127})\s*\(")
_PAIR = {"{": "}", "[": "]"}

FORMATS = {
    "json": '<tool_call>{{"name": "{tool}", "arguments": {{...}}}}</tool_call>, with every brace closed',
    "xml": "<tool_call><function={tool}><parameter=KEY>VALUE</parameter></function></tool_call>",
    "lfm": '<|tool_call_start|>[{tool}(key="value")]<|tool_call_end|>',
}


@dataclass
class Unreadable:
    """A call the model tried to make that could not be read."""

    tool: str | None
    reason: str
    style: str  # json | xml | lfm (the FORMATS key)
    fragment: str
    at: int = 0  # where the fragment starts in the text
    position: int | None = None  # its place among the calls of a reply where other calls were read (FU-205)

    @property
    def message(self) -> str:
        """The tool-style error the model gets back."""
        hint = FORMATS[self.style].format(tool=self.tool or "TOOL_NAME")
        where = f" (the {_ordinal(self.position)} call in your reply)" if self.position else ""
        return (f"Error: your tool call could not be read, so nothing ran{where}: {self.reason}; "
                f"send it again as {hint}.")


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def _ordinal(n: int) -> str:
    return _ORDINALS[n - 1] if 1 <= n <= len(_ORDINALS) else f"number {n}"


def _missing_closers(body: str) -> list[str] | str | None:
    """The closers ``body`` lacks at its end, a fault reason when something else is wrong, or None when the first
    value in it is complete (nothing to add)."""
    stack: list[str] = []
    in_string = escaped = False
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in _PAIR:
            stack.append(ch)
        elif ch in "}]":
            if not stack or _PAIR[stack[-1]] != ch:
                return f"a '{ch}' does not match the bracket it closes"
            stack.pop()
            if not stack:
                return None
    if in_string:
        return "a string is not closed"
    return [_PAIR[c] for c in reversed(stack)]


def _closers_phrase(closers: list[str]) -> str:
    braces, brackets = closers.count("}"), closers.count("]")
    parts = []
    if braces:
        parts.append(f"{braces} closing brace{'s' if braces > 1 else ''}")
    if brackets:
        parts.append(f"{brackets} closing bracket{'s' if brackets > 1 else ''}")
    return "the JSON is missing " + " and ".join(parts)


def _region(text: str, start: int) -> tuple[int, int, bool] | None:
    """Where the call starting at ``start`` ends (the close tag's start, else the text's end), the end of what it
    consumes, and whether a tag marks it as a call. None when neither tag is next to it."""
    opened = any(m.end() == start for m in _OPEN.finditer(text, 0, start + 1))
    close = re.search(r"</tool_call>", text[start:], re.IGNORECASE)
    if close is not None:
        body_end = start + close.start()
        return body_end, start + close.end(), True
    if opened:
        return len(text), len(text), True
    return None


def _call_object(payload: Any, names: Collection[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("name") not in names:
        return None
    arguments = payload.get("arguments", payload.get("parameters"))
    return payload if isinstance(arguments, dict) else None


def repair_call(text: str, names: Collection[str], *, truncated: bool = False) -> tuple[dict[str, Any], int, int] | None:
    """``(payload, start, end)`` for the one call in ``text`` that only lacks its closing braces, else None."""
    if truncated or not text or not names:
        return None
    starts = list(_CALL_START.finditer(text))
    if len(starts) != 1 or starts[0].group(1) not in names:
        return None
    start = starts[0].start()
    region = _region(text, start)
    if region is None:
        return None
    body_end, end, _tagged = region
    body = text[start:body_end].rstrip()
    closers = _missing_closers(body)
    if not isinstance(closers, list) or not closers:
        return None
    try:
        payload = json.loads(body + "".join(closers))
    except json.JSONDecodeError:
        return None
    call = _call_object(payload, names)
    if call is None:
        return None
    # Swallow an opener right before the object too, so the caller can cut the whole call out of the text.
    opener = next((m for m in _OPEN.finditer(text, 0, start + 1) if m.end() == start), None)
    return call, (opener.start() if opener else start), end


def _in_fence(text: str, pos: int) -> bool:
    return any(m.start() <= pos < m.end() for m in _FENCE.finditer(text))


_JSON_ESCAPE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt]|(.?))', re.DOTALL)


def _lone_backslash(body: str) -> bool:
    """True when ``body`` holds a backslash that starts no JSON escape (an unescaped Windows path, FU-255)."""
    return any(m.group(1) is not None for m in _JSON_ESCAPE.finditer(body))


def _json_reason(text: str, start: int, body_end: int, names: Collection[str], truncated: bool) -> tuple[str | None, str]:
    """(tool name, reason) for the JSON value at ``start`` that the parser did not take."""
    body = text[start:body_end].rstrip()
    match = _CALL_START.match(body)
    tool = match.group(1) if match else None
    try:
        payload, _end = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError as exc:
        if _lone_backslash(body):
            return tool, "a backslash in a path must be written as two (C:\\\\Users\\\\me), as JSON needs"
        closers = _missing_closers(body)
        if isinstance(closers, list) and closers:
            reason = _closers_phrase(closers)
            if truncated:
                reason += " (the reply reached the token limit before the call was complete)"
            return tool, reason
        if isinstance(closers, str):
            return tool, closers
        return tool, f"the JSON does not parse ({exc.msg})"
    if not isinstance(payload, dict):
        return tool, "it is not a call object with \"name\" and \"arguments\""
    tool = payload.get("name") if isinstance(payload.get("name"), str) else tool
    if not tool:
        return None, "the call object has no \"name\""
    if tool not in names:
        return tool, f"there is no tool named '{tool[:80]}'"
    if not isinstance(payload.get("arguments", payload.get("parameters", {})), dict):
        return tool, "\"arguments\" must be a JSON object"
    return tool, "it was not wrapped in <tool_call> tags"


def _lfm(text: str, names: Collection[str]) -> Unreadable | None:
    start = text.find(agent_lfm_calls.OPEN)
    if start < 0:
        return None
    body_start = start + len(agent_lfm_calls.OPEN)
    match = _LFM_NAME.match(text, body_start)
    tool = match.group(1) if match else None
    if _in_fence(text, start) and tool not in names:
        return None
    end = text.find(agent_lfm_calls.CLOSE, body_start)
    if end < 0:
        reason = f"the {agent_lfm_calls.OPEN} block has no {agent_lfm_calls.CLOSE}"
    else:
        reason = "the block is not a list of calls like name(key=\"value\") with plain values"
    fragment = text[start:end + len(agent_lfm_calls.CLOSE) if end >= 0 else len(text)]
    return Unreadable(tool, reason, "lfm", fragment[:2000], start)


def unreadable_call(text: str, names: Collection[str], *, truncated: bool = False) -> Unreadable | None:
    """The call-looking fragment in a reply the parsers took nothing from, or None for ordinary prose."""
    if not text:
        return None
    lfm = _lfm(text, names)
    if lfm is not None:
        return lfm
    opener = _OPEN.search(text)
    closer = re.search(r"</tool_call>", text, re.IGNORECASE)
    xml = _XML_NAME.search(text)
    calls = [m for m in _CALL_START.finditer(text) if m.group(1) in names]
    tagged_at = min((m.start() for m in (opener, closer, xml) if m is not None), default=None)
    if tagged_at is not None:
        known = any(m.group(1) in names for m in (_XML_NAME.finditer(text))) or bool(calls)
        if not _in_fence(text, tagged_at) or known:
            if xml is not None:
                tool = xml.group(1).strip()
                return Unreadable(tool, "the <function=...> block is not closed with </function>", "xml",
                                  text[xml.start():][:2000], tagged_at)
            first_json = text.find("{", opener.end() if opener else 0)
            if calls:
                first_json = calls[0].start()
            if first_json < 0 or (closer is not None and first_json > closer.start()):
                return Unreadable(None, "there is no call object next to the <tool_call> tag", "json",
                                  text[tagged_at:][:2000], tagged_at)
            body_end = closer.start() if closer is not None and closer.start() > first_json else len(text)
            tool, reason = _json_reason(text, first_json, body_end, names, truncated)
            if reason == "it was not wrapped in <tool_call> tags" and opener is not None and not text[
                    opener.end():first_json].strip():
                # A whole, tagged call is only left over when calls in another format were read from the reply.
                reason = "the reply mixed two call formats and only the other one was read"
            return Unreadable(tool, reason, "json", text[first_json:body_end][:2000], tagged_at)
    for match in calls:
        start = match.start()
        if '"arguments"' not in text[start:] and '"parameters"' not in text[start:]:
            continue
        tool, reason = _json_reason(text, start, len(text), names, truncated)
        if _in_fence(text, start) and reason == "it was not wrapped in <tool_call> tags":
            continue  # a complete call shown in a code block: the user asked to see it
        return Unreadable(tool, reason, "json", text[start:][:2000], start)
    return None
