"""Reading tool calls out of a model's reply text (extracted from ``agent.py``).

Models that write their calls as text use several shapes: Hermes ``<tool_call>{json}</tool_call>`` (with or
without the close tag, or without the opener), the XML body ``<function=NAME><parameter=K>V</parameter>`` and
LFM2.5's ``<|tool_call_start|>[name(k=v)]<|tool_call_end|>``. ``_parse_tool_calls_from_response`` turns them into
OpenAI ``tool_calls``; ``_strip_tool_call_xml`` removes what it read from the text the user sees. ``agent.py``
re-exports both, so existing imports keep working.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from backend_service import agent_call_repair, agent_lfm_calls

logger = logging.getLogger("backend_service.agent")


_TOOL_CALL_OPEN = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_TOOL_CALL_CLOSE = re.compile(r"\s*</tool_call>", re.IGNORECASE)
# XML call body used by LFM3 and Qwen3-Coder chat templates:
# <function=NAME><parameter=KEY>VALUE</parameter>...</function>
_XML_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_XML_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _decode_xml_tool_call(text: str, start: int) -> tuple[dict[str, Any], int] | None:
    """Decode an XML call body at ``start``, as ``({"name", "arguments"}, end)``.

    Parameter values are read as JSON where they parse (numbers, lists,
    objects) and kept as text otherwise, as the templates' own parser does.
    """
    match = _XML_FUNCTION.match(text, start)
    if match is None:
        return None
    arguments: dict[str, Any] = {}
    for key, value in _XML_PARAMETER.findall(match.group(2)):
        try:
            arguments[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key.strip()] = value
    return {"name": match.group(1).strip(), "arguments": arguments}, match.end()


def _bare_call_span(text: str) -> tuple[dict[str, Any], int, int] | None:
    """A call whose ``<tool_call>`` opener the model dropped: ``{"name": ..., "arguments": ...}</tool_call>``.

    Qwen3.8 does this on some turns (seen on a Storyboard chat, 2026-11-15): JSON that runs straight into the
    close tag. Only accepted when the text holds no opener at all and a call object is followed directly by the
    close tag, so ordinary prose with braces in it is left alone. Returns ``(payload, start, end)``.
    """
    if "<tool_call>" in text.lower() or "</tool_call>" not in text.lower():
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        payload, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    close = _TOOL_CALL_CLOSE.match(text, end)
    if close is None or not isinstance(payload, dict) or not (payload.get("name") or payload.get("function")):
        return None
    return payload, start, close.end()


def _strip_tool_call_xml(text: str) -> str:
    """Remove every ``<tool_call>...`` blob from a model response.

    FU-040: the chat UI shows ``result.text`` verbatim in the assistant
    bubble, so when a model emits a ``<tool_call>`` block AND we
    execute the call (either via the engine's structured field or via
    ``_parse_tool_calls_from_response``), the user sees the same call
    twice — once as raw XML noise and once as a ``ToolCallCard``. We
    strip the XML from the text we hand back to the streaming layer.

    Uses the same ``JSONDecoder.raw_decode`` walk as the parser so we
    only remove the well-formed-JSON region the parser actually
    consumed; everything around it (the model's natural-language
    framing) stays put. A trailing ``</tool_call>`` close tag, when
    present, is also swallowed.
    """
    if not text:
        return text
    text = agent_lfm_calls.strip_calls(text)  # FU-192: LFM2.5 ``<|tool_call_start|>`` blocks
    bare = _bare_call_span(text)
    if bare is not None:
        return re.sub(r"\n{3,}", "\n\n", text[:bare[1]] + text[bare[2]:]).strip()
    if "<tool_call>" not in text.lower():
        return text
    decoder = json.JSONDecoder()
    out: list[str] = []
    cursor = 0
    while True:
        match = _TOOL_CALL_OPEN.search(text, cursor)
        if match is None:
            out.append(text[cursor:])
            break
        out.append(text[cursor:match.start()])
        start = match.end()
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break
        try:
            _payload, end = _decode_xml_tool_call(text, start) or decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            # Malformed JSON after ``<tool_call>`` — drop the opener
            # alone and continue. The garbage payload stays so the
            # operator can see what the model emitted.
            cursor = match.end()
            continue
        cursor = end
        close = _TOOL_CALL_CLOSE.match(text, cursor)
        if close is not None:
            cursor = close.end()
    cleaned = "".join(out)
    # Collapse the double-blank-line that can appear when we strip a
    # mid-paragraph tool_call. ``\n\n\n+`` → ``\n\n`` keeps paragraph
    # breaks intact while removing the visible gap.
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _parse_tool_calls_from_response(
    response_text: str,
    names: Any = None,
    truncated: bool = False,
) -> list[dict[str, Any]] | None:
    """Attempt to extract tool calls from a text response.

    Models using the OpenAI tool-calling protocol return structured
    tool_calls in the response object. For models that embed tool calls
    in their text output (e.g. Hermes / NousResearch / Qwen3-Coder-Next),
    we parse them from the ``<tool_call>...</tool_call>`` XML-ish
    convention.

    FU-040 (2026-05-10): widened to handle three real-world shapes
    Coder-Next emitted in a single chat session:

      1. ``<tool_call>{"name": "x", "arguments": {...}}</tool_call>``
         — the canonical Hermes shape. Always worked.
      2. ``<tool_call>{"name": "x", "arguments": {...}}`` — no
         closing tag. The previous regex required ``</tool_call>``
         and silently dropped these, so the model's tool call
         rendered as raw XML text in the assistant bubble with no
         execution.
      3. ``<tool_call> [ {url: ...}, {url: ...} ]`` — model
         hallucinated a JSON ARRAY of pseudo-results instead of a
         call object. Rejected (the array shape has no ``name`` /
         ``arguments`` keys to dispatch from), but we keep parsing
         so any well-formed call later in the same message still
         lands.

    The parser walks each ``<tool_call>`` opener and uses the stdlib
    ``json.JSONDecoder.raw_decode`` to consume exactly the next valid
    JSON value (object OR array) — that handles both shapes (1) and
    (2) without requiring a closing tag, and shape (3) decodes to a
    list which we discard. ``raw_decode`` also correctly skips nested
    braces inside argument string values that a naive regex would
    choke on.

    FU-196: with ``names`` (the tools the model was given), a single call that only lacks its closing braces is
    mended by ``agent_call_repair.repair_call`` when nothing else parses; ``truncated`` (the reply hit the token
    limit) turns that off.
    """
    if not response_text:
        return None
    calls = _parse_tool_calls_strict(response_text)
    if calls or not names:
        return calls
    repaired = agent_call_repair.repair_call(response_text, names, truncated=truncated)
    if repaired is None:
        return None
    payload = repaired[0]
    arguments = payload.get("arguments", payload.get("parameters"))
    logger.info("Read a tool call for %s that lacked its closing braces", payload.get("name"))
    return [{
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {"name": payload["name"], "arguments": json.dumps(arguments)},
    }]


def read_calls(
    response_text: str,
    names: Any,
    truncated: bool = False,
) -> tuple[list[dict[str, Any]] | None, agent_call_repair.Unreadable | None]:
    """The calls in a reply in the order written, and a call beside them that could not be read (FU-205).

    When nothing parses this is ``_parse_tool_calls_from_response`` (the single-call repair) and no fragment: the
    loop reports a lone unreadable call itself. When a call parses, the text it came from is blanked out and the
    rest is read again: one call that only lacks its closing braces is mended (the same rules as a lone call), and
    the first call-like fragment left is returned with its place among the calls, so the model can be told."""
    found = _strict_call_spans(response_text) if response_text else []
    if not found or not names:
        return _parse_tool_calls_from_response(response_text, names, truncated), None
    masked = list(response_text)
    for _call, start, end in found:
        masked[start:end] = " " * (end - start)
    rest = "".join(masked)
    repaired = agent_call_repair.repair_call(rest, names, truncated=truncated)
    if repaired is not None:
        payload, start, end = repaired
        logger.info("Read a tool call for %s that lacked its closing braces", payload.get("name"))
        found.append((_call_dict(payload["name"], payload.get("arguments", payload.get("parameters"))), start, end))
        rest = rest[:start] + " " * (end - start) + rest[end:]
    found.sort(key=lambda item: item[1])
    bad = agent_call_repair.unreadable_call(rest, names, truncated=truncated)
    if bad is not None:
        bad.position = 1 + sum(1 for _call, start, _end in found if start < bad.at)
    return [call for call, _start, _end in found], bad


def _call_dict(name: Any, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw": arguments}
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
        },
    }


def _parse_tool_calls_strict(response_text: str) -> list[dict[str, Any]] | None:
    return [call for call, _start, _end in _strict_call_spans(response_text)] or None


def _strict_call_spans(response_text: str) -> list[tuple[dict[str, Any], int, int]]:
    """Every call the strict parser reads, with the span of text it came from."""
    lfm = agent_lfm_calls.call_spans(response_text)  # FU-192: LFM2.5 ``<|tool_call_start|>[name(k=v)]``
    if lfm:
        return lfm
    bare = _bare_call_span(response_text)
    if bare is not None:
        # Same shape handling as below, for the call that lost its opener.
        payload = bare[0]
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        return [(_call_dict(payload.get("name") or payload.get("function"), arguments), bare[1], bare[2])]
    if "<tool_call>" not in response_text.lower():
        return []

    calls: list[tuple[dict[str, Any], int, int]] = []
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        match = _TOOL_CALL_OPEN.search(response_text, cursor)
        if match is None:
            break
        start = match.end()
        # Find the first non-whitespace character; ``raw_decode`` needs
        # to start at the JSON token itself, not at preceding spaces.
        while start < len(response_text) and response_text[start].isspace():
            start += 1
        if start >= len(response_text):
            break
        try:
            payload, end = (
                _decode_xml_tool_call(response_text, start)
                or decoder.raw_decode(response_text, start)
            )
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = end
        # Shape (3): the model emitted hallucinated results as a list.
        # No ``name`` to dispatch from — skip without aborting the
        # outer loop so a later well-formed call in the same message
        # still gets picked up.
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("function")
        if not name:
            continue
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        close = _TOOL_CALL_CLOSE.match(response_text, end)
        calls.append((_call_dict(name, arguments), match.start(), close.end() if close else end))

    return calls
