"""Wire-format helpers for the TensorFold engine.

Pure functions (no sockets, no subprocesses) so the request/response shapes
can be unit-tested without a server:

* ``build_messages`` / ``build_payload`` — ChaosEngineAI chat state → the
  OpenAI-style body ``tensorfold serve`` accepts.
* ``ToolCallAssembler`` / ``tool_calls_to_text`` — TensorFold answers tool
  calls as structured OpenAI ``tool_calls``; the agent loop reads the
  ``<tool_call>{…}</tool_call>`` text form, so structured calls are rendered
  back into that form.
* ``server_error_message`` / ``parse_reply_budget`` /
  ``summarize_startup_failure`` — turn server and process errors into one
  readable line.

Field names come from TensorFold's ``docs/api.md`` and its server source
(``server/http.py``, ``server/app.py``); re-check them when the pin in
``scripts/install-tensorfold.sh`` moves.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from backend_service.inference._utils import _normalize_message_content

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL | re.IGNORECASE)
_REPLY_BUDGET = re.compile(r"request at most (\d+) reply tokens")
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit)\b.*")
_CLI_ERROR_PREFIX = "tensorfold:"

# ``reasoning_effort`` values the server accepts; anything else is dropped
# instead of turning the whole turn into an HTTP 400.
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})

# Sampling knobs TensorFold's exact sampler reads. ``stop`` is handled apart.
_PASSTHROUGH_SAMPLERS = ("top_p", "top_k", "min_p")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _structured_calls(raw: Any) -> list[dict[str, Any]]:
    """Agent-loop tool calls → OpenAI ``tool_calls`` (arguments as a JSON string)."""
    if not isinstance(raw, list):
        return []
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif arguments is None:
            arguments = "{}"
        elif not isinstance(arguments, str):
            arguments = json.dumps({"raw": arguments}, default=str)
        calls.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {"name": str(function["name"]), "arguments": arguments},
            }
        )
    return calls


def _image_part(image: str) -> dict[str, Any]:
    """An OpenAI ``image_url`` part for one attached image.

    The chat composer stores raw base64 of whatever file was chosen. TensorFold
    decodes the bytes itself (JPEG, PNG or WebP) and ignores the media type in
    the URL, so one prefix serves them all.
    """
    url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}


def build_messages(
    system_prompt: str | None,
    history: list[dict[str, Any]],
    prompt: str,
    images: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Chat history → server messages, keeping tool calls structured.

    Assistant turns that carry ``tool_calls`` keep them as data so the
    model's own chat template renders them in its native call syntax; the
    ``<tool_call>`` text this module appends for the agent loop is stripped
    from those turns so a call is never shown to the model twice. The agent
    loop sends an empty ``prompt`` on its follow-up iterations (the last
    history entry is the tool result), so an empty prompt adds no user turn.
    ``images`` (base64, only when the server was started with ``--vision``)
    ride on that final user turn, after its text, like the llama.cpp engine.
    """
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for message in history:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        text = _normalize_message_content(message.get("text", ""))
        entry: dict[str, Any] = {"role": role, "content": text}
        if role == "assistant":
            calls = _structured_calls(message.get("tool_calls"))
            if calls:
                entry["tool_calls"] = calls
                entry["content"] = _TOOL_CALL_BLOCK.sub("", text).strip()
        elif role == "tool" and message.get("tool_call_id"):
            entry["tool_call_id"] = str(message["tool_call_id"])
        messages.append(entry)
    if prompt and images:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *(_image_part(image) for image in images)],
        })
    elif prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


def build_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
    thinking_mode: str | None = None,
    samplers: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if tools:
        payload["tools"] = tools
    if (thinking_mode or "off") == "off":
        # Qwen / GLM / Gemma templates think by default; honour "off" rather
        # than spending the reply budget on hidden reasoning.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif reasoning_effort in REASONING_EFFORTS:
        payload["reasoning_effort"] = reasoning_effort
    if samplers:
        for key in _PASSTHROUGH_SAMPLERS:
            value = samplers.get(key)
            if _is_number(value):
                payload[key] = value
        seed = samplers.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0:
            payload["seed"] = seed
        stop = samplers.get("stop")
        if isinstance(stop, str) and stop:
            payload["stop"] = stop
        elif isinstance(stop, list) and stop and all(isinstance(item, str) for item in stop):
            payload["stop"] = stop
    if json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": json_schema, "strict": True},
        }
    return payload


def unsupported_sampler_note(samplers: dict[str, Any] | None) -> str | None:
    """Name the sampling knobs the user set that TensorFold's sampler has no rule for.

    TensorFold samples with temperature, top-k, top-p, min-p and a seed (that
    is what keeps drafted replies exact); repetition penalties, mirostat, DRY,
    XTC and top-n-sigma have no equivalent, so they are dropped. Say so
    rather than let a tuned preset silently do nothing.
    """
    if not samplers:
        return None
    ignored: list[str] = []
    if _is_number(samplers.get("repeat_penalty")) and samplers["repeat_penalty"] != 1:
        ignored.append("repeat_penalty")
    for key in ("frequency_penalty", "presence_penalty"):
        if _is_number(samplers.get(key)) and samplers[key] != 0:
            ignored.append(key)
    for key in ("mirostat", "dry_multiplier", "xtc_probability", "top_n_sigma"):
        if _is_number(samplers.get(key)) and samplers[key] > 0:
            ignored.append(key)
    if not ignored:
        return None
    return f"TensorFold ignored unsupported sampler settings: {', '.join(ignored)}."


def tool_calls_to_text(tool_calls: list[dict[str, Any]]) -> str:
    """Structured tool calls → the ``<tool_call>{…}</tool_call>`` blocks the agent loop parses."""
    blocks: list[str] = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            continue
        raw = function.get("arguments")
        arguments: Any = {}
        if isinstance(raw, dict):
            arguments = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                arguments = json.loads(raw)
            except ValueError:
                arguments = {"raw": raw}
            if not isinstance(arguments, dict):
                arguments = {"raw": raw}
        blocks.append(
            "<tool_call>" + json.dumps({"name": str(function["name"]), "arguments": arguments}) + "</tool_call>"
        )
    return "\n".join(blocks)


class ToolCallAssembler:
    """Rebuild streamed ``delta.tool_calls`` fragments into whole calls.

    The server sends each call as a first delta with ``index`` / ``id`` /
    ``function.name`` and then the arguments (whole, or in pieces); pieces
    with the same ``index`` are concatenated in arrival order.
    """

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def feed(self, fragments: Any) -> None:
        if not isinstance(fragments, list):
            return
        for position, fragment in enumerate(fragments):
            if not isinstance(fragment, dict):
                continue
            index = fragment.get("index")
            index = index if isinstance(index, int) and not isinstance(index, bool) else position
            call = self._calls.setdefault(
                index, {"id": f"call_{index}", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            if fragment.get("id"):
                call["id"] = str(fragment["id"])
            function = fragment.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    call["function"]["name"] = str(function["name"])
                if isinstance(function.get("arguments"), str):
                    call["function"]["arguments"] += function["arguments"]

    def calls(self) -> list[dict[str, Any]]:
        return [self._calls[index] for index in sorted(self._calls) if self._calls[index]["function"]["name"]]


def server_error_message(error: Any) -> str:
    """The message inside a server error payload (``{"error": {...}}`` or a bare string)."""
    if isinstance(error, dict):
        inner = error.get("error", error)
        if isinstance(inner, dict):
            return str(inner.get("message") or inner)
        return str(inner)
    return str(error)


def parse_reply_budget(message: str) -> int | None:
    """Reply-token room a "too many reply tokens" refusal names, if it names one.

    The server refuses an explicit ``max_tokens`` that would overflow the
    context window and says how many reply tokens do fit; that number is what
    the request retries with.
    """
    match = _REPLY_BUDGET.search(message or "")
    if not match:
        return None
    room = int(match.group(1))
    return room if room >= 1 else None


def summarize_startup_failure(log_tail: str, returncode: int | None = None) -> str:
    """One readable line for a server that exited during startup.

    The CLI reports refusals it expects (unsupported family, weights past
    the memory budget, a bad flag) as ``tensorfold: <reason>`` on stderr;
    Python tracebacks end in ``SomeError: message``. Either line is the
    cause. Falls back to the last non-empty line, then to the exit code.
    """
    lines = [line.strip() for line in (log_tail or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith(_CLI_ERROR_PREFIX) and len(line) > len(_CLI_ERROR_PREFIX):
            return line[len(_CLI_ERROR_PREFIX):].strip()[:600]
    for line in reversed(lines):
        if _EXCEPTION_LINE.match(line):
            return line[:600]
    if lines:
        return lines[-1][:600]
    if returncode is not None:
        return f"TensorFold exited during startup (code {returncode}) without printing a reason."
    return "TensorFold exited during startup."
