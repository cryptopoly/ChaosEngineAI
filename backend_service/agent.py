"""Agent loop for ChaosEngineAI tool-use.

Wraps RuntimeController.generate() with an automatic dispatch loop:
1. Call generate() with tool schemas
2. Inspect response for tool_calls
3. Execute each tool call via ToolRegistry
4. Inject tool results back into the conversation
5. Repeat until the model stops calling tools (or max iterations hit)
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from backend_service.tools import ToolRegistry, registry as default_registry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10


@dataclass
class ToolCallResult:
    """One completed tool invocation."""
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: str
    elapsed_seconds: float
    # Phase 2.8: optional structured output the frontend can render
    # natively (table / code / markdown / image / chart). When None,
    # the legacy collapsible-JSON renderer fires. The `result` text
    # field is always populated so the language model sees something
    # readable on the next turn regardless of UI rendering.
    render_as: str | None = None
    data: dict[str, Any] | None = None


@dataclass
class AgentResult:
    """Final result of an agent loop run."""
    text: str
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    iterations: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


_TOOL_CALL_OPEN = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_TOOL_CALL_CLOSE = re.compile(r"\s*</tool_call>", re.IGNORECASE)


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
    if not text or "<tool_call>" not in text.lower():
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
            _payload, end = decoder.raw_decode(text, start)
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


def _parse_tool_calls_from_response(response_text: str) -> list[dict[str, Any]] | None:
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
    """
    if not response_text or "<tool_call>" not in response_text.lower():
        return None

    calls: list[dict[str, Any]] = []
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
            payload, end = decoder.raw_decode(response_text, start)
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
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
            },
        })

    return calls if calls else None


def _execute_tool_call(
    tool_call: dict[str, Any],
    tool_registry: ToolRegistry,
) -> ToolCallResult:
    """Execute a single tool call and return the result."""
    call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")
    func = tool_call.get("function", {})
    tool_name = func.get("name", "unknown")
    raw_args = func.get("arguments", "{}")

    # FU-039 (2026-05-10): coerce ``arguments`` to a dict at the source.
    # Models occasionally emit ``{"arguments": null}`` (Coder-Next does
    # this when the tool call has no parameters) or send a non-string,
    # non-dict shape we don't recognise. Both routes used to set
    # ``arguments = None``, which then landed in ``ToolCallResult``,
    # serialised into the persisted session, and crashed the frontend's
    # ``ToolCallCard`` at ``Object.entries(null)`` on every subsequent
    # render. Result: a single bad tool turn permanently bricked the
    # Chat tab. Defaulting to ``{}`` keeps the contract consumers
    # already assume — and means the frontend boundary (also added in
    # FU-039) only fires for genuinely corrupt records, not the common
    # "no args" path.
    try:
        if raw_args is None:
            arguments = {}
        elif isinstance(raw_args, str):
            arguments = json.loads(raw_args) if raw_args.strip() else {}
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            arguments = {"raw": raw_args}
    except json.JSONDecodeError:
        arguments = {"raw": raw_args}

    tool = tool_registry.get(tool_name)
    if tool is None:
        return ToolCallResult(
            tool_call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            result=f"Error: unknown tool '{tool_name}'. Available tools: {', '.join(tool_registry.available_names())}",
            elapsed_seconds=0.0,
        )

    start = time.perf_counter()
    render_as: str | None = None
    structured_data: dict[str, Any] | None = None
    try:
        # Phase 2.8: try the structured entry first. Tools that
        # haven't migrated return None and we fall back to the
        # plain-text path below.
        structured = tool.execute_structured(**arguments)
        if structured is not None:
            result_text = structured.text
            render_as = structured.render_as
            structured_data = structured.data
        else:
            result_text = tool.execute(**arguments)
    except Exception as exc:
        result_text = f"Error executing {tool_name}: {exc}"
    elapsed = round(time.perf_counter() - start, 3)

    logger.info("Tool %s executed in %.3fs", tool_name, elapsed)

    return ToolCallResult(
        tool_call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
        result=result_text,
        elapsed_seconds=elapsed,
        render_as=render_as,
        data=structured_data,
    )


def run_agent_loop(
    *,
    generate_fn: Any,
    prompt: str,
    history: list[dict[str, Any]],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    images: list[str] | None = None,
    tool_registry: ToolRegistry | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    available_tools: list[str] | None = None,
) -> AgentResult:
    """Run the agent loop synchronously.

    Parameters
    ----------
    generate_fn : callable
        A function with signature compatible with RuntimeController.generate()
        that returns a GenerationResult with .text, .finishReason, .tool_calls,
        .promptTokens, .completionTokens attributes.
    prompt : str
        The user's initial prompt.
    history : list
        Conversation history.
    system_prompt : str | None
        System prompt.
    max_tokens : int
        Max generation tokens per iteration.
    temperature : float
        Sampling temperature.
    tool_registry : ToolRegistry | None
        Registry of available tools. Uses the global default if not provided.
    max_iterations : int
        Maximum number of tool-call/re-generate cycles.
    available_tools : list[str] | None
        Restrict to specific tool names. None means all registered tools.
    """
    reg = tool_registry or default_registry

    # Build tool schemas
    if available_tools is not None:
        tools = [
            t.openai_schema()
            for t in reg.list_tools()
            if t.name in available_tools
        ]
    else:
        tools = reg.openai_schemas()

    if not tools:
        # No tools available — just do a normal generation
        result = generate_fn(
            prompt=prompt,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
        )
        return AgentResult(
            text=result.text,
            iterations=1,
            total_prompt_tokens=result.promptTokens,
            total_completion_tokens=result.completionTokens,
        )

    all_tool_results: list[ToolCallResult] = []
    total_prompt = 0
    total_completion = 0

    # Build the messages for multi-turn tool use
    messages = list(history)  # copy
    # Add the current user message
    messages.append({"role": "user", "text": prompt})

    for iteration in range(max_iterations):
        # Generate with tools
        result = generate_fn(
            prompt=prompt if iteration == 0 else "",
            history=messages[:-1] if iteration == 0 else messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images if iteration == 0 else None,
            tools=tools,
        )

        total_prompt += result.promptTokens
        total_completion += result.completionTokens

        # Check for tool calls in the structured response
        tool_calls = getattr(result, "tool_calls", None)

        # If no structured tool calls, try parsing from text
        if not tool_calls and result.text:
            tool_calls = _parse_tool_calls_from_response(result.text)

        if not tool_calls:
            # Model is done — return the final text. Strip any
            # ``<tool_call>`` XML the parser consumed so the chat
            # bubble doesn't show raw call JSON next to a rendered
            # ToolCallCard (FU-040).
            return AgentResult(
                text=_strip_tool_call_xml(result.text),
                tool_calls=all_tool_results,
                iterations=iteration + 1,
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
            )

        # Execute each tool call
        # Add assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "text": result.text or "",
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            tc_result = _execute_tool_call(tc, reg)
            all_tool_results.append(tc_result)

            # Add tool result to conversation
            messages.append({
                "role": "tool",
                "text": tc_result.result,
                "tool_call_id": tc_result.tool_call_id,
                "name": tc_result.tool_name,
            })

    # Max iterations reached — return whatever we have
    final_text = "I've reached the maximum number of tool-use iterations. Here's what I found:\n\n"
    for tr in all_tool_results:
        final_text += f"- {tr.tool_name}: {tr.result[:200]}\n"

    return AgentResult(
        text=final_text,
        tool_calls=all_tool_results,
        iterations=max_iterations,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
    )


def run_agent_loop_streaming(
    *,
    generate_fn: Any,
    stream_generate_fn: Any,
    prompt: str,
    history: list[dict[str, Any]],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    images: list[str] | None = None,
    tool_registry: ToolRegistry | None = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    available_tools: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Streaming version of the agent loop.

    Yields dicts with keys:
    - {"token": str} for text tokens
    - {"tool_call_start": {...}} when a tool execution begins
    - {"tool_call_result": {...}} when a tool execution completes
    - {"done": True, ...} when the loop finishes
    """
    reg = tool_registry or default_registry

    if available_tools is not None:
        tools = [
            t.openai_schema()
            for t in reg.list_tools()
            if t.name in available_tools
        ]
    else:
        tools = reg.openai_schemas()

    if not tools:
        # No tools — stream normally
        yield from _passthrough_stream(
            stream_generate_fn,
            prompt=prompt,
            history=history,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images,
        )
        return

    all_tool_results: list[ToolCallResult] = []
    messages = list(history)
    messages.append({"role": "user", "text": prompt})

    for iteration in range(max_iterations):
        # Use non-streaming generate for tool-calling iterations
        # (streaming + tool calls is complex; non-streaming is reliable)
        result = generate_fn(
            prompt=prompt if iteration == 0 else "",
            history=messages[:-1] if iteration == 0 else messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            images=images if iteration == 0 else None,
            tools=tools,
        )

        tool_calls = getattr(result, "tool_calls", None)
        if not tool_calls and result.text:
            tool_calls = _parse_tool_calls_from_response(result.text)

        if not tool_calls:
            # Final response — stream it token by token for the user
            # Since we already have the full text, emit it in chunks.
            # Strip any ``<tool_call>`` XML blobs the parser already
            # consumed so the assistant bubble doesn't show raw call
            # JSON next to the rendered ToolCallCard (FU-040).
            text = _strip_tool_call_xml(result.text)
            # The final answer is already fully computed (tool-calling turns
            # are non-streaming), so the old 4-char dribble just added fake
            # latency + yields. Emit in larger chunks; the SSE layer coalesces
            # these further and the user sees the answer near-instantly.
            chunk_size = 48
            for i in range(0, len(text), chunk_size):
                yield {"token": text[i:i + chunk_size]}

            yield {
                "done": True,
                "tool_calls": [
                    {
                        "id": tr.tool_call_id,
                        "name": tr.tool_name,
                        "arguments": tr.arguments,
                        "result": tr.result,
                        "elapsed": tr.elapsed_seconds,
                    }
                    for tr in all_tool_results
                ],
                "iterations": iteration + 1,
            }
            return

        # Execute tool calls
        messages.append({
            "role": "assistant",
            "text": result.text or "",
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            func = tc.get("function", {})
            yield {
                "tool_call_start": {
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "arguments": func.get("arguments"),
                },
            }

            tc_result = _execute_tool_call(tc, reg)
            all_tool_results.append(tc_result)

            yield {
                "tool_call_result": {
                    "id": tc_result.tool_call_id,
                    "name": tc_result.tool_name,
                    "result": tc_result.result[:2000],  # Cap for streaming
                    "elapsed": tc_result.elapsed_seconds,
                    # Phase 2.8: stream the structured shape so the
                    # frontend can render it as the tool finishes
                    # rather than waiting for the final done payload.
                    "renderAs": tc_result.render_as,
                    "data": tc_result.data,
                },
            }

            messages.append({
                "role": "tool",
                "text": tc_result.result,
                "tool_call_id": tc_result.tool_call_id,
                "name": tc_result.tool_name,
            })

    # Max iterations
    yield {"token": "\n\n(Reached maximum tool-use iterations)"}
    yield {"done": True, "tool_calls": [], "iterations": max_iterations}


def _passthrough_stream(
    stream_generate_fn: Any,
    **kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Pass through a normal streaming generation without tool use."""
    for chunk in stream_generate_fn(**kwargs):
        if chunk.text:
            yield {"token": chunk.text}
        if chunk.done:
            yield {
                "done": True,
                "tool_calls": [],
                "iterations": 0,
                "finish_reason": chunk.finish_reason,
                "prompt_tokens": chunk.prompt_tokens,
                "completion_tokens": chunk.completion_tokens,
                "tok_s": chunk.tok_s,
            }
