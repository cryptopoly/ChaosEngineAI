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
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from backend_service import agent_call_repair
from backend_service.tools import StructuredToolOutput, ToolRegistry, registry as default_registry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ITERATIONS = 10
# Replies that only wait on a job already started (wait_for_job) do not use up the iteration budget. This
# bounds them anyway: a model that polls forever still ends, after about a day of 30-minute waits.
_FREE_WAIT_LIMIT = 48
_PROGRESS_SECONDS = 2.0
# Replies in a row whose tool call could not be read and got an error back (FU-196); the next one ends the turn,
# so a model that cannot format a call does not loop until the iteration budget runs out.
_UNREADABLE_LIMIT = 2


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
    # Epoch milliseconds when the call began; None for records made before this field existed.
    started_at: int | None = None


@dataclass
class AgentResult:
    """Final result of an agent loop run."""
    text: str
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    iterations: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


# Text-call parsing lives in agent_parse.py; re-exported here for existing callers and tests.
from backend_service.agent_parse import (  # noqa: E402,F401
    _TOOL_CALL_CLOSE,
    _TOOL_CALL_OPEN,
    _bare_call_span,
    _decode_xml_tool_call,
    _parse_tool_calls_from_response,
    _parse_tool_calls_strict,
    _strip_tool_call_xml,
)


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
    started_ms = int(time.time() * 1000)
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
        started_at=started_ms,
    )


def _run_tool_call_live(tool_call: dict[str, Any], tool_registry: ToolRegistry) -> Iterator[dict[str, Any]]:
    """Run one call; a long-running one (a wait on a render) runs on a worker thread while this yields
    ``tool_call_progress`` events every couple of seconds. Returns the ``ToolCallResult``."""
    func = tool_call.get("function", {})
    tool = tool_registry.get(func.get("name", ""))
    arguments = _coerce_arguments(func.get("arguments"))
    if tool is None or not tool.long_running(arguments):
        return _execute_tool_call(tool_call, tool_registry)
    box: list[ToolCallResult] = []
    worker = threading.Thread(target=lambda: box.append(_execute_tool_call(tool_call, tool_registry)),
                              name="agent-long-tool", daemon=True)
    worker.start()
    while True:
        worker.join(_PROGRESS_SECONDS)
        if not worker.is_alive():
            break
        snapshot = tool.live_progress(arguments)
        if snapshot is not None:
            yield {"tool_call_progress": {"id": tool_call.get("id"), "name": func.get("name"), "data": snapshot}}
    return box[0]


def _declined_tool_call(tool_call: dict[str, Any], text: Any) -> ToolCallResult:
    """The result a call gets when the user declined it at the plan card: nothing ran. The gate can also hand
    over a ready result (a refusal found before the card, or the job already making this render)."""
    func = tool_call.get("function", {})
    name = func.get("name", "unknown")
    if isinstance(text, StructuredToolOutput):
        return ToolCallResult(
            tool_call_id=tool_call["id"],
            tool_name=name,
            arguments=_coerce_arguments(func.get("arguments")),
            result=text.text,
            elapsed_seconds=0.0,
            render_as=text.render_as,
            data=text.data,
        )
    return ToolCallResult(
        tool_call_id=tool_call["id"],
        tool_name=name,
        arguments=_coerce_arguments(func.get("arguments")),
        result=text,
        elapsed_seconds=0.0,
        render_as="studio",
        data={"tool": name, "declined": True, "message": text},
    )


def _coerce_arguments(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw": raw_args}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {}


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {str((t.get("function") or {}).get("name") or "") for t in tools} - {""}


def _truncated(result: Any) -> bool:
    return getattr(result, "finishReason", None) == "length"


def _unreadable_turn(bad: agent_call_repair.Unreadable, reply_text: str) -> tuple[ToolCallResult, list[dict[str, Any]]]:
    """The error record for a call that could not be read (FU-196) and the two messages that tell the model: its
    reply as it wrote it, then the error in the ``<tool_response>`` wrapping a tool result gets. The error goes in
    a user turn because no call id exists to pair a ``tool`` turn with, which some chat templates require."""
    record = ToolCallResult(
        tool_call_id=f"call_{uuid.uuid4().hex[:12]}",
        tool_name=bad.tool or "tool_call",
        arguments={"unreadable": bad.fragment},
        result=bad.message,
        elapsed_seconds=0.0,
        started_at=int(time.time() * 1000),
    )
    logger.info("Tool call could not be read (%s): %s", bad.tool, bad.reason)
    return record, [
        {"role": "assistant", "text": reply_text},
        {"role": "user", "text": f"<tool_response>\n{bad.message}\n</tool_response>"},
    ]


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

    names = _tool_names(tools)
    unreadable = 0  # replies in a row that got a "could not be read" error back
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
            tool_calls = _parse_tool_calls_from_response(result.text, names, _truncated(result))

        if not tool_calls and unreadable < _UNREADABLE_LIMIT:
            bad = agent_call_repair.unreadable_call(result.text or "", names, truncated=_truncated(result))
            if bad is not None:
                unreadable += 1
                record, told = _unreadable_turn(bad, result.text or "")
                all_tool_results.append(record)
                messages.extend(told)
                continue
        if tool_calls:
            unreadable = 0

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
    gate: Any = None,
) -> Iterator[dict[str, Any]]:
    """Streaming version of the agent loop.

    Yields dicts with keys:
    - {"token": str} for text tokens
    - {"tool_call_start": {...}} when a tool execution begins
    - {"tool_call_result": {...}} when a tool execution completes
    - {"done": True, ...} when the loop finishes

    ``gate`` (Chat's studio approval, ``tools.studio_gate.StudioGate``) sees each reply's tool calls
    before they run; its events pass through and the calls it declines get its text as their result.
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

    used = 0  # iterations that count against ``max_iterations``
    last_prompt_tokens = 0
    free = 0  # replies that only waited on a job already waited on this turn
    seen_waits: set[str] = set()
    iteration = -1
    names = _tool_names(tools)
    unreadable = 0  # replies in a row that got a "could not be read" error back
    while used < max_iterations:
        iteration += 1
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

        # The prompt size of the newest step: Keep going (FU-191) stops before the context fills up.
        last_prompt_tokens = int(getattr(result, "promptTokens", 0) or 0)
        tool_calls = getattr(result, "tool_calls", None)
        if not tool_calls and result.text:
            tool_calls = _parse_tool_calls_from_response(result.text, names, _truncated(result))

        if not tool_calls and unreadable < _UNREADABLE_LIMIT:
            # FU-196: a call that could not be read is never dropped silently: the model is told why, and the
            # reply counts against the iteration budget like any other step.
            bad = agent_call_repair.unreadable_call(result.text or "", names, truncated=_truncated(result))
            if bad is not None:
                unreadable += 1
                used += 1
                record, told = _unreadable_turn(bad, result.text or "")
                all_tool_results.append(record)
                messages.extend(told)
                yield {"tool_call_start": {"id": record.tool_call_id, "name": record.tool_name,
                                           "arguments": json.dumps(record.arguments), "startedAt": record.started_at}}
                yield {"tool_call_result": {"id": record.tool_call_id, "name": record.tool_name,
                                            "arguments": record.arguments, "result": record.result,
                                            "elapsed": 0.0, "startedAt": record.started_at,
                                            "renderAs": None, "data": None}}
                continue
        if tool_calls:
            unreadable = 0

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
                        "startedAt": tr.started_at,
                    }
                    for tr in all_tool_results
                ],
                "iterations": iteration + 1,
                "prompt_tokens": last_prompt_tokens,
            }
            return

        # A reply that only waits on jobs already waited on this turn is free: a 25-minute render must not
        # end the chat with "maximum iterations". The first wait on each job still counts.
        keys: list[str | None] = []
        for tc in tool_calls:
            func = tc.get("function", {})
            tool = reg.get(func.get("name", ""))
            keys.append(tool.budget_key(_coerce_arguments(func.get("arguments"))) if tool is not None else None)
        if all(k is not None and k in seen_waits for k in keys) and free < _FREE_WAIT_LIMIT:
            free += 1
        else:
            used += 1
        seen_waits.update(k for k in keys if k is not None)

        # Execute tool calls
        # Every call gets its own id: a missing or repeated id would let one approval cover two calls.
        seen_ids: set[str] = set()
        for tc in tool_calls:
            if not tc.get("id") or tc["id"] in seen_ids:
                tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
            seen_ids.add(tc["id"])
        messages.append({
            "role": "assistant",
            "text": result.text or "",
            "tool_calls": tool_calls,
        })

        denials: dict[str, str] = {}
        if gate is not None:
            denials = yield from gate.review(tool_calls)

        for tc in tool_calls:
            func = tc.get("function", {})
            started_ms = int(time.time() * 1000)
            yield {
                "tool_call_start": {
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "arguments": func.get("arguments"),
                    "startedAt": started_ms,
                },
            }

            if tc["id"] in denials:
                tc_result = _declined_tool_call(tc, denials[tc["id"]])
            else:
                tc_result = yield from _run_tool_call_live(tc, reg)
            tc_result.started_at = started_ms
            all_tool_results.append(tc_result)

            yield {
                "tool_call_result": {
                    "id": tc_result.tool_call_id,
                    "name": tc_result.tool_name,
                    "arguments": tc_result.arguments,
                    "result": tc_result.result[:2000],  # Cap for streaming
                    "elapsed": tc_result.elapsed_seconds,
                    "startedAt": tc_result.started_at,
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
    yield {"done": True, "tool_calls": [], "iterations": max_iterations, "max_iterations_reached": True,
           "prompt_tokens": last_prompt_tokens}


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
