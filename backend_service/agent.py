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


@dataclass
class AgentResult:
    """Final result of an agent loop run."""
    text: str
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    iterations: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0


def _parse_tool_calls_from_response(response_text: str) -> list[dict[str, Any]] | None:
    """Attempt to extract tool calls from a text response.

    Models using the OpenAI tool-calling protocol return structured
    tool_calls in the response object. For models that embed tool calls
    in their text output (e.g., Hermes/Functionary format), we try to
    parse them from common patterns.
    """
    # Try the <tool_call> XML-ish format (Hermes/NousResearch)
    calls: list[dict[str, Any]] = []
    import re

    for match in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        response_text,
        re.DOTALL,
    ):
        try:
            payload = json.loads(match.group(1))
            name = payload.get("name") or payload.get("function")
            arguments = payload.get("arguments") or payload.get("parameters") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if name:
                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
                    },
                })
        except (json.JSONDecodeError, KeyError):
            continue

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

    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
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
    try:
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
            # Model is done — return the final text
            return AgentResult(
                text=result.text,
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
            # Since we already have the full text, emit it in chunks
            text = result.text
            chunk_size = 4
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
