import unittest
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend_service.agent import (
    _parse_tool_calls_from_response,
    _execute_tool_call,
    run_agent_loop,
    AgentResult,
    ToolCallResult,
)
from backend_service.tools import BaseTool, ToolRegistry


class _EchoTool(BaseTool):
    """Minimal tool for testing the agent loop."""

    name = "echo"
    description = "Echoes back its input."

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def execute(self, **kwargs):
        return f"echo: {kwargs.get('text', '')}"


class ParseToolCallsTests(unittest.TestCase):
    def test_parses_single_tool_call(self):
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        calls = _parse_tool_calls_from_response(text)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "calculator")
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["expression"], "2+2")

    def test_parses_multiple_tool_calls(self):
        text = (
            'Let me use two tools.\n'
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>\n'
            '<tool_call>{"name": "web_search", "arguments": {"query": "hello"}}</tool_call>'
        )
        calls = _parse_tool_calls_from_response(text)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["function"]["name"], "calculator")
        self.assertEqual(calls[1]["function"]["name"], "web_search")

    def test_returns_none_for_no_tool_calls(self):
        result = _parse_tool_calls_from_response("Just a normal text response.")
        self.assertIsNone(result)

    def test_parses_function_key_variant(self):
        text = '<tool_call>{"function": "echo", "parameters": {"text": "hi"}}</tool_call>'
        calls = _parse_tool_calls_from_response(text)
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "echo")

    def test_skips_malformed_json(self):
        text = '<tool_call>{not valid json}</tool_call>'
        calls = _parse_tool_calls_from_response(text)
        self.assertIsNone(calls)

    def test_parses_string_arguments(self):
        text = '<tool_call>{"name": "calc", "arguments": "{\\"x\\": 1}"}</tool_call>'
        calls = _parse_tool_calls_from_response(text)
        self.assertIsNotNone(calls)
        self.assertEqual(len(calls), 1)


class ExecuteToolCallTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(_EchoTool())

    def test_executes_known_tool(self):
        tc = {
            "id": "call_abc",
            "function": {
                "name": "echo",
                "arguments": json.dumps({"text": "hello"}),
            },
        }
        result = _execute_tool_call(tc, self.registry)
        self.assertIsInstance(result, ToolCallResult)
        self.assertEqual(result.tool_name, "echo")
        self.assertIn("hello", result.result)
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_returns_error_for_unknown_tool(self):
        tc = {
            "id": "call_xyz",
            "function": {"name": "nonexistent", "arguments": "{}"},
        }
        result = _execute_tool_call(tc, self.registry)
        self.assertIn("Error", result.result)
        self.assertIn("nonexistent", result.result)

    def test_handles_invalid_json_arguments(self):
        tc = {
            "id": "call_bad",
            "function": {"name": "echo", "arguments": "not json"},
        }
        result = _execute_tool_call(tc, self.registry)
        self.assertEqual(result.tool_name, "echo")
        # Should still produce a result (the raw args get wrapped)
        self.assertIsNotNone(result.result)


class RunAgentLoopTests(unittest.TestCase):
    def _make_generate_fn(self, responses):
        """Create a mock generate_fn that returns pre-defined responses in order."""
        call_count = [0]

        def generate_fn(**kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            resp = responses[idx]
            return SimpleNamespace(
                text=resp.get("text", ""),
                finishReason=resp.get("finishReason", "stop"),
                tool_calls=resp.get("tool_calls"),
                promptTokens=resp.get("promptTokens", 10),
                completionTokens=resp.get("completionTokens", 5),
            )

        return generate_fn

    def test_simple_generation_without_tools(self):
        gen = self._make_generate_fn([{"text": "Hello there!"}])
        result = run_agent_loop(
            generate_fn=gen,
            prompt="Hi",
            history=[],
            system_prompt=None,
            max_tokens=100,
            temperature=0.7,
            available_tools=[],  # empty tools list
        )
        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.text, "Hello there!")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls, [])

    def test_tool_call_then_final_response(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())

        responses = [
            {
                "text": '<tool_call>{"name": "echo", "arguments": {"text": "world"}}</tool_call>',
                "tool_calls": None,
            },
            {
                "text": "The echo returned: world",
                "tool_calls": None,
            },
        ]
        gen = self._make_generate_fn(responses)
        result = run_agent_loop(
            generate_fn=gen,
            prompt="Echo world",
            history=[],
            system_prompt=None,
            max_tokens=100,
            temperature=0.7,
            tool_registry=registry,
        )
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].tool_name, "echo")
        self.assertIn("world", result.text)

    def test_max_iterations_limit(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())

        # Every response triggers a tool call, so loop should hit max_iterations
        always_tool = {
            "text": '<tool_call>{"name": "echo", "arguments": {"text": "again"}}</tool_call>',
            "tool_calls": None,
        }
        gen = self._make_generate_fn([always_tool] * 20)
        result = run_agent_loop(
            generate_fn=gen,
            prompt="Loop forever",
            history=[],
            system_prompt=None,
            max_tokens=100,
            temperature=0.7,
            tool_registry=registry,
            max_iterations=3,
        )
        self.assertEqual(result.iterations, 3)
        self.assertIn("maximum", result.text.lower())
        self.assertEqual(len(result.tool_calls), 3)

    def test_empty_tools_list_skips_tool_use(self):
        gen = self._make_generate_fn([{"text": "No tools available."}])
        result = run_agent_loop(
            generate_fn=gen,
            prompt="Use a tool",
            history=[],
            system_prompt=None,
            max_tokens=100,
            temperature=0.7,
            available_tools=[],
        )
        self.assertEqual(result.text, "No tools available.")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_calls, [])

    def test_token_counts_accumulate(self):
        registry = ToolRegistry()
        registry.register(_EchoTool())

        responses = [
            {
                "text": '<tool_call>{"name": "echo", "arguments": {"text": "x"}}</tool_call>',
                "promptTokens": 20,
                "completionTokens": 15,
            },
            {
                "text": "Done",
                "promptTokens": 30,
                "completionTokens": 10,
            },
        ]
        gen = self._make_generate_fn(responses)
        result = run_agent_loop(
            generate_fn=gen,
            prompt="Test tokens",
            history=[],
            system_prompt=None,
            max_tokens=100,
            temperature=0.7,
            tool_registry=registry,
        )
        self.assertEqual(result.total_prompt_tokens, 50)
        self.assertEqual(result.total_completion_tokens, 25)


if __name__ == "__main__":
    unittest.main()
