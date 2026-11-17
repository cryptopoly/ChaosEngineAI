"""LFM2.5 text-style tool calls in the agent loop (FU-192)."""

import json
import unittest
from types import SimpleNamespace

from backend_service.agent import (
    _parse_tool_calls_from_response,
    _strip_tool_call_xml,
    run_agent_loop,
    run_agent_loop_streaming,
)
from backend_service.tools import BaseTool, ToolRegistry
from tests.test_agent import _EchoTool

S, E = "<|tool_call_start|>", "<|tool_call_end|>"


def parsed(text):
    calls = _parse_tool_calls_from_response(text) or []
    return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]


class ParseTests(unittest.TestCase):
    def test_good_call(self):
        self.assertEqual(parsed(f'{S}[echo(text="hi", n=3)]{E}'), [("echo", {"text": "hi", "n": 3})])

    def test_two_calls_in_one_block_and_in_two_blocks(self):
        self.assertEqual(parsed(f'{S}[echo(text="a"), get_weather(city="Oslo")]{E}'),
                         [("echo", {"text": "a"}), ("get_weather", {"city": "Oslo"})])
        self.assertEqual(parsed(f'{S}[echo(text="a")]{E} and {S}[echo(text="b")]{E}'),
                         [("echo", {"text": "a"}), ("echo", {"text": "b"})])

    def test_no_arguments_and_a_bare_call_without_brackets(self):
        self.assertEqual(parsed(f"{S}[list_jobs()]{E}"), [("list_jobs", {})])
        self.assertEqual(parsed(f"{S}list_jobs(){E}"), [("list_jobs", {})])

    def test_nested_values(self):
        text = (f'{S}[make(title="a, \\"quoted\\" [title]", n=-2, ratio=1.5, ok=True, off=False, none=None, '
                f'tags=["x", "y]"], opts={{"size": [512, 512], "seed": None}}, pair=(1, 2))]{E}')
        self.assertEqual(parsed(text), [("make", {
            "title": 'a, "quoted" [title]', "n": -2, "ratio": 1.5, "ok": True, "off": False, "none": None,
            "tags": ["x", "y]"], "opts": {"size": [512, 512], "seed": None}, "pair": [1, 2]})])

    def test_json_words_the_template_writes_back(self):
        self.assertEqual(parsed(f'{S}[make(opts={{"a": true, "b": null}}, on=false)]{E}'),
                         [("make", {"opts": {"a": True, "b": None}, "on": False})])

    def test_markers_inside_a_string_value(self):
        self.assertEqual(parsed(f'{S}[echo(text="say {S} here")]{E}'), [])  # a reopened block is never trusted
        self.assertEqual(parsed(f'{S}[echo(text="ends with {E[:-2]}")]{E}'), [("echo", {"text": f"ends with {E[:-2]}"})])

    def test_bad_arguments_are_not_a_call(self):
        for body in ['[echo(text=foo)]', '[echo(text=__import__("os").system("x"))]', '[echo("positional")]',
                     '[echo(**{"text": "x"})]', '[echo(text="x"', '[os.system(cmd="x")]', '[1, 2]', '[]',
                     '[echo(s={1, 2})]', '[echo(c=1j)]']:
            with self.subTest(body=body):
                self.assertIsNone(_parse_tool_calls_from_response(f"{S}{body}{E}"))
                self.assertEqual(_strip_tool_call_xml(f"{S}{body}{E}"), f"{S}{body}{E}")  # stays as the reply

    def test_unclosed_marker_is_not_a_call(self):
        text = f'I will look. {S}[echo(text="hi")]'
        self.assertIsNone(_parse_tool_calls_from_response(text))
        self.assertEqual(_strip_tool_call_xml(text), text)

    def test_an_unclosed_block_before_a_good_one(self):
        self.assertEqual(parsed(f'{S}[echo(text="lost" {S}[echo(text="ok")]{E}'), [("echo", {"text": "ok"})])

    def test_text_before_and_after_is_kept_and_markers_go(self):
        text = f'Let me check.\n\n{S}[echo(text="hi")]{E}\n\n\n\nDone soon.'
        self.assertEqual(parsed(text), [("echo", {"text": "hi"})])
        self.assertEqual(_strip_tool_call_xml(text), "Let me check.\n\nDone soon.")


class _Recorder(BaseTool):
    name = "make"
    description = "Records its arguments."

    def __init__(self):
        self.seen = []

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        self.seen.append(kwargs)
        return "made"


def scripted(*texts):
    replies = list(texts)

    def generate_fn(**_kwargs):
        text = replies.pop(0) if len(replies) > 1 else replies[0]
        return SimpleNamespace(text=text, tool_calls=None, promptTokens=1, completionTokens=1)

    return generate_fn


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(_EchoTool())
        self.maker = _Recorder()
        self.registry.register(self.maker)

    def run_loop(self, *texts):
        return run_agent_loop(generate_fn=scripted(*texts), prompt="go", history=[], system_prompt=None,
                              max_tokens=64, temperature=0.0, tool_registry=self.registry, max_iterations=3)

    def test_loop_runs_the_calls_with_their_values(self):
        result = self.run_loop(f'{S}[echo(text="hi"), make(opts={{"n": [1, None]}}, ok=True)]{E}', "All done.")
        self.assertEqual([t.tool_name for t in result.tool_calls], ["echo", "make"])
        self.assertEqual(result.tool_calls[0].result, "echo: hi")
        self.assertEqual(self.maker.seen, [{"opts": {"n": [1, None]}, "ok": True}])
        self.assertEqual(result.text, "All done.")

    def test_a_bad_block_is_reported_then_ends_the_turn_with_its_text(self):
        # FU-196: the model is told twice that the block could not be read; the third reply ends the turn.
        bad = f"{S}[echo(text=nope)]{E}"
        result = self.run_loop(bad)
        self.assertEqual((result.text, result.iterations), (bad, 3))
        self.assertEqual([t.tool_name for t in result.tool_calls], ["echo", "echo"])
        self.assertTrue(all("could not be read" in t.result for t in result.tool_calls))

    def test_unknown_tool_is_a_clear_tool_error(self):
        result = self.run_loop(f'{S}[nope(x=1)]{E}', "ok")
        self.assertIn("unknown tool 'nope'", result.tool_calls[0].result)

    def test_streaming_loop_runs_the_call_and_no_marker_reaches_the_reply(self):
        events = list(run_agent_loop_streaming(
            generate_fn=scripted(f'Checking. {S}[echo(text="a, [b]")]{E}', f'It said "a, [b]". {S}[broken'),
            stream_generate_fn=None, prompt="go", history=[], system_prompt=None, max_tokens=64,
            temperature=0.0, tool_registry=self.registry, max_iterations=5))
        results = [e["tool_call_result"] for e in events if "tool_call_result" in e]
        self.assertEqual([(r["name"], r["arguments"], r["result"]) for r in results[:1]],
                         [("echo", {"text": "a, [b]"}, "echo: a, [b]")])
        # FU-196: the unclosed block is reported to the model twice, then the turn ends on it.
        self.assertEqual(len(results), 3)
        self.assertTrue(all("has no <|tool_call_end|>" in r["result"] for r in results[1:]))
        reply = "".join(e["token"] for e in events if "token" in e)
        self.assertEqual(reply, f'It said "a, [b]". {S}[broken')  # the unclosed tail stays as text
        self.assertNotIn(E, reply)
        self.assertTrue(events[-1]["done"])


if __name__ == "__main__":
    unittest.main()
