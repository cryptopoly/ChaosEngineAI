"""FU-196: tool calls the model wrote slightly wrong are mended or reported back, never dropped silently.

The fixture holds the exact replies Qwen3.8-27B ended its turns with in a Keep going rehearsal: a
``storyboard_audio`` call without its ``<tool_call>`` opener and one closing brace short. Nothing ran and the model
was never told, so it repeated the same call seven times.
"""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend_service.agent import (
    _parse_tool_calls_from_response,
    _strip_tool_call_xml,
    run_agent_loop,
    run_agent_loop_streaming,
)
from backend_service.agent_call_repair import repair_call, unreadable_call
from backend_service.tools import BaseTool, ToolRegistry
from backend_service.tools.studio import STUDIO_TOOL_NAMES

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "keep_going_qwen_unclosed_calls.json").read_text("utf-8"))
REPLIES = FIXTURE["replies"]
NAMES = set(STUDIO_TOOL_NAMES)


class _AudioTool(BaseTool):
    name = "storyboard_audio"
    description = "Attach audio."

    def __init__(self):
        self.calls = []

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return '{"projectId": "p1", "revision": 21}'


class RepairTests(unittest.TestCase):
    def test_every_real_failing_reply_is_read_with_its_values_unchanged(self):
        self.assertEqual(len(REPLIES), 9)
        for text in REPLIES:
            self.assertIsNone(_parse_tool_calls_from_response(text))  # the parser before FU-196
            calls = _parse_tool_calls_from_response(text, NAMES)
            self.assertEqual(len(calls), 1, text)
            self.assertEqual(calls[0]["function"]["name"], "storyboard_audio")
            args = json.loads(calls[0]["function"]["arguments"])
            self.assertEqual((args["action"], args["kind"]), ("add", "music"))
            self.assertEqual(args["fields"], {"start": 0, "fadeOut": 3, "label": "Background score"})
            # Exactly the text the model wrote, plus the one brace it left off.
            body = text[text.index("{"):text.rindex("</tool_call>")].rstrip()
            self.assertEqual(json.loads(body + "}")["arguments"], args)

    def test_with_an_opener_and_no_close_tag(self):
        calls = _parse_tool_calls_from_response('<tool_call>{"name": "make_music", "arguments": {"seconds": 12}', NAMES)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"seconds": 12})

    def test_refused_repairs(self):
        cases = {
            "unknown tool": '{"name": "terminal", "arguments": {"command": "ls"}\n</tool_call>',
            "arguments not an object": '{"name": "make_music", "arguments": "loud"\n</tool_call>',
            "arguments a list": '{"name": "make_music", "arguments": [1, 2\n</tool_call>',
            "invalid after the closers": '{"name": "make_music", "arguments": {"seconds": 12,\n</tool_call>',
            "a key with no value": '{"name": "make_music", "arguments": {"seconds":\n</tool_call>',
            "mismatched bracket": '<tool_call>{"name": "make_music", "arguments": {"a": [1, 2}}</tool_call>',
            "open string": '{"name": "make_music", "arguments": {"caption": "calm\n</tool_call>',
            "no tag at all": 'I would send {"name": "make_music", "arguments": {"seconds": 12}',
            "two calls": '{"name": "make_music", "arguments": {}} {"name": "speak", "arguments": {"text": "x"}\n</tool_call>',
        }
        for label, text in cases.items():
            with self.subTest(label):
                self.assertIsNone(repair_call(text, NAMES), label)
                self.assertIsNone(_parse_tool_calls_from_response(text, NAMES), label)

    def test_a_reply_cut_at_the_token_limit_is_not_mended(self):
        self.assertIsNone(_parse_tool_calls_from_response(REPLIES[0], NAMES, truncated=True))
        bad = unreadable_call(REPLIES[0], NAMES, truncated=True)
        self.assertIn("token limit", bad.reason)

    def test_without_tool_names_nothing_is_mended(self):
        self.assertIsNone(_parse_tool_calls_from_response(REPLIES[0], None))


class UnreadableTests(unittest.TestCase):
    def test_the_real_reply_gets_a_specific_reason_and_its_format(self):
        bad = unreadable_call(REPLIES[1], NAMES)
        self.assertEqual((bad.tool, bad.reason), ("storyboard_audio", "the JSON is missing 1 closing brace"))
        self.assertTrue(bad.message.startswith("Error: your tool call could not be read, so nothing ran: "))
        self.assertIn('send it again as <tool_call>{"name": "storyboard_audio"', bad.message)

    def test_fragments_that_are_reported(self):
        cases = {
            '<tool_call>{"name": "make_music", "arguments": {"a": [1, 2}}</tool_call>': ("make_music", "does not match"),
            '<tool_call>{"name": "make_music", "arguments": "x"} oops</tool_call>': ("make_music", "must be a JSON object"),
            'Sending it: {"name": "make_music", "arguments": {"seconds": 12}}': ("make_music", "not wrapped in <tool_call>"),
            "<tool_call>\nlet me think</tool_call>": (None, "no call object"),
            "<|tool_call_start|>[make_music(seconds=12)": ("make_music", "has no <|tool_call_end|>"),
            "<|tool_call_start|>[make_music(seconds=foo())]<|tool_call_end|>": ("make_music", "plain values"),
            "<tool_call><function=make_music><parameter=seconds>12</parameter>": ("make_music", "</function>"),
        }
        for text, (tool, reason) in cases.items():
            with self.subTest(text):
                bad = unreadable_call(text, NAMES)
                self.assertIsNotNone(bad)
                self.assertEqual(bad.tool, tool)
                self.assertIn(reason, bad.reason)
        self.assertIn("<|tool_call_start|>[make_music(", unreadable_call("<|tool_call_start|>[make_music(", NAMES).message)

    def test_prose_and_code_blocks_are_left_alone(self):
        for text in [
            "The film is ready. It uses {curly} braces and a dict {\"a\": 1}.",
            'Config: {"name": "Somnus", "arguments": "none"} as asked.',  # not a tool name
            "Here is the call you asked to see:\n```json\n{\"name\": \"make_music\", \"arguments\": {\"seconds\": 12}}\n```",
            "The format looks like this:\n```\n<tool_call>{\"name\": \"some_tool\", \"arguments\": {}}</tool_call>\n```",
            "```python\nprint({'name': 'x'})\n```",
        ]:
            with self.subTest(text):
                self.assertIsNone(unreadable_call(text, NAMES))

    def test_a_broken_known_call_in_a_code_block_is_reported(self):
        text = "```json\n{\"name\": \"make_music\", \"arguments\": {\"seconds\": 12}\n```"
        self.assertIsNotNone(unreadable_call(text, NAMES))

    def test_strip_keeps_an_unreadable_fragment_visible(self):
        text = '<|tool_call_start|>[make_music(seconds=12)'
        self.assertEqual(_strip_tool_call_xml(text), text)


def _gen(replies):
    """A generate_fn answering from ``replies`` in order (the last one repeats)."""
    seen = []

    def generate_fn(**kwargs):
        seen.append({**kwargs, "history": list(kwargs.get("history") or [])})
        text = replies[min(len(seen) - 1, len(replies) - 1)]
        return SimpleNamespace(text=text, finishReason="stop", tool_calls=None, promptTokens=5, completionTokens=5)

    generate_fn.seen = seen
    return generate_fn


def _registry():
    registry = ToolRegistry()
    tool = _AudioTool()
    registry.register(tool)
    return registry, tool


class LoopTests(unittest.TestCase):
    BROKEN = '<tool_call>{"name": "storyboard_audio", "arguments": {"action": "add", "kind": "music"} junk</tool_call>'
    FIXED = '<tool_call>{"name": "storyboard_audio", "arguments": {"action": "add", "kind": "music"}}</tool_call>'

    def run_plain(self, replies, **kw):
        registry, tool = _registry()
        gen = _gen(replies)
        result = run_agent_loop(generate_fn=gen, prompt="go", history=[], system_prompt=None, max_tokens=100,
                                temperature=0.7, tool_registry=registry, **kw)
        return result, tool, gen

    def run_stream(self, replies, **kw):
        registry, tool = _registry()
        gen = _gen(replies)
        events = list(run_agent_loop_streaming(generate_fn=gen, stream_generate_fn=None, prompt="go", history=[],
                                               system_prompt=None, max_tokens=100, temperature=0.7,
                                               tool_registry=registry, **kw))
        return events, tool, gen

    def test_the_real_reply_runs_in_both_loops(self):
        result, tool, _gen_fn = self.run_plain([REPLIES[1], "Attached."])
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(tool.calls[0]["kind"], "music")
        self.assertEqual(result.text, "Attached.")
        events, tool, _gen_fn = self.run_stream([REPLIES[1], "Attached."])
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(events[-1]["tool_calls"][0]["name"], "storyboard_audio")

    def test_the_model_is_told_and_can_fix_its_call(self):
        for plain, run in ((True, self.run_plain), (False, self.run_stream)):
            with self.subTest(run.__name__):
                out, tool, gen = run([self.BROKEN, self.FIXED, "Attached."])
                self.assertEqual(len(tool.calls), 1)
                told = gen.seen[1]["history"]
                self.assertEqual(told[-2], {"role": "assistant", "text": self.BROKEN})
                self.assertEqual(told[-1]["role"], "user")
                self.assertIn("your tool call could not be read, so nothing ran", told[-1]["text"])
                self.assertTrue(told[-1]["text"].startswith("<tool_response>"))
                records = out.tool_calls if plain else out[-1]["tool_calls"]
                names = [r.tool_name if plain else r["name"] for r in records]
                self.assertEqual(names, ["storyboard_audio", "storyboard_audio"])

    def test_two_error_replies_in_a_row_then_the_turn_ends(self):
        out, tool, gen = self.run_plain([self.BROKEN])
        self.assertEqual(len(gen.seen), 3)  # two told, the third ends the turn
        self.assertEqual(tool.calls, [])
        self.assertIn("junk", out.text)  # the fragment stays visible in the final text
        self.assertEqual(len(out.tool_calls), 2)
        events, tool, gen = self.run_stream([self.BROKEN])
        self.assertEqual(len(gen.seen), 3)
        self.assertEqual(sum(1 for e in events if "tool_call_result" in e), 2)
        self.assertTrue(events[-1]["done"])
        self.assertNotIn("max_iterations_reached", events[-1])

    def test_error_replies_count_against_the_iteration_budget(self):
        out, tool, gen = self.run_plain([self.BROKEN], max_iterations=2)
        self.assertEqual(len(gen.seen), 2)
        events, tool, gen = self.run_stream([self.BROKEN], max_iterations=2)
        self.assertEqual(len(gen.seen), 2)
        self.assertTrue(events[-1].get("max_iterations_reached"))

    def test_the_streak_starts_over_after_a_call_that_reads(self):
        out, tool, gen = self.run_plain([self.BROKEN, self.BROKEN, self.FIXED, self.BROKEN, self.BROKEN, "ok"])
        self.assertEqual(len(tool.calls), 1)
        self.assertEqual(out.text, "ok")

    def test_ordinary_prose_ends_the_turn_at_once(self):
        out, tool, gen = self.run_plain(['The {"name": "Somnus"} film is finished.'])
        self.assertEqual(len(gen.seen), 1)
        self.assertEqual(out.tool_calls, [])


def _structured(name, arguments, call_id):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class CutOffStructuredCallTests(unittest.TestCase):
    """2026-11-24: llama-server handed back a ``storyboard_shot`` call cut off at the end of the 32,768-token context
    with its arguments as partial JSON. The loop ran it, then sent the broken arguments back in the next request,
    which llama-server refused ("Failed to parse tool call arguments as JSON") and the whole turn failed."""

    CUT = '{"action": "edit", "shot_id": "s_1", "fields": {"stillPrompt": "Close-up of the singer\'s hands, 16:9'

    def gen(self):
        replies = [
            SimpleNamespace(text="", finishReason="length", promptTokens=32602, completionTokens=166,
                            tool_calls=[_structured("storyboard_audio", '{"action": "add"}', "c1"),
                                        _structured("storyboard_audio", self.CUT, "c2")]),
            SimpleNamespace(text="Done.", finishReason="stop", promptTokens=5, completionTokens=5, tool_calls=None),
        ]
        seen = []

        def generate_fn(**kwargs):
            seen.append(list(kwargs.get("history") or []))
            return replies[min(len(seen) - 1, len(replies) - 1)]

        generate_fn.seen = seen
        return generate_fn

    def check(self, gen, tool, records):
        self.assertEqual(tool.calls, [{"action": "add"}])  # the whole call ran, the cut-off one did not
        history = gen.seen[1]
        sent = [c for m in history for c in (m.get("tool_calls") or [])]
        self.assertEqual([c["id"] for c in sent], ["c1"])  # the broken arguments never go back to the server
        self.assertIn("cut off inside the call", history[-1]["text"])
        self.assertIn("the second call in your reply", history[-1]["text"])
        self.assertEqual(len(records), 2)

    def test_both_loops_skip_and_report_the_cut_off_call(self):
        registry, tool = _registry()
        gen = self.gen()
        out = run_agent_loop(generate_fn=gen, prompt="go", history=[], system_prompt=None, max_tokens=100,
                             temperature=0.7, tool_registry=registry)
        self.check(gen, tool, out.tool_calls)
        registry, tool = _registry()
        gen = self.gen()
        events = list(run_agent_loop_streaming(generate_fn=gen, stream_generate_fn=None, prompt="go", history=[],
                                               system_prompt=None, max_tokens=100, temperature=0.7,
                                               tool_registry=registry))
        self.check(gen, tool, events[-1]["tool_calls"])

    def test_a_lone_cut_off_call_is_an_unreadable_turn(self):
        registry, tool = _registry()
        replies = [SimpleNamespace(text="", finishReason="length", promptTokens=9, completionTokens=9,
                                   tool_calls=[_structured("storyboard_audio", self.CUT, "c9")]),
                   SimpleNamespace(text="ok", finishReason="stop", promptTokens=5, completionTokens=5, tool_calls=None)]
        seen = []

        def generate_fn(**kwargs):
            seen.append(list(kwargs.get("history") or []))
            return replies[min(len(seen) - 1, 1)]

        out = run_agent_loop(generate_fn=generate_fn, prompt="go", history=[], system_prompt=None, max_tokens=100,
                             temperature=0.7, tool_registry=registry)
        self.assertEqual(tool.calls, [])
        self.assertEqual(out.text, "ok")
        self.assertFalse(any(m.get("tool_calls") for m in seen[1]))
        self.assertIn("your tool call could not be read, so nothing ran: the reply was cut off", seen[1][-1]["text"])


if __name__ == "__main__":
    unittest.main()
