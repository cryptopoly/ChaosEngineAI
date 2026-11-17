"""FU-205: a reply with one call that reads and one that does not runs the good one and tells the model about the
other in the same tool-results turn, instead of dropping it silently."""

import json
import unittest
from types import SimpleNamespace

from backend_service.agent import read_calls, run_agent_loop, run_agent_loop_streaming
from backend_service.keep_going.nudge import TurnOutcome, unreadable
from backend_service.tools import BaseTool, ToolRegistry

NAMES = {"storyboard_audio", "make_music"}

GOOD = '<tool_call>{"name": "storyboard_audio", "arguments": {"kind": "music"}}</tool_call>'
# A bracket that does not match: never mended.
BROKEN = '<tool_call>{"name": "make_music", "arguments": {"tags": [1, 2}}</tool_call>'
# Only its last closing brace is missing: mended like a lone call.
SHORT = '<tool_call>{"name": "make_music", "arguments": {"seconds": 12}</tool_call>'
BAD_LFM = "<|tool_call_start|>[make_music(seconds=foo())]<|tool_call_end|>"
GOOD_LFM = '<|tool_call_start|>[storyboard_audio(kind="music")]<|tool_call_end|>'

BROKEN_MESSAGE = (
    "Error: your tool call could not be read, so nothing ran (the second call in your reply): "
    "a '}' does not match the bracket it closes; send it again as "
    '<tool_call>{"name": "make_music", "arguments": {...}}</tool_call>, with every brace closed.'
)


def _names(calls):
    return [c["function"]["name"] for c in calls]


class ReadCallsTests(unittest.TestCase):
    def test_good_then_broken(self):
        calls, bad = read_calls(f"{GOOD}\n{BROKEN}", NAMES)
        self.assertEqual(_names(calls), ["storyboard_audio"])
        self.assertEqual((bad.tool, bad.position), ("make_music", 2))
        self.assertEqual(bad.message, BROKEN_MESSAGE)

    def test_broken_then_good(self):
        calls, bad = read_calls(f"{BROKEN}\n{GOOD}", NAMES)
        self.assertEqual(_names(calls), ["storyboard_audio"])
        self.assertEqual(bad.position, 1)
        self.assertIn("(the first call in your reply): a '}' does not match", bad.message)

    def test_between_two_good_calls(self):
        calls, bad = read_calls(f"{GOOD}\n{BROKEN}\n{GOOD}", NAMES)
        self.assertEqual(len(calls), 2)
        self.assertEqual(bad.position, 2)

    def test_good_json_then_broken_lfm_block_and_the_reverse(self):
        calls, bad = read_calls(f"{GOOD}\n{BAD_LFM}", NAMES)
        self.assertEqual(_names(calls), ["storyboard_audio"])
        self.assertEqual((bad.style, bad.position), ("lfm", 2))
        self.assertIn("plain values", bad.reason)
        calls, bad = read_calls(f"{GOOD_LFM}\n{BROKEN}", NAMES)
        self.assertEqual(_names(calls), ["storyboard_audio"])
        self.assertEqual((bad.style, bad.position), ("json", 2))

    def test_a_whole_call_in_a_second_format_is_named_as_such(self):
        # LFM blocks are read first, so a whole JSON call beside one used to be dropped without a word.
        calls, bad = read_calls(f"{GOOD_LFM}\n{GOOD}", NAMES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(bad.reason, "the reply mixed two call formats and only the other one was read")

    def test_two_good_calls_unchanged(self):
        calls, bad = read_calls(f"{GOOD}\n{GOOD.replace('music', 'voice')}", NAMES)
        self.assertEqual(len(calls), 2)
        self.assertIsNone(bad)

    def test_one_broken_call_unchanged(self):
        # A lone unreadable call is reported by the loop as before (FU-196), with no position in its message.
        calls, bad = read_calls(BROKEN, NAMES)
        self.assertIsNone(calls)
        self.assertIsNone(bad)

    def test_a_second_call_short_of_its_brace_is_mended_in_its_place(self):
        calls, bad = read_calls(f"{SHORT}\n{GOOD}", NAMES)
        self.assertIsNone(bad)
        self.assertEqual(_names(calls), ["make_music", "storyboard_audio"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"seconds": 12})
        calls, bad = read_calls(f"{GOOD}\n{SHORT}", NAMES)
        self.assertEqual(_names(calls), ["storyboard_audio", "make_music"])
        self.assertIsNone(bad)

    def test_mending_keeps_its_rules(self):
        cases = {
            "cut at the token limit": (SHORT, True, "token limit"),
            "unknown tool": (SHORT.replace("make_music", "terminal"), False, "missing 1 closing brace"),
            "invalid after the closers": ('<tool_call>{"name": "make_music", "arguments": {"seconds": 12,</tool_call>',
                                          False, "missing"),
            "open string": ('<tool_call>{"name": "make_music", "arguments": {"caption": "calm</tool_call>',
                            False, "string is not closed"),
        }
        for label, (second, truncated, reason) in cases.items():
            with self.subTest(label):
                calls, bad = read_calls(f"{GOOD}\n{second}", NAMES, truncated)
                self.assertEqual(_names(calls), ["storyboard_audio"], label)
                self.assertIsNotNone(bad, label)
                self.assertEqual(bad.position, 2)
                self.assertIn(reason, bad.reason)

    def test_prose_and_code_blocks_beside_a_good_call_are_left_alone(self):
        for extra in [
            "Done. It uses {curly} braces and a dict {\"a\": 1}.",
            'Config: {"name": "Somnus", "arguments": "none"} as asked.',
            "The call looks like:\n```json\n{\"name\": \"make_music\", \"arguments\": {\"seconds\": 12}}\n```",
            "```python\nprint({'name': 'x'})\n```",
        ]:
            with self.subTest(extra):
                calls, bad = read_calls(f"{GOOD}\n{extra}", NAMES)
                self.assertEqual(len(calls), 1)
                self.assertIsNone(bad)


class _Tool(BaseTool):
    name = ""
    description = "A tool."

    def __init__(self, name):
        self.name = name
        self.calls = []

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return f"{self.name} ok"


def _gen(replies):
    seen = []

    def generate_fn(**kwargs):
        seen.append({**kwargs, "history": list(kwargs.get("history") or [])})
        text = replies[min(len(seen) - 1, len(replies) - 1)]
        return SimpleNamespace(text=text, finishReason="stop", tool_calls=None, promptTokens=5, completionTokens=5)

    generate_fn.seen = seen
    return generate_fn


class LoopTests(unittest.TestCase):
    def _run(self, plain, replies, **kw):
        registry = ToolRegistry()
        audio, music = _Tool("storyboard_audio"), _Tool("make_music")
        registry.register(audio)
        registry.register(music)
        gen = _gen(replies)
        args = dict(generate_fn=gen, prompt="go", history=[], system_prompt=None, max_tokens=100, temperature=0.7,
                    tool_registry=registry, **kw)
        if plain:
            out = run_agent_loop(**args)
            records = [{"name": r.tool_name, "result": r.result} for r in out.tool_calls]
        else:
            events = list(run_agent_loop_streaming(stream_generate_fn=None, **args))
            out = events
            records = events[-1]["tool_calls"]
            streamed = [e["tool_call_result"]["name"] for e in events if "tool_call_result" in e]
            self.assertEqual(streamed, [r["name"] for r in records])
        return out, records, gen, audio, music

    def test_good_and_broken_in_one_reply_in_both_loops(self):
        for plain in (True, False):
            for reply, order in ((f"{GOOD}\n{BROKEN}", ["storyboard_audio", "make_music"]),
                                 (f"{BROKEN}\n{GOOD}", ["make_music", "storyboard_audio"])):
                with self.subTest(plain=plain, order=order):
                    out, records, gen, audio, music = self._run(plain, [reply, "Attached."])
                    self.assertEqual(len(audio.calls), 1)
                    self.assertEqual(music.calls, [])
                    self.assertEqual([r["name"] for r in records], order)  # in the order the calls were written
                    told = gen.seen[1]["history"]
                    # One tool-results turn: the good call's result, then the notice about the other one.
                    self.assertEqual(told[-3]["role"], "assistant")
                    self.assertEqual(told[-3]["text"], reply)
                    self.assertEqual(_names(told[-3]["tool_calls"]), ["storyboard_audio"])
                    self.assertEqual((told[-2]["role"], told[-2]["text"]), ("tool", "storyboard_audio ok"))
                    self.assertEqual(told[-1]["role"], "user")
                    position = "second" if order[-1] == "make_music" else "first"
                    self.assertEqual(told[-1]["text"],
                                     f"<tool_response>\n{BROKEN_MESSAGE.replace('second', position)}\n</tool_response>")

    def test_two_good_calls_add_no_notice(self):
        for plain in (True, False):
            with self.subTest(plain=plain):
                _out, records, gen, audio, music = self._run(plain, [f"{GOOD}\n{SHORT}", "Done."])
                self.assertEqual((len(audio.calls), len(music.calls)), (1, 1))
                told = gen.seen[1]["history"]
                self.assertEqual([m["role"] for m in told[-3:]], ["assistant", "tool", "tool"])
                self.assertFalse(any("could not be read" in r["result"] for r in records))

    def test_one_broken_call_is_told_as_before(self):
        for plain in (True, False):
            with self.subTest(plain=plain):
                _out, _records, gen, audio, music = self._run(plain, [BROKEN, "ok"])
                told = gen.seen[1]["history"]
                self.assertEqual(told[-2], {"role": "assistant", "text": BROKEN})
                self.assertEqual(told[-1]["text"],
                                 f"<tool_response>\n{BROKEN_MESSAGE.replace(' (the second call in your reply)', '')}"
                                 "\n</tool_response>")

    def test_a_reply_with_a_good_call_resets_the_streak(self):
        for plain in (True, False):
            with self.subTest(plain=plain):
                # Without the reset the third reply would already end the turn.
                _out, _records, gen, audio, _m = self._run(plain, [BROKEN, f"{GOOD}\n{BROKEN}", BROKEN, BROKEN, BROKEN])
                self.assertEqual(len(gen.seen), 5)
                self.assertEqual(len(audio.calls), 1)

    def test_a_mixed_reply_is_told_even_after_two_broken_replies(self):
        for plain in (True, False):
            with self.subTest(plain=plain):
                _out, _records, gen, audio, _m = self._run(plain, [BROKEN, BROKEN, f"{GOOD}\n{BROKEN}", "ok"])
                self.assertEqual(len(gen.seen), 4)
                self.assertEqual(len(audio.calls), 1)
                self.assertIn("(the second call in your reply)", gen.seen[3]["history"][-1]["text"])

    def test_keep_going_sees_the_broken_call_when_the_turn_ends_on_it(self):
        _out, records, *_rest = self._run(True, [f"{GOOD}\n{BROKEN}", "All done."])
        outcome = TurnOutcome(session_id="s", assistant={"text": "All done.", "toolCalls": records})
        self.assertEqual(unreadable(outcome), ("make_music", "a '}' does not match the bracket it closes"))


if __name__ == "__main__":
    unittest.main()
