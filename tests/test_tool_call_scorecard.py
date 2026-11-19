"""``scripts/tool-call-scorecard.py`` (FU-230): the prompts' answers and the scoring. No server, no model."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tool-call-scorecard.py"
    spec = importlib.util.spec_from_file_location("tool_call_scorecard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["tool_call_scorecard"] = module
    spec.loader.exec_module(module)
    return module


sc = _load_script()

# One right reply per prompt, written as a careful model would.
RIGHT = {
    "kind-video": [("list_models", {"kind": "video"})],
    "kind-voice": [("list_models", {"kind": "voice"})],
    "image-prompt": [("generate_image", {"prompt": "a red fox asleep in fresh snow, soft morning light"})],
    "video-size": [("generate_video", {"prompt": "waves on black rocks", "width": 768, "height": 512, "seconds": 3})],
    "speak-speed": [("speak", {"text": "Welcome to the flower shop", "style": "calm", "speed": 0.85})],
    "status": [("system_status", {})],
    "jobs-failed": [("list_jobs", {"state": "failed"})],
    "project-create": [("storyboard_project", {"action": "create", "name": "Harbour Lights", "aspect": "16:9",
                                               "resolution": "1080p", "fps": 24})],
    "shot-fields": [("storyboard_shot", {"action": "add", "project_id": "p-123", "revision": 4,
                                         "fields": {"stillPrompt": "a lighthouse at dusk", "seconds": 4}})],
    "music-flags": [("make_music", {"caption": "lo-fi beat", "instrumental": True, "seconds": 30, "bpm": 80})],
    "two-calls": [("generate_image", {"prompt": "a lighthouse at night"}), ("speak", {"text": "Safe harbour"})],
    "wait-job": [("wait_for_job", {"job_id": "job-42"})],
}


def message(calls, content=None):
    return {"role": "assistant", "content": content, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)]}


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools, cls.schemas, cls.hint = sc.studio_tools()

    def test_twelve_prompts_each_with_a_right_answer_that_scores_full(self):
        self.assertEqual(len(sc.PROMPTS), 12)
        self.assertEqual({p.id for p in sc.PROMPTS}, set(RIGHT))
        for prompt in sc.PROMPTS:
            score = sc.score_reply(prompt, message(RIGHT[prompt.id]), self.schemas)
            self.assertTrue(score.called and score.right_tool and score.valid and score.right_values,
                            (prompt.id, score))

    def test_every_expected_tool_is_one_chat_offers(self):
        offered = {t["function"]["name"] for t in self.tools}
        for prompt in sc.PROMPTS:
            for want in prompt.expect:
                self.assertTrue(set(want.tools) <= offered, (prompt.id, want.tools))

    def test_a_value_outside_the_schema_is_invalid(self):
        prompt = next(p for p in sc.PROMPTS if p.id == "kind-video")
        score = sc.score_reply(prompt, message([("list_models", {"kind": "videos"})]), self.schemas)
        self.assertTrue(score.right_tool)
        self.assertFalse(score.valid)
        self.assertFalse(score.right_values)
        self.assertIn("must be one of", score.problems[0])

    def test_valid_but_wrong_values_and_the_wrong_tool(self):
        size = next(p for p in sc.PROMPTS if p.id == "video-size")
        wrong_size = sc.score_reply(size, message([("generate_video", {"prompt": "waves", "width": 1280, "height": 720})]),
                                    self.schemas)
        self.assertTrue(wrong_size.valid)
        self.assertFalse(wrong_size.right_values)
        wrong_tool = sc.score_reply(size, message([("generate_image", {"prompt": "waves"})]), self.schemas)
        self.assertFalse(wrong_tool.right_tool)

    def test_a_look_up_before_an_action_is_counted_as_such(self):
        # The Studio hint tells the model to check the installed models first.
        size = next(p for p in sc.PROMPTS if p.id == "video-size")
        first = sc.score_reply(size, message([("system_status", {}), ("list_models", {"kind": "video"})]),
                               self.schemas)
        self.assertFalse(first.right_tool)
        self.assertTrue(first.looked_up)
        wrong = sc.score_reply(size, message([("generate_image", {"prompt": "waves"})]), self.schemas)
        self.assertFalse(wrong.looked_up)
        # Where a look-up is the answer, a different look-up is just the wrong tool.
        jobs = next(p for p in sc.PROMPTS if p.id == "jobs-failed")
        self.assertFalse(sc.score_reply(jobs, message([("system_status", {})]), self.schemas).looked_up)

    def test_an_extra_look_up_beside_the_right_call_is_fine_an_extra_action_is_not(self):
        kind = next(p for p in sc.PROMPTS if p.id == "kind-video")
        both = sc.score_reply(kind, message([("system_status", {}), ("list_models", {"kind": "video"})]), self.schemas)
        self.assertTrue(both.right_values)
        image = next(p for p in sc.PROMPTS if p.id == "image-prompt")
        extra = sc.score_reply(image, message([("generate_image", {"prompt": "a red fox"}),
                                               ("speak", {"text": "a red fox"})]), self.schemas)
        self.assertFalse(extra.right_tool)

    def test_two_calls_need_both(self):
        prompt = next(p for p in sc.PROMPTS if p.id == "two-calls")
        half = sc.score_reply(prompt, message([("generate_image", {"prompt": "a lighthouse at night"})]), self.schemas)
        self.assertFalse(half.right_tool)

    def test_a_call_left_in_the_text_counts_as_unreadable(self):
        prompt = sc.PROMPTS[0]
        text = '<tool_call>{"name": "list_models", "arguments": {"kind": "video"}'
        score = sc.score_reply(prompt, {"role": "assistant", "content": text}, self.schemas)
        self.assertFalse(score.called)
        self.assertTrue(score.unreadable)
        plain = sc.score_reply(prompt, {"role": "assistant", "content": "You have two video models."}, self.schemas)
        self.assertFalse(plain.unreadable)

    def test_summary_reports_counts(self):
        rows = [{"model": "m", "prompt": "kind-video", "tests": "enum", "called": 1, "right_tool": 1, "valid": 1,
                 "right_values": v, "looked_up": 0, "unreadable": 0, "seconds": 1.0, "prompt_tokens": 900,
                 "problems": ""} for v in (1, 0)]
        table = sc.summary(rows)
        self.assertIn("| `m` | 2 | 2 of 2 | 2 of 2 | 2 of 2 | 1 of 2 | 0 of 2 | 0 of 2 |", table)


if __name__ == "__main__":
    unittest.main()
