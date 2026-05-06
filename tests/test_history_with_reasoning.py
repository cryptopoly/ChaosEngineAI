"""Tests for `_build_history_with_reasoning`.

The history builder projects stored chat messages into the list passed to
the inference layer. When the active thread is in "auto" thinking mode,
prior assistant reasoning traces are re-emitted inside `<think>...</think>`
tags so reasoning-capable models can pick up the chain across turns.
"""

import unittest

from backend_service.state import _build_history_with_reasoning


class BuildHistoryWithReasoningTests(unittest.TestCase):
    def test_omits_reasoning_when_preserve_is_false(self):
        messages = [
            {"role": "user", "text": "What is 2+2?"},
            {"role": "assistant", "text": "Four.", "reasoning": "2 plus 2 equals 4."},
        ]
        history = _build_history_with_reasoning(messages, preserve_reasoning=False)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "text": "What is 2+2?"})
        self.assertEqual(history[1], {"role": "assistant", "text": "Four."})
        self.assertNotIn("<think>", history[1]["text"])

    def test_prepends_think_tags_when_preserve_is_true(self):
        messages = [
            {"role": "user", "text": "Solve this."},
            {"role": "assistant", "text": "Done.", "reasoning": "Step one. Step two."},
        ]
        history = _build_history_with_reasoning(messages, preserve_reasoning=True)
        self.assertIn("<think>", history[1]["text"])
        self.assertIn("</think>", history[1]["text"])
        self.assertIn("Step one. Step two.", history[1]["text"])
        self.assertTrue(history[1]["text"].endswith("Done."))

    def test_skips_assistant_messages_without_reasoning(self):
        messages = [
            {"role": "assistant", "text": "Plain answer."},
            {"role": "assistant", "text": "Another.", "reasoning": ""},
        ]
        history = _build_history_with_reasoning(messages, preserve_reasoning=True)
        self.assertEqual(history[0]["text"], "Plain answer.")
        self.assertEqual(history[1]["text"], "Another.")
        self.assertNotIn("<think>", history[0]["text"])
        self.assertNotIn("<think>", history[1]["text"])

    def test_does_not_inject_reasoning_into_user_messages(self):
        messages = [
            {"role": "user", "text": "Hi.", "reasoning": "This shouldnt happen but be safe."},
        ]
        history = _build_history_with_reasoning(messages, preserve_reasoning=True)
        self.assertEqual(history[0]["text"], "Hi.")

    def test_preserves_message_order(self):
        messages = [
            {"role": "user", "text": "Q1"},
            {"role": "assistant", "text": "A1", "reasoning": "R1"},
            {"role": "user", "text": "Q2"},
            {"role": "assistant", "text": "A2", "reasoning": "R2"},
        ]
        history = _build_history_with_reasoning(messages, preserve_reasoning=True)
        self.assertEqual([h["role"] for h in history], ["user", "assistant", "user", "assistant"])
        self.assertIn("R1", history[1]["text"])
        self.assertIn("R2", history[3]["text"])


if __name__ == "__main__":
    unittest.main()
