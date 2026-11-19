"""FU-231: the Storyboard tools reach the model only for a turn about a film."""

from __future__ import annotations

import unittest

from backend_service.tools.studio import studio_tools, wants_storyboard


class StoryboardStagingTests(unittest.TestCase):
    def test_a_plain_request_does_not_want_them(self) -> None:
        self.assertFalse(wants_storyboard("Draw me a red fox in the snow", []))
        self.assertFalse(wants_storyboard("What models do I have?", [{"role": "user", "text": "hi"}]))

    def test_film_words_in_the_message_or_the_recent_chat_do(self) -> None:
        self.assertTrue(wants_storyboard("Make a short film about a harbour", []))
        self.assertTrue(wants_storyboard("yes go on", [{"role": "user", "text": "Plan the trailer"}]))
        self.assertFalse(wants_storyboard("a filmy look", []))  # a whole word, not a part of one

    def test_an_earlier_storyboard_call_or_a_keep_going_run_do(self) -> None:
        calls = [{"role": "assistant", "text": "ok", "toolCalls": [{"name": "storyboard_shot"}]}]
        old = [{"role": "user", "text": "x"}] * 8
        self.assertTrue(wants_storyboard("next", calls + old))
        self.assertTrue(wants_storyboard("next", [], keep_going=True))

    def test_the_tool_list_drops_only_the_storyboard_tools(self) -> None:
        every = {t.name for t in studio_tools(None)}
        core = {t.name for t in studio_tools(None, storyboard=False)}
        self.assertTrue(every - core)
        self.assertTrue(all(n.startswith("storyboard_") for n in every - core))
        self.assertIn("generate_image", core)
        self.assertEqual(len(every - core), 7)


if __name__ == "__main__":
    unittest.main()
