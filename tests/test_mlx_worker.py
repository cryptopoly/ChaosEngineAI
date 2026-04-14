import unittest

from backend_service.mlx_worker import (
    RunawayGuard,
    TranscriptLoopFilter,
    _build_prompt_text,
    _strip_thinking_tokens,
    _trim_transcript_continuation,
)


class _MissingTemplateTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template is not set "
            "and no template argument was passed!"
        )


class _WorkingTemplateTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return "<templated>"


class MlxWorkerPromptTests(unittest.TestCase):
    def test_uses_chat_template_when_available(self):
        prompt, note = _build_prompt_text(
            _WorkingTemplateTokenizer(),
            history=[],
            prompt="Hello",
            system_prompt="Be concise.",
        )

        self.assertEqual(prompt, "<templated>")
        self.assertIsNone(note)

    def test_falls_back_when_chat_template_is_missing(self):
        prompt, note = _build_prompt_text(
            _MissingTemplateTokenizer(),
            history=[{"role": "assistant", "text": "Earlier reply"}],
            prompt="Hello",
            system_prompt="Be concise.",
        )

        self.assertIn("SYSTEM: Be concise.", prompt)
        self.assertIn("ASSISTANT: Earlier reply", prompt)
        self.assertIn("USER: Hello", prompt)
        self.assertTrue(prompt.endswith("ASSISTANT:"))
        self.assertIsNotNone(note)
        self.assertIn("chat template was unavailable", note.lower())

    def test_trim_transcript_continuation_keeps_first_answer(self):
        trimmed, stopped = _trim_transcript_continuation(
            "Hello!\n\nUSER: Can you check the repo?\n\nASSISTANT: Hello!"
        )

        self.assertTrue(stopped)
        self.assertEqual(trimmed, "Hello!")

    def test_trim_transcript_continuation_strips_leading_assistant_label(self):
        trimmed, stopped = _trim_transcript_continuation(
            "ASSISTANT: Hello!\n\nUSER: Can you check the repo?"
        )

        self.assertTrue(stopped)
        self.assertEqual(trimmed, "Hello!")

    def test_transcript_loop_filter_handles_split_stream_chunks(self):
        filter_ = TranscriptLoopFilter()

        first = filter_.feed("Hello!\n\nUSE")
        second = filter_.feed("R: Can you check the repo?\n\nASSISTANT: Hello!")
        flushed = filter_.flush()

        self.assertEqual(first, "Hello!\n")
        self.assertEqual(second, "")
        self.assertEqual(flushed, "")
        self.assertTrue(filter_.stopped)


class RunawayGuardTests(unittest.TestCase):
    def test_allows_normal_output(self):
        guard = RunawayGuard()
        guard.feed("Hello! How can I help you today?\n")
        guard.feed("I'm an AI assistant.\n")
        guard.flush()  # Should not raise

    def test_detects_repeated_lines(self):
        guard = RunawayGuard(min_line_length=20, max_repeats=3)
        with self.assertRaises(RuntimeError) as ctx:
            for _ in range(5):
                guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        self.assertIn("repeating itself", str(ctx.exception))

    def test_ignores_short_repeated_lines(self):
        guard = RunawayGuard(min_line_length=30, max_repeats=3)
        for _ in range(10):
            guard.feed("OK.\n")
        guard.flush()  # Should not raise — lines are too short

    def test_resets_on_different_line(self):
        guard = RunawayGuard(min_line_length=20, max_repeats=3)
        guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        guard.feed("Actually, here is a completely different line of text.\n")
        guard.feed("Wait, I will write 'Qwen3.5'. Let me try again.\n")
        guard.flush()  # Should not raise — count reset on different line

    def test_detects_thinking_heading(self):
        guard = RunawayGuard()
        guard.feed("Thinking Process:\n")
        self.assertTrue(guard.saw_thinking_heading)

    def test_no_thinking_heading_in_normal_text(self):
        guard = RunawayGuard()
        guard.feed("Hello! I can help you with that.\n")
        self.assertFalse(guard.saw_thinking_heading)


class StripThinkingTokensTests(unittest.TestCase):
    def test_strips_xml_think_tags(self):
        text = "<think>internal reasoning here</think>Hello!"
        self.assertEqual(_strip_thinking_tokens(text), "Hello!")

    def test_strips_unclosed_think_tag(self):
        text = "<think>reasoning that never closes"
        self.assertEqual(_strip_thinking_tokens(text), "")

    def test_strips_raw_thinking_heading(self):
        text = "Thinking Process:\nStep 1: analyze\nStep 2: decide\nFinal Check:\nHello!"
        result = _strip_thinking_tokens(text)
        self.assertEqual(result, "Hello!")

    def test_preserves_normal_text(self):
        text = "Hello! I'm an AI assistant."
        self.assertEqual(_strip_thinking_tokens(text), text)


if __name__ == "__main__":
    unittest.main()
