import unittest

from backend_service.mlx_worker import (
    TranscriptLoopFilter,
    _build_prompt_text,
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


if __name__ == "__main__":
    unittest.main()
