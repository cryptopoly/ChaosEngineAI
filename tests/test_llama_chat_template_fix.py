"""Phase 3.8 follow-up tests for the llama.cpp chat-template fix.

The Gemma family rejects the system role outright when llama-server
applies its embedded chat template. We fold the system message into
the first user message client-side so the template never sees a
system role and the request goes through cleanly.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from backend_service.inference import _apply_llama_chat_template_fixes


@dataclass
class _FakeLoaded:
    ref: str
    canonicalRepo: str | None = None


class LlamaChatTemplateFixTests(unittest.TestCase):
    def test_no_op_for_non_gemma(self):
        loaded = _FakeLoaded(ref="Qwen/Qwen3-8B")
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        out, note = _apply_llama_chat_template_fixes(messages, loaded)
        self.assertEqual(out, messages)
        self.assertIsNone(note)

    def test_folds_system_for_gemma_canonical_repo(self):
        loaded = _FakeLoaded(ref="local/path", canonicalRepo="google/gemma-4-26B-A4B-it")
        messages = [
            {"role": "system", "content": "Be polite."},
            {"role": "user", "content": "Hi"},
        ]
        out, note = _apply_llama_chat_template_fixes(messages, loaded)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")
        self.assertIn("Be polite.", out[0]["content"])
        self.assertIn("Hi", out[0]["content"])
        self.assertIsNotNone(note)
        self.assertIn("Gemma", note)

    def test_folds_system_for_community_gemma_ref(self):
        loaded = _FakeLoaded(ref="lmstudio-community/gemma-3-12b-it")
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Why?"},
        ]
        out, note = _apply_llama_chat_template_fixes(messages, loaded)
        # System folded into the first user; subsequent turns intact.
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["role"], "user")
        self.assertIn("Be helpful.", out[0]["content"])
        self.assertIn("What's 2+2?", out[0]["content"])
        self.assertEqual(out[1]["role"], "assistant")
        self.assertEqual(out[2]["content"], "Why?")
        self.assertIsNotNone(note)

    def test_no_note_when_no_system_message(self):
        # Gemma but no system message → fold is a no-op, so no note.
        loaded = _FakeLoaded(ref="google/gemma-4-26B-A4B-it")
        messages = [{"role": "user", "content": "Hi"}]
        out, note = _apply_llama_chat_template_fixes(messages, loaded)
        self.assertEqual(out, messages)
        self.assertIsNone(note)

    def test_handles_empty_messages(self):
        loaded = _FakeLoaded(ref="google/gemma-4-26B-A4B-it")
        out, note = _apply_llama_chat_template_fixes([], loaded)
        self.assertEqual(out, [])
        self.assertIsNone(note)

    def test_handles_missing_loaded_model(self):
        messages = [{"role": "user", "content": "Hi"}]
        out, note = _apply_llama_chat_template_fixes(messages, None)
        self.assertEqual(out, messages)
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
