"""Phase 3.8 tests for chat_template helpers."""

from __future__ import annotations

import unittest

from backend_service.helpers.chat_template import (
    ChatTemplateReport,
    fold_system_into_first_user,
    inspect_chat_template,
    is_gemma_family,
    is_multimodal_family,
)


class IsGemmaFamilyTests(unittest.TestCase):
    def test_recognises_canonical_gemma_repo(self):
        self.assertTrue(is_gemma_family("google/gemma-4-E4B-it"))
        self.assertTrue(is_gemma_family("google/gemma-2-9b"))

    def test_recognises_community_gemma_repos(self):
        self.assertTrue(is_gemma_family("mlx-community/gemma-3-9b-it-8bit"))
        self.assertTrue(is_gemma_family("lmstudio-community/gemma-3-12b-it"))

    def test_case_insensitive(self):
        self.assertTrue(is_gemma_family("GOOGLE/GEMMA-4-7B"))

    def test_rejects_non_gemma(self):
        self.assertFalse(is_gemma_family("Qwen/Qwen3-7B"))
        self.assertFalse(is_gemma_family("meta-llama/Llama-3-8B"))
        self.assertFalse(is_gemma_family(None))
        self.assertFalse(is_gemma_family(""))


class IsMultimodalFamilyTests(unittest.TestCase):
    """Bug 1: vision-capable repo prefix detection. Drives the
    mlx_lm → mlx_vlm load-path swap in mlx_worker."""

    def test_recognises_gemma_4_canonical(self):
        self.assertTrue(is_multimodal_family("google/gemma-4-E4B-it"))
        self.assertTrue(is_multimodal_family("google/gemma-4-12B-it"))
        self.assertTrue(is_multimodal_family("google/gemma-4-26B-A4B-it"))

    def test_recognises_gemma_4_community(self):
        self.assertTrue(is_multimodal_family("mlx-community/gemma-4-26b-a4b-it-5bit"))
        self.assertTrue(is_multimodal_family("lmstudio-community/gemma-4-12B-it"))

    def test_recognises_qwen_vl_family(self):
        self.assertTrue(is_multimodal_family("Qwen/Qwen2.5-VL-7B-Instruct"))
        self.assertTrue(is_multimodal_family("mlx-community/Qwen2.5-VL-72B-Instruct-4bit"))
        self.assertTrue(is_multimodal_family("Qwen/Qwen3-VL-8B"))

    def test_recognises_llava_family(self):
        self.assertTrue(is_multimodal_family("mlx-community/llava-1.5-7b-mlx"))
        self.assertTrue(is_multimodal_family("llava-hf/llava-1.5-7b-hf"))

    def test_rejects_text_only_gemma(self):
        # Earlier Gemma generations are text-only.
        self.assertFalse(is_multimodal_family("google/gemma-2-9b"))
        self.assertFalse(is_multimodal_family("google/gemma-3-12b-it"))
        self.assertFalse(is_multimodal_family("mlx-community/gemma-3-9b-it-8bit"))

    def test_rejects_text_only_qwen(self):
        self.assertFalse(is_multimodal_family("Qwen/Qwen3-7B"))
        self.assertFalse(is_multimodal_family("Qwen/Qwen2.5-7B-Instruct"))

    def test_rejects_other_text_models(self):
        self.assertFalse(is_multimodal_family("meta-llama/Llama-3-8B"))
        self.assertFalse(is_multimodal_family("deepseek-ai/DeepSeek-R1-Distill-Llama-8B"))
        self.assertFalse(is_multimodal_family(None))
        self.assertFalse(is_multimodal_family(""))

    def test_case_insensitive(self):
        self.assertTrue(is_multimodal_family("GOOGLE/GEMMA-4-12B-IT"))
        self.assertTrue(is_multimodal_family("Mlx-Community/Gemma-4-26B"))


class FoldSystemIntoFirstUserTests(unittest.TestCase):
    def test_folds_system_into_first_user(self):
        out = fold_system_into_first_user([
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "What's 2+2?"},
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")
        self.assertIn("Be concise.", out[0]["content"])
        self.assertIn("What's 2+2?", out[0]["content"])

    def test_preserves_assistant_turns_after_fold(self):
        out = fold_system_into_first_user([
            {"role": "system", "content": "Be polite."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ])
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["role"], "user")
        self.assertIn("Be polite.", out[0]["content"])
        self.assertEqual(out[1]["role"], "assistant")
        self.assertEqual(out[2]["content"], "How are you?")

    def test_idempotent_when_no_system_message(self):
        original = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        out = fold_system_into_first_user(original)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["content"], "Hi")

    def test_system_with_no_following_user_promotes_to_user(self):
        out = fold_system_into_first_user([
            {"role": "system", "content": "Be helpful."},
        ])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")
        self.assertEqual(out[0]["content"], "Be helpful.")


class InspectChatTemplateTests(unittest.TestCase):
    def test_missing_template_flagged(self):
        report = inspect_chat_template(None, "any/model")
        self.assertFalse(report.template_present)
        self.assertTrue(report.needs_attention)
        self.assertIn("no chat_template found", report.issues[0])

    def test_empty_template_flagged(self):
        report = inspect_chat_template("   ", "any/model")
        self.assertFalse(report.template_present)

    def test_gemma_family_records_system_role_fix(self):
        # Even with a healthy template, Gemma family triggers the fold
        # auto-fix — the runtime applies it transparently.
        report = inspect_chat_template(
            "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            "google/gemma-4-E4B-it",
        )
        self.assertFalse(report.accepts_system_role)
        self.assertTrue(any("Gemma" in fix for fix in report.fixes_applied))

    def test_chatml_without_generation_prompt_flagged(self):
        # ChatML template with no add_generation_prompt branch.
        template = "<|im_start|>system\n{{system}}<|im_end|><|im_start|>user\n{{user}}<|im_end|>"
        report = inspect_chat_template(template, "Qwen/Qwen3-7B")
        self.assertFalse(report.accepts_generation_prompt)
        self.assertTrue(any("add_generation_prompt" in issue for issue in report.issues))

    def test_chatml_with_generation_prompt_clean(self):
        template = (
            "<|im_start|>user\n{{user}}<|im_end|>"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        report = inspect_chat_template(template, "Qwen/Qwen3-7B")
        self.assertTrue(report.accepts_generation_prompt)

    def test_to_runtime_note_returns_none_for_clean_template(self):
        template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        report = inspect_chat_template(template, "Qwen/Qwen3-7B")
        self.assertIsNone(report.to_runtime_note())

    def test_to_runtime_note_summarises_fixes_and_issues(self):
        report = ChatTemplateReport()
        report.fixes_applied.append("test fix")
        report.issues.append("test issue")
        note = report.to_runtime_note()
        self.assertIsNotNone(note)
        self.assertIn("auto-fixed", note)
        self.assertIn("issues", note)


if __name__ == "__main__":
    unittest.main()
