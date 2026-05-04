"""Unit tests for the LLM-based prompt enhancer (FU-022).

Exercises:
* ``family_for`` mapping table — repo prefix → family id with longer
  prefix winning over shorter generic ones.
* ``enhance_prompt`` happy path returns the LLM rewrite + a note that
  cites the model + family + word delta.
* Disabled flag short-circuits without touching the singleton.
* Empty prompts return empty + no note.
* Singleton fallback path: when ``ensure_loaded`` returns
  ``(False, reason)`` the helper returns the original prompt + the
  reason as the note.
* Generation crash is caught and surfaces as a runtimeNote rather
  than a raised exception.
* Shorter-than-input rewrite is rejected — the helper falls back to
  the original to avoid clobbering the user's intent.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from backend_service.helpers.prompt_enhancer import (
    EnhancementResult,
    enhance_prompt,
    family_for,
    reset_singleton_for_test,
)


class FamilyForTests(unittest.TestCase):
    def test_wan_repo_maps_to_wan_family(self):
        self.assertEqual(family_for("Wan-AI/Wan2.1-T2V-1.3B"), "wan")
        self.assertEqual(family_for("Wan-AI/Wan2.2-TI2V-5B-Diffusers"), "wan")

    def test_wan_quantstack_mirror_also_wan(self):
        self.assertEqual(family_for("QuantStack/Wan2.2-TI2V-5B-GGUF"), "wan")

    def test_ltx_video_maps_to_ltx(self):
        self.assertEqual(family_for("Lightricks/LTX-Video"), "ltx")
        self.assertEqual(family_for("prince-canuma/LTX-2-distilled"), "ltx")

    def test_hunyuan_maps_to_hunyuan(self):
        self.assertEqual(family_for("hunyuanvideo-community/HunyuanVideo"), "hunyuan")
        self.assertEqual(family_for("tencent/HunyuanVideo"), "hunyuan")

    def test_flux_family(self):
        self.assertEqual(family_for("black-forest-labs/FLUX.1-dev"), "flux")
        self.assertEqual(family_for("black-forest-labs/FLUX.2-klein-4B"), "flux")

    def test_sd3_family_specific_before_xl(self):
        # SD3 prefix is more specific than the SDXL prefix, so it must
        # win even if the table grew SDXL entries.
        self.assertEqual(family_for("stabilityai/stable-diffusion-3.5-large"), "sd3")

    def test_sdxl_turbo_recognised_as_sdxl(self):
        self.assertEqual(family_for("stabilityai/sdxl-turbo"), "sdxl")
        self.assertEqual(family_for("ByteDance/SDXL-Lightning"), "sdxl")

    def test_unknown_repo_falls_back_to_default(self):
        self.assertEqual(family_for("foo/bar"), "default")
        self.assertEqual(family_for(""), "default")


class EnhancePromptTests(unittest.TestCase):
    def setUp(self) -> None:
        # Drop any cached model from a previous test so the
        # ensure_loaded mock has a clean slate to assert against.
        reset_singleton_for_test()

    def test_disabled_returns_original_with_no_note(self):
        result = enhance_prompt(
            "a fluffy cat",
            repo="black-forest-labs/FLUX.1-dev",
            enabled=False,
        )
        self.assertEqual(result.enhanced, "a fluffy cat")
        self.assertIsNone(result.note)
        self.assertIsNone(result.modelUsed)
        self.assertEqual(result.family, "flux")

    def test_empty_prompt_returns_empty(self):
        result = enhance_prompt(
            "   ",
            repo="black-forest-labs/FLUX.1-dev",
            enabled=True,
        )
        self.assertEqual(result.enhanced, "")
        self.assertIsNone(result.note)
        self.assertIsNone(result.modelUsed)

    def test_singleton_load_failure_returns_original_with_note(self):
        with patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.ensure_loaded"
        ) as mock_load:
            mock_load.return_value = (False, "mlx_lm not installed.")
            result = enhance_prompt(
                "a fluffy cat",
                repo="black-forest-labs/FLUX.1-dev",
                enabled=True,
            )
        self.assertEqual(result.enhanced, "a fluffy cat")
        self.assertEqual(result.note, "mlx_lm not installed.")
        self.assertIsNone(result.modelUsed)

    def test_happy_path_returns_rewritten_with_note(self):
        with patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.ensure_loaded"
        ) as mock_load, patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.generate"
        ) as mock_gen:
            mock_load.return_value = (True, None)
            mock_gen.return_value = (
                "A fluffy orange tabby cat lounging on a sunlit windowsill, "
                "shallow depth of field, golden hour lighting, photorealistic "
                "style, sharp fur details."
            )
            result = enhance_prompt(
                "a fluffy cat",
                repo="black-forest-labs/FLUX.1-dev",
                enabled=True,
                model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            )
        self.assertNotEqual(result.enhanced, "a fluffy cat")
        self.assertIn("fluffy", result.enhanced.lower())
        self.assertIsNotNone(result.note)
        self.assertIn("flux", result.note.lower())
        self.assertEqual(result.modelUsed, "mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        self.assertEqual(result.family, "flux")

    def test_generation_crash_returns_original_with_note(self):
        with patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.ensure_loaded"
        ) as mock_load, patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.generate"
        ) as mock_gen:
            mock_load.return_value = (True, None)
            mock_gen.side_effect = RuntimeError("CUDA OOM")
            result = enhance_prompt(
                "a fluffy cat",
                repo="black-forest-labs/FLUX.1-dev",
                enabled=True,
            )
        self.assertEqual(result.enhanced, "a fluffy cat")
        self.assertIn("crashed", (result.note or "").lower())
        self.assertIn("CUDA OOM", result.note or "")

    def test_shorter_than_input_rewrite_is_rejected(self):
        # Some 0.5B models occasionally produce a single-word completion
        # ("Cat.") instead of a real rewrite. The helper detects this
        # by word-count and falls back to the original prompt rather
        # than clobbering the user's intent with garbage output.
        with patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.ensure_loaded"
        ) as mock_load, patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.generate"
        ) as mock_gen:
            mock_load.return_value = (True, None)
            mock_gen.return_value = "Cat."
            result = enhance_prompt(
                "a fluffy cat sitting on a windowsill",
                repo="black-forest-labs/FLUX.1-dev",
                enabled=True,
            )
        self.assertEqual(result.enhanced, "a fluffy cat sitting on a windowsill")
        self.assertIn("shorter", (result.note or "").lower())

    def test_rewrite_strips_quotes_and_trailing_whitespace(self):
        # Some 0.5B chat models wrap their output in quotation marks.
        # Strip a single layer of leading/trailing quotes so the user
        # doesn't see them in the textarea.
        with patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.ensure_loaded"
        ) as mock_load, patch(
            "backend_service.helpers.prompt_enhancer._SINGLETON.generate"
        ) as mock_gen:
            mock_load.return_value = (True, None)
            mock_gen.return_value = (
                '  "An orange tabby cat lounging on a sunny windowsill in '
                "golden afternoon light, photorealistic, shallow depth of field, "
                'warm tones."  '
            )
            result = enhance_prompt(
                "a fluffy cat",
                repo="black-forest-labs/FLUX.1-dev",
                enabled=True,
            )
        self.assertFalse(result.enhanced.startswith('"'))
        self.assertFalse(result.enhanced.endswith('"'))
        self.assertTrue(result.enhanced.startswith("An orange tabby"))


class EnhancementResultTests(unittest.TestCase):
    def test_result_dataclass_is_frozen(self):
        result = EnhancementResult(
            enhanced="x", note=None, modelUsed=None, family="flux",
        )
        with self.assertRaises(Exception):
            result.enhanced = "y"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
