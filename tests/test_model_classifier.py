"""Tests for ``backend_service.helpers.model_classifier``.

Pin the keyword lists against the tracked-latest seeds in
``backend_service/catalog/image_models.py`` so newly-tracked image families
don't silently leak into the chat-oriented My Models list.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from backend_service.helpers.model_classifier import (
    _looks_like_image_model,
    _looks_like_video_model,
)


class ImageClassifierTests(unittest.TestCase):
    def test_flux_variants_are_image(self):
        for name in (
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.2-dev",
            "black-forest-labs/FLUX.2-klein-9B",
        ):
            self.assertTrue(
                _looks_like_image_model(Path("/tmp/none"), name),
                msg=f"{name} should be classified as image",
            )

    def test_tracked_latest_seeds_are_image(self):
        for name in (
            "baidu/ERNIE-Image",
            "baidu/ERNIE-Image-Turbo",
            "NucleusAI/Nucleus-Image",
            "Tongyi-MAI/Z-Image",
            "Tongyi-MAI/Z-Image-Turbo",
            "HiDream-ai/HiDream-I1-Full",
            "zai-org/GLM-Image",
        ):
            self.assertTrue(
                _looks_like_image_model(Path("/tmp/none"), name),
                msg=f"{name} should be classified as image (tracked-latest seed)",
            )

    def test_text_llms_not_classified_as_image(self):
        for name in (
            "Qwen/Qwen3-Coder-Next",
            "mlx-community/Qwen3-0.6B-4bit",
            "google/gemma-4-9b-it",
            "MiniMaxAI/MiniMax-M2.5",
        ):
            self.assertFalse(
                _looks_like_image_model(Path("/tmp/none"), name),
                msg=f"{name} should NOT be classified as image",
            )


class VideoClassifierTests(unittest.TestCase):
    def test_known_video_families(self):
        for name in (
            "Lightricks/LTX-Video",
            "prince-canuma/LTX-2-distilled",
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            "tencent/HunyuanVideo",
            "genmoai/Mochi-1-preview",
        ):
            self.assertTrue(
                _looks_like_video_model(name),
                msg=f"{name} should be classified as video",
            )

    def test_qwen_image_not_video(self):
        self.assertFalse(_looks_like_video_model("Qwen/Qwen-Image"))


if __name__ == "__main__":
    unittest.main()
