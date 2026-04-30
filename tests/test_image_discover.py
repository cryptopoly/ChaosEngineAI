"""Tests for Image Discover latest-model selection."""

from __future__ import annotations

import unittest

from backend_service.helpers.images import (
    _is_latest_image_candidate,
    _tracked_latest_seed_payloads,
)


class ImageDiscoverLatestTests(unittest.TestCase):
    def test_tracked_latest_fallback_includes_post_august_2025_models(self):
        payloads = _tracked_latest_seed_payloads([])
        repos = [str(item.get("repo") or "") for item in payloads]

        self.assertEqual(payloads[0].get("releaseDate"), "2026-04")
        self.assertIn("baidu/ERNIE-Image", repos)
        self.assertIn("black-forest-labs/FLUX.2-dev", repos)
        self.assertIn("Qwen/Qwen-Image-Edit-2511", repos)

    def test_latest_candidate_accepts_current_official_diffusers_repos(self):
        self.assertTrue(
            _is_latest_image_candidate(
                {
                    "id": "baidu/ERNIE-Image",
                    "tags": ["diffusers", "text-to-image"],
                    "pipeline_tag": "text-to-image",
                    "downloads": 4100,
                    "likes": 488,
                },
                curated_repos=set(),
            )
        )

    def test_latest_candidate_filters_low_signal_loras(self):
        self.assertFalse(
            _is_latest_image_candidate(
                {
                    "id": "random-user/current-model-lora",
                    "tags": ["diffusers", "text-to-image", "lora"],
                    "pipeline_tag": "text-to-image",
                    "downloads": 50_000,
                    "likes": 500,
                },
                curated_repos=set(),
            )
        )


if __name__ == "__main__":
    unittest.main()
