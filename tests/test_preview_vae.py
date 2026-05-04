"""Tests for FU-018 TAESD / TAEHV preview VAE swap helper."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend_service.helpers.preview_vae import (
    maybe_apply_preview_vae,
    resolve_preview_vae_id,
)


class ResolvePreviewVaeIdTests(unittest.TestCase):
    def test_flux1_dev_maps_to_taef1(self):
        self.assertEqual(
            resolve_preview_vae_id("black-forest-labs/FLUX.1-dev"),
            "madebyollin/taef1",
        )

    def test_flux1_schnell_maps_to_taef1(self):
        self.assertEqual(
            resolve_preview_vae_id("black-forest-labs/FLUX.1-schnell"),
            "madebyollin/taef1",
        )

    def test_flux2_klein_4b_maps_to_taef2(self):
        self.assertEqual(
            resolve_preview_vae_id("black-forest-labs/FLUX.2-klein-4B"),
            "madebyollin/taef2",
        )

    def test_flux2_klein_9b_maps_to_taef2(self):
        # Longest-prefix-wins: FLUX.2 must beat FLUX.1 even though both
        # share the black-forest-labs/FLUX prefix.
        self.assertEqual(
            resolve_preview_vae_id("black-forest-labs/FLUX.2-klein-9B"),
            "madebyollin/taef2",
        )

    def test_sdxl_maps_to_taesdxl(self):
        self.assertEqual(
            resolve_preview_vae_id("stabilityai/stable-diffusion-xl-base-1.0"),
            "madebyollin/taesdxl",
        )

    def test_sd3_maps_to_taesd3(self):
        self.assertEqual(
            resolve_preview_vae_id("stabilityai/stable-diffusion-3.5-large"),
            "madebyollin/taesd3",
        )

    def test_wan22_maps_to_taew2_2(self):
        self.assertEqual(
            resolve_preview_vae_id("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            "madebyollin/taew2_2",
        )

    def test_wan21_maps_to_taew2_2(self):
        self.assertEqual(
            resolve_preview_vae_id("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            "madebyollin/taew2_2",
        )

    def test_ltx_video_maps_to_taeltx2_3_wide(self):
        self.assertEqual(
            resolve_preview_vae_id("Lightricks/LTX-Video"),
            "madebyollin/taeltx2_3_wide",
        )

    def test_ltx_2_maps_to_taeltx2_3_wide(self):
        self.assertEqual(
            resolve_preview_vae_id("prince-canuma/LTX-2-distilled"),
            "madebyollin/taeltx2_3_wide",
        )

    def test_hunyuan_maps_to_taehv1_5(self):
        self.assertEqual(
            resolve_preview_vae_id("hunyuanvideo-community/HunyuanVideo"),
            "madebyollin/taehv1_5",
        )

    def test_cogvideox_maps_to_taecogvideox(self):
        self.assertEqual(
            resolve_preview_vae_id("THUDM/CogVideoX-5b"),
            "madebyollin/taecogvideox",
        )

    def test_mochi_maps_to_taemochi(self):
        self.assertEqual(
            resolve_preview_vae_id("genmo/mochi-1-preview"),
            "madebyollin/taemochi",
        )

    def test_qwen_image_maps_to_taeqwenimage(self):
        self.assertEqual(
            resolve_preview_vae_id("Qwen/Qwen-Image"),
            "madebyollin/taeqwenimage",
        )

    def test_qwen_image_2512_maps_to_taeqwenimage(self):
        self.assertEqual(
            resolve_preview_vae_id("Qwen/Qwen-Image-2512"),
            "madebyollin/taeqwenimage",
        )

    def test_unmapped_repo_returns_none(self):
        self.assertIsNone(
            resolve_preview_vae_id("some-org/UnknownModel"),
        )


class MaybeApplyPreviewVaeTests(unittest.TestCase):
    def test_disabled_is_noop(self):
        pipeline = SimpleNamespace(vae=object())
        original_vae = pipeline.vae
        note = maybe_apply_preview_vae(
            pipeline,
            repo="black-forest-labs/FLUX.1-dev",
            enabled=False,
        )
        self.assertIsNone(note)
        self.assertIs(pipeline.vae, original_vae)

    def test_unmapped_repo_is_noop(self):
        pipeline = SimpleNamespace(vae=object())
        original_vae = pipeline.vae
        note = maybe_apply_preview_vae(
            pipeline,
            repo="some-org/UnknownModel",
            enabled=True,
        )
        self.assertIsNone(note)
        self.assertIs(pipeline.vae, original_vae)

    def test_pipeline_without_vae_returns_skip_note(self):
        pipeline = SimpleNamespace()  # no .vae
        note = maybe_apply_preview_vae(
            pipeline,
            repo="black-forest-labs/FLUX.1-dev",
            enabled=True,
        )
        self.assertIsNotNone(note)
        self.assertIn("vae", note.lower())

    def test_swap_failure_falls_back_to_stock(self):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            self.skipTest("diffusers not available")

        original_vae = SimpleNamespace(dtype="fp16")
        pipeline = SimpleNamespace(vae=original_vae)

        with patch("diffusers.AutoencoderTiny") as mock_cls:
            mock_cls.from_pretrained.side_effect = Exception("not cached")
            note = maybe_apply_preview_vae(
                pipeline,
                repo="black-forest-labs/FLUX.1-dev",
                enabled=True,
            )

        self.assertIsNotNone(note)
        self.assertIn("madebyollin/taef1", note)
        self.assertIn("download failed", note)
        # On failure, the stock VAE stays in place.
        self.assertIs(pipeline.vae, original_vae)

    def test_local_load_succeeds_swaps_vae(self):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            self.skipTest("diffusers not available")

        original_vae = SimpleNamespace(dtype="fp16")
        pipeline = SimpleNamespace(vae=original_vae)
        sentinel = SimpleNamespace(name="fake-preview-vae")

        with patch("diffusers.AutoencoderTiny") as mock_cls:
            mock_cls.from_pretrained.return_value = sentinel
            note = maybe_apply_preview_vae(
                pipeline,
                repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                enabled=True,
            )

        self.assertIsNotNone(note)
        self.assertIn("madebyollin/taew2_2", note)
        self.assertIs(pipeline.vae, sentinel)
        # First call should be the local-cache attempt.
        first_call = mock_cls.from_pretrained.call_args_list[0]
        self.assertEqual(first_call.args, ("madebyollin/taew2_2",))
        self.assertTrue(first_call.kwargs.get("local_files_only"))

    def test_remote_fallback_succeeds_when_local_misses(self):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            self.skipTest("diffusers not available")

        original_vae = SimpleNamespace(dtype="fp16")
        pipeline = SimpleNamespace(vae=original_vae)
        sentinel = SimpleNamespace(name="fake-preview-vae-remote")

        with patch("diffusers.AutoencoderTiny") as mock_cls:
            mock_cls.from_pretrained.side_effect = [
                Exception("local cache miss"),
                sentinel,
            ]
            note = maybe_apply_preview_vae(
                pipeline,
                repo="Lightricks/LTX-Video",
                enabled=True,
            )

        self.assertIsNotNone(note)
        self.assertIn("madebyollin/taeltx2_3_wide", note)
        self.assertIs(pipeline.vae, sentinel)
        self.assertEqual(mock_cls.from_pretrained.call_count, 2)


if __name__ == "__main__":
    unittest.main()
