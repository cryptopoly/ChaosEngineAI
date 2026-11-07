"""D-48: Wan 2.2 image-to-video (start + end frame) wiring.

Covers the catalog row, the shared "who takes an image" rule, the pipeline
kwargs (``image`` / ``last_image``), the two-expert GGUF load, the route
checks, the download patterns and the agent-bridge listing. No weights load.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

from backend_service.helpers.video_image_input import (
    WAN_I2V_REPOS,
    accepts_image_input,
    requires_start_image,
)
from backend_service.models import VideoGenerationRequest
from backend_service.video_runtime import DiffusersVideoEngine
from backend_service.video_runtime.pipeline_helpers import load_frame_image
from backend_service.video_runtime.types import VideoGenerationConfig

ROW_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers-distill-gguf-q4km"
BASE = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def _png(path: Path, size=(1024, 576), colour="red") -> Path:
    from PIL import Image

    Image.new("RGB", size, colour).save(path)
    return path


def _row() -> dict:
    from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES

    for family in VIDEO_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == ROW_ID:
                return variant
    raise AssertionError(f"{ROW_ID} missing from the video catalog")


class CatalogRowTests(unittest.TestCase):
    def test_row_pins_both_gguf_experts_and_the_distill_schedule(self):
        row = _row()
        self.assertEqual(row["repo"], BASE)
        self.assertTrue(row["ggufFile"].startswith("high_noise/"))
        self.assertTrue(row["ggufFile2"].startswith("low_noise/"))
        self.assertEqual(row["defaultSteps"], 4)
        self.assertEqual(row["cfgOverride"], 1.0)
        self.assertEqual(row["taskSupport"], ["img2video"])


class ImageInputRuleTests(unittest.TestCase):
    def test_wan_i2v_takes_and_needs_a_start_image(self):
        row = _row()
        self.assertTrue(accepts_image_input(row))
        self.assertTrue(requires_start_image(row))

    def test_ltx_takes_but_does_not_need_one(self):
        row = {"repo": "dgrauet/ltx-2.5-mlx-q4"}
        self.assertTrue(accepts_image_input(row))
        self.assertFalse(requires_start_image(row))

    def test_text_to_video_wan_takes_none(self):
        row = {"repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}
        self.assertFalse(accepts_image_input(row))
        self.assertFalse(requires_start_image(row))

    def test_rule_matches_the_runtime_registry(self):
        from backend_service.video_runtime.repos import PIPELINE_REGISTRY

        self.assertEqual(
            {r for r, e in PIPELINE_REGISTRY.items() if e["task"] == "img2video"}, set(WAN_I2V_REPOS)
        )


class LoadFrameImageTests(unittest.TestCase):
    def test_wide_image_is_centre_cropped_to_the_clip_shape(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.png"
            image = Image.new("RGB", (400, 100), "black")
            image.paste(Image.new("RGB", (200, 100), "white"), (100, 0))
            image.save(path)
            out = load_frame_image(str(path), 64, 32)
        self.assertEqual(out.size, (64, 32))
        # The 200x100 centre (all white) survives; the black sides are cropped away.
        self.assertEqual(out.getpixel((0, 16)), (255, 255, 255))
        self.assertEqual(out.getpixel((63, 16)), (255, 255, 255))


class PipelineKwargsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.start = _png(Path(self.tmp.name) / "s.png")
        self.end = _png(Path(self.tmp.name) / "e.png", colour="blue")
        self.engine = DiffusersVideoEngine()
        self.engine._pipeline = type("WanImageToVideoPipeline", (), {})()

    def config(self, **extra):
        return VideoGenerationConfig(
            modelId=ROW_ID, modelName="Wan I2V", repo=BASE, prompt="p", negativePrompt="",
            width=768, height=448, numFrames=33, fps=16, steps=4, guidance=1.0, **extra,
        )

    def test_start_and_end_images_become_image_and_last_image(self):
        kwargs = self.engine._build_pipeline_kwargs(
            self.config(startImagePath=str(self.start), endImagePath=str(self.end)), generator=None
        )
        self.assertEqual(kwargs["image"].size, (768, 448))
        self.assertEqual(kwargs["last_image"].getpixel((0, 0)), (0, 0, 255))

    def test_start_only_has_no_last_image(self):
        kwargs = self.engine._build_pipeline_kwargs(self.config(startImagePath=str(self.start)), generator=None)
        self.assertIn("image", kwargs)
        self.assertNotIn("last_image", kwargs)

    def test_no_start_image_is_a_readable_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.engine._build_pipeline_kwargs(self.config(), generator=None)
        self.assertIn("start image", str(ctx.exception))

    def test_text_to_video_pipelines_get_no_image(self):
        self.engine._pipeline = type("WanPipeline", (), {})()
        kwargs = self.engine._build_pipeline_kwargs(self.config(startImagePath=str(self.start)), generator=None)
        self.assertNotIn("image", kwargs)


class TwoExpertGgufLoadTests(unittest.TestCase):
    """The low-noise GGUF goes in as ``transformer_2``, shaped by that subfolder's config."""

    def test_both_experts_are_loaded_and_passed_to_from_pretrained(self):
        engine = DiffusersVideoEngine()
        high, low = object(), object()
        pipeline = mock.MagicMock()
        pipeline_cls = mock.MagicMock()
        pipeline_cls.from_pretrained.return_value = pipeline
        calls = []

        def fake_gguf(**kw):
            calls.append((kw["gguf_file"], kw.get("subfolder", "transformer")))
            return (high if kw.get("subfolder", "transformer") == "transformer" else low), "ok"

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(engine, "_pipeline_class", return_value=pipeline_cls), \
                mock.patch("backend_service.helpers.hf_local.local_snapshot_path", return_value=tmp), \
                mock.patch("backend_service.video_runtime.validate_local_diffusers_snapshot", return_value=None), \
                mock.patch.object(engine, "_detect_device", return_value="mps"), \
                mock.patch.object(engine, "_preferred_torch_dtype", return_value="bf16"), \
                mock.patch.object(engine, "_try_load_gguf_transformer", side_effect=fake_gguf), \
                mock.patch("backend_service.video_runtime._detect_device_memory_gb", return_value=64.0):
            engine._ensure_pipeline(BASE, gguf_repo="g/r", gguf_file="hi.gguf", gguf_file2="lo.gguf")

        self.assertEqual(calls, [("hi.gguf", "transformer"), ("lo.gguf", "transformer_2")])
        kwargs = pipeline_cls.from_pretrained.call_args.kwargs
        self.assertIs(kwargs["transformer"], high)
        self.assertIs(kwargs["transformer_2"], low)
        # Quantized experts fit whole on a Mac: no CPU offload.
        pipeline.enable_sequential_cpu_offload.assert_not_called()
        pipeline.to.assert_called_with("mps")


class RouteTests(unittest.TestCase):
    def check(self, body):
        from backend_service.routes.video import _checked_input_images

        return _checked_input_images(body, {"name": "Wan I2V", "repo": BASE})

    def test_wan_i2v_without_a_start_image_is_refused(self):
        for kw in ({}, {"endImagePath": "/e.png"}):
            with self.assertRaises(HTTPException) as ctx:
                self.check(VideoGenerationRequest(modelId=ROW_ID, prompt="p", **kw))
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertIn("start image", ctx.exception.detail)

    def test_start_and_end_images_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            start = _png(Path(tmp) / "s.png")
            end = _png(Path(tmp) / "e.png")
            out = self.check(VideoGenerationRequest(
                modelId=ROW_ID, prompt="p", startImagePath=str(start), endImagePath=str(end)))
        self.assertEqual(out.startImagePath, str(start.resolve()))
        self.assertEqual(out.endImagePath, str(end.resolve()))


class GgufValidationTests(unittest.TestCase):
    def test_a_missing_low_noise_expert_keeps_the_row_uninstalled(self):
        from backend_service.helpers.video import _video_variant_gguf_validation_error

        row = _row()

        def fake_download(repo_id, filename, local_files_only):
            if filename == row["ggufFile2"]:
                raise FileNotFoundError(filename)
            return __file__

        with mock.patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
            error = _video_variant_gguf_validation_error(row)
        self.assertIsNotNone(error)
        self.assertIn(row["ggufFile2"], error)

    def test_both_experts_present_validates(self):
        from backend_service.helpers.video import _video_variant_gguf_validation_error

        with mock.patch("huggingface_hub.hf_hub_download", return_value=__file__):
            self.assertIsNone(_video_variant_gguf_validation_error(_row()))


class BridgeListingTests(unittest.TestCase):
    def test_image_to_video_rows_are_listed_but_never_the_default(self):
        from backend_service.mcp_bridge import models as bridge_models

        self.assertTrue(bridge_models._text_to_media(_row(), "video"))
        self.assertFalse(bridge_models._text_to_media({"repo": "x/other", "taskSupport": ["img2video"]}, "video"))
        rows = [
            {"id": ROW_ID, "fit": "fits", "requiresStartImage": True},
            {"id": "ltx", "fit": "fits", "requiresStartImage": False},
        ]
        with mock.patch.object(bridge_models, "installed_models", return_value=rows):
            self.assertEqual(bridge_models.default_model(object(), "video")["id"], "ltx")


if __name__ == "__main__":
    unittest.main()
