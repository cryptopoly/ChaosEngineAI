"""Image-to-video on the diffusers engine (Wan 2.2 TI2V-5B), and the one place that says who takes a picture."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend_service.video_runtime import DiffusersVideoEngine
from backend_service.video_runtime.pipeline_helpers import (
    build_pipeline_kwargs,
    load_start_image,
    pipeline_class_for_repo,
)
from backend_service.video_runtime.repos import IMAGE_TO_VIDEO_REGISTRY, image_input_support
from backend_service.video_runtime.types import VideoGenerationConfig

TI2V = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_21 = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LTX2 = "dgrauet/ltx-2.3-mlx-q4"


def _config(repo: str = TI2V, start: str | None = None, end: str | None = None, **extra) -> VideoGenerationConfig:
    return VideoGenerationConfig(
        modelId=repo, modelName="m", repo=repo, prompt="waves", negativePrompt="", width=832, height=480,
        numFrames=33, fps=24, steps=20, guidance=5.0, seed=1, startImagePath=start, endImagePath=end, **extra,
    )


def _png(path: Path, size=(640, 360), colour=(200, 30, 30)) -> Path:
    from PIL import Image

    Image.new("RGB", size, colour).save(path)
    return path


class CapabilityTests(unittest.TestCase):
    def test_who_takes_a_start_and_an_end_picture(self) -> None:
        self.assertEqual(image_input_support(LTX2), (True, True))
        self.assertEqual(image_input_support(TI2V), (True, False))  # the first frame only
        self.assertEqual(image_input_support(WAN_21), (False, False))
        self.assertEqual(image_input_support("Lightricks/LTX-Video"), (False, False))

    def test_catalog_payload_flags_match(self) -> None:
        from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES

        variants = {v["repo"]: v for f in VIDEO_MODEL_FAMILIES for v in f["variants"]}
        self.assertIn("img2video", variants[TI2V]["taskSupport"])
        from backend_service.video_runtime.repos import LTX2_DIFFUSERS_REPOS

        self.assertEqual(set(IMAGE_TO_VIDEO_REGISTRY), {TI2V, *LTX2_DIFFUSERS_REPOS})
        self.assertNotIn("img2video", variants[WAN_21]["taskSupport"])

    def test_the_enriched_catalog_carries_both_flags(self) -> None:
        from backend_service.helpers import video as video_helpers

        source = Path(video_helpers.__file__).read_text(encoding="utf-8")
        self.assertIn('enriched["acceptsImageInput"], enriched["acceptsEndImage"] = image_input_support(', source)


class PlacementTests(unittest.TestCase):
    def test_model_offload_across_card_sizes(self) -> None:
        """Pipeline, largest component, free GPU memory (GB) -> offload by model? Cards from 24 GB up to unified memory."""
        from backend_service.video_runtime.defaults import _prefers_model_cpu_offload as prefers

        wan_5b = (22.0, 11.4)       # Wan 2.2 TI2V-5B: pipeline, text encoder (its largest part)
        ltx_4bit = (25.0, 14.4)     # LTX-2.3 distilled 4-bit: pipeline, transformer
        ltx_bf16 = (76.0, 38.0)     # LTX-2.3 bf16: pipeline, transformer
        small = (9.0, 4.0)          # Wan 2.1 1.3B
        cases = [
            # (label, free GB, (pipeline, largest), expected)
            ("24 GB: Wan 5B", 24.0, wan_5b, True),
            ("24 GB: LTX 4-bit", 24.0, ltx_4bit, True),
            ("24 GB: small model fits whole", 24.0, small, False),
            ("24 GB: LTX bf16 cannot (transformer alone too big): old path", 24.0, ltx_bf16, False),
            ("32 GB (RTX 5090): LTX 4-bit leaves too little for activations", 32.0, ltx_4bit, True),
            ("32 GB: Wan 5B", 32.0, wan_5b, True),
            ("48 GB: LTX bf16, transformer 38 GB does not fit twice: old path", 48.0, ltx_bf16, False),
            ("80 GB (H100): LTX bf16", 80.0, ltx_bf16, True),
            ("48 GB: LTX 4-bit has 23 GB to spare: in place", 48.0, ltx_4bit, False),
            ("96 GB (RTX PRO 6000): LTX bf16 has 20 GB to spare: in place", 96.0, ltx_bf16, False),
            ("96 GB: 4-bit sits in place", 96.0, ltx_4bit, False),
            ("20 GB free of 24 (a chat model holds 4): LTX 4-bit", 20.0, ltx_4bit, True),
        ]
        for label, free, (footprint, largest), expected in cases:
            self.assertEqual(prefers("cuda", free, footprint, largest), expected, label)

    def test_no_offload_on_unified_memory_or_other_devices_or_unknown_figures(self) -> None:
        from backend_service.video_runtime.defaults import _prefers_model_cpu_offload as prefers

        self.assertFalse(prefers("cuda", 100.0, 76.0, 38.0, unified_memory=True))   # DGX Spark: one pool, copying helps nothing
        self.assertFalse(prefers("cuda", 24.0, 22.0, 11.0, unified_memory=True))
        self.assertFalse(prefers("mps", 24.0, 22.0, 11.0))     # CUDA only
        self.assertFalse(prefers("cuda", None, 22.0, 11.0))    # unknown figures: the old path
        self.assertFalse(prefers("cuda", 24.0, None, 11.0))
        self.assertFalse(prefers("cuda", 24.0, 22.0, None))

    def test_the_gpu_memory_probe_reads_free_and_total(self) -> None:
        from backend_service.video_runtime.defaults import _cuda_memory_gb

        torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (20_000_000_000, 24_000_000_000)))
        self.assertEqual(_cuda_memory_gb(torch), (20.0, 24.0))
        # weights this process already put on the card count as available to it: the 4-bit LTX-2.3 pack is on the GPU
        # (22.7 GB of 24) before placement runs, and judging by the 1.3 GB left over left it thrashing on step 1
        loaded = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda: (1_300_000_000, 24_000_000_000), memory_reserved=lambda: 22_700_000_000))
        self.assertEqual(_cuda_memory_gb(loaded), (24.0, 24.0))
        self.assertEqual(_cuda_memory_gb(SimpleNamespace(cuda=SimpleNamespace())), (None, None))  # no CUDA

        def broken() -> tuple[int, int]:
            raise RuntimeError("no device")

        self.assertEqual(_cuda_memory_gb(SimpleNamespace(cuda=SimpleNamespace(mem_get_info=broken))), (None, None))

    def test_a_pipeline_already_on_the_card_is_still_offloaded(self) -> None:
        """The 4-bit LTX-2.3 case end to end: 22.7 GB resident, 1.3 GB free, 14.4 GB largest part, 25 GB pipeline."""
        from backend_service.video_runtime.defaults import _cuda_memory_gb, _prefers_model_cpu_offload as prefers

        loaded = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=lambda: (1_300_000_000, 24_000_000_000), memory_reserved=lambda: 22_700_000_000))
        budget, _ = _cuda_memory_gb(loaded)
        self.assertTrue(prefers("cuda", budget, 25.0, 14.4))
        self.assertFalse(prefers("cuda", 1.3, 25.0, 14.4))  # the bug: the free memory alone rules it out

    def test_unified_memory_is_a_gpu_total_close_to_the_machines_ram(self) -> None:
        from backend_service.video_runtime.defaults import _is_unified_memory

        with mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(total=128_000_000_000)):
            self.assertTrue(_is_unified_memory(119.0))    # DGX Spark: ~119 GB visible of 128 GB
            self.assertFalse(_is_unified_memory(24.0))    # a discrete card in a 128 GB workstation
            self.assertFalse(_is_unified_memory(96.0))    # a 96 GB card in a 128 GB workstation
        with mock.patch("psutil.virtual_memory", return_value=SimpleNamespace(total=64_000_000_000)):
            self.assertFalse(_is_unified_memory(24.0))    # a 4090 in a 64 GB PC
        self.assertFalse(_is_unified_memory(None))

    def test_the_largest_component_is_measured_from_its_parameters(self) -> None:
        from backend_service.video_runtime.defaults import _largest_component_gb

        class Param:
            def __init__(self, count: int, size: int) -> None:
                self._count, self._size = count, size

            def numel(self) -> int:
                return self._count

            def element_size(self) -> int:
                return self._size

        class Module:
            def __init__(self, *params: Param) -> None:
                self._params = params

            def parameters(self):
                return iter(self._params)

        pipeline = SimpleNamespace(components={
            "transformer": Module(Param(5_000_000_000, 2)),           # 10 GB
            "text_encoder": Module(Param(5_700_000_000, 2)),          # 11.4 GB
            "tokenizer": object(),                                    # not a module
        })
        self.assertAlmostEqual(_largest_component_gb(pipeline), 11.4, places=1)
        self.assertIsNone(_largest_component_gb(SimpleNamespace(components=None)))

    def test_the_ti2v_footprint_counts_the_text_encoder(self) -> None:
        from backend_service.video_runtime.defaults import _VIDEO_MODEL_FOOTPRINT_BF16_GB

        self.assertGreaterEqual(_VIDEO_MODEL_FOOTPRINT_BF16_GB[TI2V], 20.0)


class PipelineChoiceTests(unittest.TestCase):
    def test_a_picture_run_loads_the_image_class_and_a_text_run_the_text_class(self) -> None:
        import diffusers

        self.assertIs(pipeline_class_for_repo(TI2V), diffusers.WanPipeline)
        self.assertIs(pipeline_class_for_repo(TI2V, image_input=True), diffusers.WanImageToVideoPipeline)
        # a repo with no image class keeps its own, whatever the run asks
        self.assertIs(pipeline_class_for_repo(WAN_21, image_input=True), diffusers.WanPipeline)

    def test_the_mode_is_part_of_the_loaded_pipeline_key(self) -> None:
        engine = DiffusersVideoEngine()
        with_picture = engine._compute_variant_key(_config(start="C:/x.png"))
        without = engine._compute_variant_key(_config())
        self.assertNotEqual(with_picture, without)
        self.assertIn("i2v", with_picture)
        # a start picture on a repo with no image class does not rebuild anything
        self.assertEqual(engine._compute_variant_key(_config(WAN_21, start="C:/x.png")),
                         engine._compute_variant_key(_config(WAN_21)))
        self.assertTrue(engine._uses_image_pipeline(_config(start="C:/x.png")))
        self.assertFalse(engine._uses_image_pipeline(_config()))
        self.assertFalse(engine._uses_image_pipeline(_config(WAN_21, start="C:/x.png")))


class StartImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def test_the_picture_is_scaled_to_cover_the_clip_and_cropped_not_stretched(self) -> None:
        wide = self.dir / "wide.png"
        from PIL import Image

        image = Image.new("RGB", (1000, 500), (255, 0, 0))
        for x in range(500, 1000):
            for y in range(500):
                image.putpixel((x, y), (0, 0, 255))  # right half blue
        image.save(wide)
        out = load_start_image(str(wide), 480, 480)  # square crop of a 2:1 picture: the centre, half red half blue
        self.assertEqual(out.size, (480, 480))
        self.assertEqual(out.getpixel((10, 240)), (255, 0, 0))
        self.assertEqual(out.getpixel((470, 240)), (0, 0, 255))

    def test_rgba_and_rotated_pictures_come_out_rgb(self) -> None:
        from PIL import Image

        path = self.dir / "alpha.png"
        Image.new("RGBA", (64, 64), (10, 20, 30, 128)).save(path)
        self.assertEqual(load_start_image(str(path), 128, 64).mode, "RGB")

    def test_the_image_pipeline_gets_the_picture_and_the_text_pipeline_does_not(self) -> None:
        still = _png(self.dir / "still.png")
        i2v = type("WanImageToVideoPipeline", (), {})()
        text = type("WanPipeline", (), {})()
        kwargs = build_pipeline_kwargs(_config(start=str(still)), None, i2v)
        self.assertEqual(kwargs["image"].size, (832, 480))
        self.assertNotIn("image", build_pipeline_kwargs(_config(start=str(still)), None, text))
        with self.assertRaises(RuntimeError):  # the image class with no picture would fail deep inside diffusers
            build_pipeline_kwargs(_config(), None, i2v)


class RouteGateTests(unittest.TestCase):
    def _body(self, **fields):
        from backend_service.models import VideoGenerationRequest

        return VideoGenerationRequest(modelId="x", prompt="p", **fields)

    def test_ti2v_takes_a_start_picture_and_refuses_an_end_picture(self) -> None:
        from fastapi import HTTPException

        from backend_service.routes.video import _checked_input_images

        with tempfile.TemporaryDirectory() as tmp:
            still = _png(Path(tmp) / "s.png")
            variant = {"name": "Wan 2.2 TI2V 5B", "repo": TI2V}
            with mock.patch("backend_service.helpers.media_input.resolve_input_image", side_effect=lambda p: Path(p)):
                body = _checked_input_images(self._body(startImagePath=str(still)), variant)
                self.assertEqual(body.startImagePath, str(still))
                with self.assertRaises(HTTPException) as end:
                    _checked_input_images(self._body(startImagePath=str(still), endImagePath=str(still)), variant)
                self.assertIn("only a start image", end.exception.detail)
                with self.assertRaises(HTTPException) as none:
                    _checked_input_images(self._body(startImagePath=str(still)), {"name": "Wan 2.1", "repo": WAN_21})
                self.assertIn("cannot start from an image", none.exception.detail)

    def test_the_agent_bridge_lists_both_flags(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "backend_service" / "mcp_bridge" / "models.py").read_text(encoding="utf-8")
        self.assertIn('row["acceptsImageInput"], row["acceptsEndImage"] = image_input_support(repo)', source)


if __name__ == "__main__":
    unittest.main()
