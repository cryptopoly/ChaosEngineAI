"""Per-model image and video size limits, and the memory gate's pixel term.

The global 2048 px cap is gone: each catalog variant carries ``maxSide``,
``maxPixels`` and ``nativeSide`` (``backend_service/catalog/size_limits.py``), the
size guard (``helpers/generation_limits.py``) applies them, ``HARD_MAX_SIDE`` is
the safety-net ceiling, and the memory gate charges sizes above the model's
reference resolution.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from pydantic import ValidationError

from backend_service.app import _apply_min_side, _resolve_image_size
from backend_service.catalog import (
    IMAGE_MODEL_FAMILIES,
    LATEST_IMAGE_TRACKED_SEEDS,
    VIDEO_MODEL_FAMILIES,
)
from backend_service.catalog.size_limits import IMAGE_FAMILY_LIMITS, VIDEO_FAMILY_LIMITS
from backend_service.helpers.generation_limits import (
    DEFAULT_MAX_SIDE,
    HARD_MAX_SIDE,
    normalize_image_size,
    normalize_video_size,
    parse_resolution,
    reference_frames,
    reference_pixels,
    size_limit_fields,
    size_limits,
    snap_to_multiple,
)
from backend_service.helpers.memory_gate import (
    gate_image_generation,
    gate_video_generation,
    size_extra_gb,
)
from backend_service.models import ImageGenerationRequest, VideoGenerationRequest
from backend_service.state import ChaosEngineState
from tests.test_backend_service import (
    FakeRuntime,
    fake_library,
    fake_system_snapshot,
    make_test_client,
)


def _image_variant(variant_id: str) -> dict:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == variant_id:
                return variant
    raise KeyError(variant_id)


def _video_variant(variant_id: str) -> dict:
    for family in VIDEO_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == variant_id:
                return variant
    raise KeyError(variant_id)


SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
FLUX = "black-forest-labs/FLUX.1-schnell"
SD35_MEDIUM = "stabilityai/stable-diffusion-3.5-medium"


def _normalize_image(variant: dict, width: int, height: int):
    limits = size_limits(variant)
    return normalize_image_size(
        width, height, min_side=variant.get("minSide"), model_name=variant["name"], **limits.as_kwargs()
    )


def _normalize_video(variant: dict, width: int, height: int):
    return normalize_video_size(
        width, height, variant["repo"], model_name=variant["name"], **size_limits(variant).as_kwargs()
    )


class CatalogAuditTests(unittest.TestCase):
    def test_every_image_variant_carries_the_three_fields(self):
        for family in IMAGE_MODEL_FAMILIES:
            for variant in family["variants"]:
                with self.subTest(variant=variant["id"]):
                    for key in ("maxSide", "maxPixels", "nativeSide"):
                        self.assertIsInstance(variant.get(key), int, key)
                        self.assertGreater(variant[key], 0, key)
                    self.assertLessEqual(variant["maxSide"], HARD_MAX_SIDE)
                    self.assertLessEqual(variant["maxPixels"], variant["maxSide"] ** 2)
                    self.assertGreaterEqual(variant["maxSide"], variant["nativeSide"])

    def test_every_video_variant_carries_the_three_fields(self):
        for family in VIDEO_MODEL_FAMILIES:
            for variant in family["variants"]:
                with self.subTest(variant=variant["id"]):
                    for key in ("maxSide", "maxPixels", "nativeSide"):
                        self.assertIsInstance(variant.get(key), int, key)
                        self.assertGreater(variant[key], 0, key)
                    self.assertLessEqual(variant["maxSide"], HARD_MAX_SIDE)
                    self.assertLessEqual(variant["maxPixels"], variant["maxSide"] ** 2)

    def test_tracked_seeds_carry_the_three_fields(self):
        for seed in LATEST_IMAGE_TRACKED_SEEDS:
            with self.subTest(seed=seed["repo"]):
                for key in ("maxSide", "maxPixels", "nativeSide"):
                    self.assertIn(key, seed)

    def test_every_family_has_a_row_in_the_table(self):
        """A new family must be given a limit on purpose, not inherit the unverified default."""
        for family in IMAGE_MODEL_FAMILIES:
            self.assertIn(family["id"], IMAGE_FAMILY_LIMITS, f"image family {family['id']} has no size-limit row")
        for family in VIDEO_MODEL_FAMILIES:
            self.assertIn(family["id"], VIDEO_FAMILY_LIMITS, f"video family {family['id']} has no size-limit row")

    def test_the_recommended_resolution_is_inside_the_limits(self):
        """A model must accept its own recommended size."""
        for families in (IMAGE_MODEL_FAMILIES, VIDEO_MODEL_FAMILIES):
            for family in families:
                for variant in family["variants"]:
                    resolution = parse_resolution(variant.get("recommendedResolution"))
                    if not resolution:
                        continue
                    with self.subTest(variant=variant["id"]):
                        self.assertLessEqual(max(resolution), variant["maxSide"])
                        self.assertLessEqual(resolution[0] * resolution[1], variant["maxPixels"])

    def test_variant_payload_fields_match_the_catalog(self):
        from backend_service.helpers.images import _image_model_payloads

        with mock.patch("backend_service.helpers.images._image_repo_live_metadata", return_value={}), \
                mock.patch("backend_service.helpers.images.annotate_fit", side_effect=lambda f, *_a, **_k: f, create=True):
            families = _image_model_payloads([])
        by_id = {v["id"]: v for f in families for v in f["variants"]}
        sdxl = by_id[SDXL]
        self.assertEqual((sdxl["maxSide"], sdxl["nativeSide"]), (2048, 1024))
        self.assertEqual(by_id[FLUX]["maxSide"], 4096)


class SizeLimitsReaderTests(unittest.TestCase):
    def test_a_variant_without_limits_gets_the_defaults(self):
        limits = size_limits({"recommendedResolution": "1360x768"})
        self.assertEqual(limits.max_side, DEFAULT_MAX_SIDE)
        self.assertEqual(limits.max_pixels, DEFAULT_MAX_SIDE ** 2)
        self.assertEqual(limits.native_side, 1360)
        self.assertEqual(size_limits(None).native_side, 1024)

    def test_declared_values_are_read(self):
        limits = size_limits({"maxSide": 2048, "maxPixels": 2_000_000, "nativeSide": 1024, "minSide": 768})
        self.assertEqual((limits.max_side, limits.max_pixels, limits.native_side, limits.min_side), (2048, 2_000_000, 1024, 768))
        self.assertEqual(size_limit_fields({"maxSide": 2048})["maxPixels"], 2048 * 2048)

    def test_garbage_falls_back(self):
        limits = size_limits({"maxSide": "big", "maxPixels": -5, "nativeSide": None})
        self.assertEqual(limits.max_side, DEFAULT_MAX_SIDE)

    def test_the_global_ceiling_holds_whatever_a_row_says(self):
        limits = size_limits({"maxSide": 100_000, "maxPixels": 10 ** 12})
        self.assertEqual(limits.max_side, HARD_MAX_SIDE)
        self.assertEqual(limits.max_pixels, HARD_MAX_SIDE ** 2)

    def test_reference_pixels_follow_the_recommended_resolution(self):
        self.assertEqual(reference_pixels({"recommendedResolution": "768x512"}), 768 * 512)
        self.assertEqual(reference_pixels({"nativeSide": 1328}), 1328 * 1328)
        self.assertEqual(reference_pixels(None), 1024 * 1024)

    def test_reference_frames_are_the_default_clip_at_24_fps(self):
        self.assertEqual(reference_frames({"defaultDurationSeconds": 4.0}), 96)
        self.assertIsNone(reference_frames({}))


class ImageLimitTests(unittest.TestCase):
    def test_sdxl_is_capped_at_its_limit(self):
        width, height, note = _normalize_image(_image_variant(SDXL), 4096, 4096)
        self.assertEqual((width, height), (2048, 2048))
        self.assertIn("Stable Diffusion XL Base 1.0", note)
        self.assertIn("2048 px per side", note)

    def test_flux_allows_4096(self):
        self.assertEqual(_normalize_image(_image_variant(FLUX), 4096, 4096), (4096, 4096, None))
        self.assertEqual(_normalize_image(_image_variant(FLUX), 4096, 2304), (4096, 2304, None))

    def test_flux_is_still_capped_at_its_own_limit(self):
        width, height, note = _normalize_image(_image_variant(FLUX), 6000, 3000)
        self.assertEqual((width, height), (4096, 2048))  # the aspect ratio survives
        self.assertIn("4096 px per side", note)

    def test_the_pixel_limit_applies_and_keeps_the_aspect_ratio(self):
        variant = _image_variant(SD35_MEDIUM)  # 2048 px per side, 1440 x 1440 pixels
        width, height, note = _normalize_image(variant, 2048, 2048)
        self.assertLessEqual(width * height, variant["maxPixels"])
        self.assertEqual(width, height)
        self.assertIn("megapixels", note)
        self.assertEqual(_normalize_image(variant, 1440, 1440), (1440, 1440, None))
        wide_w, wide_h, _ = _normalize_image(variant, 2048, 1024)
        self.assertLessEqual(wide_w * wide_h, variant["maxPixels"])
        self.assertAlmostEqual(wide_w / wide_h, 2.0, delta=0.1)

    def test_flux2_stops_at_four_megapixels(self):
        variant = _image_variant("black-forest-labs/FLUX.2-klein-4B")
        self.assertEqual(_normalize_image(variant, 2048, 2048), (2048, 2048, None))
        width, height, _ = _normalize_image(variant, 4096, 4096)
        self.assertLessEqual(width * height, 4 * 1024 * 1024)

    def test_snapping_is_unchanged(self):
        self.assertEqual(normalize_image_size(1001, 777)[:2], (1008, 784))
        self.assertEqual(normalize_image_size(4090, 4090, max_side=4096)[:2], (4096, 4096))
        self.assertEqual(snap_to_multiple(24, 16, lo=16, hi=2048), 32)
        self.assertEqual(snap_to_multiple(5000, 16, lo=16, hi=2048), 2048)

    def test_a_snap_only_note_is_the_old_one(self):
        _, _, note = normalize_image_size(1001, 777)
        self.assertEqual(note, "Size 1001x777 changed to 1008x784: image models need multiples of 16.")

    def test_a_size_inside_the_limits_has_no_note(self):
        self.assertEqual(normalize_image_size(1024, 768, max_side=2048), (1024, 768, None))

    def test_the_model_minimum_still_applies(self):
        width, height, _ = normalize_image_size(300, 300, min_side=768, max_side=2048)
        self.assertEqual((width, height), (768, 768))

    def test_the_ceiling_of_8192_holds_even_without_a_model_limit(self):
        width, height, note = normalize_image_size(20_000, 20_000, max_side=1_000_000)
        self.assertEqual((width, height), (HARD_MAX_SIDE, HARD_MAX_SIDE))
        self.assertIn(f"{HARD_MAX_SIDE} px per side", note)

    def test_no_limit_given_uses_the_default(self):
        self.assertEqual(normalize_image_size(9000, 9000)[:2], (DEFAULT_MAX_SIDE, DEFAULT_MAX_SIDE))

    def test_the_min_side_helper_uses_the_same_limit(self):
        self.assertEqual(_apply_min_side(256, 2048, 768, max_side=2048), (768, 2048))
        w, h = _apply_min_side(2048, 256, 768, max_side=2048)
        self.assertLessEqual(max(w, h), 2048)
        w, h = _apply_min_side(2048, 256, 768, max_side=4096)
        self.assertEqual((w, h), (4096, 768))
        w, h = _apply_min_side(2048, 256, 768, max_side=1_000_000)
        self.assertLessEqual(max(w, h), HARD_MAX_SIDE)

    def test_resolve_image_size_uses_the_variants_limits(self):
        request = SimpleNamespace(width=4096, height=4096, draftMode=False)
        self.assertEqual(_resolve_image_size(request, _image_variant(SDXL))[:2], (2048, 2048))
        self.assertEqual(_resolve_image_size(request, _image_variant(FLUX))[:2], (4096, 4096))
        note = _resolve_image_size(request, _image_variant(SDXL))[2]
        self.assertIn("2048 px per side", note)

    def test_resolve_image_size_draft_mode_still_shrinks(self):
        request = SimpleNamespace(width=4096, height=4096, draftMode=True)
        self.assertEqual(_resolve_image_size(request, _image_variant(FLUX))[:2], (512, 512))

    def test_resolve_image_size_raises_a_small_request_to_the_minimum(self):
        iris = _image_variant("speridlabs/iris-3b-mflux")
        request = SimpleNamespace(width=256, height=256, draftMode=False)
        width, height, note = _resolve_image_size(request, iris)
        self.assertEqual((width, height), (768, 768))
        self.assertIn("768", note)


class VideoLimitTests(unittest.TestCase):
    def test_ltx_video_stops_at_its_card_limit(self):
        variant = _video_variant("Lightricks/LTX-Video")
        width, height, note = _normalize_video(variant, 1920, 1080)
        self.assertLessEqual(max(width, height), 1280)
        self.assertLessEqual(width * height, variant["maxPixels"])
        self.assertEqual((width % 32, height % 32), (0, 0))
        self.assertIn("1280 px per side", note)
        self.assertIn("LTX-Video", note)

    def test_wan_accepts_720p(self):
        variant = _video_variant("Wan-AI/Wan2.1-T2V-14B-Diffusers")
        self.assertEqual(_normalize_video(variant, 1280, 720), (1280, 720, None))
        width, height, _ = _normalize_video(variant, 1920, 1088)
        self.assertLessEqual(max(width, height), 1280)

    def test_ltx2_allows_beyond_the_old_cap(self):
        variant = _video_variant("dgrauet/ltx-2.3-mlx-q4")
        self.assertEqual(_normalize_video(variant, 3840, 2176)[:2], (3840, 2176))

    def test_cogvideox_is_held_to_its_only_resolution(self):
        variant = _video_variant("THUDM/CogVideoX-5b")
        self.assertEqual(_normalize_video(variant, 720, 480), (720, 480, None))
        width, height, _ = _normalize_video(variant, 1920, 1080)
        self.assertLessEqual(width * height, 720 * 480)

    def test_video_snapping_is_unchanged(self):
        self.assertEqual(normalize_video_size(770, 500, "Lightricks/LTX-Video")[:2], (768, 512))
        self.assertEqual(normalize_video_size(833, 481, "Wan-AI/Wan2.1-T2V-14B-Diffusers")[:2], (832, 480))

    def test_video_ceiling_of_8192(self):
        width, height, _ = normalize_video_size(20_000, 20_000, "Lightricks/LTX-2", max_side=1_000_000)
        self.assertLessEqual(max(width, height), HARD_MAX_SIDE)


class RequestModelTests(unittest.TestCase):
    def test_the_requests_accept_up_to_the_ceiling(self):
        self.assertEqual(ImageGenerationRequest(modelId="m", prompt="p", width=HARD_MAX_SIDE, height=4096).width, HARD_MAX_SIDE)
        self.assertEqual(VideoGenerationRequest(modelId="m", prompt="p", width=4096).width, 4096)

    def test_the_requests_reject_above_the_ceiling(self):
        with self.assertRaises(ValidationError):
            ImageGenerationRequest(modelId="m", prompt="p", width=HARD_MAX_SIDE + 16)
        with self.assertRaises(ValidationError):
            VideoGenerationRequest(modelId="m", prompt="p", height=HARD_MAX_SIDE + 16)


class MemoryGateSizeTests(unittest.TestCase):
    NATIVE = 1024 * 1024

    def _image_gate(self, available_gb, width, height, **kwargs):
        return gate_image_generation(
            available_gb,
            30.0,
            width=width,
            height=height,
            reference_pixels=self.NATIVE,
            footprint_gb=kwargs.pop("footprint_gb", 12.0),
            model_name="FLUX.1 Schnell",
            **kwargs,
        )

    def test_extra_memory_is_zero_at_or_below_the_reference(self):
        for size in (256, 512, 1024):
            self.assertEqual(size_extra_gb(size, size, reference_pixels=self.NATIVE, footprint_gb=12.0), 0.0)
        self.assertEqual(size_extra_gb(1024, 1024, reference_pixels=None, footprint_gb=12.0), 0.0)

    def test_extra_memory_grows_with_the_pixel_count(self):
        two_k = size_extra_gb(2048, 2048, reference_pixels=self.NATIVE, footprint_gb=12.0)
        four_k = size_extra_gb(4096, 4096, reference_pixels=self.NATIVE, footprint_gb=12.0)
        self.assertGreater(two_k, 0)
        self.assertGreater(four_k, 4 * two_k * 0.9)

    def test_a_small_machine_refuses_an_oversize_request_and_says_why(self):
        refusal = self._image_gate(10.0, 4096, 4096)  # a 16 GB machine with ~10 GB free
        self.assertIsNotNone(refusal)
        self.assertEqual(refusal["code"], "memory_gate_image_size_too_large")
        self.assertIn("4096x4096", refusal["message"])
        self.assertIn("FLUX.1 Schnell", refusal["message"])
        self.assertIn("10.0 GB is available", refusal["message"])
        self.assertIn("Try ", refusal["message"])

    def test_the_suggested_size_actually_fits(self):
        refusal = self._image_gate(10.0, 4096, 4096)
        suggestion = refusal["message"].split("Try ")[1].split(" ")[0]
        width, height = (int(part) for part in suggestion.split("x"))
        self.assertLess(width, 4096)
        self.assertIsNone(self._image_gate(10.0, width, height))

    def test_a_big_machine_runs_the_same_request(self):
        self.assertIsNone(self._image_gate(150.0, 4096, 4096))  # a 192 GB Mac Studio

    def test_a_small_machine_is_unaffected_at_1024(self):
        self.assertIsNone(self._image_gate(10.0, 1024, 1024))
        self.assertIsNone(self._image_gate(10.0, 768, 512))

    def test_estimates_at_or_below_native_do_not_change(self):
        with_size = gate_image_generation(
            6.0, 30.0, width=1024, height=1024, reference_pixels=self.NATIVE, footprint_gb=12.0
        )
        without = gate_image_generation(6.0, 30.0)
        self.assertEqual(with_size, without)

    def test_a_gate_called_without_a_size_behaves_as_before(self):
        self.assertIsNone(gate_image_generation(10.0, 30.0))
        self.assertEqual(
            gate_image_generation(2.0, 30.0)["code"], "memory_gate_image_low_available"
        )

    def test_the_model_too_large_check_still_comes_first(self):
        refusal = self._image_gate(10.0, 4096, 4096, model_gb=20.0)
        self.assertEqual(refusal["code"], "memory_gate_image_model_too_large")

    def test_the_load_and_the_size_add_up(self):
        # 12 GB model not yet resident: 13.2 GB to load, so 15 GB free fits 1024 but not 2048 (5.4 GB extra).
        self.assertIsNone(self._image_gate(15.0, 1024, 1024, model_gb=12.0))
        refusal = self._image_gate(15.0, 2048, 2048, model_gb=12.0)
        self.assertEqual(refusal["code"], "memory_gate_image_size_too_large")
        self.assertIn("to load the model", refusal["message"])

    def test_video_size_and_length(self):
        kwargs = dict(reference_pixels=768 * 512, reference_frames=96, footprint_gb=20.0, model_name="LTX")
        self.assertIsNone(gate_video_generation(30.0, 30.0, width=768, height=512, frames=97, **kwargs))
        self.assertIsNone(gate_video_generation(30.0, 30.0, width=768, height=512, frames=257, **kwargs))
        refusal = gate_video_generation(30.0, 30.0, width=4096, height=2304, frames=121, **kwargs)
        self.assertEqual(refusal["code"], "memory_gate_video_size_too_large")
        self.assertIsNone(gate_video_generation(200.0, 30.0, width=4096, height=2304, frames=121, **kwargs))
        # More frames cost more at a size above the reference.
        short = size_extra_gb(1920, 1088, reference_pixels=768 * 512, footprint_gb=20.0, frames=96, reference_frames=96)
        long = size_extra_gb(1920, 1088, reference_pixels=768 * 512, footprint_gb=20.0, frames=192, reference_frames=96)
        self.assertAlmostEqual(long, short * 2)

    def test_the_measured_ltx2_run_fits_its_footprint(self):
        """Measured: 1280x704 x 121 frames peaked at 18 GB against a 20 GB footprint."""
        extra = size_extra_gb(
            1280, 704, reference_pixels=768 * 512, footprint_gb=20.0, frames=121, reference_frames=96
        )
        self.assertLess(20.0 + extra, 26.0)


class ImageRouteGateTests(unittest.TestCase):
    """The route judges the size that will render, after the model's limits."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=root / "settings.json",
            benchmarks_path=root / "benchmark-history.json",
            chat_sessions_path=root / "chat-sessions.json",
        )
        state.runtime = FakeRuntime()
        self.client = make_test_client(state)

    def tearDown(self):
        self.tempdir.cleanup()

    def _post(self, model_id, width, height, available_gb):
        with mock.patch(
            "backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(available_gb, 20.0)
        ), mock.patch(
            "backend_service.routes.images._generate_image_artifacts",
            return_value=([], {"activeEngine": "diffusers"}),
        ) as generate:
            response = self.client.post(
                "/api/images/generate",
                json={"modelId": model_id, "prompt": "x", "width": width, "height": height, "steps": 4},
            )
        return response, generate

    @staticmethod
    def _message(response):
        detail = response.json()["detail"]
        return detail["message"] if isinstance(detail, dict) else detail

    def test_an_oversize_request_is_refused_on_a_small_machine(self):
        # SDXL is capped at 2048, so the gate must name 2048x2048, not the typed 4096x4096.
        response, generate = self._post(SDXL, 4096, 4096, available_gb=18.0)
        self.assertEqual(response.status_code, 503, response.text)
        message = self._message(response)
        self.assertIn("2048x2048", message)
        self.assertNotIn("4096x4096", message)
        generate.assert_not_called()

    def test_the_same_request_runs_with_enough_memory(self):
        response, generate = self._post(SDXL, 4096, 4096, available_gb=64.0)
        self.assertEqual(response.status_code, 200, response.text)
        generate.assert_called_once()

    def test_1024_is_never_refused_for_its_size(self):
        response, _ = self._post(SDXL, 1024, 1024, available_gb=18.0)
        self.assertEqual(response.status_code, 200, response.text)

    def test_draft_mode_is_judged_at_its_small_size(self):
        with mock.patch(
            "backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(18.0, 20.0)
        ), mock.patch(
            "backend_service.routes.images._generate_image_artifacts",
            return_value=([], {"activeEngine": "diffusers"}),
        ):
            response = self.client.post(
                "/api/images/generate",
                json={"modelId": SDXL, "prompt": "x", "width": 4096, "height": 4096, "steps": 4, "draftMode": True},
            )
        self.assertEqual(response.status_code, 200, response.text)


class VideoGateRouteHelperTests(unittest.TestCase):
    def _run_gate(self, available_gb, width, height, frames=97):
        from backend_service.routes import video as video_routes

        variant = _video_variant("dgrauet/ltx-2.3-mlx-q4")
        state = SimpleNamespace(
            add_log=mock.Mock(),
            video_runtime=SimpleNamespace(capabilities=lambda: {"loadedModelRepo": variant["repo"]}),
        )
        request = mock.MagicMock()
        request.app.state.chaosengine = state
        body = SimpleNamespace(width=width, height=height, numFrames=frames)
        with mock.patch(
            "backend_service.helpers.memory_gate.snapshot_memory_signals", return_value=(available_gb, 20.0)
        ):
            video_routes._video_memory_gate(request, variant, "video generation", body)

    def test_native_size_passes_on_a_modest_machine(self):
        self._run_gate(24.0, 768, 512)

    def test_an_oversize_clip_is_refused_with_the_size_in_the_message(self):
        with self.assertRaises(HTTPException) as caught:
            self._run_gate(24.0, 4096, 2304, frames=121)
        self.assertEqual(caught.exception.status_code, 503)
        detail = caught.exception.detail
        message = detail["message"] if isinstance(detail, dict) else detail
        self.assertIn("4096x2304", message)

    def test_a_big_machine_runs_it(self):
        self._run_gate(200.0, 4096, 2304, frames=121)


def _load_ceiling_script():
    """The measuring script (hyphenated file name); only its pure helpers are used here, nothing is run."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "measure-image-ceiling.py"
    spec = importlib.util.spec_from_file_location("measure_image_ceiling", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CeilingScriptHelperTests(unittest.TestCase):
    """The helpers of scripts/measure-image-ceiling.py. The script itself generates images and is not run by tests."""

    @classmethod
    def setUpClass(cls):
        cls.script = _load_ceiling_script()

    def test_steps_are_native_one_and_a_half_two_three_four_times(self):
        steps = self.script.size_steps(1024, self.script.DEFAULT_FACTORS)
        self.assertEqual(
            [(width, height) for _factor, width, height in steps],
            [(1024, 1024), (1536, 1536), (2048, 2048), (3072, 3072), (4096, 4096)],
        )

    def test_steps_keep_the_aspect_ratio_on_the_long_edge(self):
        steps = self.script.size_steps(1024, (1.0, 2.0), self.script.parse_aspect("16:9"))
        self.assertEqual([(w, h) for _f, w, h in steps], [(1024, 576), (2048, 1152)])

    def test_steps_are_valid_sizes_for_every_engine(self):
        for _factor, width, height in self.script.size_steps(1328, (1, 1.5, 2, 3, 4), (3, 2)):
            self.assertEqual((width % 64, height % 64), (0, 0))

    def test_steps_that_round_to_the_same_size_are_skipped(self):
        steps = self.script.size_steps(64, (1.0, 1.2))
        self.assertEqual(len(steps), 1)

    def test_the_model_limit_check(self):
        self.assertTrue(self.script.exceeds_model_limit(4096, 4096, 2048, 2048 * 2048))
        self.assertTrue(self.script.exceeds_model_limit(2048, 2048, 4096, 2_000_000))
        self.assertFalse(self.script.exceeds_model_limit(2048, 2048, 2048, 2048 * 2048))
        self.assertFalse(self.script.exceeds_model_limit(8000, 8000, None, None))

    def test_footprint_output_is_parsed(self):
        sample = "python3.11 [4242]: 64-bit    Footprint: 18 GB (16384 bytes per page)\n"
        self.assertAlmostEqual(self.script.parse_footprint(sample), 18.0)
        self.assertAlmostEqual(self.script.parse_footprint("Footprint: 512 MB"), 0.5)
        self.assertIsNone(self.script.parse_footprint("nothing here"))

    def test_a_bad_aspect_is_rejected(self):
        with self.assertRaises(ValueError):
            self.script.parse_aspect("wide")


if __name__ == "__main__":
    unittest.main()
