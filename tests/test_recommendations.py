"""Recommended Studio settings (``backend_service/helpers/recommendations.py``).

Covers: no recommendation for a setting the variant's engine cannot use (every catalog variant,
several machines), the table lookups per hardware bucket, measured versus estimated labelling
(bucket and chip must both match), and the payload shape, including through the real image and
video catalog builders. Nothing loads a model; Hub metadata lookups are patched out.
"""

from __future__ import annotations

import unittest
from unittest import mock

from backend_service.catalog import IMAGE_MODEL_FAMILIES, VIDEO_MODEL_FAMILIES
from backend_service.helpers import recommendations as rec_mod
from backend_service.helpers.hardware_class import HardwareClass
from backend_service.helpers.recommendations import (
    IMAGE_KEYS,
    VIDEO_KEYS,
    accepted_settings,
    annotate_recommendations,
    recommended_settings,
)

APPLE_64 = HardwareClass("apple", "apple-64", "Apple Silicon, 64 GB", 64.0, 51.8)
APPLE_16 = HardwareClass("apple", "apple-16", "Apple Silicon, 16 GB", 16.0, 10.7)
CUDA_16 = HardwareClass("cuda", "cuda-16", "NVIDIA GPU, 16 GB VRAM", 32.0, 16.0)
CUDA_48 = HardwareClass("cuda", "cuda-48", "NVIDIA GPU, 48 GB VRAM", 128.0, 48.0)
CPU_32 = HardwareClass("cpu", "cpu", "CPU only, 32 GB RAM", 32.0, 19.2)
MACHINES = (APPLE_64, APPLE_16, CUDA_16, CUDA_48, CPU_32)
CHIP = "Apple M4 Max"

# Settings that exist only in the diffusers pipelines.
DIFFUSERS_ONLY = {"cacheStrategy", "cacheThreshold", "cfgDecay", "previewVae", "sampler", "fp8LayerwiseCasting"}


def _variants(families):
    for family in families:
        yield from family["variants"]


def _variant(families, variant_id):
    return next(v for v in _variants(families) if v["id"] == variant_id)


class EngineAcceptsOnlyItsSettingsTests(unittest.TestCase):
    def test_every_catalog_recommendation_is_a_setting_the_engine_reads(self):
        for kind, families, keys in (("image", IMAGE_MODEL_FAMILIES, IMAGE_KEYS), ("video", VIDEO_MODEL_FAMILIES, VIDEO_KEYS)):
            for hardware in MACHINES:
                for variant in _variants(families):
                    with self.subTest(kind=kind, variant=variant["id"], bucket=hardware.bucket):
                        rec = recommended_settings(variant, kind, hardware, chip=CHIP)
                        if rec is None:
                            continue
                        accepted = accepted_settings(variant, kind, hardware)
                        self.assertLessEqual(set(rec["settings"]), accepted)
                        self.assertLessEqual(set(rec["settings"]), set(keys))

    def test_mflux_sdcpp_and_ltx2_mlx_never_get_diffusers_only_settings(self):
        external = [
            ("image", v) for v in _variants(IMAGE_MODEL_FAMILIES) if v.get("engine") in {"mflux", "sdcpp"}
        ] + [
            ("video", v) for v in _variants(VIDEO_MODEL_FAMILIES) if "ltx-2-mlx" in str(v.get("runtime") or "")
        ]
        self.assertGreater(len(external), 5)
        for kind, variant in external:
            for hardware in MACHINES:
                with self.subTest(variant=variant["id"], bucket=hardware.bucket):
                    self.assertFalse(accepted_settings(variant, kind, hardware) & DIFFUSERS_ONLY)

    def test_mflux_models_without_a_guidance_input_get_no_guidance(self):
        from backend_service.image_runtime.mflux_engine import mflux_family_for_repo

        for variant in _variants(IMAGE_MODEL_FAMILIES):
            family = mflux_family_for_repo(str(variant.get("repo") or "")) if variant.get("engine") == "mflux" else None
            if family is None or family.takes_guidance:
                continue
            with self.subTest(variant=variant["id"]):
                rec = recommended_settings(variant, "image", APPLE_64, chip=CHIP)
                self.assertNotIn("guidance", (rec or {}).get("settings", {}))

    def test_ltx2_mlx_gets_only_size_frames_and_fps(self):
        variant = _variant(VIDEO_MODEL_FAMILIES, "dgrauet/ltx-2.3-mlx-q4")
        rec = recommended_settings(variant, "video", APPLE_64, chip=CHIP)
        self.assertEqual(set(rec["settings"]), {"width", "height", "numFrames", "fps"})

    def test_a_setting_outside_the_engine_is_filtered_even_if_the_table_has_it(self):
        bogus = rec_mod._Entry(
            "image", ("speridlabs/iris-3b-mflux",), {"cacheStrategy": "fbcache", "cfgDecay": True}, {"cacheStrategy": "x"},
        )
        with mock.patch.object(rec_mod, "_ENTRIES", rec_mod._ENTRIES + (bogus,)):
            rec = recommended_settings(_variant(IMAGE_MODEL_FAMILIES, "speridlabs/iris-3b-mflux"), "image", APPLE_64, chip=CHIP)
        self.assertNotIn("cacheStrategy", rec["settings"])
        self.assertNotIn("cfgDecay", rec["settings"])

    def test_teacache_is_never_recommended_for_wan_and_a_threshold_needs_a_strategy(self):
        wan = _variant(VIDEO_MODEL_FAMILIES, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        teacache = rec_mod._Entry("video", (wan["id"],), {"cacheStrategy": "teacache", "cacheThreshold": 0.4}, {})
        with mock.patch.object(rec_mod, "_ENTRIES", rec_mod._ENTRIES + (teacache,)):
            rec = recommended_settings(wan, "video", APPLE_64, chip=CHIP)
        self.assertNotIn("cacheStrategy", rec["settings"])
        self.assertNotIn("cacheThreshold", rec["settings"])

    def test_unet_gets_a_sampler_and_no_cache_flow_match_gets_cache(self):
        sdxl = _variant(IMAGE_MODEL_FAMILIES, "stabilityai/stable-diffusion-xl-base-1.0")
        flux = _variant(IMAGE_MODEL_FAMILIES, "black-forest-labs/FLUX.1-dev")
        self.assertIn("sampler", accepted_settings(sdxl, "image", APPLE_64))
        self.assertNotIn("cacheStrategy", accepted_settings(sdxl, "image", APPLE_64))
        self.assertNotIn("cfgDecay", accepted_settings(sdxl, "image", APPLE_64))
        self.assertIn("cacheStrategy", accepted_settings(flux, "image", APPLE_64))
        self.assertIn("cfgDecay", accepted_settings(flux, "image", APPLE_64))
        self.assertNotIn("sampler", accepted_settings(flux, "image", APPLE_64))


class TableLookupTests(unittest.TestCase):
    def test_distilled_video_rows_get_their_tuned_steps_and_no_guidance(self):
        # The Video Studio sends 30 steps / guidance 5 and the backend substitutes the
        # catalog values only for the request schema defaults (50 / 3.0).
        variant = _variant(VIDEO_MODEL_FAMILIES, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers-causvid")
        rec = recommended_settings(variant, "video", APPLE_64, chip=CHIP)
        self.assertEqual(rec["settings"]["steps"], 4)
        self.assertEqual(rec["settings"]["guidance"], 1.0)
        self.assertNotIn("cacheStrategy", rec["settings"])

    def test_guidance_below_one_is_raised_to_the_studio_minimum(self):
        variant = _variant(IMAGE_MODEL_FAMILIES, "leejet/FLUX.1-schnell-gguf-q4k")  # cfgOverride 0.0
        rec = recommended_settings(variant, "image", APPLE_64, chip=CHIP)
        self.assertEqual(rec["settings"]["guidance"], 1.0)

    def test_fp8_only_on_small_cuda_cards(self):
        flux = _variant(IMAGE_MODEL_FAMILIES, "black-forest-labs/FLUX.1-dev")
        self.assertTrue(recommended_settings(flux, "image", CUDA_16, chip="")["settings"].get("fp8LayerwiseCasting"))
        self.assertNotIn("fp8LayerwiseCasting", recommended_settings(flux, "image", CUDA_48, chip="")["settings"])
        self.assertNotIn("fp8LayerwiseCasting", recommended_settings(flux, "image", APPLE_64, chip=CHIP)["settings"])

    def test_cpu_hosts_get_a_small_size_for_diffusers_images(self):
        sd35 = _variant(IMAGE_MODEL_FAMILIES, "stabilityai/stable-diffusion-3.5-medium")
        cpu = recommended_settings(sd35, "image", CPU_32, chip="")
        mac = recommended_settings(sd35, "image", APPLE_64, chip=CHIP)
        self.assertEqual((cpu["settings"]["width"], cpu["settings"]["height"]), (512, 512))
        self.assertNotIn("width", mac["settings"])

    def test_bucket_and_usable_memory_entries(self):
        entry = rec_mod._Entry("image", ("x",), {"steps": 3}, {"steps": "r"}, hardware="apple-64")
        small = rec_mod._Entry("image", ("x",), {"steps": 2}, {"steps": "r"}, hardware="apple", below_usable_gb=20)
        variant = {"id": "x", "repo": "x/y"}
        self.assertTrue(entry.matches(variant, "image", APPLE_64))
        self.assertFalse(entry.matches(variant, "image", APPLE_16))
        self.assertFalse(entry.matches(variant, "video", APPLE_64))
        self.assertTrue(small.matches(variant, "image", APPLE_16))
        self.assertFalse(small.matches(variant, "image", APPLE_64))
        self.assertFalse(small.matches(variant, "image", CUDA_16))

    def test_entries_match_variant_ids_not_shared_repos(self):
        # The FLUX.1-dev LoRA rows share the base repo but are tuned for 8 steps.
        lora = _variant(IMAGE_MODEL_FAMILIES, "black-forest-labs/FLUX.1-dev-hyper-sd-8step")
        rec = recommended_settings(lora, "image", APPLE_64, chip=CHIP)
        self.assertEqual(rec["settings"]["steps"], 8)
        self.assertNotIn("cacheStrategy", rec["settings"])

    def test_sizes_are_clamped_to_the_variant_limits(self):
        variant = dict(_variant(IMAGE_MODEL_FAMILIES, "speridlabs/iris-3b-mflux"), maxSide=900)
        rec = recommended_settings(variant, "image", APPLE_64, chip=CHIP)
        self.assertEqual(rec["settings"]["width"], 900)
        variant = dict(variant, maxSide=4096, minSide=1100)
        rec = recommended_settings(variant, "image", APPLE_64, chip=CHIP)
        self.assertEqual(rec["settings"]["width"], 1100)

    def test_a_model_with_nothing_to_recommend_returns_none(self):
        self.assertIsNone(recommended_settings({"id": "nobody/nothing", "repo": "nobody/nothing"}, "image", APPLE_64, chip=CHIP))

    def test_kind_must_be_image_or_video(self):
        with self.assertRaises(ValueError):
            recommended_settings({"id": "x"}, "chat", APPLE_64, chip=CHIP)


class MeasuredBasisTests(unittest.TestCase):
    VARIANT = {"id": "speridlabs/iris-3b-mflux", "repo": "speridlabs/iris-3b", "engine": "mflux", "name": "Iris", "defaultSteps": 4}

    def _measured(self, keys=frozenset({"steps"})):
        return {
            self.VARIANT["id"]: (
                rec_mod._Measurement("apple-64", CHIP, frozenset(keys), {"steps": "4 steps: 11 s (measured)."}),
            )
        }

    def test_measured_needs_the_same_bucket_and_chip(self):
        with mock.patch.object(rec_mod, "_MEASURED", self._measured()):
            same = recommended_settings(self.VARIANT, "image", APPLE_64, chip=CHIP)
            other_chip = recommended_settings(self.VARIANT, "image", APPLE_64, chip="Apple M1 Max")
            unknown_chip = recommended_settings(self.VARIANT, "image", APPLE_64, chip="")
            other_bucket = recommended_settings(self.VARIANT, "image", APPLE_16, chip=CHIP)
        self.assertEqual(same["basisByKey"]["steps"], "measured")
        self.assertEqual(same["reasons"]["steps"], "4 steps: 11 s (measured).")
        self.assertEqual(same["measuredOn"], "Apple M4 Max, 64 GB")
        for rec in (other_chip, unknown_chip, other_bucket):
            self.assertEqual(rec["basisByKey"]["steps"], "estimated")
            self.assertEqual(rec["basis"], "estimated")
            self.assertIsNone(rec["measuredOn"])

    def test_overall_basis_is_measured_only_when_every_setting_is(self):
        with mock.patch.object(rec_mod, "_MEASURED", self._measured({"steps"})):
            partial = recommended_settings(self.VARIANT, "image", APPLE_64, chip=CHIP)
        with mock.patch.object(rec_mod, "_MEASURED", self._measured({"steps", "width", "height"})):
            full = recommended_settings(self.VARIANT, "image", APPLE_64, chip=CHIP)
        self.assertEqual(set(partial["settings"]), {"steps", "width", "height"})
        self.assertEqual(partial["basis"], "estimated")
        self.assertEqual(full["basis"], "measured")

    def test_without_measurements_everything_is_estimated(self):
        with mock.patch.object(rec_mod, "_MEASURED", {}):
            rec = recommended_settings(self.VARIANT, "image", APPLE_64, chip=CHIP)
        self.assertEqual(set(rec["basisByKey"].values()), {"estimated"})

    def test_recorded_measurements_describe_settings_the_table_recommends(self):
        # Every measured key must be a setting the variant actually gets on that bucket,
        # with a reason that states the timing.
        for variant_id, measurements in rec_mod._MEASURED.items():
            kind = "video" if any(v["id"] == variant_id for v in _variants(VIDEO_MODEL_FAMILIES)) else "image"
            families = VIDEO_MODEL_FAMILIES if kind == "video" else IMAGE_MODEL_FAMILIES
            variant = _variant(families, variant_id)
            for measurement in measurements:
                hardware = HardwareClass("apple", measurement.bucket, "x", float(measurement.bucket.split("-")[1]), 51.8)
                rec = recommended_settings(variant, kind, hardware, chip=measurement.chip)
                with self.subTest(variant=variant_id):
                    self.assertLessEqual(set(measurement.keys), set(rec["settings"]))
                    for key in measurement.keys:
                        self.assertEqual(rec["basisByKey"][key], "measured")
                        self.assertRegex(rec["reasons"][key], r"\d")


class PayloadShapeTests(unittest.TestCase):
    def _check(self, rec):
        self.assertEqual(
            set(rec), {"hardwareBucket", "hardwareLabel", "basis", "basisByKey", "measuredOn", "settings", "reasons"},
        )
        self.assertIn(rec["basis"], {"measured", "estimated"})
        self.assertEqual(set(rec["basisByKey"]), set(rec["settings"]))
        self.assertTrue(set(rec["basisByKey"].values()) <= {"measured", "estimated"})
        self.assertEqual(set(rec["reasons"]), set(rec["settings"]))
        for reason in rec["reasons"].values():
            self.assertTrue(reason and isinstance(reason, str))

    def test_every_catalog_recommendation_has_the_contract_shape(self):
        for kind, families in (("image", IMAGE_MODEL_FAMILIES), ("video", VIDEO_MODEL_FAMILIES)):
            for hardware in MACHINES:
                for variant in _variants(families):
                    rec = recommended_settings(variant, kind, hardware, chip=CHIP)
                    if rec is not None:
                        with self.subTest(variant=variant["id"], bucket=hardware.bucket):
                            self._check(rec)
                            self.assertEqual(rec["hardwareBucket"], hardware.bucket)

    def test_annotate_copies_and_sets_every_variant(self):
        families = [{"id": "f", "variants": [{"id": "speridlabs/iris-3b-mflux", "repo": "speridlabs/iris-3b", "engine": "mflux"}, {"id": "none", "repo": "none"}]}]
        out = annotate_recommendations(families, "image", APPLE_64, chip=CHIP)
        self.assertNotIn("recommendedSettings", families[0]["variants"][0])
        self.assertIsNotNone(out[0]["variants"][0]["recommendedSettings"])
        self.assertIsNone(out[0]["variants"][1]["recommendedSettings"])

    def test_catalog_payload_builders_attach_recommendations(self):
        from backend_service.helpers import images as images_mod
        from backend_service.helpers import video as video_mod

        # The MLX-only families are dropped off Apple Silicon; keep them so the test runs anywhere.
        with mock.patch.object(images_mod, "_image_repo_live_metadata", return_value={}), \
                mock.patch.object(video_mod, "_image_repo_live_metadata", return_value={}), \
                mock.patch.object(images_mod, "is_apple_silicon", return_value=True), \
                mock.patch.object(video_mod, "is_apple_silicon", return_value=True), \
                mock.patch.object(rec_mod, "host_chip", return_value=CHIP):
            image_families = images_mod._image_model_payloads([], APPLE_64)
            video_families = video_mod._video_model_payloads([], APPLE_64)
        for families in (image_families, video_families):
            variants = list(_variants(families))
            self.assertTrue(variants)
            self.assertTrue(all("recommendedSettings" in v for v in variants))
            for v in variants:
                if v["recommendedSettings"] is not None:
                    self._check(v["recommendedSettings"])
        iris = next(v for v in _variants(image_families) if v["id"] == "speridlabs/iris-3b-mflux")
        self.assertEqual(iris["recommendedSettings"]["settings"]["steps"], 4)


if __name__ == "__main__":
    unittest.main()
