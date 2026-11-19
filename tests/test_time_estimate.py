"""FU-165: Studio time estimates scale with size, frames and steps (shared cases with src/utils/timeEstimate.ts)."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from backend_service.helpers.time_estimate import scaled_seconds, variant_seconds

CASES = json.loads((Path(__file__).parent / "fixtures" / "time_estimate_cases.json").read_text())["cases"]
REF = {"width": 1024, "height": 1024, "frames": 1, "steps": 8}
VREF = {"width": 768, "height": 512, "frames": 97, "steps": 8}


def _run(case: dict) -> float | None:
    req = case["request"]
    variant = {"estimatedGenerationSeconds": case["catalog"], "estimateRef": case["ref"], "engine": case.get("engine")}
    return variant_seconds(variant, case["kind"], req["width"], req["height"], req["frames"], req["steps"])


class SharedCasesTest(unittest.TestCase):
    def test_every_case_matches(self) -> None:
        self.assertGreaterEqual(len(CASES), 10)
        for case in CASES:
            with self.subTest(case["name"]):
                got = _run(case)
                if case["expected"] is None:
                    self.assertIsNone(got)
                else:
                    self.assertTrue(math.isclose(got, case["expected"], rel_tol=1e-9), (got, case["expected"]))

    def test_measured_cases_within_tolerance(self) -> None:
        for case in CASES:
            if "measured" not in case:
                continue
            with self.subTest(case["name"]):
                error = abs(case["expected"] - case["measured"]) / case["measured"] * 100
                self.assertLessEqual(error, case["tolerancePct"])


class EdgeTest(unittest.TestCase):
    def test_missing_reference_is_flat(self) -> None:
        self.assertEqual(scaled_seconds(40, None, 512, 512, 1, 4), 40.0)
        self.assertEqual(scaled_seconds(40, {}, 512, 512, 1, 4), 40.0)
        self.assertEqual(scaled_seconds(40, "junk", 512, 512, 1, 4), 40.0)

    def test_bad_catalog_is_none(self) -> None:
        for catalog in (None, 0, -3, "12", True, float("nan"), float("inf")):
            self.assertIsNone(scaled_seconds(catalog, REF, 512, 512, 1, 4))

    def test_zero_and_bad_inputs_never_raise(self) -> None:
        for args in ((0, 512, 1, 4), (512, 0, 1, 4), (None, None, None, None), ("a", "b", "c", "d"), (-5, 512, -1, -2)):
            got = scaled_seconds(40, REF, *args)
            self.assertIsInstance(got, float)
            self.assertGreater(got, 0)

    def test_huge_inputs_stay_finite(self) -> None:
        got = scaled_seconds(40, REF, 10**6, 10**6, 10**6, 10**6)
        self.assertTrue(math.isfinite(got))
        self.assertEqual(scaled_seconds(40, REF, 10**400, 512, 1, 4), 40.0)

    def test_monotonic_in_size_frames_and_steps(self) -> None:
        sizes = [scaled_seconds(100, VREF, s, s, 97, 8) for s in (256, 512, 768, 1024, 1536)]
        frames = [scaled_seconds(100, VREF, 768, 512, f, 8) for f in (9, 25, 49, 97, 193, 257)]
        steps = [scaled_seconds(100, VREF, 768, 512, 97, s) for s in (1, 4, 8, 16, 50)]
        for series in (sizes, frames, steps):
            self.assertEqual(series, sorted(series))
            self.assertLess(series[0], series[-1])

    def test_fixed_share_is_the_floor(self) -> None:
        ref = {**REF, "fixed": 0.5}
        self.assertAlmostEqual(scaled_seconds(40, ref, 16, 16, 1, 1), 20.0, delta=0.2)

    def test_ignores_steps(self) -> None:
        ref = {**VREF, "ignoresSteps": True}
        self.assertEqual(scaled_seconds(100, ref, 768, 512, 97, 50), 100.0)

    def test_variant_none(self) -> None:
        self.assertIsNone(variant_seconds(None, "image", 512, 512))


def _catalog_rows():
    from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
    from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES

    for kind, families in (("image", IMAGE_MODEL_FAMILIES), ("video", VIDEO_MODEL_FAMILIES)):
        for family in families:
            for variant in family["variants"]:
                yield kind, variant


class CatalogRefTest(unittest.TestCase):
    def test_every_ref_has_four_positive_integers(self) -> None:
        rows = [(kind, v) for kind, v in _catalog_rows() if "estimateRef" in v]
        self.assertGreaterEqual(len(rows), 40)
        for kind, variant in rows:
            with self.subTest(variant["id"]):
                ref = variant["estimateRef"]
                for key in ("width", "height", "frames", "steps"):
                    self.assertIsInstance(ref[key], int)
                    self.assertNotIsInstance(ref[key], bool)
                    self.assertGreater(ref[key], 0)
                self.assertLessEqual(set(ref), {"width", "height", "frames", "steps", "fixed", "quad", "ignoresSteps"})
                if "fixed" in ref:
                    self.assertTrue(0 < ref["fixed"] < 1)
                if kind == "image":
                    self.assertEqual(ref["frames"], 1)
                self.assertIsNotNone(variant.get("estimatedGenerationSeconds"))

    def test_reference_settings_give_the_catalog_figure(self) -> None:
        for kind, variant in _catalog_rows():
            ref = variant.get("estimateRef")
            if not ref:
                continue
            with self.subTest(variant["id"]):
                got = variant_seconds(variant, kind, ref["width"], ref["height"], ref["frames"], ref["steps"])
                self.assertEqual(got, round(float(variant["estimatedGenerationSeconds"]), 1))

    def test_payloads_carry_the_ref(self) -> None:
        from unittest import mock

        from backend_service.helpers.images import _image_model_payloads
        from backend_service.helpers.video import _video_model_payloads

        with mock.patch("backend_service.helpers.video.is_apple_silicon", return_value=False):
            video = _video_model_payloads([])
        image = _image_model_payloads([])
        rows = {v["id"]: v for family in video + image for v in family["variants"]}
        self.assertEqual(rows["Wan-AI/Wan2.2-TI2V-5B-Diffusers"]["estimateRef"]["steps"], 20)
        if "speridlabs/iris-3b-mflux" in rows:  # the mflux rows are listed on Apple Silicon only
            self.assertEqual(rows["speridlabs/iris-3b-mflux"]["estimateRef"]["width"], 1024)


if __name__ == "__main__":
    unittest.main()
