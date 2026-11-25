"""The structured licence on text, image and video catalogue rows (D0135): every row that has one has a valid shape,
the known-restricted rows say so, and the catalogue routes carry the object to the screens.

No model is downloaded and nothing runs; live Hugging Face lookups are stubbed out.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.catalog import (
    IMAGE_MODEL_FAMILIES,
    LATEST_IMAGE_TRACKED_SEEDS,
    MODEL_FAMILIES,
    VIDEO_MODEL_FAMILIES,
    licences,
)
from backend_service.catalog.video_upscale_models import VIDEO_UPSCALE_MODELS

H = {"Authorization": "Bearer app-token"}
REQUIRED = {"id", "name", "commercial", "gated", "url", "summary"}


def _rows(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [v for f in families for v in f["variants"]]


def _check_shape(case: unittest.TestCase, row_id: str, lic: Any) -> None:
    case.assertIsInstance(lic, dict, row_id)
    case.assertTrue(REQUIRED <= set(lic), f"{row_id}: missing {REQUIRED - set(lic)}")
    case.assertIn(lic["commercial"], licences.COMMERCIAL_VALUES, row_id)
    case.assertIsInstance(lic["gated"], bool, row_id)
    case.assertTrue(str(lic["url"]).startswith("https://"), f"{row_id}: {lic['url']}")
    case.assertTrue(str(lic["name"]).strip() and str(lic["id"]).strip(), row_id)
    case.assertTrue(str(lic["summary"]).strip().endswith("."), f"{row_id}: summary is not one sentence")


class LicenceShapeTests(unittest.TestCase):
    def test_every_licence_a_row_carries_has_a_valid_shape(self):
        seen = 0
        sources = {
            "text": _rows(MODEL_FAMILIES),
            "image": _rows(IMAGE_MODEL_FAMILIES),
            "video": _rows(VIDEO_MODEL_FAMILIES),
            "upscale": VIDEO_UPSCALE_MODELS,
        }
        for kind, rows in sources.items():
            for row in rows:
                if row.get("licence") is None:
                    continue
                _check_shape(self, f"{kind}:{row['id']}", row["licence"])
                seen += 1
        for seed in LATEST_IMAGE_TRACKED_SEEDS:
            if seed.get("licence") is not None:
                _check_shape(self, f"seed:{seed['repo']}", seed["licence"])
                seen += 1
        self.assertGreater(seen, 200)

    def test_every_text_image_and_video_row_has_one(self):
        # The notices table covers every family in these catalogues; a new row without one is a gap to fill or
        # to list as unknown, so it fails here until someone decides.
        missing = [
            f"{kind}:{row['id']}"
            for kind, families in (("text", MODEL_FAMILIES), ("image", IMAGE_MODEL_FAMILIES), ("video", VIDEO_MODEL_FAMILIES))
            for row in _rows(families)
            if not row.get("licence")
        ]
        self.assertEqual(missing, [])

    def test_the_commercial_word_follows_the_licence(self):
        by_id = {r["id"]: r["licence"] for fams in (MODEL_FAMILIES, IMAGE_MODEL_FAMILIES, VIDEO_MODEL_FAMILIES) for r in _rows(fams)}
        # FLUX.1 dev is non-commercial and gated, whatever Hugging Face reports as "other".
        self.assertEqual(by_id["black-forest-labs/FLUX.1-dev"]["commercial"], "no")
        self.assertTrue(by_id["black-forest-labs/FLUX.1-dev"]["gated"])
        # A conversion keeps the upstream licence but is not itself gated unless its notes say so.
        self.assertEqual(by_id["leejet/FLUX.1-dev-gguf-q4k"]["commercial"], "no")
        self.assertFalse(by_id["leejet/FLUX.1-dev-gguf-q4k"]["gated"])
        self.assertEqual(by_id["Qwen/Qwen2.5-3B-Instruct"]["name"], "Qwen Research Licence")
        self.assertEqual(by_id["Qwen/Qwen2.5-7B-Instruct"]["name"], "Apache 2.0")
        self.assertEqual(by_id["Qwen/Qwen-Image-2.1-mflux"]["commercial"], "no")
        self.assertEqual(by_id["Qwen/Qwen-Image"]["commercial"], "yes")
        self.assertEqual(by_id["black-forest-labs/FLUX.2-klein-4B"]["name"], "Apache 2.0")
        self.assertEqual(by_id["black-forest-labs/FLUX.2-dev"]["commercial"], "no")
        self.assertEqual(by_id["meta-llama/Llama-3.3-70B-Instruct"]["gated"], True)
        self.assertEqual(by_id["mlx-community/Llama-3.3-70B-Instruct-4bit"]["name"], "Llama 3.3 Community License")
        self.assertEqual(by_id["THUDM/CogVideoX-2b"]["name"], "Apache 2.0")
        self.assertEqual(by_id["THUDM/CogVideoX-5b"]["commercial"], "limited")
        self.assertTrue(by_id["Lightricks/LTX-2.5-Diffusers"]["gated"])
        self.assertTrue(by_id["dgrauet/ltx-2.5-mlx-q4"]["gated"])
        self.assertEqual(by_id["MiniMaxAI/MiniMax-M2.7"]["commercial"], "no")

    def test_the_licence_objects_are_not_shared_across_kinds_by_mistake(self):
        # as_gated / as_open / with_url return copies; the constants stay as defined.
        self.assertFalse(licences.APACHE_2_0["gated"])
        self.assertTrue(licences.as_gated(licences.APACHE_2_0)["gated"])
        self.assertFalse(licences.APACHE_2_0["gated"])
        self.assertFalse(licences.as_open(licences.FLUX_1_DEV_NC)["gated"])
        self.assertTrue(licences.FLUX_1_DEV_NC["gated"])

    def test_the_h3_row_keeps_its_own_object(self):
        h3 = next(r for r in _rows(VIDEO_MODEL_FAMILIES) if r["id"].startswith("leejet/MiniMax-H3"))
        self.assertTrue(h3["licence"]["territoryRestricted"])
        _check_shape(self, h3["id"], h3["licence"])


class _ClientCase(unittest.TestCase):
    def setUp(self):
        for patcher in (
            mock.patch("backend_service.helpers.video._image_repo_live_metadata", lambda repo: {}),
            mock.patch("backend_service.helpers.images._image_repo_live_metadata", lambda repo: {}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = TestClient(create_app(api_token="app-token"))

    def _variants(self, families: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [v for f in families for v in f.get("variants") or []]


class RoutePayloadTests(_ClientCase):
    def test_the_image_catalogue_route_carries_it(self):
        payload = self.client.get("/api/images/catalog", headers=H).json()
        variants = self._variants(payload["families"])
        self.assertTrue(variants)
        for variant in variants:
            _check_shape(self, variant["id"], variant.get("licence"))
        dev = next(v for v in variants if v["id"] == "black-forest-labs/FLUX.1-dev")
        self.assertEqual(dev["licence"]["commercial"], "no")
        for row in payload["latest"]:
            if row.get("licence") is not None:
                _check_shape(self, str(row.get("repo")), row["licence"])

    def test_the_tracked_latest_rows_carry_it(self):
        payload = self.client.get("/api/images/catalog", headers=H).json()
        with_licence = [row for row in payload["latest"] if row.get("licence")]
        self.assertTrue(with_licence, "no tracked latest image row carries a licence")

    def test_the_video_catalogue_route_carries_it(self):
        with mock.patch("backend_service.state.ChaosEngineState._library", return_value=[]):
            payload = self.client.get("/api/video/catalog", headers=H).json()
        variants = self._variants(payload["families"])
        self.assertTrue(variants)
        for variant in variants:
            _check_shape(self, variant["id"], variant.get("licence"))

    def test_the_text_catalogue_in_the_workspace_carries_it(self):
        payload = self.client.get("/api/workspace", headers=H).json()
        variants = self._variants(payload["featuredModels"])
        self.assertGreater(len(variants), 100)
        for variant in variants:
            _check_shape(self, variant["id"], variant.get("licence"))


class E2eCheckTests(_ClientCase):
    """The suite's check (scripts/e2e_phases/model_licences.py) against the in-process routes."""

    def setUp(self):
        super().setUp()
        import importlib.util
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        path = Path(__file__).resolve().parents[1] / "scripts" / "e2e_phases" / "model_licences.py"
        spec = importlib.util.spec_from_file_location("e2e_model_licences_for_test", path)
        self.phase = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.phase
        spec.loader.exec_module(self.phase)
        routes = {"image-catalog": "/api/images/catalog", "video-catalog": "/api/video/catalog"}

        def cli_json(*args, timeout=0.0):
            route = routes.get(args[0]) or args[2]
            with mock.patch("backend_service.state.ChaosEngineState._library", return_value=[]):
                response = self.client.get(route, headers=H)
            return 0, response.json(), ""

        self.suite = SimpleNamespace(_cli_json=cli_json)

    def test_the_check_passes_for_each_catalogue(self):
        for kind in ("text", "image", "video"):
            status, reason, detail = self.phase.catalog_licences(self.suite, kind)
            self.assertEqual((status, reason), ("pass", ""), kind)
            self.assertGreater(detail["rows"], 20, kind)
            self.assertEqual(detail["yes"] + detail["limited"] + detail["no"], detail["rows"], kind)

    def test_the_check_fails_on_a_row_without_one_and_on_a_bad_value(self):
        original = self.suite._cli_json

        def broken(*args, timeout=0.0):
            rc, payload, err = original(*args, timeout=timeout)
            variants = payload["families"][0]["variants"]
            variants[0].pop("licence")
            variants[1]["licence"] = {**variants[1]["licence"], "commercial": "maybe"}
            return rc, payload, err

        self.suite._cli_json = broken
        status, reason, detail = self.phase.catalog_licences(self.suite, "image")
        self.assertEqual(status, "fail")
        self.assertIn("no licence", reason)
        self.assertIn("'maybe'", reason)
        self.assertEqual(detail["count"], 2)


if __name__ == "__main__":
    unittest.main()
