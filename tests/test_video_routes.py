"""Contract tests for the video generation API.

The video runtime isn't wired yet, so these tests lock in the shape of the
scaffolded endpoints: catalog returns the four planned engines, runtime
reports as unavailable, library/outputs are empty, and generate/preload/
download all surface a 501 until the engine lands.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState

from tests.test_backend_service import (
    TEST_API_TOKEN,
    FakeRuntime,
    fake_library,
    fake_system_snapshot,
)


def make_client() -> tuple[TestClient, tempfile.TemporaryDirectory]:
    tempdir = tempfile.TemporaryDirectory()
    settings_path = Path(tempdir.name) / "settings.json"
    benchmarks_path = Path(tempdir.name) / "benchmark-history.json"
    chat_sessions_path = Path(tempdir.name) / "chat-sessions.json"
    state = ChaosEngineState(
        system_snapshot_provider=fake_system_snapshot,
        library_provider=fake_library,
        settings_path=settings_path,
        benchmarks_path=benchmarks_path,
        chat_sessions_path=chat_sessions_path,
    )
    state.runtime = FakeRuntime()
    client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
    client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})
    return client, tempdir


class VideoCatalogRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_catalog_returns_families_and_latest_shape(self):
        response = self.client.get("/api/video/catalog")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("families", payload)
        self.assertIn("latest", payload)
        self.assertIsInstance(payload["families"], list)
        self.assertIsInstance(payload["latest"], list)

    def test_catalog_surfaces_all_first_wave_engines(self):
        payload = self.client.get("/api/video/catalog").json()
        family_ids = {family["id"] for family in payload["families"]}
        # First-wave candidates — these must appear until we explicitly retire one.
        for expected in ("ltx-video", "wan-2-2", "hunyuan-video", "mochi-1"):
            self.assertIn(expected, family_ids, f"expected {expected} in video catalog")

    def test_catalog_variants_have_frontend_ready_fields(self):
        payload = self.client.get("/api/video/catalog").json()
        for family in payload["families"]:
            self.assertIn("name", family)
            self.assertIn("variants", family)
            self.assertGreater(len(family["variants"]), 0, f"{family['id']} has no variants")
            for variant in family["variants"]:
                # Fields the frontend ImageModelVariant type needs to reuse.
                for key in ("id", "repo", "name", "provider", "sizeGb", "taskSupport"):
                    self.assertIn(key, variant, f"{variant.get('id')} missing {key}")
                self.assertIn("txt2video", variant["taskSupport"])
                self.assertEqual(variant.get("availableLocally"), False)
                self.assertEqual(variant.get("familyName"), family["name"])


class VideoRuntimeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_runtime_reports_unavailable_until_engine_ships(self):
        response = self.client.get("/api/video/runtime")
        self.assertEqual(response.status_code, 200)
        runtime = response.json()["runtime"]
        self.assertEqual(runtime["realGenerationAvailable"], False)
        self.assertEqual(runtime["activeEngine"], "placeholder")
        self.assertIsNone(runtime["loadedModelRepo"])
        self.assertIn("diffusers", runtime["missingDependencies"])

    def test_library_is_empty_until_downloads_land(self):
        response = self.client.get("/api/video/library")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"models": []})

    def test_outputs_is_empty_until_generation_lands(self):
        response = self.client.get("/api/video/outputs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"outputs": []})


class VideoNotImplementedRouteTests(unittest.TestCase):
    """Generate/preload/download must surface 501 until the runtime ships."""

    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_generate_returns_501(self):
        response = self.client.post("/api/video/generate", json={})
        self.assertEqual(response.status_code, 501)

    def test_preload_returns_501(self):
        response = self.client.post("/api/video/preload", json={})
        self.assertEqual(response.status_code, 501)

    def test_download_returns_501(self):
        response = self.client.post("/api/video/download", json={})
        self.assertEqual(response.status_code, 501)


if __name__ == "__main__":
    unittest.main()
