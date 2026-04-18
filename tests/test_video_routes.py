"""Contract tests for the video generation API.

Routes covered:
- GET  /api/video/catalog      -> always lists the curated engines
- GET  /api/video/runtime      -> delegates to VideoRuntimeManager.capabilities
- GET  /api/video/library      -> filters catalog by local snapshot readiness
- GET  /api/video/outputs      -> empty until generation lands
- POST /api/video/preload      -> 404 unknown, 409 not-installed, 200 happy
- POST /api/video/unload       -> 404 unknown, 200 default
- POST /api/video/generate     -> 501 until generation lands
- POST /api/video/download     -> 501 until downloads land
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState
from backend_service import video_runtime as video_runtime_mod
from backend_service.routes import video as video_routes

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
        for expected in ("ltx-video", "wan-2-2", "hunyuan-video", "mochi-1"):
            self.assertIn(expected, family_ids, f"expected {expected} in video catalog")

    def test_catalog_variants_have_frontend_ready_fields(self):
        payload = self.client.get("/api/video/catalog").json()
        for family in payload["families"]:
            self.assertIn("name", family)
            self.assertIn("variants", family)
            self.assertGreater(len(family["variants"]), 0, f"{family['id']} has no variants")
            for variant in family["variants"]:
                for key in ("id", "repo", "name", "provider", "sizeGb", "taskSupport"):
                    self.assertIn(key, variant, f"{variant.get('id')} missing {key}")
                self.assertIn("txt2video", variant["taskSupport"])
                # availableLocally should be False on a fresh test env (no snapshots).
                self.assertEqual(variant.get("availableLocally"), False)
                self.assertEqual(variant.get("familyName"), family["name"])


class VideoRuntimeRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_runtime_delegates_to_video_runtime_manager(self):
        response = self.client.get("/api/video/runtime")
        self.assertEqual(response.status_code, 200)
        runtime = response.json()["runtime"]
        # Shape is fixed regardless of whether diffusers is installed.
        for key in ("activeEngine", "realGenerationAvailable", "message"):
            self.assertIn(key, runtime)

    def test_runtime_reports_placeholder_when_core_deps_missing(self):
        with mock.patch.object(
            video_runtime_mod,
            "_find_missing",
            side_effect=[["diffusers", "torch"], []],
        ):
            runtime = self.client.get("/api/video/runtime").json()["runtime"]
        self.assertFalse(runtime["realGenerationAvailable"])
        self.assertEqual(runtime["activeEngine"], "placeholder")

    def test_library_is_empty_when_no_snapshots_are_installed(self):
        response = self.client.get("/api/video/library")
        self.assertEqual(response.status_code, 200)
        # Fresh test env: no local snapshots, so every variant is excluded.
        self.assertEqual(response.json(), {"models": []})

    def test_outputs_is_empty_until_generation_lands(self):
        response = self.client.get("/api/video/outputs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"outputs": []})


class VideoPreloadRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_preload_returns_404_for_unknown_model(self):
        response = self.client.post("/api/video/preload", json={"modelId": "ghost/nope"})
        self.assertEqual(response.status_code, 404)

    def test_preload_returns_409_when_model_not_installed_locally(self):
        response = self.client.post(
            "/api/video/preload",
            json={"modelId": "Lightricks/LTX-Video"},
        )
        # Nothing is downloaded in this test env — expect 409 Conflict.
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertTrue(
            "not installed" in detail.lower() or "did not produce" in detail.lower(),
            f"unexpected detail: {detail}",
        )

    def test_preload_happy_path_returns_runtime_status(self):
        # Pretend the model is installed and diffusers is ready; monkey-patch
        # the engine preload seam so we never actually load weights.
        with mock.patch.object(
            video_routes,
            "_video_variant_available_locally",
            return_value=True,
        ), mock.patch.object(video_runtime_mod, "_find_missing", return_value=[]), \
                mock.patch.object(
                    video_runtime_mod.DiffusersVideoEngine,
                    "_ensure_pipeline",
                    return_value=mock.MagicMock(),
                ):
            response = self.client.post(
                "/api/video/preload",
                json={"modelId": "Lightricks/LTX-Video"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        runtime = response.json()["runtime"]
        self.assertTrue(runtime["realGenerationAvailable"])


class VideoUnloadRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_unload_no_body_returns_runtime_status(self):
        response = self.client.post("/api/video/unload")
        self.assertEqual(response.status_code, 200)
        self.assertIn("runtime", response.json())

    def test_unload_unknown_model_returns_404(self):
        response = self.client.post("/api/video/unload", json={"modelId": "ghost/nope"})
        self.assertEqual(response.status_code, 404)


class VideoNotImplementedRouteTests(unittest.TestCase):
    """Generate + download must still surface 501 in this phase."""

    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_generate_returns_501(self):
        response = self.client.post("/api/video/generate", json={})
        self.assertEqual(response.status_code, 501)

    def test_download_returns_501(self):
        response = self.client.post("/api/video/download", json={})
        self.assertEqual(response.status_code, 501)


if __name__ == "__main__":
    unittest.main()
