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
        for expected in ("ltx-video", "wan-2-1", "wan-2-2", "hunyuan-video", "mochi-1"):
            self.assertIn(expected, family_ids, f"expected {expected} in video catalog")

    def test_catalog_includes_wan_2_1_small_starter_variant(self):
        """The 1.3B variant is our recommended first-download target — make sure it's visible."""
        payload = self.client.get("/api/video/catalog").json()
        variant_ids = {
            variant["id"]
            for family in payload["families"]
            for variant in family["variants"]
        }
        self.assertIn("Wan-AI/Wan2.1-T2V-1.3B", variant_ids)
        # Sanity-check: the 1.3B is materially smaller than the 14B / A14B options.
        wan_21_family = next(family for family in payload["families"] if family["id"] == "wan-2-1")
        sizes = {variant["id"]: variant["sizeGb"] for variant in wan_21_family["variants"]}
        self.assertLess(sizes["Wan-AI/Wan2.1-T2V-1.3B"], sizes["Wan-AI/Wan2.1-T2V-14B"])

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
    """Generate still surfaces 501 until the generation loop lands."""

    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_generate_returns_501(self):
        response = self.client.post("/api/video/generate", json={})
        self.assertEqual(response.status_code, 501)


class VideoDownloadRouteTests(unittest.TestCase):
    """Contract tests for /api/video/download* endpoints.

    We deliberately do not exercise a real HF snapshot_download from a unit
    test — that's an integration test path, taken by pulling ``Wan-AI/Wan2.1-T2V-1.3B``
    via the real endpoint. These tests assert the validation surface and the
    shape of the ``not_found`` / ``deleted`` paths which can be exercised
    without any network or subprocess activity.
    """

    def setUp(self) -> None:
        self.client, self.tempdir = make_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_download_rejects_unknown_repo(self):
        response = self.client.post("/api/video/download", json={"repo": "ghost/nope"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("video model catalog", response.json()["detail"])

    def test_download_requires_repo_field(self):
        response = self.client.post("/api/video/download", json={})
        # Pydantic validation rejects the missing repo field before the route runs.
        self.assertEqual(response.status_code, 422)

    def test_download_status_is_empty_when_nothing_is_downloading(self):
        response = self.client.get("/api/video/download/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"downloads": []})

    def test_download_status_filters_out_non_video_repos(self):
        # Seed an unrelated repo directly in state — /api/video/download/status
        # should ignore it so the Video UI never sees image/text download rows.
        state = self.client.app.state.chaosengine
        state._downloads["fake/other-repo"] = {
            "repo": "fake/other-repo",
            "state": "downloading",
            "progress": 0.5,
            "downloadedGb": 1.0,
            "totalGb": 2.0,
            "error": None,
        }
        response = self.client.get("/api/video/download/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"downloads": []})

    def test_download_status_surfaces_video_repos(self):
        # Seed a valid video repo directly so we don't need to call the real
        # start_download (which would hit Hugging Face). The filter must keep it.
        state = self.client.app.state.chaosengine
        state._downloads["Wan-AI/Wan2.1-T2V-1.3B"] = {
            "repo": "Wan-AI/Wan2.1-T2V-1.3B",
            "state": "downloading",
            "progress": 0.25,
            "downloadedGb": 0.6,
            "totalGb": 2.5,
            "error": None,
        }
        response = self.client.get("/api/video/download/status")
        self.assertEqual(response.status_code, 200)
        downloads = response.json()["downloads"]
        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0]["repo"], "Wan-AI/Wan2.1-T2V-1.3B")

    def test_cancel_rejects_repo_outside_video_catalog(self):
        response = self.client.post(
            "/api/video/download/cancel",
            json={"repo": "ghost/nope"},
        )
        self.assertEqual(response.status_code, 404)

    def test_cancel_known_but_not_downloading_returns_not_found_state(self):
        response = self.client.post(
            "/api/video/download/cancel",
            json={"repo": "Wan-AI/Wan2.1-T2V-1.3B"},
        )
        self.assertEqual(response.status_code, 200)
        download = response.json()["download"]
        self.assertEqual(download["state"], "not_found")

    def test_delete_rejects_repo_outside_video_catalog(self):
        response = self.client.post(
            "/api/video/download/delete",
            json={"repo": "ghost/nope"},
        )
        self.assertEqual(response.status_code, 404)

    def test_delete_known_but_not_downloaded_is_noop(self):
        response = self.client.post(
            "/api/video/download/delete",
            json={"repo": "Wan-AI/Wan2.1-T2V-1.3B"},
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertEqual(result["state"], "not_found")


if __name__ == "__main__":
    unittest.main()
