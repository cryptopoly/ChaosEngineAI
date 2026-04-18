"""Contract tests for the video generation API.

Routes covered:
- GET  /api/video/catalog           -> always lists the curated engines
- GET  /api/video/runtime           -> delegates to VideoRuntimeManager.capabilities
- GET  /api/video/library           -> filters catalog by local snapshot readiness
- GET  /api/video/outputs           -> empty until generation lands
- POST /api/video/preload           -> 404 unknown, 409 not-installed, 200 happy
- POST /api/video/unload            -> 404 unknown, 200 default
- POST /api/video/generate          -> 501 until generation lands
- POST /api/video/download          -> 404 unknown, 200 starts HF snapshot download
- GET  /api/video/download/status   -> filters download list to video repos
- POST /api/video/download/cancel   -> 404 unknown, 200 pauses
- POST /api/video/download/delete   -> 404 unknown, 200 wipes cache + evicts runtime

All tests redirect HF_HUB_CACHE into a per-test tempdir so no test can ever
observe or wipe the user's real downloaded model snapshots.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient

from backend_service import app as app_module
from backend_service.app import create_app
from backend_service.state import ChaosEngineState
from backend_service import video_runtime as video_runtime_mod
from backend_service.routes import video as video_routes
from backend_service.video_runtime import GeneratedVideo

from tests.test_backend_service import (
    TEST_API_TOKEN,
    FakeRuntime,
    fake_library,
    fake_system_snapshot,
)


# Environment variables we redirect into the per-test tempdir so the test
# process never observes or touches the user's real Hugging Face cache.
# Critical: without this, a ``delete_download`` test against a valid video
# repo would physically wipe the user's in-progress snapshot on disk.
_HF_CACHE_ENV_VARS: tuple[str, ...] = (
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
)


def make_client() -> tuple[TestClient, tempfile.TemporaryDirectory, dict[str, str | None]]:
    tempdir = tempfile.TemporaryDirectory()
    settings_path = Path(tempdir.name) / "settings.json"
    benchmarks_path = Path(tempdir.name) / "benchmark-history.json"
    chat_sessions_path = Path(tempdir.name) / "chat-sessions.json"
    hf_cache = Path(tempdir.name) / "hf-cache"
    hf_cache.mkdir(parents=True, exist_ok=True)

    env_snapshot: dict[str, str | None] = {key: os.environ.get(key) for key in _HF_CACHE_ENV_VARS}
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"] = str(tempdir.name)

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
    return client, tempdir, env_snapshot


def restore_env(env_snapshot: dict[str, str | None]) -> None:
    for key, value in env_snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class VideoCatalogRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client, self.tempdir, self.env_snapshot = make_client()

    def tearDown(self) -> None:
        restore_env(self.env_snapshot)
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
        self.assertIn("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", variant_ids)
        # Sanity-check: the 1.3B is materially smaller than the 14B / A14B options.
        wan_21_family = next(family for family in payload["families"] if family["id"] == "wan-2-1")
        sizes = {variant["id"]: variant["sizeGb"] for variant in wan_21_family["variants"]}
        self.assertLess(
            sizes["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"],
            sizes["Wan-AI/Wan2.1-T2V-14B-Diffusers"],
        )

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
        self.client, self.tempdir, self.env_snapshot = make_client()

    def tearDown(self) -> None:
        restore_env(self.env_snapshot)
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
        self.client, self.tempdir, self.env_snapshot = make_client()

    def tearDown(self) -> None:
        restore_env(self.env_snapshot)
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
        self.client, self.tempdir, self.env_snapshot = make_client()

    def tearDown(self) -> None:
        restore_env(self.env_snapshot)
        self.tempdir.cleanup()

    def test_unload_no_body_returns_runtime_status(self):
        response = self.client.post("/api/video/unload")
        self.assertEqual(response.status_code, 200)
        self.assertIn("runtime", response.json())

    def test_unload_unknown_model_returns_404(self):
        response = self.client.post("/api/video/unload", json={"modelId": "ghost/nope"})
        self.assertEqual(response.status_code, 404)


class VideoGenerateRouteTests(unittest.TestCase):
    """End-to-end contract tests for POST /api/video/generate.

    We cannot run a real diffusion pass inside a unit test — the weights are
    10+ GB and the pipeline needs a GPU. Instead we stub ``VideoRuntimeManager.generate``
    to return a synthetic ``GeneratedVideo`` with a tiny fake mp4 payload, then
    walk the full pipeline: persistence -> /outputs listing -> /outputs/{id}/file
    streaming -> /outputs/{id} delete. That lets us verify the route wiring,
    the filesystem layout, and the FileResponse headers without touching a GPU.
    """

    # A minimal 'mp4-looking' payload. We don't require it to be a valid video
    # — only that bytes round-trip through disk unchanged.
    FAKE_MP4_BYTES = b"\x00\x00\x00\x20ftypmp42" + b"\x00" * 64

    def setUp(self) -> None:
        self.client, self.tempdir, self.env_snapshot = make_client()
        # Redirect the global VIDEO_OUTPUTS_DIR into our tempdir so tests never
        # touch the real ~/Library/Application Support location.
        self.outputs_dir = Path(self.tempdir.name) / "video-outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._original_outputs_dir = app_module.VIDEO_OUTPUTS_DIR
        app_module.VIDEO_OUTPUTS_DIR = self.outputs_dir

    def tearDown(self) -> None:
        app_module.VIDEO_OUTPUTS_DIR = self._original_outputs_dir
        restore_env(self.env_snapshot)
        self.tempdir.cleanup()

    def _fake_generated_video(self, seed: int = 42) -> GeneratedVideo:
        return GeneratedVideo(
            seed=seed,
            bytes=self.FAKE_MP4_BYTES,
            extension="mp4",
            mimeType="video/mp4",
            durationSeconds=3.5,
            frameCount=24,
            fps=24,
            width=768,
            height=512,
            runtimeLabel="diffusers-test-stub",
            runtimeNote=None,
        )

    def _payload(self, **overrides: Any) -> dict[str, Any]:
        body = {
            "modelId": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt": "A cinematic shot of a misty pine forest at dawn.",
            "width": 768,
            "height": 512,
            "numFrames": 24,
            "fps": 24,
            "steps": 20,
            "guidance": 3.0,
            "seed": 42,
        }
        body.update(overrides)
        return body

    def test_generate_rejects_unknown_model_with_404(self):
        response = self.client.post(
            "/api/video/generate",
            json=self._payload(modelId="ghost/nope"),
        )
        self.assertEqual(response.status_code, 404)

    def test_generate_rejects_not_installed_model_with_409(self):
        response = self.client.post("/api/video/generate", json=self._payload())
        # No snapshot on disk in this test env -> expect 409 Conflict.
        self.assertEqual(response.status_code, 409)

    def test_generate_requires_fields(self):
        response = self.client.post("/api/video/generate", json={})
        # Pydantic validation rejects the missing required fields.
        self.assertEqual(response.status_code, 422)

    def test_generate_happy_path_persists_artifact_and_returns_outputs(self):
        state = self.client.app.state.chaosengine
        runtime_capabilities = {
            "activeEngine": "diffusers",
            "realGenerationAvailable": True,
            "message": "Ready",
            "missingDependencies": [],
            "loadedModelRepo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        }
        generate_mock = mock.MagicMock(
            return_value=(self._fake_generated_video(), runtime_capabilities)
        )
        state.video_runtime.generate = generate_mock  # type: ignore[method-assign]

        with mock.patch.object(
            video_routes,
            "_video_variant_available_locally",
            return_value=True,
        ):
            response = self.client.post("/api/video/generate", json=self._payload())

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertIn("artifact", payload)
        self.assertIn("outputs", payload)
        self.assertIn("runtime", payload)

        artifact = payload["artifact"]
        self.assertTrue(artifact["artifactId"].startswith("vid-"))
        self.assertEqual(artifact["modelId"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        self.assertEqual(artifact["prompt"], self._payload()["prompt"])
        self.assertEqual(artifact["fps"], 24)
        self.assertEqual(artifact["numFrames"], 24)
        self.assertEqual(artifact["steps"], 20)
        self.assertEqual(artifact["seed"], 42)
        self.assertEqual(artifact["videoMimeType"], "video/mp4")
        self.assertEqual(artifact["videoExtension"], "mp4")
        self.assertTrue(artifact["videoPath"], "videoPath should be set after save")
        self.assertTrue(artifact["metadataPath"], "metadataPath should be set after save")

        # Bytes actually landed on disk in the day-bucketed directory.
        video_path = Path(artifact["videoPath"])
        self.assertTrue(video_path.exists(), f"video not written to {video_path}")
        self.assertEqual(video_path.read_bytes(), self.FAKE_MP4_BYTES)

        # The outputs list in the response mirrors the newly saved artifact.
        self.assertEqual(len(payload["outputs"]), 1)
        self.assertEqual(payload["outputs"][0]["artifactId"], artifact["artifactId"])

        # And a fresh GET /outputs finds it too — proving metadata persisted.
        listing = self.client.get("/api/video/outputs").json()["outputs"]
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["artifactId"], artifact["artifactId"])

    def test_generate_then_stream_file_then_delete_round_trip(self):
        state = self.client.app.state.chaosengine
        state.video_runtime.generate = mock.MagicMock(  # type: ignore[method-assign]
            return_value=(
                self._fake_generated_video(),
                {
                    "activeEngine": "diffusers",
                    "realGenerationAvailable": True,
                    "message": "Ready",
                    "missingDependencies": [],
                },
            )
        )

        with mock.patch.object(
            video_routes,
            "_video_variant_available_locally",
            return_value=True,
        ):
            generate_resp = self.client.post("/api/video/generate", json=self._payload())
        self.assertEqual(generate_resp.status_code, 200, generate_resp.text)
        artifact_id = generate_resp.json()["artifact"]["artifactId"]

        # Detail endpoint returns the freshly saved artifact.
        detail_resp = self.client.get(f"/api/video/outputs/{artifact_id}")
        self.assertEqual(detail_resp.status_code, 200)
        self.assertEqual(detail_resp.json()["artifact"]["artifactId"], artifact_id)

        # File endpoint streams the mp4 bytes through with the right headers.
        file_resp = self.client.get(f"/api/video/outputs/{artifact_id}/file")
        self.assertEqual(file_resp.status_code, 200)
        self.assertEqual(file_resp.headers["content-type"], "video/mp4")
        self.assertEqual(file_resp.content, self.FAKE_MP4_BYTES)

        # Delete clears it and the listing goes empty again.
        delete_resp = self.client.delete(f"/api/video/outputs/{artifact_id}")
        self.assertEqual(delete_resp.status_code, 200)
        self.assertEqual(delete_resp.json()["deleted"], artifact_id)
        self.assertEqual(delete_resp.json()["outputs"], [])

        # After delete, follow-up GETs surface 404 for detail and file.
        self.assertEqual(
            self.client.get(f"/api/video/outputs/{artifact_id}").status_code,
            404,
        )
        self.assertEqual(
            self.client.get(f"/api/video/outputs/{artifact_id}/file").status_code,
            404,
        )

    def test_generate_surfaces_runtime_error_as_400(self):
        state = self.client.app.state.chaosengine
        state.video_runtime.generate = mock.MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("Pipeline did not return any frames."),
        )
        with mock.patch.object(
            video_routes,
            "_video_variant_available_locally",
            return_value=True,
        ):
            response = self.client.post("/api/video/generate", json=self._payload())
        self.assertEqual(response.status_code, 400)
        self.assertIn("Pipeline did not return any frames", response.json()["detail"])

    def test_delete_nonexistent_output_returns_404(self):
        response = self.client.delete("/api/video/outputs/vid-doesnotexist")
        self.assertEqual(response.status_code, 404)


class VideoOutputFileMissingTests(unittest.TestCase):
    """Cover the 410 Gone path when metadata exists but the mp4 is gone."""

    def setUp(self) -> None:
        self.client, self.tempdir, self.env_snapshot = make_client()
        self.outputs_dir = Path(self.tempdir.name) / "video-outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._original_outputs_dir = app_module.VIDEO_OUTPUTS_DIR
        app_module.VIDEO_OUTPUTS_DIR = self.outputs_dir

    def tearDown(self) -> None:
        app_module.VIDEO_OUTPUTS_DIR = self._original_outputs_dir
        restore_env(self.env_snapshot)
        self.tempdir.cleanup()

    def test_file_endpoint_returns_410_when_mp4_missing_on_disk(self):
        # Seed metadata that points at a non-existent file.
        day = self.outputs_dir / "2026-04-18"
        day.mkdir(parents=True, exist_ok=True)
        stub_id = "vid-ghost123abc"
        metadata_path = day / f"{stub_id}.json"
        fake_video_path = day / f"{stub_id}.mp4"
        import json
        metadata_path.write_text(
            json.dumps(
                {
                    "artifactId": stub_id,
                    "modelId": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    "modelName": "Wan 2.1 T2V 1.3B",
                    "prompt": "orphaned metadata",
                    "width": 768,
                    "height": 512,
                    "numFrames": 24,
                    "fps": 24,
                    "steps": 20,
                    "guidance": 3.0,
                    "seed": 7,
                    "createdAt": "2026-04-18T00:00:00Z",
                    "durationSeconds": 3.5,
                    "clipDurationSeconds": 1.0,
                    "videoPath": str(fake_video_path),
                    "videoMimeType": "video/mp4",
                    "videoExtension": "mp4",
                    "runtimeLabel": "test",
                }
            )
        )
        # NOTE: we deliberately do NOT create fake_video_path.

        resp = self.client.get(f"/api/video/outputs/{stub_id}/file")
        self.assertEqual(resp.status_code, 410)


class VideoDownloadRouteTests(unittest.TestCase):
    """Contract tests for /api/video/download* endpoints.

    We deliberately do not exercise a real HF snapshot_download from a unit
    test — that's an integration test path, taken by pulling ``Wan-AI/Wan2.1-T2V-1.3B``
    via the real endpoint. These tests assert the validation surface and the
    shape of the ``not_found`` / ``deleted`` paths which can be exercised
    without any network or subprocess activity.
    """

    def setUp(self) -> None:
        self.client, self.tempdir, self.env_snapshot = make_client()

    def tearDown(self) -> None:
        restore_env(self.env_snapshot)
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
        state._downloads["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"] = {
            "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
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
        self.assertEqual(downloads[0]["repo"], "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")

    def test_cancel_rejects_repo_outside_video_catalog(self):
        response = self.client.post(
            "/api/video/download/cancel",
            json={"repo": "ghost/nope"},
        )
        self.assertEqual(response.status_code, 404)

    def test_cancel_known_but_not_downloading_returns_not_found_state(self):
        response = self.client.post(
            "/api/video/download/cancel",
            json={"repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
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
            json={"repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertEqual(result["state"], "not_found")


if __name__ == "__main__":
    unittest.main()
