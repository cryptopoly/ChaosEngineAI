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

# Sentinel keys used in the snapshot dict to carry original module-level
# attributes that we patch in ``make_client``. Prefixed with ``__`` so the
# env-var restore loop skips them (env vars can't begin with that in POSIX).
_MODULE_ATTR_KEYS: tuple[str, ...] = (
    "__video_outputs_dir__",
    "__image_outputs_dir__",
    "__settings_path__",
)


def make_client() -> tuple[TestClient, tempfile.TemporaryDirectory, dict[str, Any]]:
    """Spin up an isolated FastAPI test client.

    Beyond the obvious tempdir for settings/benchmarks/chat-sessions and the
    HF cache env vars, we also redirect the module-level output-dir and
    settings-path constants in ``backend_service.app``. Those get captured
    at import time from the user's real ``~/.chaosengine/`` dir and are not
    affected by the test's ``settings_path`` argument — so without patching
    them, the ``GET /api/video/outputs`` route reads the user's real saved
    clips, and any settings-driven output override would leak in too.

    The returned snapshot carries both the env-var originals and the
    module-attribute originals. ``restore_env`` uses both halves on teardown.
    """
    tempdir = tempfile.TemporaryDirectory()
    settings_path = Path(tempdir.name) / "settings.json"
    benchmarks_path = Path(tempdir.name) / "benchmark-history.json"
    chat_sessions_path = Path(tempdir.name) / "chat-sessions.json"
    hf_cache = Path(tempdir.name) / "hf-cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    # Pre-create the output dirs so the /outputs listing helpers can ``rglob``
    # them without tripping a missing-dir path.
    video_outputs_dir = Path(tempdir.name) / "video-outputs"
    image_outputs_dir = Path(tempdir.name) / "image-outputs"
    video_outputs_dir.mkdir(parents=True, exist_ok=True)
    image_outputs_dir.mkdir(parents=True, exist_ok=True)

    snapshot: dict[str, Any] = {key: os.environ.get(key) for key in _HF_CACHE_ENV_VARS}
    snapshot["__video_outputs_dir__"] = app_module.VIDEO_OUTPUTS_DIR
    snapshot["__image_outputs_dir__"] = app_module.IMAGE_OUTPUTS_DIR
    snapshot["__settings_path__"] = app_module.SETTINGS_PATH

    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["HF_HOME"] = str(tempdir.name)
    app_module.VIDEO_OUTPUTS_DIR = video_outputs_dir
    app_module.IMAGE_OUTPUTS_DIR = image_outputs_dir
    # Point the module-level settings path at the tempdir so the no-arg
    # ``_load_settings()`` calls inside ``_current_video_outputs_dir`` read
    # defaults instead of the user's real settings file (which may contain a
    # ``videoOutputsDirectory`` override that would bypass our patched
    # VIDEO_OUTPUTS_DIR and point back at a real location).
    app_module.SETTINGS_PATH = settings_path

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
    return client, tempdir, snapshot


def restore_env(env_snapshot: dict[str, Any]) -> None:
    """Reverse the env-var and module-attr patching done by ``make_client``."""
    # Module attrs first, so any env-var-triggered import-time re-resolution
    # (none today, but future-proofing) sees the final state.
    if "__video_outputs_dir__" in env_snapshot:
        app_module.VIDEO_OUTPUTS_DIR = env_snapshot["__video_outputs_dir__"]
    if "__image_outputs_dir__" in env_snapshot:
        app_module.IMAGE_OUTPUTS_DIR = env_snapshot["__image_outputs_dir__"]
    if "__settings_path__" in env_snapshot:
        app_module.SETTINGS_PATH = env_snapshot["__settings_path__"]
    for key, value in env_snapshot.items():
        if key in _MODULE_ATTR_KEYS:
            continue
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
                runtime_fields = (
                    variant.get("runtimeFootprintGb"),
                    variant.get("runtimeFootprintMpsGb"),
                    variant.get("runtimeFootprintCudaGb"),
                    variant.get("runtimeFootprintCpuGb"),
                )
                self.assertTrue(
                    any(float(value or 0) > 0 for value in runtime_fields),
                    f"{variant.get('id')} missing runtime footprint metadata",
                )


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
            # Three calls now: core, output, model-specific deps. The probe
            # short-circuits on missing core deps but still asks the other
            # two so they're surfaced in the install hint.
            side_effect=[["diffusers", "torch"], [], []],
        ):
            runtime = self.client.get("/api/video/runtime").json()["runtime"]
        self.assertFalse(runtime["realGenerationAvailable"])
        self.assertEqual(runtime["activeEngine"], "placeholder")

    def test_mlx_runtime_returns_shape(self):
        """`/api/video/mlx-runtime` proxies VideoRuntimeManager.mlx_video_capabilities.

        Result varies by host (Apple Silicon vs other) but the contract
        is fixed: 200 OK, ``runtime`` object with the standard
        VideoRuntimeStatus keys, ``activeEngine == "mlx-video"``.
        """
        response = self.client.get("/api/video/mlx-runtime")
        self.assertEqual(response.status_code, 200)
        runtime = response.json()["runtime"]
        for key in ("activeEngine", "realGenerationAvailable", "message"):
            self.assertIn(key, runtime)
        self.assertEqual(runtime["activeEngine"], "mlx-video")

    def test_mlx_runtime_delegates_to_manager(self):
        """Endpoint must call `mlx_video_capabilities()` (not the diffusers probe)."""
        state = self.client.app.state.chaosengine
        with mock.patch.object(
            state.video_runtime,
            "mlx_video_capabilities",
            return_value={
                "activeEngine": "mlx-video",
                "realGenerationAvailable": False,
                "message": "stub",
                "device": "mps",
                "missingDependencies": ["mlx-video"],
            },
        ) as mocked:
            runtime = self.client.get("/api/video/mlx-runtime").json()["runtime"]
        mocked.assert_called_once()
        self.assertEqual(runtime["device"], "mps")
        self.assertIn("mlx-video", runtime["missingDependencies"])

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

    def test_generate_unloads_idle_image_runtime_first(self):
        state = self.client.app.state.chaosengine
        state.image_runtime = mock.MagicMock()
        state.image_runtime.capabilities.return_value = {
            "activeEngine": "diffusers",
            "realGenerationAvailable": True,
            "loadedModelRepo": "black-forest-labs/FLUX.1-schnell",
        }
        state.image_runtime.unload.return_value = {"loadedModelRepo": None}
        state.video_runtime.generate = mock.MagicMock(  # type: ignore[method-assign]
            return_value=(
                self._fake_generated_video(),
                {
                    "activeEngine": "diffusers",
                    "realGenerationAvailable": True,
                    "message": "Ready",
                    "missingDependencies": [],
                    "loadedModelRepo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                },
            )
        )

        with mock.patch.object(
            video_routes,
            "_video_variant_available_locally",
            return_value=True,
        ):
            response = self.client.post("/api/video/generate", json=self._payload())

        self.assertEqual(response.status_code, 200, response.text)
        state.image_runtime.unload.assert_called_once_with()

    def test_generate_rejects_while_image_generation_active(self):
        from backend_service.progress import IMAGE_PROGRESS

        IMAGE_PROGRESS.begin(run_label="FLUX test", total_steps=4, message="Diffusing")
        try:
            with mock.patch.object(
                video_routes,
                "_video_variant_available_locally",
                return_value=True,
            ):
                response = self.client.post("/api/video/generate", json=self._payload())
        finally:
            IMAGE_PROGRESS.finish()

        self.assertEqual(response.status_code, 409)
        self.assertIn("image generation is still running", response.json()["detail"])

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

    def test_generate_threads_phase_b_request_fields_into_config(self):
        """``scheduler``, ``useNf4``, ``enableLtxRefiner`` reach VideoGenerationConfig.

        Pydantic accepts the fields, the route handler propagates them, and the
        runtime manager sees the resolved config. We don't run a real pass —
        just assert the kwargs the manager would've used.
        """
        state = self.client.app.state.chaosengine
        captured: dict[str, Any] = {}

        def _capture(config):
            captured["config"] = config
            return self._fake_generated_video(), {
                "activeEngine": "diffusers",
                "realGenerationAvailable": True,
                "message": "Ready",
                "missingDependencies": [],
            }

        state.video_runtime.generate = mock.MagicMock(  # type: ignore[method-assign]
            side_effect=_capture
        )

        with mock.patch.object(
            video_routes,
            "_video_variant_available_locally",
            return_value=True,
        ):
            response = self.client.post(
                "/api/video/generate",
                json=self._payload(
                    scheduler="unipc",
                    useNf4=True,
                    enableLtxRefiner=False,
                ),
            )
        self.assertEqual(response.status_code, 200, response.text)
        config = captured["config"]
        self.assertEqual(config.scheduler, "unipc")
        self.assertTrue(config.useNf4)
        self.assertFalse(config.enableLtxRefiner)

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

    def test_cancel_accepts_shared_gguf_repo(self):
        response = self.client.post(
            "/api/video/download/cancel",
            json={"repo": "city96/LTX-Video-gguf"},
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

    def test_delete_accepts_shared_gguf_repo(self):
        response = self.client.post(
            "/api/video/download/delete",
            json={"repo": "city96/LTX-Video-gguf"},
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertEqual(result["state"], "not_found")


class MlxVideoSnapshotValidationTests(unittest.TestCase):
    """mlx-video repos (e.g. ``prince-canuma/LTX-2-*``) ship MLX layouts
    WITHOUT ``model_index.json``. The diffusers-shape validator must not flag
    these as incomplete; ``_video_download_validation_error`` must route
    through the mlx-video schema check instead.
    """

    def test_complete_mlx_snapshot_validates_clean(self):
        from backend_service.helpers.video import _validate_mlx_video_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "text_encoder",
                "tokenizer",
                "text_projections",
                "transformer",
                "vae",
            ):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text("{}")
            self.assertIsNone(_validate_mlx_video_snapshot(str(root)))

    def test_missing_component_reports_which_one(self):
        from backend_service.helpers.video import _validate_mlx_video_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "text_encoder",
                "tokenizer",
                "text_projections",
                "transformer",
            ):  # vae absent
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text("{}")
            err = _validate_mlx_video_snapshot(str(root))
            self.assertIsNotNone(err)
            self.assertIn("vae", err)

    def test_empty_component_dir_reports_as_incomplete(self):
        from backend_service.helpers.video import _validate_mlx_video_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "text_encoder",
                "tokenizer",
                "text_projections",
                "transformer",
                "vae",
            ):
                (root / name).mkdir()
            # All folders exist but are empty — partial download.
            err = _validate_mlx_video_snapshot(str(root))
            self.assertIsNotNone(err)
            self.assertIn("empty", err.lower())

    def test_ltx2_missing_text_projections_reports_incomplete(self):
        from backend_service.helpers.video import _validate_mlx_video_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("text_encoder", "tokenizer", "transformer", "vae"):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text("{}")
            err = _validate_mlx_video_snapshot(
                str(root),
                "prince-canuma/LTX-2-distilled",
            )
            self.assertIsNotNone(err)
            self.assertIn("text_projections", err)

    def test_ltx23_snapshot_uses_current_mlx_layout(self):
        from backend_service.helpers.video import _validate_mlx_video_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "audio_vae",
                "text_projections",
                "transformer",
                "vae",
                "vocoder",
            ):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text("{}")
            self.assertIsNone(
                _validate_mlx_video_snapshot(str(root), "prince-canuma/LTX-2.3-dev")
            )

    def test_ltx23_missing_component_reports_current_name(self):
        from backend_service.helpers.video import _validate_mlx_video_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("audio_vae", "transformer", "vae", "vocoder"):
                folder = root / name
                folder.mkdir()
                (folder / "config.json").write_text("{}")
            err = _validate_mlx_video_snapshot(str(root), "prince-canuma/LTX-2.3-dev")
            self.assertIsNotNone(err)
            self.assertIn("text_projections", err)

    def test_mlx_routed_repo_skips_diffusers_check(self):
        from backend_service.helpers.video import (
            _is_mlx_video_routed_repo,
            _video_download_validation_error,
        )
        repo = "prince-canuma/LTX-2-distilled"
        # Confirm the routing predicate matches.
        self.assertTrue(_is_mlx_video_routed_repo(repo))
        # Non-existent snapshot returns the standard "no snapshot" error
        # rather than tripping the diffusers-shape check (which would say
        # "missing model_index.json" — the original bug).
        with mock.patch(
            "backend_service.helpers.video._hf_repo_snapshot_dir",
            return_value=None,
        ):
            err = _video_download_validation_error(repo)
            self.assertIsNotNone(err)
            self.assertIn("Download did not produce", err)
            # Crucially, no mention of model_index.json — that's the
            # diffusers-shape error that misled users into thinking
            # their LTX-2 download was broken.
            self.assertNotIn("model_index", err)


class VideoGgufVariantValidationTests(unittest.TestCase):
    def test_gguf_partial_local_data_reports_shared_repo_delete_target(self):
        from backend_service.helpers.video import _video_model_payloads

        with tempfile.TemporaryDirectory() as tmp:
            gguf_snapshot = Path(tmp) / "gguf"
            gguf_snapshot.mkdir()
            (gguf_snapshot / "partial.gguf").write_bytes(b"partial")

            def snapshot(repo: str):
                if repo == "city96/LTX-Video-gguf":
                    return gguf_snapshot
                return None

            with mock.patch(
                "backend_service.helpers.video._hf_repo_snapshot_dir",
                side_effect=snapshot,
            ), mock.patch(
                "backend_service.helpers.video._image_repo_live_metadata",
                return_value={},
            ):
                families = _video_model_payloads([])

        variants = [
            variant
            for family in families
            for variant in family["variants"]
            if variant["id"] == "Lightricks/LTX-Video-gguf-q6k"
        ]
        self.assertEqual(len(variants), 1)
        variant = variants[0]
        self.assertTrue(variant["hasLocalData"])
        self.assertFalse(variant["availableLocally"])
        self.assertEqual(variant["localDataRepos"], ["city96/LTX-Video-gguf"])
        self.assertEqual(variant["primaryLocalRepo"], "city96/LTX-Video-gguf")
        self.assertEqual(variant["localPath"], str(gguf_snapshot))

    def test_gguf_variant_requires_cached_transformer_file(self):
        from backend_service.helpers.video import _video_variant_validation_error

        variant = {
            "name": "Wan 2.2 TI2V 5B · GGUF Q8_0",
            "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "ggufRepo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
            "ggufFile": "Wan2.2-TI2V-5B-Q8_0.gguf",
        }
        with mock.patch(
            "backend_service.helpers.video._video_download_validation_error",
            return_value=None,
        ), mock.patch(
            "huggingface_hub.hf_hub_download",
            side_effect=FileNotFoundError("not cached"),
        ):
            err = _video_variant_validation_error(variant)

        self.assertIsNotNone(err)
        self.assertIn("GGUF transformer file is missing", err)
        self.assertIn("Wan2.2-TI2V-5B-Q8_0.gguf", err)

    def test_gguf_missing_file_status_reason_is_specific(self):
        from backend_service.helpers.video import _video_variant_local_status_reason

        variant = {
            "name": "Wan 2.2 TI2V 5B · GGUF Q8_0",
            "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "ggufRepo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
            "ggufFile": "Wan2.2-TI2V-5B-Q8_0.gguf",
        }
        reason = _video_variant_local_status_reason(
            variant,
            (
                "The base diffusers snapshot is installed, but the selected GGUF "
                "transformer file is missing: QuantStack/Wan2.2-TI2V-5B-GGUF/"
                "Wan2.2-TI2V-5B-Q8_0.gguf."
            ),
        )
        self.assertEqual(
            reason,
            (
                "Base model installed; missing GGUF transformer: "
                "QuantStack/Wan2.2-TI2V-5B-GGUF/Wan2.2-TI2V-5B-Q8_0.gguf."
            ),
        )

    def test_mlx_missing_components_status_reason_is_specific(self):
        from backend_service.helpers.video import _video_variant_local_status_reason

        reason = _video_variant_local_status_reason(
            {"repo": "prince-canuma/LTX-2-distilled"},
            (
                "The local snapshot is incomplete. Missing mlx-video components: "
                "text_projections (empty). Re-download the model and keep "
                "ChaosEngineAI open until the download completes."
            ),
        )
        self.assertEqual(reason, "Missing MLX components: text_projections (empty).")

    def test_ltx23_variant_requires_shared_text_encoder(self):
        from backend_service.helpers.video import _video_variant_validation_error

        variant = {
            "name": "LTX-2.3 · dev (MLX)",
            "repo": "prince-canuma/LTX-2.3-dev",
            "textEncoderRepo": "prince-canuma/LTX-2-distilled",
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "ltx23"
            shared = Path(tmp) / "ltx2"
            base.mkdir()
            shared.mkdir()

            def snapshot(repo: str):
                if repo == "prince-canuma/LTX-2.3-dev":
                    return base
                if repo == "prince-canuma/LTX-2-distilled":
                    return shared
                return None

            with mock.patch(
                "backend_service.helpers.video._video_download_validation_error",
                return_value=None,
            ), mock.patch(
                "backend_service.helpers.video._hf_repo_snapshot_dir",
                side_effect=snapshot,
            ):
                err = _video_variant_validation_error(variant)

        self.assertIsNotNone(err)
        self.assertIn("shared mlx-video text components", err)
        self.assertIn("prince-canuma/LTX-2-distilled", err)

    def test_ltx23_variant_accepts_shared_text_encoder_repo(self):
        from backend_service.helpers.video import _video_variant_validation_error

        variant = {
            "name": "LTX-2.3 · dev (MLX)",
            "repo": "prince-canuma/LTX-2.3-dev",
            "textEncoderRepo": "prince-canuma/LTX-2-distilled",
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "ltx23"
            shared = Path(tmp) / "ltx2"
            base.mkdir()
            (shared / "text_encoder").mkdir(parents=True)
            (shared / "text_encoder" / "config.json").write_text("{}")
            (shared / "text_encoder" / "model.safetensors.index.json").write_text("{}")
            (shared / "tokenizer").mkdir()
            (shared / "tokenizer" / "tokenizer.json").write_text("{}")
            (shared / "tokenizer" / "tokenizer.model").write_text("tokenizer")

            def snapshot(repo: str):
                if repo == "prince-canuma/LTX-2.3-dev":
                    return base
                if repo == "prince-canuma/LTX-2-distilled":
                    return shared
                return None

            with mock.patch(
                "backend_service.helpers.video._video_download_validation_error",
                return_value=None,
            ), mock.patch(
                "backend_service.helpers.video._hf_repo_snapshot_dir",
                side_effect=snapshot,
            ):
                self.assertIsNone(_video_variant_validation_error(variant))

    def test_gguf_variant_validates_when_base_and_gguf_are_cached(self):
        from backend_service.helpers.video import _video_variant_validation_error

        variant = {
            "name": "Wan 2.2 TI2V 5B · GGUF Q8_0",
            "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "ggufRepo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
            "ggufFile": "Wan2.2-TI2V-5B-Q8_0.gguf",
        }
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "Wan2.2-TI2V-5B-Q8_0.gguf"
            gguf.write_bytes(b"gguf")
            with mock.patch(
                "backend_service.helpers.video._video_download_validation_error",
                return_value=None,
            ), mock.patch(
                "huggingface_hub.hf_hub_download",
                return_value=str(gguf),
            ):
                self.assertIsNone(_video_variant_validation_error(variant))


if __name__ == "__main__":
    unittest.main()
