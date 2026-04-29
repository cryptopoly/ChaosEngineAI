"""Tests for the diagnostics endpoints.

Focus on contract shape (the frontend Copy-to-Markdown flow depends on
each section existing) and the redaction rules (API tokens must not
leak into the shared snapshot).
"""

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState
from tests.test_setup_routes import _fake_system_snapshot, TEST_API_TOKEN


class FakeRuntime:
    class _Caps:
        pythonExecutable = "/usr/bin/python3"

        def to_dict(self):
            return {"pythonExecutable": self.pythonExecutable, "ggufAvailable": True}

    class _Engine:
        engine_name = "mock"
        engine_label = "Idle"

    capabilities = _Caps()
    engine = _Engine()
    loaded_model = None

    def refresh_capabilities(self, *, force=False):
        return self.capabilities

    def status(self, **kwargs):
        return {"engine": "mock", "loadedModel": None, "nativeBackends": {}}

    def warm_models(self):
        return []


class DiagnosticsSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=Path(self.tempdir.name) / "settings.json",
            benchmarks_path=Path(self.tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self.tempdir.name) / "chats.json",
            library_cache_path=Path(self.tempdir.name) / "library_cache.json",
        )
        state.runtime = FakeRuntime()
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_snapshot_returns_all_sections(self):
        """Contract for the frontend Copy-to-Markdown flow: every section
        key must be present so the renderer doesn't have to defend
        against missing fields.
        """
        resp = self.client.get("/api/diagnostics/snapshot")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("generatedAt", "app", "os", "hardware", "python", "runtime", "gpu", "extras", "environment", "logs"):
            self.assertIn(key, body, f"missing top-level section: {key}")

    def test_snapshot_surfaces_app_version(self):
        resp = self.client.get("/api/diagnostics/snapshot")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("appVersion", body["app"])
        self.assertTrue(body["app"]["appVersion"])

    def test_snapshot_redacts_secret_env_vars(self):
        """Any env var with token/secret/password/api_key in the name
        must be redacted before leaving the backend — users paste these
        payloads into support threads.
        """
        import os
        os.environ["CHAOSENGINE_TEST_API_TOKEN"] = "super-secret-value-do-not-leak"
        os.environ["CHAOSENGINE_TEST_NORMAL_VAR"] = "public-value"
        try:
            resp = self.client.get("/api/diagnostics/snapshot")
            self.assertEqual(resp.status_code, 200)
            env = resp.json()["environment"]
            self.assertEqual(env.get("CHAOSENGINE_TEST_API_TOKEN"), "***redacted***")
            self.assertEqual(env.get("CHAOSENGINE_TEST_NORMAL_VAR"), "public-value")
        finally:
            os.environ.pop("CHAOSENGINE_TEST_API_TOKEN", None)
            os.environ.pop("CHAOSENGINE_TEST_NORMAL_VAR", None)

    def test_snapshot_pinned_env_vars_present_even_when_unset(self):
        """Pinned vars appear in the payload even with null values so
        users can see what the backend inherited vs didn't."""
        resp = self.client.get("/api/diagnostics/snapshot")
        env = resp.json()["environment"]
        # PYTHONPATH / PATH are pinned; they may be null on some test
        # environments but the KEY must still be in the response.
        self.assertIn("PATH", env)
        self.assertIn("PYTHONPATH", env)
        self.assertIn("CHAOSENGINE_EXTRAS_SITE_PACKAGES", env)

    def test_snapshot_gpu_uses_find_spec_without_importing_torch(self):
        """Regression lock: the snapshot endpoint must not trigger a
        torch import in the backend process. Importing torch pins
        torch/lib/*.dll in the handle table on Windows and breaks the
        GPU bundle install flow.
        """
        import sys
        saved = sys.modules.pop("torch", None)
        try:
            resp = self.client.get("/api/diagnostics/snapshot")
            self.assertEqual(resp.status_code, 200)
            self.assertNotIn(
                "torch",
                sys.modules,
                "diagnostics snapshot must not import torch into the backend process",
            )
        finally:
            if saved is not None:
                sys.modules["torch"] = saved


class DiagnosticsLogTailTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=Path(self.tempdir.name) / "settings.json",
            benchmarks_path=Path(self.tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self.tempdir.name) / "chats.json",
            library_cache_path=Path(self.tempdir.name) / "library_cache.json",
        )
        state.runtime = FakeRuntime()
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_log_tail_clamps_to_ceiling(self):
        # 999999 requested → clamped to the MAX constant. We don't
        # assert the exact max here because it's an implementation
        # detail; we assert that the endpoint doesn't OOM and returns
        # a reasonable count.
        resp = self.client.get("/api/diagnostics/log-tail?lines=999999")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("path", body)
        self.assertIn("lines", body)
        self.assertLessEqual(len(body["lines"]), 500)

    def test_log_tail_no_log_returns_empty(self):
        # When there's no matching backend log file on this system,
        # the endpoint returns an empty list rather than 404.
        resp = self.client.get("/api/diagnostics/log-tail?lines=50")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body["lines"], list)


class DiagnosticsReextractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=Path(self.tempdir.name) / "settings.json",
            benchmarks_path=Path(self.tempdir.name) / "benchmarks.json",
            chat_sessions_path=Path(self.tempdir.name) / "chats.json",
            library_cache_path=Path(self.tempdir.name) / "library_cache.json",
        )
        state.runtime = FakeRuntime()
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_reextract_is_idempotent_when_no_cache(self):
        """Calling re-extract when there's no cached extraction is a
        soft success — the endpoint reports deleted=False with no
        error, matching the 'nothing to do' semantics.
        """
        # Patch the path resolver to point at a guaranteed-empty dir
        # so we don't accidentally delete the real cache in dev envs.
        from unittest import mock
        import backend_service.routes.diagnostics as diag

        fake_path = Path(self.tempdir.name) / "nonexistent-runtime-cache"
        with mock.patch.object(diag, "_runtime_extraction_root", return_value=fake_path):
            resp = self.client.post("/api/diagnostics/reextract-runtime")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["deleted"])
        self.assertIsNone(body["error"])
        self.assertEqual(body["path"], str(fake_path))


if __name__ == "__main__":
    unittest.main()
