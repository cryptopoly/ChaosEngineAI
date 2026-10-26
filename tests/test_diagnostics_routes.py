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


class StorageChildrenTests(unittest.TestCase):
    """Disk usage drill-down: lists one folder, refuses anything outside the model directories."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.models = base / "models"
        (self.models / "tool" / "weights").mkdir(parents=True)
        (self.models / "tool" / "weights" / "a.bin").write_bytes(b"x" * 4096)
        (self.models / "tool" / "small.txt").write_bytes(b"y" * 10)
        (self.models / "models--org--repo" / "blobs").mkdir(parents=True)
        (self.models / "models--org--repo" / "blobs" / "h").write_bytes(b"z" * 100)
        self.outside = base / "outside"
        self.outside.mkdir()
        (self.outside / "f").write_bytes(b"q")
        state = ChaosEngineState(
            system_snapshot_provider=_fake_system_snapshot,
            library_provider=lambda: [],
            settings_path=base / "settings.json",
            benchmarks_path=base / "benchmarks.json",
            chat_sessions_path=base / "chats.json",
            library_cache_path=base / "library_cache.json",
        )
        state.runtime = FakeRuntime()
        state.settings["modelDirectories"] = [
            {"id": "m", "label": "Models", "path": str(self.models), "enabled": True}
        ]
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_top_marks_folders_expandable_but_not_hf_repos(self):
        body = self.client.get("/api/diagnostics/storage-top").json()
        flags = {e["repoLabel"]: e["expandable"] for e in body["entries"]}
        self.assertEqual(flags, {"tool": True, "org/repo": False})

    def test_shared_blob_store_is_not_deletable(self):
        (self.models / "blobs").mkdir()
        (self.models / "blobs" / "h").write_bytes(b"b" * 50)
        body = self.client.get("/api/diagnostics/storage-top").json()
        deletable = {e["repoLabel"]: e["deletable"] for e in body["entries"]}
        self.assertFalse(deletable["blobs"])
        self.assertTrue(deletable["tool"])
        inner = self.client.get(
            "/api/diagnostics/storage-children", params={"path": str(self.models / "blobs")}
        ).json()["entries"]
        self.assertEqual([e["deletable"] for e in inner], [False])

    def test_shared_blob_store_is_grouped_by_repo(self):
        blobs = self.models / "blobs"
        (blobs / "ab").mkdir(parents=True)
        (blobs / ".huggingface-shared-blobs").write_bytes(b"1")
        for name, size, ref in (
            ("abc1", 300, "models--org--big/blobs/abc1"),
            ("abc2", 100, "models--org--big/blobs/abc2"),
            ("abc3", 50, "models--other--small/blobs/abc3"),
        ):
            (blobs / "ab" / name).write_bytes(b"x" * size)
            (blobs / "ab" / f"{name}.refs").write_text(ref + "\n")
        entries = self.client.get(
            "/api/diagnostics/storage-children", params={"path": str(blobs)}
        ).json()["entries"]
        self.assertEqual(
            [(e["repoLabel"], e["sizeBytes"]) for e in entries],
            [("org/big (2 files)", 400), ("other/small (1 file)", 50)],
        )
        self.assertEqual([e["hfRepo"] for e in entries], ["org/big", "other/small"])
        self.assertTrue(all(e["deletable"] for e in entries))

    def test_delete_hf_repo_frees_it_and_keeps_other_repos(self):
        import os
        from unittest import mock

        hub = self.models / "hub"
        for name, size in (("org--gone", 300), ("org--kept", 100)):
            repo = hub / f"models--{name}"
            (repo / "blobs").mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "snapshots" / f"rev-{name}").mkdir(parents=True)
            (repo / "blobs" / name).write_bytes(b"x" * size)
            (repo / "snapshots" / f"rev-{name}" / "w.bin").symlink_to(repo / "blobs" / name)
            (repo / "refs" / "main").write_text(f"rev-{name}")
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            missing = self.client.post("/api/diagnostics/hf-repo/delete", json={"repo": "org/nope"})
            self.assertEqual(missing.status_code, 404)
            resp = self.client.post("/api/diagnostics/hf-repo/delete", json={"repo": "org/gone"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["freedBytes"], 300)
        self.assertFalse((hub / "models--org--gone").exists())
        self.assertTrue((hub / "models--org--kept" / "blobs" / "org--kept").exists())

    def test_delete_hf_repo_rejects_a_path_shaped_repo(self):
        resp = self.client.post("/api/diagnostics/hf-repo/delete", json={"repo": "../etc/passwd"})
        self.assertEqual(resp.status_code, 422)

    def test_children_lists_files_and_folders_biggest_first(self):
        resp = self.client.get(
            "/api/diagnostics/storage-children", params={"path": str(self.models / "tool")}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([e["repoLabel"] for e in resp.json()["entries"]], ["weights", "small.txt"])

    def test_children_refuses_paths_outside_model_directories(self):
        for path in (str(self.outside), str(self.models / ".." / "outside")):
            resp = self.client.get("/api/diagnostics/storage-children", params={"path": path})
            self.assertEqual(resp.status_code, 403, path)


class StopProcessRouteTests(unittest.TestCase):
    """The Dashboard Stop buttons: contract of the two endpoints."""

    class _Runtime(FakeRuntime):
        def stop_process(self, pid):
            if pid == 1:
                raise ValueError("That process serves the loaded model. Unload the model instead.")
            return {"pid": pid, "action": "terminated"}

        def stop_idle(self):
            return {"warmEvicted": 2, "orphansStopped": 1}

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
        state.runtime = self._Runtime()
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()

    def test_stop_a_process(self):
        resp = self.client.post("/api/models/processes/stop", json={"pid": 4321})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["action"], "terminated")

    def test_the_loaded_models_process_is_a_conflict(self):
        resp = self.client.post("/api/models/processes/stop", json={"pid": 1})
        self.assertEqual(resp.status_code, 409)
        self.assertIn("Unload the model", resp.json()["detail"])

    def test_pid_must_be_an_integer(self):
        for bad in ("12", None, True):
            self.assertEqual(self.client.post("/api/models/processes/stop", json={"pid": bad}).status_code, 400)

    def test_stop_idle(self):
        resp = self.client.post("/api/models/processes/stop-idle")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"warmEvicted": 2, "orphansStopped": 1})
