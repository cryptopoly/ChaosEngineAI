"""Contract tests for route modules that previously had no dedicated tests.

Phase 0 of the v0.8.0 refactor — the existing integration tests in
``test_backend_service.py`` cover the heavy chat / image / model paths,
but the smaller endpoints (benchmarks, cache, finetuning, metrics,
openai_compat, server, storage, prompts) were never wired to a
``TestClient``. Without those, a refactor of ``state.py`` or
``routes/__init__.py`` could silently break them.

Each test mounts ``create_app`` with a tempdir-rooted ``ChaosEngineState``
and a minimal ``FakeRuntime`` and exercises one happy-path or one
deterministic error per route. The goal is contract preservation across
the refactor, not deep behaviour coverage.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend_service.app import create_app
from backend_service.state import ChaosEngineState
from tests.test_setup_routes import _fake_system_snapshot, TEST_API_TOKEN


class _FakeRuntime:
    """Minimal runtime stub — same shape as the diagnostics tests use."""

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
        # state.server_status() reads serverReady + activeRequests +
        # requestsServed off this dict — keep the keys complete.
        return {
            "engine": "mock",
            "loadedModel": None,
            "nativeBackends": {},
            "serverReady": False,
            "recentOrphanedWorkers": [],
            "activeRequests": 0,
            "requestsServed": 0,
        }

    def warm_models(self):
        return []


class _RouteContractCase(unittest.TestCase):
    """Shared setup: tempdir state + FakeRuntime + TestClient with auth."""

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
        state.runtime = _FakeRuntime()
        self.state = state
        self.client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})

    def tearDown(self):
        self.tempdir.cleanup()


class BenchmarkRouteTests(_RouteContractCase):
    def test_run_benchmark_returns_dict_or_400(self):
        # No model is loaded in the fake runtime — the endpoint either
        # surfaces a 400 with a "model not loaded" detail or runs a
        # placeholder benchmark and returns a dict. Both shapes are
        # acceptable; we only assert the contract (status 200 → dict,
        # status 400 → has detail).
        resp = self.client.post("/api/benchmarks/run", json={"prompt": "hi"})
        self.assertIn(resp.status_code, (200, 400, 422))
        body = resp.json()
        if resp.status_code == 200:
            self.assertIsInstance(body, dict)
        else:
            self.assertIn("detail", body)

    def test_unknown_mode_is_refused_not_run_as_throughput(self):
        resp = self.client.post("/api/benchmarks/run", json={"mode": "nonsense"})
        self.assertEqual(resp.status_code, 422)

    def test_delete_removes_one_saved_run_and_persists(self):
        self.state._append_benchmark_run({"id": "bench-keep", "label": "Keep"})
        self.state._append_benchmark_run({"id": "bench-drop", "label": "Drop"})
        resp = self.client.delete("/api/benchmarks/bench-drop")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["deleted"], "bench-drop")
        self.assertEqual([run["id"] for run in body["benchmarks"]], ["bench-keep"])
        saved = json.loads(Path(self.state._benchmarks_path).read_text(encoding="utf-8"))
        self.assertEqual([run["id"] for run in saved], ["bench-keep"])

    def test_delete_unknown_run_is_404(self):
        resp = self.client.delete("/api/benchmarks/bench-missing")
        self.assertEqual(resp.status_code, 404)


class ConversionCancelRouteTests(_RouteContractCase):
    def test_cancel_reports_whether_a_conversion_was_stopped(self):
        self.state.runtime.cancel_conversion = lambda: False  # type: ignore[attr-defined]
        resp = self.client.post("/api/models/convert/cancel")
        self.assertEqual((resp.status_code, resp.json()), (200, {"cancelled": False}))
        self.state.runtime.cancel_conversion = lambda: True  # type: ignore[attr-defined]
        self.assertEqual(self.client.post("/api/models/convert/cancel").json(), {"cancelled": True})


class CachePreviewRouteTests(_RouteContractCase):
    def test_cache_preview_default_params_returns_shape(self):
        # The preview no longer builds a system snapshot (speed comes from
        # memory bandwidth or measured history), so no runtime stub is needed.
        resp = self.client.get("/api/cache/preview")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, dict)
        self.assertEqual(body["archSource"], "estimate")
        self.assertIn(body["speedSource"], {"estimate", "measured"})

    def test_cache_preview_rejects_out_of_range_bits(self):
        resp = self.client.get("/api/cache/preview", params={"bits": 99})
        self.assertEqual(resp.status_code, 422)

    def test_cache_preview_accepts_a_model_under_half_a_billion_parameters(self):
        # Qwen2.5-0.5B reports 0.48; the old lower bound of 0.5 answered 422.
        resp = self.client.get(
            "/api/cache/preview",
            params={"bits": 0, "fp16_layers": 0, "num_layers": 24, "num_heads": 14, "hidden_size": 896,
                    "context_tokens": 8192, "params_b": 0.48, "num_kv_heads": 2, "strategy": "native"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["paramsB"], 0.48)


class FineTuningRouteTests(_RouteContractCase):
    def test_get_adapters_returns_list(self):
        # Phase 0 fix: route used ``app.state.engine`` which was never
        # set — would 500 on every call. Now uses ``chaosengine``.
        resp = self.client.get("/api/adapters")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("adapters", body)
        self.assertIn("count", body)
        self.assertIsInstance(body["adapters"], list)

    def test_finetuning_status_returns_idle_initially(self):
        resp = self.client.get("/api/finetuning/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("status", body)

    def test_finetuning_start_validates_inputs(self):
        # Empty modelPath fails Pydantic min_length=1.
        resp = self.client.post(
            "/api/finetuning/start",
            json={"modelPath": "", "datasetPath": "/tmp/ds.jsonl"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_finetuning_start_says_it_is_not_built(self):
        # It used to answer 200 "accepted" for a job that never ran and left the status at "accepted".
        resp = self.client.post(
            "/api/finetuning/start",
            json={"modelPath": "/nonexistent/model", "datasetPath": "/nonexistent/ds.jsonl"},
        )
        self.assertEqual(resp.status_code, 501)
        self.assertIn("not available", resp.json()["detail"]["message"])
        self.assertEqual(self.client.get("/api/finetuning/status").json()["status"], "idle")


class MetricsRouteTests(_RouteContractCase):
    def test_gpu_metrics_returns_dict(self):
        resp = self.client.get("/api/metrics/gpu")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), dict)


class OpenAICompatRouteTests(_RouteContractCase):
    def test_v1_models_returns_dict(self):
        resp = self.client.get("/v1/models")
        # Endpoint is deterministic — returns a {data: [...]} shape even
        # when no models are loaded.
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, dict)

    def test_v1_chat_completions_validates_body(self):
        resp = self.client.post("/v1/chat/completions", json={})
        self.assertIn(resp.status_code, (400, 422, 503))

    def test_v1_embeddings_validates_body(self):
        resp = self.client.post("/v1/embeddings", json={})
        self.assertIn(resp.status_code, (400, 422, 503))


class ServerRouteTests(_RouteContractCase):
    def test_server_status_returns_dict(self):
        resp = self.client.get("/api/server/status")
        self.assertEqual(resp.status_code, 200)
        self.assertIsInstance(resp.json(), dict)


class StorageRouteTests(_RouteContractCase):
    def test_storage_settings_returns_paths(self):
        resp = self.client.get("/api/settings/storage")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        for key in ("configuredPath", "effectivePath", "defaultPath", "currentHubSizeBytes"):
            self.assertIn(key, body, f"missing key: {key}")

    def test_storage_path_update_rejects_relative_path(self):
        resp = self.client.post("/api/settings/storage", json={"hfCachePath": "relative/path"})
        self.assertEqual(resp.status_code, 400)

    def test_storage_path_update_accepts_empty_to_reset(self):
        resp = self.client.post("/api/settings/storage", json={"hfCachePath": ""})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["configuredPath"], "")

    def test_storage_move_status_idle_when_no_job(self):
        resp = self.client.get("/api/settings/storage/move/status")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["phase"], "idle")


class PromptsRouteTests(_RouteContractCase):
    """The prompt-template CRUD endpoints (separate from FU-022 enhance).

    Phase 0 fix: route used ``app.state.engine`` which was never set.
    Now uses ``chaosengine`` and the singleton library cache.
    """

    def setUp(self):
        super().setUp()
        # The library is module-level and, left to itself, opens the real data directory
        # (``DATA_LOCATION``, not this test's tempdir): every run of this class used to add a
        # "Phase 0 test template" row to the developer's own Prompts page (505 of them were found
        # on 2026-11-18). Point it at the tempdir, and drop it again afterwards.
        from backend_service.helpers.prompts import PromptLibrary
        from backend_service.routes import prompts as prompts_module
        prompts_module._library = PromptLibrary(Path(self.tempdir.name))
        self.addCleanup(setattr, prompts_module, "_library", None)

    def test_list_prompts_returns_seeded_templates(self):
        resp = self.client.get("/api/prompts")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("templates", body)
        self.assertGreater(body["count"], 0)

    def test_create_prompt_persists(self):
        resp = self.client.post(
            "/api/prompts",
            json={
                "name": "Phase 0 test template",
                "systemPrompt": "Be concise.",
                "tags": ["test"],
                "category": "Testing",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["created"])
        self.assertEqual(body["template"]["name"], "Phase 0 test template")
        # It was written to this test's folder, not to the real library.
        stored = (Path(self.tempdir.name) / "prompt_templates.json").read_text(encoding="utf-8")
        self.assertIn("Phase 0 test template", stored)

    def test_delete_unknown_prompt_returns_404(self):
        resp = self.client.delete("/api/prompts/no-such-template")
        self.assertEqual(resp.status_code, 404)


class RemovedPluginsRouteTests(_RouteContractCase):
    """D-87: the Plugins feature was removed. Its routes are gone and nothing else answers in their place."""

    def test_plugins_routes_are_gone(self):
        self.assertEqual(self.client.get("/api/plugins").status_code, 404)
        for action in ("enable", "disable"):
            self.assertEqual(self.client.post(f"/api/plugins/tool.calculator/{action}").status_code, 404)

    def test_old_plugin_data_on_disk_is_ignored_not_deleted(self):
        # A settings file or a ``plugins`` folder left from an older build must not stop start-up. No build wrote
        # plugin state to settings.json (the switches lived in memory), but an unknown key is ignored either way,
        # and nothing in the backend reads or removes the old ``~/.chaosengine/plugins/`` folder.
        from backend_service.helpers.settings import _load_settings

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            manifest = data_dir / "plugins" / "old" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"id": "old", "name": "Old", "type": "tool"}', encoding="utf-8")
            settings_path = data_dir / "settings.json"
            settings_path.write_text(
                json.dumps({"preferredServerPort": 8877, "plugins": {"tool.calculator": {"enabled": False}}}),
                encoding="utf-8",
            )
            loaded = _load_settings(settings_path, 8876, data_dir)
            self.assertEqual(loaded["preferredServerPort"], 8877)
            self.assertNotIn("plugins", loaded)
            self.assertTrue(manifest.exists())
        backend = Path(__file__).resolve().parents[1] / "backend_service"
        self.assertFalse((backend / "plugins" / "__init__.py").exists())
        self.assertFalse((backend / "routes" / "plugins.py").exists())


if __name__ == "__main__":
    unittest.main()
