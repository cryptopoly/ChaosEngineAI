import unittest
import os
import json
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import compute_cache_preview, create_app
from backend_service.helpers.huggingface import (
    _search_huggingface_hub,
    _find_quantized_variants,
)
from backend_service.inference import GenerationResult, LoadedModelInfo, StreamChunk, _resolve_gguf_path
from backend_service.state import ChaosEngineState, _spawn_snapshot_download
from backend_service.helpers.discovery import _discover_local_models

TEST_API_TOKEN = "test-api-token"


def fake_system_snapshot():
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "Apple Silicon / 48 GB unified memory",
        "backendLabel": "Python sidecar",
        "appVersion": "0.5.1",
        "mlxAvailable": False,
        "mlxLmAvailable": False,
        "mlxUsable": False,
        "ggufAvailable": False,
        "converterAvailable": False,
        "nativePython": "/tmp/python",
        "llamaServerPath": None,
        "llamaCliPath": None,
        "nativeRuntimeMessage": None,
        "totalMemoryGb": 48.0,
        "availableMemoryGb": 30.0,
        "usedMemoryGb": 18.0,
        "swapUsedGb": 0.0,
        "cpuUtilizationPercent": 12.0,
        "gpuUtilizationPercent": None,
        "spareHeadroomGb": 24.0,
        "runningLlmProcesses": [],
        "uptimeMinutes": 1.0,
    }


def fake_library():
    return [
        {
            "name": "google/gemma-4-E4B-it",
            "path": "/tmp/gemma-4-e4b",
            "format": "HF cache",
            "sizeGb": 5.4,
            "lastModified": "2026-04-05 12:00",
            "actions": ["Run Chat", "Run Server", "Cache Preview", "Delete"],
        }
    ]


def fake_urlopen_json(payload):
    response = mock.MagicMock()
    response.read.return_value = json.dumps(payload).encode()
    context_manager = mock.MagicMock()
    context_manager.__enter__.return_value = response
    context_manager.__exit__.return_value = False
    return context_manager


def make_test_client(state: ChaosEngineState) -> TestClient:
    client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
    client.headers.update({"Authorization": f"Bearer {TEST_API_TOKEN}"})
    return client


class FakeRuntime:
    def __init__(self) -> None:
        self.engine = SimpleNamespace(engine_name="mock", engine_label="Idle")
        self.loaded_model = None
        self.runtime_note = None
        self.last_generate_kwargs = None
        self.load_requests: list[dict[str, object]] = []
        self.profile_updates: list[dict[str, object]] = []
        self._warm_pool = {}
        self.clear_warm_pool_calls = 0
        self.recent_orphaned_workers = []
        self.capabilities = SimpleNamespace(
            pythonExecutable="/tmp/python",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=False,
            llamaCliPath=None,
            llamaServerPath=None,
            converterAvailable=False,
            vllmAvailable=False,
            vllmVersion=None,
            to_dict=lambda: {
                "pythonExecutable": "/tmp/python",
                "mlxAvailable": False,
                "mlxLmAvailable": False,
                "mlxUsable": False,
                "ggufAvailable": False,
                "llamaCliPath": None,
                "llamaServerPath": None,
                "converterAvailable": False,
                "vllmAvailable": False,
                "vllmVersion": None,
            }
        )

    def warm_models(self) -> list[dict[str, object]]:
        return [
            info.to_dict() if hasattr(info, "to_dict") else {}
            for _, info in self._warm_pool.values()
        ]

    def status(self, *, active_requests: int = 0, requests_served: int = 0) -> dict[str, object]:
        return {
            "state": "loaded" if self.loaded_model is not None else "idle",
            "engine": self.engine.engine_name,
            "engineLabel": self.engine.engine_label,
            "loadedModel": self.loaded_model.to_dict() if self.loaded_model is not None else None,
            "warmModels": self.warm_models(),
            "supportsGeneration": True,
            "serverReady": self.loaded_model is not None,
            "activeRequests": active_requests,
            "requestsServed": requests_served,
            "runtimeNote": self.runtime_note,
            "nativeBackends": self.capabilities.to_dict(),
            "recentOrphanedWorkers": list(self.recent_orphaned_workers),
        }

    def load_model(
        self,
        *,
        model_ref: str,
        model_name: str | None = None,
        canonical_repo: str | None = None,
        source: str = "catalog",
        backend: str = "auto",
        path: str | None = None,
        runtime_target: str | None = None,
        cache_strategy: str = "native",
        cache_bits: int = 0,
        fp16_layers: int = 0,
        fused_attention: bool = False,
        fit_model_in_memory: bool = True,
        context_tokens: int = 8192,
        speculative_decoding: bool = False,
        tree_budget: int = 0,
        keep_warm_previous: bool = True,
        progress_callback=None,
    ) -> LoadedModelInfo:
        self.load_requests.append(
            {
                "model_ref": model_ref,
                "runtime_target": runtime_target,
                "keep_warm_previous": keep_warm_previous,
            }
        )
        if callable(progress_callback):
            progress_callback({"phase": "loading", "percent": 100, "message": "Fake runtime loaded"})
        draft_model = "z-lab/Qwen3-4B-DFlash" if speculative_decoding else None
        runtime_note = (
            f"Fake runtime. DFLASH speculative decoding active (draft: {draft_model}, budget={tree_budget})."
            if speculative_decoding
            else "Fake runtime"
        )
        loaded = LoadedModelInfo(
            ref=model_ref,
            name=model_name or model_ref,
            backend=backend,
            source=source,
            engine=self.engine.engine_name,
            cacheStrategy=cache_strategy,
            cacheBits=cache_bits,
            fp16Layers=fp16_layers,
            fusedAttention=fused_attention,
            fitModelInMemory=fit_model_in_memory,
            contextTokens=context_tokens,
            loadedAt="2026-04-13 00:00:00",
            canonicalRepo=canonical_repo,
            path=path,
            runtimeTarget=runtime_target or path or model_ref,
            runtimeNote=runtime_note,
            speculativeDecoding=speculative_decoding,
            dflashDraftModel=draft_model,
            treeBudget=tree_budget if speculative_decoding else 0,
        )
        self.loaded_model = loaded
        self.runtime_note = loaded.runtimeNote
        return loaded

    def update_profile(
        self,
        *,
        canonical_repo: str | None = None,
        cache_strategy: str,
        cache_bits: int,
        fp16_layers: int,
        fused_attention: bool,
    ) -> LoadedModelInfo:
        if self.loaded_model is None:
            raise RuntimeError("No model is loaded.")
        self.profile_updates.append(
            {
                "canonical_repo": canonical_repo,
                "cache_strategy": cache_strategy,
                "cache_bits": cache_bits,
                "fp16_layers": fp16_layers,
                "fused_attention": fused_attention,
            }
        )
        self.loaded_model.cacheStrategy = cache_strategy
        self.loaded_model.cacheBits = cache_bits
        self.loaded_model.fp16Layers = fp16_layers
        self.loaded_model.fusedAttention = fused_attention
        if canonical_repo is not None:
            self.loaded_model.canonicalRepo = canonical_repo
        return self.loaded_model

    def unload_model(self) -> None:
        self.loaded_model = None
        self.runtime_note = None

    def unload_warm_model_by_ref(self, ref: str) -> bool:
        return False

    def clear_warm_pool(self) -> int:
        count = len(self._warm_pool)
        self._warm_pool = {}
        self.clear_warm_pool_calls += 1
        return count

    def generate(
        self,
        *,
        prompt: str,
        history,
        system_prompt,
        max_tokens: int,
        temperature: float,
        images=None,
        tools=None,
        engine=None,
    ) -> GenerationResult:
        self.last_generate_kwargs = {
            "prompt": prompt,
            "history": history,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "images": images,
            "tools": tools,
        }
        text = (
            "Cache compression shrinks KV memory so longer contexts fit, "
            "with a small trade-off in token throughput."
        )
        prompt_tokens = max(1, len(str(prompt).split()))
        completion_tokens = max(1, len(text.split()))
        return GenerationResult(
            text=text,
            finishReason="stop",
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
            tokS=42.0,
            responseSeconds=0.1,
            runtimeNote=self.runtime_note,
        )

    def stream_generate(
        self,
        *,
        prompt: str,
        history,
        system_prompt,
        max_tokens: int,
        temperature: float,
        images=None,
        tools=None,
        engine=None,
    ):
        self.last_generate_kwargs = {
            "prompt": prompt,
            "history": history,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "images": images,
            "tools": tools,
        }
        text = "Streaming compare output."
        prompt_tokens = max(1, len(str(prompt).split()))
        completion_tokens = max(1, len(text.split()))
        yield StreamChunk(text=text)
        yield StreamChunk(
            done=True,
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            tok_s=42.0,
            runtime_note=self.runtime_note,
            dflash_acceptance_rate=4.5 if self.loaded_model and self.loaded_model.speculativeDecoding else None,
            cache_strategy=self.loaded_model.cacheStrategy if self.loaded_model else "native",
            cache_bits=self.loaded_model.cacheBits if self.loaded_model else 0,
            fp16_layers=self.loaded_model.fp16Layers if self.loaded_model else 0,
            speculative_decoding=self.loaded_model.speculativeDecoding if self.loaded_model else False,
            tree_budget=self.loaded_model.treeBudget if self.loaded_model else 0,
        )

    def get_engine_for_request(self, model_ref: str | None):
        if self.loaded_model is None:
            raise RuntimeError("Load a model before sending prompts.")
        return self.engine, self.loaded_model


class FakeWarmEngine:
    def __init__(self) -> None:
        self.unload_calls = 0

    def unload_model(self) -> None:
        self.unload_calls += 1


class ChaosEngineBackendTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.tempdir.name) / "settings.json"
        self.benchmarks_path = Path(self.tempdir.name) / "benchmark-history.json"
        self.chat_sessions_path = Path(self.tempdir.name) / "chat-sessions.json"
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )
        state.runtime = FakeRuntime()
        self.client = make_test_client(state)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_health_reports_runtime_metadata(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("appVersion", payload)
        self.assertEqual(payload["engine"], "mock")

    def test_system_gpu_status_reports_expected_keys(self):
        response = self.client.get("/api/system/gpu-status")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        for key in (
            "platform",
            "nvidiaGpuDetected",
            "torchImported",
            "torchCudaAvailable",
            "torchMpsAvailable",
            "cpuFallbackWarning",
            "recommendation",
        ):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["platform"], str)
        self.assertIsInstance(payload["nvidiaGpuDetected"], bool)
        self.assertIsInstance(payload["cpuFallbackWarning"], bool)

    def test_system_gpu_status_exempt_from_auth(self):
        # The banner polls this endpoint before the token is wired up, so it
        # must stay reachable even when require_api_auth is enforced.
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )
        state.runtime = FakeRuntime()
        client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        response = client.get("/api/system/gpu-status")
        self.assertEqual(response.status_code, 200)

    def test_auth_session_bootstrap_returns_local_token(self):
        response = self.client.get(
            "/api/auth/session",
            headers={"Origin": "http://127.0.0.1:5174"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["apiToken"], TEST_API_TOKEN)

    def test_protected_route_rejects_missing_auth(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )
        state.runtime = FakeRuntime()
        client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        response = client.post("/api/chat/sessions", json={"title": "Blocked"})
        self.assertEqual(response.status_code, 401)

    def test_protected_route_allows_missing_auth_when_require_api_auth_disabled(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )
        state.runtime = FakeRuntime()
        state.settings["requireApiAuth"] = False
        app = create_app(state=state, api_token=TEST_API_TOKEN)
        client = TestClient(app)
        # With the toggle off, a tokenless call from an external client
        # should succeed instead of returning 401.
        response = client.post("/api/chat/sessions", json={"title": "Allowed"})
        self.assertIn(response.status_code, (200, 201))

    def test_require_api_auth_hot_applies_after_settings_update(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )
        state.runtime = FakeRuntime()
        app = create_app(state=state, api_token=TEST_API_TOKEN)
        client = TestClient(app)
        # Rejected while auth is required.
        self.assertEqual(client.get("/api/workspace").status_code, 401)
        # Toggle off via the PATCH endpoint (with the token, since auth is
        # still on at this point).
        patch_response = client.patch(
            "/api/settings",
            json={"requireApiAuth": False},
            headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
        )
        self.assertEqual(patch_response.status_code, 200)
        self.assertIs(patch_response.json()["settings"]["requireApiAuth"], False)
        # Without restarting the server, anonymous requests now succeed.
        self.assertEqual(client.get("/api/workspace").status_code, 200)

    def test_require_api_auth_env_override_disables_auth(self):
        with mock.patch.dict(os.environ, {"CHAOSENGINE_REQUIRE_AUTH": "0"}):
            state = ChaosEngineState(
                system_snapshot_provider=fake_system_snapshot,
                library_provider=fake_library,
                settings_path=self.settings_path,
                benchmarks_path=self.benchmarks_path,
                chat_sessions_path=self.chat_sessions_path,
            )
            state.runtime = FakeRuntime()
            client = TestClient(create_app(state=state, api_token=TEST_API_TOKEN))
        self.assertEqual(client.get("/api/workspace").status_code, 200)

    def test_fresh_state_starts_without_seeded_workspace_data(self):
        workspace = self.client.get("/api/workspace").json()

        self.assertEqual(workspace["chatSessions"], [])
        self.assertEqual(workspace["benchmarks"], [])

    @mock.patch("backend_service.routes.models._search_huggingface_hub", return_value=[])
    def test_model_search_matches_normalized_multi_token_queries(self, _hub_search):
        for query in ("qwen coder", "coder qwen", "qwen 3 coder", "qwen next 32b"):
            with self.subTest(query=query):
                response = self.client.get("/api/models/search", params={"q": query})
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertIn("qwen3-coder", [family["id"] for family in payload["results"]])

    @mock.patch("backend_service.routes.models._search_huggingface_hub", return_value=[])
    def test_model_search_does_not_false_positive_qwen36_against_qwen35_catalog(self, _hub_search):
        response = self.client.get("/api/models/search", params={"q": "Qwen3.6-35B-A3B"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["results"], [])

    @mock.patch("backend_service.helpers.huggingface.urllib.request.urlopen")
    def test_hub_search_includes_glm_and_multimodal_gemma_matches(self, urlopen_mock):
        payload = [
            {
                "id": "zai-org/glm-4.7-flash",
                "pipeline_tag": "text-generation",
                "tags": ["glm", "tool-use", "gguf"],
                "downloads": 307123,
                "likes": 94,
            },
            {
                "id": "google/gemma-4-31b",
                "pipeline_tag": "image-text-to-text",
                "tags": ["gemma", "vision-language", "mlx"],
                "downloads": 301147,
                "likes": 98,
            },
            {
                "id": "google/gemma-4-e4b-it",
                "pipeline_tag": "image-text-to-text",
                "tags": ["gemma", "vision-language", "mlx"],
                "downloads": 400000,
                "likes": 120,
            },
            {
                "id": "runwayml/stable-diffusion-v1-5",
                "pipeline_tag": "text-to-image",
                "tags": ["diffusers"],
                "downloads": 999999,
                "likes": 1000,
            },
        ]
        urlopen_mock.return_value = fake_urlopen_json(payload)

        glm_results = _search_huggingface_hub("glm", fake_library(), limit=20)
        gemma_results = _search_huggingface_hub("gemma-4-31B", fake_library(), limit=20)

        self.assertIn("zai-org/glm-4.7-flash", [model["repo"] for model in glm_results])
        self.assertIn("google/gemma-4-31b", [model["repo"] for model in gemma_results])
        self.assertNotIn("google/gemma-4-e4b-it", [model["repo"] for model in gemma_results])
        self.assertNotIn("runwayml/stable-diffusion-v1-5", [model["repo"] for model in glm_results])
        self.assertNotIn("runwayml/stable-diffusion-v1-5", [model["repo"] for model in gemma_results])

    @mock.patch("backend_service.helpers.huggingface.urllib.request.urlopen")
    def test_hub_search_orders_results_by_most_recent_update_by_default(self, urlopen_mock):
        payload = [
            {
                "id": "org/older-model-popular",
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
                "downloads": 999999,
                "likes": 1000,
                "lastModified": "2026-04-10T10:00:00Z",
            },
            {
                "id": "org/newer-model",
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
                "downloads": 10,
                "likes": 1,
                "lastModified": "2026-04-16T10:00:00Z",
            },
            {
                "id": "org/middle-model",
                "pipeline_tag": "text-generation",
                "tags": ["text-generation"],
                "downloads": 100,
                "likes": 5,
                "lastModified": "2026-04-14T10:00:00Z",
            },
        ]
        urlopen_mock.return_value = fake_urlopen_json(payload)

        results = _search_huggingface_hub("model", fake_library(), limit=20)

        self.assertEqual(
            [model["repo"] for model in results],
            ["org/newer-model", "org/middle-model", "org/older-model-popular"],
        )

    @mock.patch("backend_service.helpers.huggingface.urllib.request.urlopen")
    def test_quantized_variants_finds_city96_gguf_mirrors(self, urlopen_mock):
        """``_find_quantized_variants`` must pull city96 GGUF mirrors
        for a base FLUX repo and tag them with ``format=GGUF`` so the
        Discover panel can render them as alternate variants."""
        # urlopen is called once per mirror author — return the same
        # payload for all of them; only city96's contains real matches.
        def _fake_urlopen(req, timeout=8):
            if "author=city96" in req.full_url:
                return fake_urlopen_json([
                    {
                        "id": "city96/FLUX.1-dev-gguf",
                        "tags": ["gguf", "text-to-image"],
                        "downloads": 50000,
                        "likes": 1200,
                        "lastModified": "2024-09-01T10:00:00Z",
                    },
                    {
                        "id": "city96/FLUX.1-dev-extra",
                        "tags": ["text-to-image"],
                        "downloads": 5,
                        "likes": 0,
                    },
                ])
            return fake_urlopen_json([])

        urlopen_mock.side_effect = _fake_urlopen
        results = _find_quantized_variants("black-forest-labs/FLUX.1-dev")

        repos = [r["repo"] for r in results]
        self.assertIn("city96/FLUX.1-dev-gguf", repos)
        # Extra repo has no quant tag and "gguf" not in its id — filtered.
        self.assertNotIn("city96/FLUX.1-dev-extra", repos)
        gguf_entry = next(r for r in results if r["repo"] == "city96/FLUX.1-dev-gguf")
        self.assertEqual(gguf_entry["format"], "GGUF")
        self.assertEqual(gguf_entry["baseRepo"], "black-forest-labs/FLUX.1-dev")

    @mock.patch("backend_service.helpers.huggingface.urllib.request.urlopen")
    def test_quantized_variants_detects_nf4_from_repo_name(self, urlopen_mock):
        """NF4 mirrors sometimes ship without a tag — fall back to
        matching ``nf4`` / ``bnb`` in the repo id so those still surface."""
        def _fake_urlopen(req, timeout=8):
            if "author=QuantStack" in req.full_url:
                return fake_urlopen_json([
                    {
                        "id": "QuantStack/FLUX.1-dev-nf4",
                        "tags": [],
                        "downloads": 1000,
                        "likes": 40,
                    }
                ])
            return fake_urlopen_json([])

        urlopen_mock.side_effect = _fake_urlopen
        results = _find_quantized_variants("black-forest-labs/FLUX.1-dev")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["format"], "NF4")

    def test_quantized_variants_route_returns_wrapped_payload(self):
        with mock.patch(
            "backend_service.routes.models._find_quantized_variants",
            return_value=[{"repo": "city96/FLUX.1-dev-gguf", "format": "GGUF"}],
        ):
            response = self.client.get(
                "/api/models/quantized-variants",
                params={"repo": "black-forest-labs/FLUX.1-dev"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["repo"], "black-forest-labs/FLUX.1-dev")
        self.assertEqual(len(payload["variants"]), 1)

    @mock.patch("backend_service.helpers.huggingface.urllib.request.urlopen")
    def test_hub_search_matches_exact_qwen36_repo_name(self, urlopen_mock):
        payload = [
            {
                "id": "Qwen/Qwen3.6-35B-A3B",
                "pipeline_tag": "image-text-to-text",
                "tags": ["transformers", "safetensors", "qwen3_5_moe", "conversational"],
                "downloads": 0,
                "likes": 201,
                "lastModified": "2026-04-15T05:59:19Z",
            },
            {
                "id": "Qwen/Qwen3.5-35B-A3B",
                "pipeline_tag": "image-text-to-text",
                "tags": ["transformers", "safetensors", "qwen3_5_moe", "conversational"],
                "downloads": 1000,
                "likes": 1000,
                "lastModified": "2026-04-01T05:59:19Z",
            },
        ]
        urlopen_mock.return_value = fake_urlopen_json(payload)

        results = _search_huggingface_hub("Qwen3.6-35B-A3B", fake_library(), limit=20)

        self.assertEqual([model["repo"] for model in results], ["Qwen/Qwen3.6-35B-A3B"])

    @mock.patch("backend_service.helpers.huggingface.urllib.request.urlopen")
    def test_hub_search_accepts_huggingface_model_urls(self, urlopen_mock):
        payload = [
            {
                "id": "Qwen/Qwen3.6-35B-A3B",
                "pipeline_tag": "image-text-to-text",
                "tags": ["transformers", "safetensors", "qwen3_5_moe", "conversational"],
                "downloads": 0,
                "likes": 201,
                "lastModified": "2026-04-15T05:59:19Z",
            },
        ]
        urlopen_mock.return_value = fake_urlopen_json(payload)

        direct_results = _search_huggingface_hub("https://huggingface.co/Qwen/Qwen3.6-35B-A3B", fake_library(), limit=20)
        short_results = _search_huggingface_hub("https://hf.co/Qwen/Qwen3.6-35B-A3B", fake_library(), limit=20)

        self.assertEqual([model["repo"] for model in direct_results], ["Qwen/Qwen3.6-35B-A3B"])
        self.assertEqual([model["repo"] for model in short_results], ["Qwen/Qwen3.6-35B-A3B"])

    @mock.patch("backend_service.state.threading.Thread")
    @mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0)
    @mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=74.9)
    def test_model_download_preflight_sets_live_size_before_starting(
        self,
        _preflight_size,
        _downloaded_bytes,
        thread_cls,
    ):
        fake_thread = mock.Mock()
        thread_cls.return_value = fake_thread

        response = self.client.post("/api/models/download", json={"repo": "Qwen/Qwen3-Coder-Next-FP8"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()["download"]
        self.assertEqual(payload["state"], "downloading")
        self.assertEqual(payload["totalGb"], 74.9)
        thread_cls.assert_called_once()
        fake_thread.start.assert_called_once()

    @mock.patch("backend_service.state._spawn_snapshot_download")
    @mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0)
    @mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=66.99)
    def test_model_download_disables_hf_xet_for_snapshot_download(
        self,
        _preflight_size,
        _downloaded_bytes,
        spawn_download,
    ):
        state = self.client.app.state.chaosengine

        process = mock.Mock()
        process.poll.return_value = 0
        process.returncode = 0
        process.wait.return_value = 0
        spawn_download.return_value = process

        created_threads = []

        class ImmediateThread:
            def __init__(self, *, target=None, daemon=None):
                self.target = target
                self.daemon = daemon
                self.run_target = not created_threads
                created_threads.append(self)

            def start(self):
                if self.run_target and self.target is not None:
                    self.target()

            def join(self, timeout=None):
                return None

        with mock.patch("backend_service.state.threading.Thread", side_effect=ImmediateThread):
            payload = state.start_download("Qwen/Qwen3.6-35B-A3B")

        self.assertEqual(payload["state"], "completed")
        spawn_download.assert_called_once()
        env = spawn_download.call_args.args[1]
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
        self.assertEqual(env["HF_HUB_DISABLE_PROGRESS_BARS"], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")

    @mock.patch("backend_service.state.threading.Thread")
    @mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0)
    @mock.patch(
        "backend_service.helpers.huggingface._hf_repo_preflight_size_gb",
        side_effect=RuntimeError("Hugging Face repository not found: org/missing-model"),
    )
    def test_model_download_preflight_returns_failed_for_missing_repo(
        self,
        _preflight_size,
        _downloaded_bytes,
        thread_cls,
    ):
        response = self.client.post("/api/models/download", json={"repo": "org/missing-model"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()["download"]
        self.assertEqual(payload["state"], "failed")
        self.assertIn("not found on Hugging Face", payload["error"])
        thread_cls.assert_not_called()

    @mock.patch("backend_service.state.threading.Thread")
    @mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", return_value=0)
    @mock.patch(
        "backend_service.helpers.huggingface._hf_repo_preflight_size_gb",
        side_effect=RuntimeError("Hugging Face refused access to org/private-model (HTTP 401). Set HF_TOKEN in Settings."),
    )
    def test_model_download_preflight_returns_failed_for_auth_errors(
        self,
        _preflight_size,
        _downloaded_bytes,
        thread_cls,
    ):
        response = self.client.post("/api/models/download", json={"repo": "org/private-model"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()["download"]
        self.assertEqual(payload["state"], "failed")
        self.assertIn("HF_TOKEN", payload["error"])
        self.assertIn("refused access", payload["error"].lower())
        thread_cls.assert_not_called()

    def test_legacy_seeded_workspace_data_is_filtered_from_saved_files(self):
        self.chat_sessions_path.write_text(
            json.dumps(
                [
                    {
                        "id": "ui-direction",
                        "title": "Compact desktop layout",
                        "updatedAt": "2026-04-13 12:00:00",
                        "model": "Seeded model",
                        "cacheLabel": "Native f16",
                        "messages": [],
                    },
                    {
                        "id": "session-real",
                        "title": "Real chat",
                        "updatedAt": "2026-04-13 12:00:00",
                        "model": "Gemma",
                        "cacheLabel": "Native f16",
                        "messages": [],
                    },
                ]
            ),
            encoding="utf-8",
        )
        self.benchmarks_path.write_text(
            json.dumps(
                [
                    {
                        "id": "baseline",
                        "label": "Seeded baseline",
                    },
                    {
                        "id": "bench-real",
                        "label": "Real benchmark",
                    },
                ]
            ),
            encoding="utf-8",
        )

        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )

        self.assertEqual([session["id"] for session in state.chat_sessions], ["session-real"])
        self.assertEqual([run["id"] for run in state.benchmark_runs], ["bench-real"])

    def test_duplicate_auto_generated_session_titles_are_normalized_on_load(self):
        self.chat_sessions_path.write_text(
            json.dumps(
                [
                    {
                        "id": "session-a",
                        "title": "Explain how cache compression",
                        "updatedAt": "2026-04-13 12:00:00",
                        "model": "Gemma",
                        "cacheLabel": "Native f16",
                        "messages": [
                            {"role": "user", "text": "Explain how cache compression helps long contexts."},
                            {"role": "assistant", "text": "Answer 1"},
                        ],
                    },
                    {
                        "id": "session-b",
                        "title": "Explain how cache compression",
                        "updatedAt": "2026-04-13 11:00:00",
                        "model": "Gemma",
                        "cacheLabel": "Native f16",
                        "messages": [
                            {"role": "user", "text": "Explain how cache compression helps long contexts."},
                            {"role": "assistant", "text": "Answer 2"},
                        ],
                    },
                ]
            ),
            encoding="utf-8",
        )

        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )

        self.assertEqual(
            [session["title"] for session in state.chat_sessions],
            ["Explain how cache compression", "Explain how cache compression (2)"],
        )
        saved = json.loads(self.chat_sessions_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [session["title"] for session in saved],
            ["Explain how cache compression", "Explain how cache compression (2)"],
        )

    def test_model_load_and_chat_generation(self):
        load_response = self.client.post(
            "/api/models/load",
            json={
                "modelRef": "google/gemma-4-E4B-it",
                "modelName": "Gemma 4 E4B Instruct",
                "source": "catalog",
                "backend": "mock",
                "cacheBits": 0,
                "fp16Layers": 0,
            },
        )
        self.assertEqual(load_response.status_code, 200)
        runtime = load_response.json()["runtime"]
        self.assertEqual(runtime["state"], "loaded")
        self.assertEqual(runtime["loadedModel"]["name"], "Gemma 4 E4B Instruct")

        generate_response = self.client.post(
            "/api/chat/generate",
            json={
                "prompt": "Explain how cache compression helps long contexts.",
                "temperature": 0.7,
                "maxTokens": 128,
            },
        )
        self.assertEqual(generate_response.status_code, 200)
        payload = generate_response.json()
        self.assertEqual(payload["assistant"]["role"], "assistant")
        self.assertIn("cache", payload["assistant"]["text"].lower())
        self.assertGreater(payload["assistant"]["metrics"]["completionTokens"], 0)
        self.assertEqual(len(payload["session"]["messages"]), 2)
        self.assertEqual(self.client.app.state.chaosengine.runtime.last_generate_kwargs["system_prompt"], "")

        workspace = self.client.get("/api/workspace").json()
        self.assertEqual(workspace["runtime"]["state"], "loaded")
        self.assertEqual(workspace["server"]["status"], "running")
        self.assertGreaterEqual(workspace["server"]["requestsServed"], 1)

    def test_model_load_rejects_models_not_on_disk(self):
        """Regression: previously a load request for a model not in the library
        would fall through to llama-server's HuggingFace auto-fetch, which on
        Windows blew up with an opaque SSL error (no CA bundle in the bundled
        llama-server.exe). The guard now raises a clear error before we ever
        get there, so users see 'download it first' instead of 'HTTPLIB failed:
        SSL server verification failed'.
        """
        response = self.client.post(
            "/api/models/load",
            json={
                # Deliberately a repo that isn't in fake_library() and has no
                # path — same shape as the phantom Nemotron load that caused
                # the user's SSL failure.
                "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
                "modelName": "Nemotron 3 Nano 4B GGUF",
                "source": "catalog",
                "backend": "mock",
            },
        )
        self.assertEqual(response.status_code, 500)
        detail = response.json()["detail"]
        # Error text must mention the model and point at the fix.
        self.assertIn("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF", detail)
        self.assertIn("Discover", detail)
        # Crucially: llama-server / the runtime should NEVER be invoked.
        self.assertEqual(self.client.app.state.chaosengine.runtime.load_requests, [])

    def test_speculative_mlx_load_clears_warm_pool_and_does_not_keep_previous_model_warm(self):
        state = self.client.app.state.chaosengine
        state.runtime.engine = SimpleNamespace(engine_name="mlx", engine_label="MLX")
        state.runtime.loaded_model = LoadedModelInfo(
            ref="Qwen3-Coder-Next-MLX-4bit",
            name="Qwen3-Coder-Next-MLX-4bit",
            backend="mlx",
            source="library",
            engine="mlx",
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=True,
            contextTokens=32768,
            loadedAt="2026-04-15 00:00:00",
            path="/tmp/qwen3-coder-next-mlx-4bit",
            runtimeTarget="/tmp/qwen3-coder-next-mlx-4bit",
            runtimeNote="Fake runtime",
            speculativeDecoding=False,
            treeBudget=0,
        )
        state.runtime._warm_pool = {
            "stale-warm-model": (
                FakeWarmEngine(),
                LoadedModelInfo(
                    ref="warm/model",
                    name="Warm model",
                    backend="mlx",
                    source="catalog",
                    engine="mlx",
                    cacheStrategy="native",
                    cacheBits=0,
                    fp16Layers=0,
                    fusedAttention=False,
                    fitModelInMemory=True,
                    contextTokens=8192,
                    loadedAt="2026-04-15 00:00:00",
                    runtimeTarget="warm/model",
                    runtimeNote="Fake runtime",
                ),
            )
        }

        response = self.client.post(
            "/api/models/load",
            json={
                "modelRef": "Qwen3-Coder-Next-MLX-4bit",
                "modelName": "Qwen3-Coder-Next-MLX-4bit",
                "source": "library",
                "backend": "mlx",
                "path": "/tmp/qwen3-coder-next-mlx-4bit",
                "cacheStrategy": "native",
                "cacheBits": 0,
                "fp16Layers": 0,
                "contextTokens": 32768,
                "speculativeDecoding": True,
                "treeBudget": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(state.runtime.clear_warm_pool_calls, 1)
        self.assertEqual(state.runtime._warm_pool, {})
        self.assertFalse(state.runtime.load_requests[-1]["keep_warm_previous"])

    def test_repeated_auto_generated_session_titles_get_numbered(self):
        self.client.post(
            "/api/models/load",
            json={
                "modelRef": "google/gemma-4-E4B-it",
                "modelName": "Gemma 4 E4B Instruct",
                "source": "catalog",
                "backend": "mock",
                "cacheBits": 0,
                "fp16Layers": 0,
            },
        )

        first = self.client.post(
            "/api/chat/generate",
            json={
                "prompt": "Explain how cache compression helps long contexts.",
                "temperature": 0.7,
                "maxTokens": 128,
            },
        )
        second = self.client.post(
            "/api/chat/generate",
            json={
                "prompt": "Explain how cache compression helps long contexts.",
                "temperature": 0.7,
                "maxTokens": 128,
            },
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["session"]["title"], "Explain how cache compression")
        self.assertEqual(second.json()["session"]["title"], "Explain how cache compression (2)")

    def test_model_download_delete_clears_tracked_attempt(self):
        repo = "org/stuck-model"
        state = self.client.app.state.chaosengine
        state.image_runtime = SimpleNamespace(unload=mock.Mock(return_value={"activeEngine": "placeholder"}))
        state._downloads[repo] = {
            "repo": repo,
            "state": "cancelled",
            "progress": 0.01,
            "downloadedGb": 0.2,
            "totalGb": 12.0,
            "error": None,
        }
        state._download_cancel[repo] = True
        state._download_tokens[repo] = "tok-1"

        response = self.client.post("/api/models/download/delete", json={"repo": repo})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["state"], "deleted")
        self.assertNotIn(repo, state._downloads)
        self.assertNotIn(repo, state._download_cancel)
        self.assertNotIn(repo, state._download_tokens)

    def test_image_download_delete_removes_cache_and_unloads_matching_models(self):
        repo = "black-forest-labs/FLUX.1-dev"
        hf_cache = Path(self.tempdir.name) / "hf-cache"
        snapshot_dir = hf_cache / "models--black-forest-labs--FLUX.1-dev" / "snapshots" / "rev-1"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model_index.json").write_text("{}", encoding="utf-8")
        (snapshot_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 16)

        state = self.client.app.state.chaosengine
        state.image_runtime = SimpleNamespace(unload=mock.Mock(return_value={"activeEngine": "placeholder"}))
        state._downloads[repo] = {
            "repo": repo,
            "state": "failed",
            "progress": 0.01,
            "downloadedGb": 0.3,
            "totalGb": 24.0,
            "error": "stuck",
        }
        state.runtime.loaded_model = LoadedModelInfo(
            ref=repo,
            name="FLUX.1 Dev",
            backend="mlx",
            source="catalog",
            engine="mock",
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=True,
            contextTokens=8192,
            loadedAt="2026-04-13 00:00:00",
            path=str(snapshot_dir),
            runtimeTarget=repo,
            runtimeNote="Fake runtime",
        )
        warm_engine = FakeWarmEngine()
        state.runtime._warm_pool = {
            "warm-1": (
                warm_engine,
                LoadedModelInfo(
                    ref=repo,
                    name="FLUX.1 Dev",
                    backend="mlx",
                    source="catalog",
                    engine="mock",
                    cacheStrategy="native",
                    cacheBits=0,
                    fp16Layers=0,
                    fusedAttention=False,
                    fitModelInMemory=True,
                    contextTokens=8192,
                    loadedAt="2026-04-13 00:00:00",
                    path=str(snapshot_dir),
                    runtimeTarget=repo,
                    runtimeNote="Fake runtime",
                ),
            )
        }

        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hf_cache)}):
            response = self.client.post("/api/images/download/delete", json={"repo": repo})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result"]["state"], "deleted")
        self.assertFalse((hf_cache / "models--black-forest-labs--FLUX.1-dev").exists())
        self.assertIsNone(state.runtime.loaded_model)
        self.assertEqual(state.runtime._warm_pool, {})
        self.assertEqual(warm_engine.unload_calls, 1)
        state.image_runtime.unload.assert_called_once_with(repo)

    def test_openai_compatible_completion_autoloads_model(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "google/gemma-4-E4B-it",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Summarize the cache compression advantage."},
                ],
                "max_tokens": 80,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertGreater(payload["usage"]["total_tokens"], 0)

    def test_compare_stream_includes_requested_and_actual_runtime_metadata(self):
        response = self.client.post(
            "/api/chat/compare",
            json={
                "prompt": "Test compare",
                "modelA": {
                    "modelRef": "google/gemma-4-E4B-it",
                    "modelName": "Gemma 4 E4B Instruct",
                    "source": "catalog",
                    "backend": "mock",
                    "launch": {
                        "temperature": 0.7,
                        "maxTokens": 32,
                        "cacheStrategy": "native",
                        "cacheBits": 0,
                        "fp16Layers": 0,
                        "fusedAttention": False,
                        "fitModelInMemory": True,
                        "contextTokens": 8192,
                        "speculativeDecoding": True,
                        "treeBudget": 64,
                    },
                },
                "modelB": {
                    "modelRef": "google/gemma-4-E4B-it",
                    "modelName": "Gemma 4 E4B Instruct",
                    "source": "catalog",
                    "backend": "mock",
                    "launch": {
                        "temperature": 0.7,
                        "maxTokens": 32,
                        "cacheStrategy": "native",
                        "cacheBits": 0,
                        "fp16Layers": 0,
                        "fusedAttention": False,
                        "fitModelInMemory": True,
                        "contextTokens": 8192,
                        "speculativeDecoding": False,
                        "treeBudget": 0,
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

        loaded_a = next(event for event in events if event.get("model") == "a" and event.get("loaded"))
        done_a = next(event for event in events if event.get("model") == "a" and event.get("done"))

        self.assertTrue(loaded_a["requestedSpeculativeDecoding"])
        self.assertEqual(loaded_a["requestedTreeBudget"], 64)
        self.assertTrue(loaded_a["speculativeDecoding"])
        self.assertEqual(loaded_a["dflashDraftModel"], "z-lab/Qwen3-4B-DFlash")
        self.assertEqual(loaded_a["cacheLabel"], "Native f16")

        self.assertTrue(done_a["speculativeDecoding"])
        self.assertEqual(done_a["treeBudget"], 64)
        self.assertEqual(done_a["dflashDraftModel"], "z-lab/Qwen3-4B-DFlash")
        self.assertEqual(done_a["dflashAcceptanceRate"], 4.5)

        models_response = self.client.get("/v1/models")
        self.assertEqual(models_response.status_code, 200)
        models = models_response.json()["data"]
        self.assertEqual(models[0]["id"], "google/gemma-4-E4B-it")

    def test_compare_stream_uses_exclusive_loading_and_clears_warm_pool(self):
        state = self.client.app.state.chaosengine
        state.runtime._warm_pool = {
            "stale-warm-model": (
                FakeWarmEngine(),
                LoadedModelInfo(
                    ref="warm/model",
                    name="Warm model",
                    backend="mlx",
                    source="catalog",
                    engine="mock",
                    cacheStrategy="native",
                    cacheBits=0,
                    fp16Layers=0,
                    fusedAttention=False,
                    fitModelInMemory=True,
                    contextTokens=8192,
                    loadedAt="2026-04-15 00:00:00",
                    runtimeTarget="warm/model",
                    runtimeNote="Fake runtime",
                ),
            )
        }

        response = self.client.post(
            "/api/chat/compare",
            json={
                "prompt": "Test compare exclusive loading",
                "modelA": {
                    "modelRef": "google/gemma-4-E4B-it",
                    "modelName": "Gemma 4 E4B Instruct",
                    "source": "catalog",
                    "backend": "mock",
                },
                "modelB": {
                    "modelRef": "meta-llama/Llama-3.2-3B-Instruct",
                    "modelName": "Llama 3.2 3B Instruct",
                    "source": "catalog",
                    "backend": "mock",
                    # Path provided so the on-disk guard in ``state.load_model``
                    # lets the load through — this test isn't exercising the
                    # guard itself, just verifying warm-pool eviction behaviour.
                    "path": "/tmp/llama-3.2-3b-instruct",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(state.runtime.clear_warm_pool_calls, 1)
        self.assertEqual(state.runtime._warm_pool, {})
        self.assertEqual(len(state.runtime.load_requests), 2)
        self.assertTrue(all(not req["keep_warm_previous"] for req in state.runtime.load_requests))

    def test_workspace_surfaces_tracked_runtime_process_when_system_scan_misses_it(self):
        state = self.client.app.state.chaosengine
        state._system_snapshot_provider = lambda: {
            **fake_system_snapshot(),
            "runningLlmProcesses": [],
        }
        state.runtime.engine = SimpleNamespace(
            engine_name="mlx",
            engine_label="MLX",
            process_pid=lambda: 4242,
        )
        state.runtime.loaded_model = LoadedModelInfo(
            ref="google/gemma-4-E4B-it",
            name="Gemma 4 E4B Instruct",
            backend="mlx",
            source="catalog",
            engine="mlx",
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=True,
            contextTokens=8192,
            loadedAt="2026-04-13 00:00:00",
            canonicalRepo="google/gemma-4-E4B-it",
            runtimeTarget="google/gemma-4-E4B-it",
            runtimeNote="Fake runtime",
        )

        with mock.patch(
            "backend_service.state._describe_process",
            return_value={
                "pid": 4242,
                "name": "python",
                "owner": "ChaosEngineAI",
                "memoryGb": 14.9,
                "cpuPercent": 0.0,
                "kind": "mlx_worker",
            },
        ):
            response = self.client.get("/api/workspace")

        self.assertEqual(response.status_code, 200)
        process = response.json()["system"]["runningLlmProcesses"][0]
        self.assertEqual(process["pid"], 4242)
        self.assertEqual(process["kind"], "mlx_worker")
        self.assertEqual(process["modelName"], "Gemma 4 E4B Instruct")
        self.assertEqual(process["modelStatus"], "active")

    def test_workspace_refreshes_stale_tracked_runtime_process_details(self):
        state = self.client.app.state.chaosengine
        state._system_snapshot_provider = lambda: {
            **fake_system_snapshot(),
            "runningLlmProcesses": [
                {
                    "pid": 4242,
                    "name": "chaosengineai",
                    "owner": "ChaosEngineAI",
                    "memoryGb": 0.0,
                    "cpuPercent": 0.0,
                    "kind": "other",
                }
            ],
        }
        state.runtime.engine = SimpleNamespace(
            engine_name="mlx",
            engine_label="MLX",
            process_pid=lambda: 4242,
        )
        state.runtime.loaded_model = LoadedModelInfo(
            ref="mlx-community/Qwen3-Coder-Next-MLX-4bit",
            name="Qwen3-Coder-Next-MLX-4bit",
            backend="mlx",
            source="catalog",
            engine="mlx",
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=True,
            contextTokens=32768,
            loadedAt="2026-04-15 00:00:00",
            canonicalRepo="Qwen/Qwen3-Coder-Next",
            runtimeTarget="mlx-community/Qwen3-Coder-Next-MLX-4bit",
            runtimeNote="Fake runtime",
        )

        with mock.patch(
            "backend_service.state._describe_process",
            return_value={
                "pid": 4242,
                "name": "python",
                "owner": "ChaosEngineAI",
                "memoryGb": 54.6,
                "cpuPercent": 0.0,
                "kind": "mlx_worker",
            },
        ):
            response = self.client.get("/api/workspace")

        self.assertEqual(response.status_code, 200)
        process = response.json()["system"]["runningLlmProcesses"][0]
        self.assertEqual(process["pid"], 4242)
        self.assertEqual(process["name"], "python")
        self.assertEqual(process["kind"], "mlx_worker")
        self.assertEqual(process["memoryGb"], 54.6)
        self.assertEqual(process["modelName"], "Qwen3-Coder-Next-MLX-4bit")
        self.assertEqual(process["modelStatus"], "active")

    def test_reload_logs_launch_settings_reason(self):
        first = self.client.post(
            "/api/models/load",
            json={
                "modelRef": "google/gemma-4-E4B-it",
                "modelName": "Gemma 4 E4B Instruct",
                "source": "catalog",
                "backend": "mock",
                "contextTokens": 8192,
            },
        )
        self.assertEqual(first.status_code, 200)

        second = self.client.post(
            "/api/models/load",
            json={
                "modelRef": "google/gemma-4-E4B-it",
                "modelName": "Gemma 4 E4B Instruct",
                "source": "catalog",
                "backend": "mock",
                "contextTokens": 16384,
            },
        )
        self.assertEqual(second.status_code, 200)

        log_messages = [entry["message"] for entry in self.client.app.state.chaosengine.logs]
        self.assertTrue(
            any(
                "launch settings changed" in message and "context 8192 -> 16384" in message
                for message in log_messages
            )
        )

    def test_mlx_cache_only_change_uses_profile_update_without_weight_reload(self):
        state = self.client.app.state.chaosengine
        state.runtime.engine = SimpleNamespace(engine_name="mlx", engine_label="MLX")
        state.runtime.loaded_model = LoadedModelInfo(
            ref="google/gemma-4-E4B-it",
            name="Gemma 4 E4B Instruct",
            backend="mlx",
            source="catalog",
            engine="mlx",
            cacheStrategy="native",
            cacheBits=0,
            fp16Layers=0,
            fusedAttention=False,
            fitModelInMemory=True,
            contextTokens=8192,
            loadedAt="2026-04-13 00:00:00",
            canonicalRepo="google/gemma-4-E4B-it",
            runtimeTarget="google/gemma-4-E4B-it",
            runtimeNote="Fake runtime",
        )

        with mock.patch.object(state.runtime, "load_model", wraps=state.runtime.load_model) as load_model_spy:
            response = self.client.post(
                "/api/models/load",
                json={
                    "modelRef": "google/gemma-4-E4B-it",
                    "modelName": "Gemma 4 E4B Instruct",
                    "canonicalRepo": "google/gemma-4-E4B-it",
                    "source": "catalog",
                    "backend": "mlx",
                    "cacheStrategy": "rotorquant",
                    "cacheBits": 4,
                    "fp16Layers": 2,
                    "fusedAttention": True,
                    "contextTokens": 8192,
                },
            )

        self.assertEqual(response.status_code, 200)
        load_model_spy.assert_not_called()
        self.assertEqual(len(state.runtime.profile_updates), 1)
        self.assertEqual(state.runtime.loaded_model.cacheStrategy, "rotorquant")
        self.assertEqual(state.runtime.loaded_model.cacheBits, 4)
        self.assertEqual(state.runtime.loaded_model.fp16Layers, 2)
        self.assertTrue(state.runtime.loaded_model.fusedAttention)

    def test_snapshot_download_process_redirects_progress_output_to_log_file(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
            with mock.patch("backend_service.state.subprocess.Popen") as popen:
                _spawn_snapshot_download("org/model", {"PYTHONUNBUFFERED": "1"}, handle)

        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stdout"], handle)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["text"], True)

    def test_snapshot_download_passes_empty_allowlist_for_standard_repos(self):
        """Non-video repos should not receive an allowlist arg (empty string)."""
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
            with mock.patch("backend_service.state.subprocess.Popen") as popen:
                _spawn_snapshot_download("org/model", {}, handle)

        args = popen.call_args.args[0]
        # args are [python, "-c", helper, repo, allow_patterns_json]. The
        # final slot is the empty string when no allowlist is set.
        self.assertEqual(args[3], "org/model")
        self.assertEqual(args[4], "")

    def test_snapshot_download_passes_allowlist_when_supplied(self):
        """A supplied allowlist arrives at the subprocess as JSON."""
        patterns = ["model_index.json", "transformer/**"]
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
            with mock.patch("backend_service.state.subprocess.Popen") as popen:
                _spawn_snapshot_download(
                    "org/video-model", {}, handle, allow_patterns=patterns,
                )

        args = popen.call_args.args[0]
        self.assertEqual(args[3], "org/video-model")
        self.assertEqual(json.loads(args[4]), patterns)

    def test_preview_math_reduces_cache_size(self):
        preview = compute_cache_preview(
            bits=3,
            fp16_layers=4,
            num_layers=32,
            num_heads=32,
            hidden_size=4096,
            context_tokens=8192,
            params_b=7.0,
            system_stats=fake_system_snapshot(),
            strategy="rotorquant",
        )
        self.assertLess(preview["optimizedCacheGb"], preview["baselineCacheGb"])
        self.assertGreater(preview["compressionRatio"], 1.0)

    def test_manual_cache_backend_install_returns_helpful_error(self):
        for package_name in ("chaosengine", "chaos-engine"):
            response = self.client.post(
                "/api/setup/install-package",
                json={"package": package_name},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("not published on PyPI", response.json()["detail"])
            self.assertIn("pip install -e /path/to/ChaosEngine", response.json()["detail"])

    def test_convert_endpoint_returns_conversion_payload(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )
        state.runtime.convert_model = lambda **kwargs: {  # type: ignore[method-assign]
            "sourceRef": kwargs["source_ref"],
            "sourcePath": kwargs["source_path"],
            "sourceLabel": "Gemma 4 E4B Instruct",
            "hfRepo": kwargs["hf_repo"],
            "outputPath": "/tmp/gemma-4-e4b-it-mlx",
            "quantize": kwargs["quantize"],
            "qBits": kwargs["q_bits"],
            "dtype": kwargs["dtype"],
            "ggufMetadata": None,
            "log": "[INFO] Quantized model with 4.5 bits per weight.",
        }
        client = make_test_client(state)

        response = client.post(
            "/api/models/convert",
            json={
                "modelRef": "google/gemma-4-E4B-it",
                "hfRepo": "google/gemma-4-E4B-it",
                "outputPath": "/tmp/gemma-4-e4b-it-mlx",
                "quantize": True,
                "qBits": 4,
                "dtype": "float16",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["conversion"]["outputPath"], "/tmp/gemma-4-e4b-it-mlx")
        self.assertTrue(payload["conversion"]["quantize"])
        self.assertEqual(payload["conversion"]["qBits"], 4)
        self.assertIn("library", payload)

    def test_session_update_renames_and_switches_model(self):
        create_response = self.client.post("/api/chat/sessions", json={"title": "New chat"})
        self.assertEqual(create_response.status_code, 200)
        session = create_response.json()["session"]

        update_response = self.client.patch(
            f"/api/chat/sessions/{session['id']}",
            json={
                "title": "Gemma thread",
                "model": "Gemma 4 E4B Instruct",
                "modelRef": "google/gemma-4-E4B-it",
                "modelSource": "catalog",
                "modelBackend": "mlx",
                "thinkingMode": "auto",
            },
        )
        self.assertEqual(update_response.status_code, 200)
        updated = update_response.json()["session"]
        self.assertEqual(updated["title"], "Gemma thread")
        self.assertEqual(updated["modelRef"], "google/gemma-4-E4B-it")
        self.assertEqual(updated["thinkingMode"], "auto")

    def test_session_update_allows_explicit_null_to_clear_stale_model_metadata(self):
        create_response = self.client.post("/api/chat/sessions", json={"title": "New chat"})
        self.assertEqual(create_response.status_code, 200)
        session = create_response.json()["session"]

        seeded = self.client.patch(
            f"/api/chat/sessions/{session['id']}",
            json={
                "model": "Qwen3.5-9B",
                "modelRef": "mlx-community/Qwen3.5-9B-4bit",
                "canonicalRepo": "Qwen/Qwen3.5-9B",
                "modelSource": "library",
                "modelPath": "/tmp/qwen3.5-9b",
                "modelBackend": "mlx",
                "speculativeDecoding": True,
                "treeBudget": 64,
                "dflashDraftModel": "z-lab/Qwen3.5-9B-DFlash",
            },
        )
        self.assertEqual(seeded.status_code, 200)

        cleared = self.client.patch(
            f"/api/chat/sessions/{session['id']}",
            json={
                "model": "Qwen3-Coder-Next-MLX-4bit",
                "modelRef": "Qwen3-Coder-Next-MLX-4bit",
                "canonicalRepo": None,
                "modelSource": "library",
                "modelPath": None,
                "modelBackend": "mlx",
                "dflashDraftModel": None,
            },
        )
        self.assertEqual(cleared.status_code, 200)
        updated = cleared.json()["session"]
        self.assertEqual(updated["model"], "Qwen3-Coder-Next-MLX-4bit")
        self.assertEqual(updated["modelRef"], "Qwen3-Coder-Next-MLX-4bit")
        self.assertIsNone(updated["canonicalRepo"])
        self.assertIsNone(updated["modelPath"])
        self.assertIsNone(updated["dflashDraftModel"])
        self.assertTrue(updated["speculativeDecoding"])
        self.assertEqual(updated["treeBudget"], 64)

    def test_settings_endpoint_updates_model_directories_and_launch_defaults(self):
        response = self.client.patch(
            "/api/settings",
            json={
                "modelDirectories": [
                    {
                        "label": "AI Models",
                        "path": "~/AI_Models",
                        "enabled": True,
                        "source": "user",
                    }
                ],
                "preferredServerPort": 8899,
                "allowRemoteConnections": True,
                "launchPreferences": {
                    "contextTokens": 16384,
                    "maxTokens": 1024,
                    "temperature": 0.5,
                    "cacheBits": 0,
                    "fp16Layers": 0,
                    "fusedAttention": True,
                    "cacheStrategy": "native",
                    "fitModelInMemory": False,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        settings = response.json()["settings"]
        self.assertEqual(settings["preferredServerPort"], 8899)
        self.assertTrue(settings["allowRemoteConnections"])
        self.assertEqual(settings["launchPreferences"]["contextTokens"], 16384)
        self.assertEqual(settings["launchPreferences"]["cacheStrategy"], "native")
        self.assertEqual(settings["modelDirectories"][0]["label"], "AI Models")

    def test_settings_change_of_remote_provider_base_requires_new_key(self):
        seeded = self.client.patch(
            "/api/settings",
            json={
                "remoteProviders": [
                    {
                        "id": "remote-1",
                        "label": "Primary",
                        "apiBase": "https://api.openai.com/v1",
                        "apiKey": "sk-test-secret",
                        "model": "gpt-4o-mini",
                    }
                ]
            },
        )
        self.assertEqual(seeded.status_code, 200)

        changed = self.client.patch(
            "/api/settings",
            json={
                "remoteProviders": [
                    {
                        "id": "remote-1",
                        "label": "Primary",
                        "apiBase": "https://attacker.example/v1",
                        "apiKey": "",
                        "model": "gpt-4o-mini",
                    }
                ]
            },
        )
        self.assertEqual(changed.status_code, 400)
        self.assertIn("Re-enter the API key", changed.json()["detail"])

    def test_explicit_gguf_path_wins_over_hf_cache_library_entry(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=lambda: [
                {
                    "name": "ggml-org/test-model-stories260K",
                    "path": "/tmp/hf-cache-dir",
                    "format": "HF cache",
                    "sizeGb": 0.1,
                    "lastModified": "2026-04-05 12:00",
                    "actions": ["Run Chat"],
                }
            ],
            settings_path=self.settings_path,
            benchmarks_path=self.benchmarks_path,
            chat_sessions_path=self.chat_sessions_path,
        )

        runtime_target, resolved_backend = state._resolve_model_target(
            model_ref="ggml-org/test-model-stories260K",
            path="/tmp/test-model.gguf",
            backend="auto",
        )

        self.assertEqual(runtime_target, "/tmp/test-model.gguf")
        self.assertEqual(resolved_backend, "llama.cpp")

    def test_recursive_discovery_finds_nested_model_directories(self):
        models_root = Path(self.tempdir.name) / "AI_Models"
        nested_model = models_root / "publisher" / "family" / "variant"
        nested_model.mkdir(parents=True)
        (nested_model / "config.json").write_text("{}", encoding="utf-8")
        (nested_model / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        (nested_model / "model-00001-of-00001.safetensors").write_bytes(b"x" * 4096)

        library = _discover_local_models(
            [
                {
                    "label": "AI Models",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], "variant")
        self.assertGreater(library[0]["sizeGb"], 0)

    def test_discovery_classifies_sharded_mlx_directory_as_mlx(self):
        models_root = Path(self.tempdir.name) / "AI_Models"
        mlx_dir = models_root / "mlx-community" / "Qwen3.5-9B-MLX-4bit"
        mlx_dir.mkdir(parents=True)
        (mlx_dir / "config.json").write_text(
            '{"quantization":{"bits":4,"group_size":64,"mode":"affine"},"dtype":"bfloat16"}',
            encoding="utf-8",
        )
        (mlx_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (mlx_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        (mlx_dir / "model-00001-of-00002.safetensors").write_bytes(b"x" * 4096)
        (mlx_dir / "model-00002-of-00002.safetensors").write_bytes(b"y" * 4096)

        library = _discover_local_models(
            [
                {
                    "label": "AI Models",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], mlx_dir.name)
        self.assertEqual(library[0]["format"], "MLX")
        self.assertEqual(library[0]["quantization"], "4-bit")
        self.assertEqual(library[0]["backend"], "mlx")
        self.assertEqual(library[0]["sourceKind"], "Directory")
        self.assertFalse(library[0]["broken"])

    def test_discovery_classifies_hf_cache_mlx_repo_by_storage_format(self):
        models_root = Path(self.tempdir.name) / "HF"
        hf_repo = models_root / "models--mlx-community--Qwen3.5-9B-4bit" / "snapshots" / "1234"
        hf_repo.mkdir(parents=True)
        (hf_repo / "config.json").write_text(
            '{"quantization":{"bits":4,"group_size":64,"mode":"affine"},"dtype":"bfloat16"}',
            encoding="utf-8",
        )
        (hf_repo / "tokenizer.json").write_text("{}", encoding="utf-8")
        (hf_repo / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
        (hf_repo / "model-00001-of-00002.safetensors").write_bytes(b"x" * 4096)
        (hf_repo / "model-00002-of-00002.safetensors").write_bytes(b"y" * 4096)

        library = _discover_local_models(
            [
                {
                    "label": "HF",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], "mlx-community/Qwen3.5-9B-4bit")
        self.assertEqual(library[0]["format"], "MLX")
        self.assertEqual(library[0]["quantization"], "4-bit")
        self.assertEqual(library[0]["backend"], "mlx")
        self.assertEqual(library[0]["sourceKind"], "HF cache")
        self.assertFalse(library[0]["broken"])

    def test_discovery_marks_nvfp4_modelopt_repo_as_unsupported_for_mlx(self):
        models_root = Path(self.tempdir.name) / "HF"
        hf_repo = models_root / "models--LilaRest--gemma-4-31B-it-NVFP4-turbo" / "snapshots" / "1234"
        hf_repo.mkdir(parents=True)
        (hf_repo / "config.json").write_text(
            '{"quantization_config":{"bits":4,"quant_algo":"NVFP4","quant_method":"modelopt"},"torch_dtype":"bfloat16"}',
            encoding="utf-8",
        )
        (hf_repo / "tokenizer.json").write_text("{}", encoding="utf-8")
        (hf_repo / "model-00001-of-00001.safetensors").write_bytes(b"x" * 4096)

        library = _discover_local_models(
            [
                {
                    "label": "HF",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], "LilaRest/gemma-4-31B-it-NVFP4-turbo")
        self.assertEqual(library[0]["format"], "Transformers")
        self.assertEqual(library[0]["quantization"], "NVFP4")
        self.assertEqual(library[0]["backend"], "mlx")
        self.assertEqual(library[0]["sourceKind"], "HF cache")
        self.assertTrue(library[0]["broken"])
        self.assertIn("NVFP4", library[0]["brokenReason"])
        self.assertIn("not supported by the MLX runtime", library[0]["brokenReason"])

    def test_discovery_marks_partial_hf_sharded_snapshot_as_broken(self):
        models_root = Path(self.tempdir.name) / "HF"
        hf_repo = models_root / "models--Qwen--Qwen3.6-35B-A3B"
        snapshot = hf_repo / "snapshots" / "1234"
        blobs = hf_repo / "blobs"
        snapshot.mkdir(parents=True)
        blobs.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        for index in range(1, 8):
            (snapshot / f"model-{index:05d}-of-00026.safetensors").write_bytes(b"x" * 4096)
        (blobs / "partial-shard.incomplete").write_bytes(b"x" * 1024)

        library = _discover_local_models(
            [
                {
                    "label": "HF",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(library[0]["format"], "Transformers")
        self.assertEqual(library[0]["sourceKind"], "HF cache")
        self.assertTrue(library[0]["broken"])
        self.assertIn("incomplete", library[0]["brokenReason"].lower())

    def test_discovery_marks_partial_local_gguf_directory_as_broken(self):
        models_root = Path(self.tempdir.name) / "AI_Models"
        gguf_dir = models_root / "unsloth" / "Qwen3.6-35B-A3B-GGUF"
        gguf_dir.mkdir(parents=True)
        (gguf_dir / "mmproj-F32.gguf").write_bytes(b"x" * 4096)
        (gguf_dir / "downloading_Qwen3.6-35B-A3B-UD-Q4_K_S.gguf.part").write_bytes(b"y" * 4096)

        library = _discover_local_models(
            [
                {
                    "label": "AI Models",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], "Qwen3.6-35B-A3B-GGUF")
        self.assertEqual(library[0]["format"], "GGUF")
        self.assertEqual(library[0]["sourceKind"], "Directory")
        self.assertTrue(library[0]["broken"])
        self.assertIn("still downloading", library[0]["brokenReason"])

    def test_resolve_gguf_path_ignores_mmproj_only_directory(self):
        gguf_dir = Path(self.tempdir.name) / "Qwen3.6-35B-A3B-GGUF"
        gguf_dir.mkdir(parents=True)
        (gguf_dir / "mmproj-F32.gguf").write_bytes(b"x" * 4096)
        (gguf_dir / "downloading_Qwen3.6-35B-A3B-UD-Q4_K_S.gguf.part").write_bytes(b"y" * 4096)

        self.assertIsNone(_resolve_gguf_path(str(gguf_dir), None))
        self.assertIsNone(_resolve_gguf_path(str(gguf_dir / "mmproj-F32.gguf"), None))

    def test_discovery_prefers_gguf_for_config_plus_gguf_directory(self):
        models_root = Path(self.tempdir.name) / "AI_Models"
        gguf_dir = models_root / "Jackrong" / "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
        gguf_dir.mkdir(parents=True)
        (gguf_dir / "config.json").write_text("{}", encoding="utf-8")
        (gguf_dir / "Qwen3.5-27B.Q4_K_M.gguf").write_bytes(b"x" * 4096)
        (gguf_dir / "mmproj-BF16.gguf").write_bytes(b"y" * 2048)

        library = _discover_local_models(
            [
                {
                    "label": "AI Models",
                    "path": str(models_root),
                    "enabled": True,
                    "source": "user",
                }
            ]
        )

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["name"], gguf_dir.name)
        self.assertEqual(library[0]["format"], "GGUF")
        self.assertEqual(library[0]["quantization"], "Q4_K_M")
        self.assertEqual(library[0]["backend"], "llama.cpp")
        self.assertEqual(library[0]["sourceKind"], "Directory")
        self.assertFalse(library[0]["broken"])
        self.assertIsNone(library[0]["brokenReason"])

    def test_model_load_rejects_broken_library_entry(self):
        target = Path(self.tempdir.name) / "broken-model"
        target.mkdir(parents=True)

        self.client.app.state.chaosengine._library_provider = lambda: [
            {
                "name": "LilaRest/gemma-4-31B-it-NVFP4-turbo",
                "path": str(target),
                "format": "Transformers",
                "sourceKind": "Directory",
                "quantization": "NVFP4",
                "backend": "mlx",
                "sizeGb": 1.0,
                "lastModified": "2026-04-13 12:00",
                "actions": ["Run Chat"],
                "broken": True,
                "brokenReason": "This model uses NVFP4 quantisation (via modelopt), which is not supported by the MLX runtime.",
            }
        ]

        response = self.client.post(
            "/api/models/load",
            json={
                "modelRef": "LilaRest/gemma-4-31B-it-NVFP4-turbo",
                "modelName": "Gemma 4 31B NVFP4",
                "path": str(target),
                "source": "library",
                "backend": "auto",
                "cacheBits": 0,
                "fp16Layers": 0,
            },
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Cannot load", response.json()["detail"])
        self.assertIn("NVFP4", response.json()["detail"])

    def test_reveal_model_path_endpoint_returns_resolved_path(self):
        target = Path(self.tempdir.name) / "example.gguf"
        target.write_bytes(b"model")

        with mock.patch("backend_service.helpers.discovery.subprocess.Popen") as popen:
            response = self.client.post("/api/models/reveal", json={"path": str(target)})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["revealed"], str(target.resolve()))
        popen.assert_called()


class VideoRepoAllowPatternsTests(unittest.TestCase):
    """``_video_repo_allow_patterns`` scopes video downloads to the diffusers
    layout. Without this guard ``snapshot_download`` pulls every historical
    checkpoint sibling in repos like Lightricks/LTX-Video — turning a 2 GB
    pipeline into a 200+ GB download.
    """

    def test_returns_none_for_non_video_repos(self):
        from backend_service.helpers.video import _video_repo_allow_patterns

        self.assertIsNone(_video_repo_allow_patterns("meta-llama/Llama-2-7b-hf"))
        self.assertIsNone(_video_repo_allow_patterns("stabilityai/stable-diffusion-xl-base-1.0"))
        self.assertIsNone(_video_repo_allow_patterns(""))

    def test_returns_diffusers_layout_for_known_video_repo(self):
        from backend_service.helpers.video import _video_repo_allow_patterns

        patterns = _video_repo_allow_patterns("Lightricks/LTX-Video")
        self.assertIsNotNone(patterns)
        assert patterns is not None  # for the type-checker
        # These folders are the core of every diffusers video pipeline we
        # ship. If any of them disappears the download will start and then
        # fail to load — so they're worth asserting on explicitly.
        self.assertIn("model_index.json", patterns)
        self.assertIn("transformer/**", patterns)
        self.assertIn("vae/**", patterns)
        self.assertIn("text_encoder/**", patterns)
        self.assertIn("scheduler/**", patterns)
        self.assertIn("tokenizer/**", patterns)

    def test_returns_mlx_layout_for_ltx2_repo(self):
        from backend_service.helpers.video import _video_repo_allow_patterns

        patterns = _video_repo_allow_patterns("prince-canuma/LTX-2-distilled")
        self.assertIsNotNone(patterns)
        assert patterns is not None
        self.assertIn("transformer/**", patterns)
        self.assertIn("vae/**", patterns)
        self.assertIn("text_encoder/**", patterns)
        self.assertIn("tokenizer/**", patterns)
        self.assertIn("*spatial-upscaler*.safetensors", patterns)
        self.assertNotIn("model_index.json", patterns)

    def test_returns_fresh_list_each_call(self):
        """Callers get their own copy so mutating the list doesn't leak
        back into the module-level constant."""
        from backend_service.helpers.video import _video_repo_allow_patterns

        first = _video_repo_allow_patterns("Lightricks/LTX-Video")
        second = _video_repo_allow_patterns("Lightricks/LTX-Video")
        self.assertEqual(first, second)
        assert first is not None  # for the type-checker
        first.append("leak-check")
        again = _video_repo_allow_patterns("Lightricks/LTX-Video")
        assert again is not None
        self.assertNotIn("leak-check", again)


if __name__ == "__main__":
    unittest.main()
