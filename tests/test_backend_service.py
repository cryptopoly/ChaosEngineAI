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
from backend_service.inference import GenerationResult, LoadedModelInfo
from backend_service.state import ChaosEngineState, _spawn_snapshot_download
from backend_service.helpers.discovery import _discover_local_models


def fake_system_snapshot():
    return {
        "platform": "Darwin",
        "arch": "arm64",
        "hardwareSummary": "Apple Silicon / 48 GB unified memory",
        "backendLabel": "Python sidecar",
        "appVersion": "0.5.0",
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


class FakeRuntime:
    def __init__(self) -> None:
        self.engine = SimpleNamespace(engine_name="mock", engine_label="No backend")
        self.loaded_model = None
        self.runtime_note = None
        self.last_generate_kwargs = None
        self._warm_pool = {}
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
        return []

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
        progress_callback=None,
    ) -> LoadedModelInfo:
        if callable(progress_callback):
            progress_callback({"phase": "loading", "percent": 100, "message": "Fake runtime loaded"})
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
            path=path,
            runtimeTarget=runtime_target or path or model_ref,
            runtimeNote="Fake runtime",
        )
        self.loaded_model = loaded
        self.runtime_note = loaded.runtimeNote
        return loaded

    def unload_model(self) -> None:
        self.loaded_model = None
        self.runtime_note = None

    def unload_warm_model_by_ref(self, ref: str) -> bool:
        return False

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
        self.client = TestClient(create_app(state=state))

    def tearDown(self):
        self.tempdir.cleanup()

    def test_health_reports_runtime_metadata(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("appVersion", payload)
        self.assertEqual(payload["engine"], "mock")

    def test_fresh_state_starts_without_seeded_workspace_data(self):
        workspace = self.client.get("/api/workspace").json()

        self.assertEqual(workspace["chatSessions"], [])
        self.assertEqual(workspace["benchmarks"], [])

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
        self.assertIn("Thinking mode is OFF", self.client.app.state.chaosengine.runtime.last_generate_kwargs["system_prompt"])

        workspace = self.client.get("/api/workspace").json()
        self.assertEqual(workspace["runtime"]["state"], "loaded")
        self.assertEqual(workspace["server"]["status"], "running")
        self.assertGreaterEqual(workspace["server"]["requestsServed"], 1)

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

        models_response = self.client.get("/v1/models")
        self.assertEqual(models_response.status_code, 200)
        models = models_response.json()["data"]
        self.assertEqual(models[0]["id"], "google/gemma-4-E4B-it")

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

    def test_snapshot_download_process_redirects_progress_output_to_log_file(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as handle:
            with mock.patch("backend_service.state.subprocess.Popen") as popen:
                _spawn_snapshot_download("org/model", {"PYTHONUNBUFFERED": "1"}, handle)

        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stdout"], handle)
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["text"], True)

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
        client = TestClient(create_app(state=state))

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


if __name__ == "__main__":
    unittest.main()
