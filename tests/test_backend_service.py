import unittest
from pathlib import Path
import tempfile
from unittest import mock

from fastapi.testclient import TestClient

from backend_service.app import compute_cache_preview, create_app
from backend_service.state import ChaosEngineState
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


class ChaosEngineBackendTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.settings_path = Path(self.tempdir.name) / "settings.json"
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
        )
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

        workspace = self.client.get("/api/workspace").json()
        self.assertEqual(workspace["runtime"]["state"], "loaded")
        self.assertEqual(workspace["server"]["status"], "running")
        self.assertGreaterEqual(workspace["server"]["requestsServed"], 1)

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
        )
        self.assertLess(preview["optimizedCacheGb"], preview["baselineCacheGb"])
        self.assertGreater(preview["compressionRatio"], 1.0)

    def test_convert_endpoint_returns_conversion_payload(self):
        state = ChaosEngineState(
            system_snapshot_provider=fake_system_snapshot,
            library_provider=fake_library,
            settings_path=self.settings_path,
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
            },
        )
        self.assertEqual(update_response.status_code, 200)
        updated = update_response.json()["session"]
        self.assertEqual(updated["title"], "Gemma thread")
        self.assertEqual(updated["modelRef"], "google/gemma-4-E4B-it")

    def test_settings_endpoint_updates_model_directories_and_launch_defaults(self):
        response = self.client.patch(
            "/api/settings",
            json={
                "modelDirectories": [
                    {
                        "label": "AI Models",
                        "path": "/Users/dan/AI_Models",
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
