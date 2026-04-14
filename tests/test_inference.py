import unittest
from unittest import mock

from backend_service.inference import (
    BackendCapabilities,
    LlamaCppEngine,
    RepeatedLineGuard,
    RuntimeController,
    LoadedModelInfo,
    _gguf_startup_fallback_note,
    _is_local_target,
)
from backend_service.state import _compose_chat_system_prompt


class RepeatedLineGuardTests(unittest.TestCase):
    def test_raises_for_runaway_repeated_long_lines(self):
        guard = RepeatedLineGuard(max_repeats=4)
        repeated = 'Wait, I need to check if there is a specific character in the context of the "Tongyi" model.'

        with self.assertRaises(RuntimeError):
            for _ in range(4):
                guard.feed(repeated + "\n")

    def test_ignores_short_repeated_lines(self):
        guard = RepeatedLineGuard(max_repeats=3)

        for _ in range(6):
            guard.feed("hello\n")

        guard.flush()


class LlamaCppCommandTests(unittest.TestCase):
    def _capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=True,
            llamaServerPath="/usr/local/bin/llama-server",
        )

    def test_build_command_enables_reasoning_flags_when_supported(self):
        engine = LlamaCppEngine(self._capabilities())

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=True),
        ):
            command, _runtime_note = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )

        self.assertIn("--reasoning-format", command)
        self.assertIn("deepseek", command)
        self.assertIn("--reasoning", command)
        self.assertIn("off", command)

    def test_build_command_skips_reasoning_flags_when_unsupported(self):
        engine = LlamaCppEngine(self._capabilities())

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
        ):
            command, _runtime_note = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )

        self.assertNotIn("--reasoning-format", command)
        self.assertNotIn("--reasoning", command)


class GgufTargetDetectionTests(unittest.TestCase):
    def test_treats_huggingface_repo_id_as_non_local(self):
        self.assertFalse(_is_local_target("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF"))

    def test_treats_absolute_and_relative_paths_as_local_candidates(self):
        self.assertTrue(_is_local_target("/tmp/model.gguf"))
        self.assertTrue(_is_local_target("./models/model.gguf"))

    def test_formats_startup_fallback_note_with_requested_strategy(self):
        self.assertEqual(
            _gguf_startup_fallback_note("RotorQuant"),
            "GGUF startup failed with RotorQuant cache, so ChaosEngineAI retried with the standard f16 KV cache.",
        )


class LlamaCppFallbackMetadataTests(unittest.TestCase):
    def _capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=True,
            llamaServerPath="/usr/local/bin/llama-server",
        )

    def test_startup_fallback_records_native_actual_runtime(self):
        engine = LlamaCppEngine(self._capabilities())

        fake_process = mock.Mock()
        fake_process.poll.return_value = None

        with (
            mock.patch.object(engine, "_build_command", side_effect=[(["llama-server"], None), (["llama-server"], None)]),
            mock.patch.object(engine, "_wait_for_server", side_effect=[RuntimeError("unknown cache type"), None]),
            mock.patch.object(engine, "_cleanup_process"),
            mock.patch("backend_service.inference.subprocess.Popen", return_value=fake_process),
        ):
            loaded = engine.load_model(
                model_ref="qwen",
                model_name="Qwen",
                canonical_repo=None,
                source="library",
                backend="llama.cpp",
                path=None,
                runtime_target="lmstudio-community/Qwen3.5-35B-A3B-GGUF",
                cache_strategy="rotorquant",
                cache_bits=3,
                fp16_layers=4,
                fused_attention=False,
                fit_model_in_memory=True,
                context_tokens=8192,
            )

        self.assertEqual(loaded.cacheStrategy, "native")
        self.assertEqual(loaded.cacheBits, 0)
        self.assertEqual(loaded.fp16Layers, 0)
        self.assertEqual(
            loaded.runtimeNote,
            "GGUF startup failed with RotorQuant cache, so ChaosEngineAI retried with the standard f16 KV cache.",
        )


class ChatSystemPromptTests(unittest.TestCase):
    def test_chat_policy_is_prepended(self):
        prompt = _compose_chat_system_prompt("Answer in one sentence.", "off")

        self.assertIn("Give the final answer directly", prompt)
        self.assertIn("Thinking mode is OFF", prompt)
        self.assertTrue(prompt.endswith("Answer in one sentence."))

    def test_auto_mode_skips_explicit_thinking_off_policy(self):
        prompt = _compose_chat_system_prompt("Answer in one sentence.", "auto")

        self.assertIn("Give the final answer directly", prompt)
        self.assertIn("Thinking mode is AUTO", prompt)
        self.assertNotIn("Thinking mode is OFF", prompt)


class DummyEngine:
    engine_name = "dummy"
    engine_label = "Dummy runtime"

    _next_pid = 1000

    def __init__(self, label: str) -> None:
        self.label = label
        self.load_requests: list[dict[str, object]] = []
        self.unload_calls = 0
        self.pid = DummyEngine._next_pid
        DummyEngine._next_pid += 1

    def load_model(self, **kwargs: object) -> LoadedModelInfo:
        self.load_requests.append(dict(kwargs))
        return LoadedModelInfo(
            ref=str(kwargs["model_ref"]),
            name=str(kwargs["model_name"]),
            backend=str(kwargs["backend"]),
            source=str(kwargs["source"]),
            engine=self.engine_name,
            cacheStrategy=str(kwargs["cache_strategy"]),
            cacheBits=int(kwargs["cache_bits"]),
            fp16Layers=int(kwargs["fp16_layers"]),
            fusedAttention=bool(kwargs["fused_attention"]),
            fitModelInMemory=bool(kwargs["fit_model_in_memory"]),
            contextTokens=int(kwargs["context_tokens"]),
            loadedAt="now",
            path=kwargs.get("path") if isinstance(kwargs.get("path"), str) else None,
            runtimeTarget=kwargs.get("runtime_target") if isinstance(kwargs.get("runtime_target"), str) else None,
            runtimeNote=self.label,
        )

    def unload_model(self) -> None:
        self.unload_calls += 1

    def process_pid(self) -> int | None:
        return self.pid


class RuntimeControllerWarmPoolTests(unittest.TestCase):
    def _capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=False,
        )

    def _controller(self) -> RuntimeController:
        with mock.patch("backend_service.inference.get_backend_capabilities", return_value=self._capabilities()):
            return RuntimeController()

    def _load(self, controller: RuntimeController, model_ref: str, *, context_tokens: int) -> LoadedModelInfo:
        return controller.load_model(
            model_ref=model_ref,
            model_name=model_ref,
            source="catalog",
            backend="mlx",
            path=None,
            runtime_target=model_ref,
            cache_strategy="native",
            cache_bits=0,
            fp16_layers=0,
            fused_attention=False,
            fit_model_in_memory=True,
            context_tokens=context_tokens,
        )

    def test_same_profile_reuses_warm_model(self):
        controller = self._controller()
        engine_a = DummyEngine("A-8k")
        engine_b = DummyEngine("B-8k")
        controller._select_engine = mock.Mock(side_effect=[engine_a, engine_b])

        self._load(controller, "model-a", context_tokens=8192)
        self._load(controller, "model-b", context_tokens=8192)
        loaded = self._load(controller, "model-a", context_tokens=8192)

        self.assertEqual(controller._select_engine.call_count, 2)
        self.assertIs(controller.engine, engine_a)
        self.assertEqual(loaded.contextTokens, 8192)
        self.assertEqual(engine_a.unload_calls, 0)

    def test_profile_change_does_not_reuse_stale_warm_model(self):
        controller = self._controller()
        engine_a_8k = DummyEngine("A-8k")
        engine_b = DummyEngine("B-8k")
        engine_a_1m = DummyEngine("A-1m")
        controller._select_engine = mock.Mock(side_effect=[engine_a_8k, engine_b, engine_a_1m])

        self._load(controller, "model-a", context_tokens=8192)
        self._load(controller, "model-b", context_tokens=8192)
        loaded = self._load(controller, "model-a", context_tokens=1_000_000)

        self.assertEqual(controller._select_engine.call_count, 3)
        self.assertIs(controller.engine, engine_a_1m)
        self.assertEqual(loaded.contextTokens, 1_000_000)
        self.assertEqual(len(engine_a_8k.load_requests), 1)
        self.assertEqual(len(engine_a_1m.load_requests), 1)

    def test_profile_change_unloads_same_model_instead_of_parking_duplicate(self):
        controller = self._controller()
        engine_a_8k = DummyEngine("A-8k")
        engine_a_1m = DummyEngine("A-1m")
        controller._select_engine = mock.Mock(side_effect=[engine_a_8k, engine_a_1m])

        self._load(controller, "model-a", context_tokens=8192)
        loaded = self._load(controller, "model-a", context_tokens=1_000_000)

        self.assertEqual(controller._select_engine.call_count, 2)
        self.assertIs(controller.engine, engine_a_1m)
        self.assertEqual(loaded.contextTokens, 1_000_000)
        self.assertEqual(engine_a_8k.unload_calls, 1)
        self.assertEqual(controller.warm_models(), [{**loaded.to_dict(), "warm": True, "active": True}])

    def test_warm_pool_hit_does_not_grow_beyond_capacity(self):
        controller = self._controller()
        engine_a = DummyEngine("A")
        engine_b = DummyEngine("B")
        engine_c = DummyEngine("C")
        controller._select_engine = mock.Mock(side_effect=[engine_a, engine_b, engine_c])

        self._load(controller, "model-a", context_tokens=8192)
        self._load(controller, "model-b", context_tokens=8192)
        self._load(controller, "model-c", context_tokens=8192)
        self._load(controller, "model-b", context_tokens=8192)

        self.assertLessEqual(len(controller._warm_pool), controller.MAX_WARM_MODELS)


class RuntimeControllerOrphanWorkerTests(unittest.TestCase):
    def _capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=False,
        )

    def _controller(self) -> RuntimeController:
        with mock.patch("backend_service.inference.get_backend_capabilities", return_value=self._capabilities()):
            return RuntimeController()

    def test_status_reports_recent_pruned_orphaned_workers(self):
        controller = self._controller()
        controller._tracked_process_pids = mock.Mock(return_value={101})

        tracked = mock.Mock()
        tracked.pid = 101
        tracked.cmdline.return_value = ["python", "-m", "backend_service.mlx_worker", "serve"]
        tracked.name.return_value = "python"

        orphan = mock.Mock()
        orphan.pid = 202
        orphan.cmdline.return_value = ["python", "-m", "backend_service.mlx_worker", "serve"]
        orphan.name.return_value = "python"

        parent = mock.Mock()
        parent.children.return_value = [tracked, orphan]

        with mock.patch("psutil.Process", return_value=parent):
            status = controller.status()

        tracked.terminate.assert_not_called()
        orphan.terminate.assert_called_once()
        self.assertEqual(status["recentOrphanedWorkers"][0]["pid"], 202)
        self.assertEqual(status["recentOrphanedWorkers"][0]["kind"], "mlx_worker")


if __name__ == "__main__":
    unittest.main()
