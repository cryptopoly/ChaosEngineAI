import unittest
from unittest import mock

from backend_service.inference import (
    BackendCapabilities,
    LlamaCppEngine,
    RepeatedLineGuard,
    RuntimeController,
    LoadedModelInfo,
    _friendly_llama_error,
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
            command, _runtime_note, _, _mmproj = engine._build_command(
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
            command, _runtime_note, _, _mmproj = engine._build_command(
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

    def test_hides_info_only_metal_startup_lines(self):
        logs = (
            "load_backend: loaded BLAS backend from /opt/homebrew/Cellar/ggml/0.9.11/libexec/libggml-blas.so\n"
            "ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices\n"
            "ggml_metal_library_init: using embedded metal library"
        )
        self.assertEqual(
            _friendly_llama_error(logs),
            "llama.cpp exited during startup before reporting a specific error. "
            "The visible ggml/Metal lines are informational startup messages, not the cause. "
            "Retry with Native f16 or inspect the full server log for the real failure.",
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

    def test_startup_fallback_tries_chaosengine_then_native(self):
        """When the turbo binary fails, the fallback chain is:
        rotorquant → chaosengine → native f16."""
        engine = LlamaCppEngine(self._capabilities())

        fake_process = mock.Mock()
        fake_process.poll.return_value = None

        # 3 attempts: rotorquant (fail) → chaosengine (fail) → native (succeed)
        with (
            mock.patch.object(engine, "_build_command", side_effect=[
                (["llama-server-turbo"], None, False, None),
                (["llama-server"], None, False, None),
                (["llama-server"], None, False, None),
            ]),
            mock.patch.object(engine, "_wait_for_server", side_effect=[
                RuntimeError("unknown architecture"),
                RuntimeError("cache type unsupported"),
                None,
            ]),
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
        self.assertIn("RotorQuant", loaded.runtimeNote)

    def test_startup_fallback_lands_on_chaosengine_when_it_works(self):
        """When turbo binary fails but ChaosEngine succeeds, use ChaosEngine."""
        engine = LlamaCppEngine(self._capabilities())

        fake_process = mock.Mock()
        fake_process.poll.return_value = None

        # 2 attempts: rotorquant (fail) → chaosengine (succeed)
        with (
            mock.patch.object(engine, "_build_command", side_effect=[
                (["llama-server-turbo"], None, False, None),
                (["llama-server"], None, False, None),
            ]),
            mock.patch.object(engine, "_wait_for_server", side_effect=[
                RuntimeError("unknown architecture"),
                None,
            ]),
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

        self.assertEqual(loaded.cacheStrategy, "chaosengine")
        self.assertEqual(loaded.fp16Layers, 0)
        self.assertIn("RotorQuant", loaded.runtimeNote)
        self.assertIn("turbo binary", loaded.runtimeNote)

    def test_successful_gguf_load_reports_fp16_layers_as_ignored(self):
        engine = LlamaCppEngine(self._capabilities())

        fake_process = mock.Mock()
        fake_process.poll.return_value = None

        with (
            mock.patch.object(engine, "_build_command", return_value=(["llama-server-turbo"], None, False, None)),
            mock.patch.object(engine, "_wait_for_server", return_value=None),
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

        self.assertEqual(loaded.cacheStrategy, "rotorquant")
        self.assertEqual(loaded.cacheBits, 3)
        self.assertEqual(loaded.fp16Layers, 0)
        self.assertIn("Rotor 3-bit 0+0 cache", loaded.runtimeNote)
        self.assertIn("ignores the FP16 layers setting", loaded.runtimeNote)


class ChatSystemPromptTests(unittest.TestCase):
    def test_user_system_prompt_passed_through(self):
        prompt = _compose_chat_system_prompt("Answer in one sentence.", "off")
        self.assertEqual(prompt, "Answer in one sentence.")

    def test_no_thinking_policy_injected(self):
        prompt = _compose_chat_system_prompt("Answer in one sentence.", "auto")
        self.assertEqual(prompt, "Answer in one sentence.")
        self.assertNotIn("Thinking mode", prompt)

    def test_empty_system_prompt(self):
        self.assertEqual(_compose_chat_system_prompt(None), "")
        self.assertEqual(_compose_chat_system_prompt(""), "")


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

    def _load(
        self,
        controller: RuntimeController,
        model_ref: str,
        *,
        context_tokens: int,
        keep_warm_previous: bool = True,
    ) -> LoadedModelInfo:
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
            keep_warm_previous=keep_warm_previous,
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

    def test_exclusive_load_unloads_previous_model_instead_of_parking_it(self):
        controller = self._controller()
        engine_a = DummyEngine("A")
        engine_b = DummyEngine("B")
        controller._select_engine = mock.Mock(side_effect=[engine_a, engine_b])

        self._load(controller, "model-a", context_tokens=8192)
        loaded = self._load(
            controller,
            "model-b",
            context_tokens=8192,
            keep_warm_previous=False,
        )

        self.assertIs(controller.engine, engine_b)
        self.assertEqual(loaded.ref, "model-b")
        self.assertEqual(engine_a.unload_calls, 1)
        self.assertEqual(controller._warm_pool, {})

    def test_large_incoming_load_unloads_previous_model_instead_of_parking_it(self):
        controller = self._controller()
        engine_a = DummyEngine("A")
        engine_b = DummyEngine("B")
        controller._select_engine = mock.Mock(side_effect=[engine_a, engine_b])

        self._load(controller, "model-a", context_tokens=8192)
        with (
            mock.patch.object(RuntimeController, "_target_resident_bytes", return_value=20),
            mock.patch.object(controller, "_memory_budget_bytes", return_value=10),
        ):
            loaded = self._load(controller, "model-b", context_tokens=8192)

        self.assertIs(controller.engine, engine_b)
        self.assertEqual(loaded.ref, "model-b")
        self.assertEqual(engine_a.unload_calls, 1)
        self.assertEqual(controller._warm_pool, {})

    def test_clear_warm_pool_unloads_every_parked_engine(self):
        controller = self._controller()
        engine_a = DummyEngine("A")
        engine_b = DummyEngine("B")
        controller._select_engine = mock.Mock(side_effect=[engine_a, engine_b])

        self._load(controller, "model-a", context_tokens=8192)
        self._load(controller, "model-b", context_tokens=8192)
        cleared = controller.clear_warm_pool()

        self.assertEqual(cleared, 1)
        self.assertEqual(engine_a.unload_calls, 1)
        self.assertEqual(controller._warm_pool, {})


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

        # `create_time` needs to be older than the detection grace window,
        # otherwise orphans are deliberately skipped. Use an epoch value so
        # now - create_time is huge and positive.
        tracked = mock.Mock()
        tracked.pid = 101
        tracked.cmdline.return_value = ["python", "-m", "backend_service.mlx_worker", "serve"]
        tracked.name.return_value = "python"
        tracked.create_time.return_value = 0.0

        orphan = mock.Mock()
        orphan.pid = 202
        orphan.cmdline.return_value = ["python", "-m", "backend_service.mlx_worker", "serve"]
        orphan.name.return_value = "python"
        orphan.create_time.return_value = 0.0

        parent = mock.Mock()
        parent.children.return_value = [tracked, orphan]

        with mock.patch("psutil.Process", return_value=parent):
            status = controller.status()

        tracked.terminate.assert_not_called()
        orphan.terminate.assert_called_once()
        self.assertEqual(status["recentOrphanedWorkers"][0]["pid"], 202)
        self.assertEqual(status["recentOrphanedWorkers"][0]["kind"], "mlx_worker")


class TurboBinaryRoutingTests(unittest.TestCase):
    """Tests for multi-binary (standard vs turbo) llama-server selection."""

    def _capabilities(self, *, turbo_path: str | None = None) -> BackendCapabilities:
        return BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=True,
            llamaServerPath="/usr/local/bin/llama-server",
            llamaServerTurboPath=turbo_path,
        )

    def test_native_strategy_uses_standard_binary(self):
        engine = LlamaCppEngine(self._capabilities(turbo_path="/usr/local/bin/llama-server-turbo"))

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
            mock.patch("backend_service.inference._llama_server_cache_types", return_value=frozenset({"f16", "q8_0", "q4_0"})),
        ):
            command, _, _, _mmproj = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )
        self.assertEqual(command[0], "/usr/local/bin/llama-server")

    def test_rotorquant_uses_turbo_binary_when_available(self):
        engine = LlamaCppEngine(self._capabilities(turbo_path="/usr/local/bin/llama-server-turbo"))

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
            mock.patch("backend_service.inference._llama_server_cache_types", return_value=frozenset({"f16", "q8_0", "turbo2", "turbo3", "turbo4"})),
        ):
            command, _, _, _mmproj = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="rotorquant",
                cache_bits=3,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )
        self.assertEqual(command[0], "/usr/local/bin/llama-server-turbo")
        self.assertIn("turbo3", command)

    def test_rotorquant_falls_back_to_f16_without_turbo_binary(self):
        engine = LlamaCppEngine(self._capabilities(turbo_path=None))

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
            mock.patch("backend_service.inference._llama_server_cache_types", return_value=frozenset({"f16", "q8_0", "q4_0"})),
        ):
            command, runtime_note, _, _mmproj = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="rotorquant",
                cache_bits=3,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )
        # Should fall back to standard binary with f16 cache
        self.assertEqual(command[0], "/usr/local/bin/llama-server")
        self.assertIn("f16", command)
        self.assertNotIn("turbo3", command)
        self.assertIn("llama-server-turbo", runtime_note)

    def test_chaosengine_uses_standard_binary(self):
        engine = LlamaCppEngine(self._capabilities(turbo_path="/usr/local/bin/llama-server-turbo"))

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
            mock.patch("backend_service.inference._llama_server_cache_types", return_value=frozenset({"f16", "q8_0", "q4_0", "q5_0"})),
        ):
            command, _, _, _mmproj = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="chaosengine",
                cache_bits=4,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )
        self.assertEqual(command[0], "/usr/local/bin/llama-server")
        self.assertIn("q4_0", command)

    def test_turbo_only_binary_serves_all_strategies(self):
        """When only llama-server-turbo exists (no standard binary), it should
        serve as the binary for all strategies since it's a superset."""
        caps = BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False,
            mlxLmAvailable=False,
            mlxUsable=False,
            ggufAvailable=True,
            llamaServerPath=None,
            llamaServerTurboPath="/usr/local/bin/llama-server-turbo",
        )
        engine = LlamaCppEngine(caps)

        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
            mock.patch("backend_service.inference._llama_server_cache_types", return_value=frozenset({"f16", "q8_0", "q4_0"})),
        ):
            command, _, _, _mmproj = engine._build_command(
                path="/tmp/model.gguf",
                runtime_target=None,
                cache_strategy="native",
                cache_bits=0,
                context_tokens=8192,
                fit_enabled=True,
                is_fallback=False,
            )
        self.assertEqual(command[0], "/usr/local/bin/llama-server-turbo")


class CacheTypeValidationTests(unittest.TestCase):
    """Tests for pre-validation of cache types against binary capabilities."""

    def test_llama_server_cache_types_parses_help_text(self):
        from backend_service.inference import _llama_server_cache_types, _CACHE_TYPE_CACHE

        help_text = (
            "-ctk,  --cache-type-k type\n"
            "                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1\n"
        )
        _CACHE_TYPE_CACHE.pop("/test/binary", None)

        with mock.patch("backend_service.inference._llama_server_help_text", return_value=help_text):
            types = _llama_server_cache_types("/test/binary")

        self.assertIn("q8_0", types)
        self.assertIn("q4_0", types)
        self.assertNotIn("iso3", types)
        _CACHE_TYPE_CACHE.pop("/test/binary", None)

    def test_cache_types_returns_standard_set_for_missing_binary(self):
        from backend_service.inference import _llama_server_cache_types, _STANDARD_CACHE_TYPES

        types = _llama_server_cache_types(None)
        self.assertEqual(types, _STANDARD_CACHE_TYPES)

    def test_cache_types_parses_turbo_help_text_multiline(self):
        """The turbo binary wraps allowed values across multiple lines."""
        from backend_service.inference import _llama_server_cache_types, _CACHE_TYPE_CACHE

        # Realistic multi-line output from llama-server-turbo --help
        turbo_help = (
            "-ctk,  --cache-type-k type              kv cache data type for k\n"
            "                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1,\n"
            "                                        turbo2, turbo3, turbo4\n"
            "                                        (default: f16)\n"
        )
        _CACHE_TYPE_CACHE.pop("/test/turbo", None)

        with mock.patch("backend_service.inference._llama_server_help_text", return_value=turbo_help):
            types = _llama_server_cache_types("/test/turbo")

        self.assertIn("turbo3", types)
        self.assertIn("turbo4", types)
        self.assertIn("q8_0", types)  # still includes standard types
        _CACHE_TYPE_CACHE.pop("/test/turbo", None)

    def test_build_command_prevalidation_catches_unsupported_type(self):
        """When cache type is unsupported by the binary, _build_command should
        fall back to f16 and set an informative runtime note."""
        caps = BackendCapabilities(
            pythonExecutable="/usr/bin/python3",
            mlxAvailable=False, mlxLmAvailable=False, mlxUsable=False,
            ggufAvailable=True,
            llamaServerPath="/usr/local/bin/llama-server",
            llamaServerTurboPath=None,
        )
        engine = LlamaCppEngine(caps)

        # Standard binary doesn't know about turbo3
        with (
            mock.patch("backend_service.inference._find_open_port", return_value=9999),
            mock.patch("backend_service.inference._llama_server_supports", return_value=False),
            mock.patch("backend_service.inference._llama_server_cache_types",
                       return_value=frozenset({"f16", "q8_0", "q4_0"})),
        ):
            command, note, _, _mmproj = engine._build_command(
                path="/tmp/model.gguf", runtime_target=None,
                cache_strategy="rotorquant", cache_bits=3,
                context_tokens=8192, fit_enabled=True, is_fallback=False,
            )

        self.assertIn("f16", command)
        self.assertNotIn("turbo3", command)
        self.assertIn("llama-server-turbo", note)


if __name__ == "__main__":
    unittest.main()
