"""Tests for the in-process ``VLLMEngine.generate`` signature.

A live regression on WSL2 + RTX 4090 (2026-05-18) caught the engine
missing the ``samplers``, ``reasoning_effort``, and ``json_schema``
kwargs that ``RuntimeController.generate`` passes through. With Qwen3
loaded via ``vllm`` backend, any chat turn raised
``TypeError: VLLMEngine.generate() got an unexpected keyword argument
'samplers'``.

These tests don't load a real model — they patch the LLM and assert
the signature accepts every kwarg the controller passes and forwards
sampler overrides into ``SamplingParams``. Cheap to run on any host
(no CUDA / no vLLM wheel required) because we stub the upstream
import.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


_VLLM_AVAILABLE = importlib.util.find_spec("vllm") is not None


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm wheel not installed (CUDA-only)")
class VLLMEngineSignatureTests(unittest.TestCase):
    """Locked in by the FU-052 follow-up: every engine ``generate``
    method must accept the full controller kwarg surface, even if
    individual kwargs are no-ops for that backend."""

    def test_generate_signature_matches_controller_call_shape(self):
        from backend_service.vllm_engine import VLLMEngine

        sig = inspect.signature(VLLMEngine.generate)
        params = sig.parameters
        # Every kwarg the controller can pass on a chat turn must be
        # named here. Add new ones to the engine in lockstep with the
        # controller — otherwise users on the CUDA path crash mid-turn.
        for required in (
            "prompt",
            "history",
            "system_prompt",
            "max_tokens",
            "temperature",
            "images",
            "tools",
            "samplers",
            "reasoning_effort",
            "json_schema",
        ):
            self.assertIn(
                required,
                params,
                f"VLLMEngine.generate() missing kwarg '{required}' — "
                f"keep it in sync with RuntimeController.generate "
                f"(see backend_service/inference/controller.py).",
            )


@unittest.skipUnless(_VLLM_AVAILABLE, "vllm wheel not installed (CUDA-only)")
class VLLMSamplerForwardingTests(unittest.TestCase):
    """SamplingParams should reflect the sampler overrides the user
    set in the UI. We stub the heavy vllm.LLM call and just inspect
    the SamplingParams kwargs the engine constructs."""

    def setUp(self) -> None:
        from backend_service.vllm_engine import VLLMEngine

        self.engine = VLLMEngine.__new__(VLLMEngine)
        # Pretend a model is loaded so the generate() guard passes.
        self.engine._llm = MagicMock()
        self.engine.loaded_model = SimpleNamespace(
            ref="Qwen/Qwen3-0.6B",
            runtimeNote=None,
        )
        # Fake vllm output shape so the result-decoding code doesn't
        # blow up while we inspect SamplingParams.
        fake_output = SimpleNamespace(
            outputs=[SimpleNamespace(
                text="ok",
                finish_reason="stop",
                token_ids=[1, 2, 3],
            )],
            prompt_token_ids=[1, 2],
        )
        self.engine._llm.generate.return_value = [fake_output]

    def _capture_params(self, **generate_kwargs):
        """Run engine.generate and return the SamplingParams kwargs
        it constructed."""
        captured: dict[str, object] = {}

        def fake_sampling_params(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        # SamplingParams is imported inside generate() so patch the
        # vllm.SamplingParams attribute at call site.
        with patch("vllm.SamplingParams", side_effect=fake_sampling_params):
            self.engine.generate(
                prompt="hi",
                history=[],
                system_prompt=None,
                max_tokens=32,
                temperature=0.7,
                **generate_kwargs,
            )
        return captured

    def test_no_samplers_uses_only_max_tokens_and_temperature(self):
        params = self._capture_params()
        self.assertEqual(params.get("max_tokens"), 32)
        self.assertAlmostEqual(params.get("temperature"), 0.7)
        # No sampler keys leaked through.
        for key in ("top_p", "top_k", "min_p", "repetition_penalty", "seed"):
            self.assertNotIn(key, params)

    def test_passthrough_samplers_forwarded_to_sampling_params(self):
        params = self._capture_params(
            samplers={
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.05,
                "seed": 42,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            },
        )
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["top_k"], 40)
        self.assertEqual(params["min_p"], 0.05)
        self.assertEqual(params["seed"], 42)
        self.assertAlmostEqual(params["frequency_penalty"], 0.1)
        self.assertAlmostEqual(params["presence_penalty"], 0.2)

    def test_repeat_penalty_renamed_to_repetition_penalty(self):
        # llama-server calls the knob ``repeat_penalty``; vLLM calls
        # the same concept ``repetition_penalty``. Translate on the
        # way in so the UI doesn't need to know which engine is live.
        params = self._capture_params(samplers={"repeat_penalty": 1.15})
        self.assertAlmostEqual(params["repetition_penalty"], 1.15)
        self.assertNotIn("repeat_penalty", params)

    def test_zero_temperature_is_bumped_to_avoid_vllm_floor(self):
        # vLLM's SamplingParams rejects temperature exactly 0 — the
        # engine must floor it to a tiny positive value. Without this
        # the controller's ``temperature=0`` shorthand would crash.
        from backend_service.vllm_engine import VLLMEngine

        engine = VLLMEngine.__new__(VLLMEngine)
        engine._llm = MagicMock()
        engine.loaded_model = SimpleNamespace(ref="x", runtimeNote=None)
        engine._llm.generate.return_value = [SimpleNamespace(
            outputs=[SimpleNamespace(text="ok", finish_reason="stop", token_ids=[1])],
            prompt_token_ids=[1],
        )]
        captured: dict[str, object] = {}
        with patch("vllm.SamplingParams",
                   side_effect=lambda **k: (captured.update(k), MagicMock())[1]):
            engine.generate(
                prompt="hi", history=[], system_prompt=None,
                max_tokens=4, temperature=0.0,
            )
        self.assertGreater(captured["temperature"], 0.0)


class VLLMEnvPreparationTests(unittest.TestCase):
    """The env the vLLM engine subprocess inherits (no vllm import needed)."""

    def _run(self, strategy, env):
        import os
        from backend_service.vllm_engine import _prepare_vllm_env

        with patch.dict(os.environ, env, clear=True):
            notes = _prepare_vllm_env(strategy)
            return notes, dict(os.environ)

    def test_triattention_plugin_off_for_other_strategies(self) -> None:
        for strategy in ("native", "turboquant"):
            _notes, env = self._run(strategy, {"VLLM_USE_FLASHINFER_SAMPLER": "1"})
            self.assertEqual(env["ENABLE_TRIATTENTION"], "0", strategy)

    def test_triattention_plugin_left_on_for_its_own_strategy(self) -> None:
        _notes, env = self._run("triattention", {"ENABLE_TRIATTENTION": "0", "VLLM_USE_FLASHINFER_SAMPLER": "1"})
        self.assertNotIn("ENABLE_TRIATTENTION", env)

    def test_flashinfer_sampler_off_without_nvcc(self) -> None:
        with patch("backend_service.vllm_engine.shutil.which", return_value=None),                 patch("backend_service.vllm_engine.Path.exists", return_value=False):
            notes, env = self._run("native", {})
        self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
        self.assertTrue(notes)

    def test_flashinfer_sampler_untouched_with_nvcc_or_explicit_setting(self) -> None:
        with patch("backend_service.vllm_engine.shutil.which", return_value="/usr/bin/nvcc"):
            _notes, env = self._run("native", {})
        self.assertNotIn("VLLM_USE_FLASHINFER_SAMPLER", env)
        with patch("backend_service.vllm_engine.shutil.which", return_value=None):
            _notes, env = self._run("native", {"VLLM_USE_FLASHINFER_SAMPLER": "1"})
        self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "1")


class VLLMPromptRenderingTests(unittest.TestCase):
    def _engine(self, tokenizer):
        from backend_service.vllm_engine import VLLMEngine

        engine = VLLMEngine.__new__(VLLMEngine)
        engine._llm = MagicMock()
        engine._llm.get_tokenizer.return_value = tokenizer
        return engine

    def test_chat_template_is_applied_when_the_tokenizer_has_one(self) -> None:
        tok = MagicMock()
        tok.chat_template = "{{ messages }}"
        tok.apply_chat_template.return_value = "<|im_start|>user" + chr(10) + "hi<|im_end|>" + chr(10) + "<|im_start|>assistant" + chr(10)
        text = self._engine(tok)._render_prompt([{"role": "user", "content": "hi"}])
        self.assertTrue(text.startswith("<|im_start|>user"))
        tok.apply_chat_template.assert_called_once()
        self.assertTrue(tok.apply_chat_template.call_args.kwargs["add_generation_prompt"])

    def test_plain_text_when_there_is_no_template_or_it_fails(self) -> None:
        tok = MagicMock()
        tok.chat_template = None
        self.assertIn("USER: hi", self._engine(tok)._render_prompt([{"role": "user", "content": "hi"}]))
        tok = MagicMock()
        tok.chat_template = "x"
        tok.apply_chat_template.side_effect = ValueError("bad template")
        self.assertIn("USER: hi", self._engine(tok)._render_prompt([{"role": "user", "content": "hi"}]))


class TriAttentionVllmGuardTests(unittest.TestCase):
    """FU-171: TriAttention's vLLM patches only work on vLLM 0.18 and older."""

    def _strategy(self):
        from cache_compression.triattention import TriAttentionStrategy

        return TriAttentionStrategy()

    def test_version_gate(self) -> None:
        from cache_compression.triattention import vllm_unsupported_reason

        for ok in ("0.15.1", "0.18.1", "0.18.2rc1", None, "", "dev"):
            self.assertIsNone(vllm_unsupported_reason(ok), ok)
        for bad in ("0.19.0", "0.19.1", "0.33.0", "1.0.0"):
            reason = vllm_unsupported_reason(bad)
            self.assertIn(bad, reason)
            self.assertIn("0.18", reason)

    def test_unsupported_vllm_is_refused_before_patching(self) -> None:
        with patch("cache_compression.triattention._module_available", return_value=True), \
                patch("cache_compression.triattention._installed_vllm_version", return_value="0.33.0"):
            with self.assertRaisesRegex(RuntimeError, "vLLM 0.33.0 is installed"):
                self._strategy().apply_vllm_patches()

    def test_supported_vllm_goes_on_to_the_install(self) -> None:
        with patch("cache_compression.triattention._module_available", return_value=True), \
                patch("cache_compression.triattention._installed_vllm_version", return_value="0.18.1"), \
                patch.dict(sys.modules, {"triattention": None}):
            # The integration module is not importable here, which is the step after the gate.
            with self.assertRaises(NotImplementedError):
                self._strategy().apply_vllm_patches()

    def test_settings_say_unavailable_with_the_reason_on_new_vllm(self) -> None:
        with patch("cache_compression.triattention._module_available", return_value=True), \
                patch("cache_compression.triattention._has_mlx_entrypoint", return_value=False), \
                patch("cache_compression.triattention._installed_vllm_version", return_value="0.33.0"):
            tri = self._strategy()
            self.assertFalse(tri.is_available())
            self.assertEqual(tri.availability_badge(), "Unavailable")
            self.assertEqual(tri.availability_tone(), "warning")
            self.assertIn("0.18", tri.availability_reason())

    def test_settings_on_supported_vllm_mention_the_stats_file(self) -> None:
        with patch("cache_compression.triattention._module_available", return_value=True), \
                patch("cache_compression.triattention._has_mlx_entrypoint", return_value=False), \
                patch("cache_compression.triattention._installed_vllm_version", return_value="0.18.1"):
            tri = self._strategy()
            self.assertTrue(tri.is_available())
            self.assertEqual(tri.availability_badge(), "Experimental")
            self.assertIn("stats file", tri.availability_reason())


class TriAttentionVllmRuntimeTests(unittest.TestCase):
    """Stats lookup, env and LLM kwargs for a TriAttention load (FU-171)."""

    def setUp(self) -> None:
        import os
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        env = patch.dict(os.environ, {"CHAOSENGINE_TRIATTENTION_STATS_DIR": self.dir})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("CHAOSENGINE_TRIATTENTION_KV_BUDGET", None)

    def _write(self, name: str) -> None:
        from pathlib import Path

        (Path(self.dir) / name).write_bytes(b"x")

    def test_stats_found_by_repo_name_or_local_folder(self) -> None:
        from cache_compression.triattention import find_stats_file

        self._write("qwen3-0.6b.pt")
        self.assertEqual(find_stats_file([None, "Qwen/Qwen3-0.6B"]).name, "qwen3-0.6b.pt")
        self.assertIsNotNone(find_stats_file(["/models/Qwen3-0.6B/"]))
        self.assertIsNotNone(find_stats_file(["C:\\models\\Qwen3-0.6B"]))
        self.assertIsNone(find_stats_file(["Qwen/Qwen3-8B"]))
        self.assertIsNone(find_stats_file(["..", "", None]))

    def test_no_stats_file_is_a_readable_error(self) -> None:
        from cache_compression.triattention import TriAttentionStrategy

        with self.assertRaisesRegex(RuntimeError, "qwen3-8b.pt"):
            TriAttentionStrategy().vllm_runtime(["Qwen/Qwen3-8B"])

    def test_runtime_env_and_kwargs(self) -> None:
        import os
        from cache_compression.triattention import TriAttentionStrategy

        self._write("qwen3-0.6b.pt")
        with patch.dict(os.environ, {"CHAOSENGINE_TRIATTENTION_KV_BUDGET": "1024"}):
            env, kwargs, note = TriAttentionStrategy().vllm_runtime(["Qwen/Qwen3-0.6B"])
        self.assertEqual(env["TRIATTN_RUNTIME_KV_BUDGET"], "1024")
        self.assertTrue(env["TRIATTN_RUNTIME_SPARSE_STATS_PATH"].endswith("qwen3-0.6b.pt"))
        self.assertEqual(env["TRIATTN_RUNTIME_PROTECT_PREFILL"], "true")
        self.assertEqual(env["TRIATTN_RUNTIME_INCLUDE_PREFILL_IN_BUDGET"], "false")
        self.assertEqual(env["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
        self.assertEqual(kwargs, {
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "max_num_batched_tokens": 2048,
        })
        self.assertIn("1024", note)

    def test_bad_budget_falls_back_to_default(self) -> None:
        import os
        from cache_compression.triattention import TriAttentionStrategy

        self._write("qwen3-0.6b.pt")
        for raw in ("abc", "0", "-5"):
            with patch.dict(os.environ, {"CHAOSENGINE_TRIATTENTION_KV_BUDGET": raw}):
                env, _kwargs, _note = TriAttentionStrategy().vllm_runtime(["Qwen/Qwen3-0.6B"])
            self.assertEqual(env["TRIATTN_RUNTIME_KV_BUDGET"], "2048", raw)


class TriAttentionSkipStampFixTests(unittest.TestCase):
    """A skipped compression must not re-arm the one-step dedup guard (FU-171)."""

    def _fake_state_module(self):
        import types

        class RequestStateStore:
            def __init__(self):
                self._states = {}

            def get(self, req_id):
                return self._states.get(req_id)

            def mark_compression_skipped(self, req_id, reason, step):
                state = self._states.get(req_id)
                state.last_trigger_reason = f"skipped:{reason}"
                state.last_compression_step = step

        module = types.ModuleType("triattention.vllm.runtime.state")
        module.RequestStateStore = RequestStateStore
        return module, RequestStateStore

    def test_skip_keeps_the_last_applied_step(self) -> None:
        from cache_compression.triattention import _install_skip_stamp_fix

        module, store_cls = self._fake_state_module()
        with patch.dict(sys.modules, {"triattention.vllm.runtime.state": module}):
            self.assertTrue(_install_skip_stamp_fix())
            self.assertTrue(_install_skip_stamp_fix())  # idempotent
        store = store_cls()
        store._states["r"] = SimpleNamespace(last_compression_step=580, last_trigger_reason="")
        for step in (706, 707, 708):
            store.mark_compression_skipped(req_id="r", reason="batch_queue_dedup", step=step)
        self.assertEqual(store._states["r"].last_compression_step, 580)
        self.assertEqual(store._states["r"].last_trigger_reason, "skipped:batch_queue_dedup")

    def test_missing_module_is_reported(self) -> None:
        from cache_compression.triattention import _install_skip_stamp_fix

        with patch.dict(sys.modules, {"triattention.vllm.runtime.state": None}):
            self.assertFalse(_install_skip_stamp_fix())


class VLLMEngineTriAttentionLoadTests(unittest.TestCase):
    """load_model passes the strategy's env and kwargs, or falls back to native."""

    def _load(self, strategy):
        import os
        import types
        from backend_service.vllm_engine import VLLMEngine, _STRATEGY_ENV_KEYS

        captured: dict[str, object] = {}

        class LLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                captured["env_multiproc"] = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = LLM
        fake_vllm.__version__ = "0.18.1"
        engine = VLLMEngine(capabilities=MagicMock())
        with patch.dict(sys.modules, {"vllm": fake_vllm}), \
                patch("cache_compression.registry.get", return_value=strategy), \
                patch.dict(os.environ, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}, clear=False):
            info = engine.load_model(
                model_ref="Qwen/Qwen3-0.6B", model_name="Qwen3-0.6B",
                canonical_repo="Qwen/Qwen3-0.6B", source="hub", backend="vllm",
                path=None, runtime_target=None, cache_strategy="triattention",
                cache_bits=3, fp16_layers=0, fused_attention=False,
                fit_model_in_memory=True, context_tokens=8192,
            )
            leftover = {k: os.environ.get(k) for k in list(_STRATEGY_ENV_KEYS)}
            from backend_service.vllm_engine import _apply_strategy_env
            _apply_strategy_env({})
        return info, captured, leftover

    def test_triattention_kwargs_and_env_reach_vllm(self) -> None:
        strategy = MagicMock()
        strategy.name = "TriAttention"
        strategy.vllm_runtime.return_value = (
            {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRIATTN_RUNTIME_KV_BUDGET": "2048"},
            {"enforce_eager": True, "enable_prefix_caching": False, "max_num_batched_tokens": 2048},
            "TriAttention: note.",
        )
        info, captured, leftover = self._load(strategy)
        self.assertEqual(info.cacheStrategy, "triattention")
        self.assertTrue(captured["enforce_eager"])
        self.assertFalse(captured["enable_prefix_caching"])
        self.assertEqual(captured["env_multiproc"], "0")
        self.assertEqual(leftover["TRIATTN_RUNTIME_KV_BUDGET"], "2048")
        self.assertIn("TriAttention: note.", info.runtimeNote)
        strategy.vllm_runtime.assert_called_once_with(["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B", None])

    def test_missing_stats_falls_back_to_native_with_the_reason(self) -> None:
        strategy = MagicMock()
        strategy.name = "TriAttention"
        strategy.vllm_runtime.side_effect = RuntimeError("needs a calibration stats file.")
        info, captured, _leftover = self._load(strategy)
        self.assertEqual(info.cacheStrategy, "native")
        self.assertNotIn("enforce_eager", captured)
        self.assertIsNone(captured["env_multiproc"])
        self.assertIn("calibration stats file", info.runtimeNote)
        strategy.apply_vllm_patches.assert_not_called()

    def test_strategy_env_is_cleared_for_the_next_load(self) -> None:
        import os
        from backend_service.vllm_engine import _apply_strategy_env

        with patch.dict(os.environ, {}, clear=False):
            _apply_strategy_env({"VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
            self.assertEqual(os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"], "0")
            _apply_strategy_env({})
            self.assertNotIn("VLLM_ENABLE_V1_MULTIPROCESSING", os.environ)


if __name__ == "__main__":
    unittest.main()
