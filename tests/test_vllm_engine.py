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
    """apply_vllm_patches refuses a vLLM whose engine core it cannot patch."""

    def _strategy(self):
        from cache_compression.triattention import TriAttentionStrategy

        return TriAttentionStrategy()

    def _stub_vllm(self, engine_core_cls):
        import sys
        import types

        modules = {
            "vllm": types.ModuleType("vllm"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
            "vllm.v1.engine.core": types.ModuleType("vllm.v1.engine.core"),
        }
        modules["vllm"].__version__ = "0.33.0"
        modules["vllm.v1.engine.core"].EngineCoreProc = engine_core_cls
        return patch.dict(sys.modules, modules)

    def test_engine_core_without_the_hook_is_refused(self) -> None:
        class EngineCoreProc:
            def _attach_iteration_details(self): ...

        with patch("cache_compression.triattention._module_available", return_value=True),                 self._stub_vllm(EngineCoreProc):
            with self.assertRaisesRegex(RuntimeError, "does not support vLLM 0.33.0"):
                self._strategy().apply_vllm_patches()

    def test_engine_core_with_the_hook_goes_on_to_the_install(self) -> None:
        class EngineCoreProc:
            def log_iteration_details(self): ...

        with patch("cache_compression.triattention._module_available", return_value=True),                 self._stub_vllm(EngineCoreProc):
            # The integration module is not importable here, which is the step after the guard.
            with self.assertRaises(NotImplementedError):
                self._strategy().apply_vllm_patches()


if __name__ == "__main__":
    unittest.main()
