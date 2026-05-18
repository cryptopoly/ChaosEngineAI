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


if __name__ == "__main__":
    unittest.main()
