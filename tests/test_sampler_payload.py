"""Tests for the Phase 2.2 sampler-override plumbing.

Two helpers cover the backend half of the contract:
  - `_apply_sampler_kwargs` (in inference.py) merges Phase 2.2 fields
    into a llama-server `/v1/chat/completions` payload.
  - `_build_sampler_overrides` (in state.py) projects a GenerateRequest
    into the dict shape `_apply_sampler_kwargs` consumes.

Together they ensure the user's per-thread overrides reach
llama-server / mlx-lm without ad-hoc casing in three different code
paths.
"""

import unittest
from types import SimpleNamespace

from backend_service.inference import _apply_sampler_kwargs
from backend_service.state import _build_sampler_overrides


class ApplySamplerKwargsTests(unittest.TestCase):
    def test_no_op_when_all_inputs_none(self):
        payload = {"temperature": 0.7, "max_tokens": 512}
        _apply_sampler_kwargs(
            payload,
            samplers=None,
            reasoning_effort=None,
            json_schema=None,
        )
        self.assertEqual(payload, {"temperature": 0.7, "max_tokens": 512})

    def test_merges_all_supported_sampler_keys(self):
        payload: dict = {}
        _apply_sampler_kwargs(
            payload,
            samplers={
                "top_p": 0.9,
                "top_k": 40,
                "min_p": 0.05,
                "repeat_penalty": 1.1,
                "seed": 42,
                "mirostat": 2,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
            },
            reasoning_effort=None,
            json_schema=None,
        )
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["top_k"], 40)
        self.assertEqual(payload["min_p"], 0.05)
        self.assertEqual(payload["repeat_penalty"], 1.1)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["mirostat"], 2)
        self.assertEqual(payload["mirostat_tau"], 5.0)
        self.assertEqual(payload["mirostat_eta"], 0.1)

    def test_none_values_in_samplers_skip_merge(self):
        # The frontend may send the union of fields with most set to null —
        # explicit nulls must not override server defaults.
        payload: dict = {"temperature": 0.7}
        _apply_sampler_kwargs(
            payload,
            samplers={"top_p": None, "top_k": 40, "seed": None},
            reasoning_effort=None,
            json_schema=None,
        )
        self.assertNotIn("top_p", payload)
        self.assertEqual(payload["top_k"], 40)
        self.assertNotIn("seed", payload)

    def test_unknown_sampler_keys_are_ignored(self):
        # Forward-compat: a future field not yet in _LLAMA_SAMPLER_KEYS
        # should be silently ignored rather than poisoning the payload.
        payload: dict = {}
        _apply_sampler_kwargs(
            payload,
            samplers={"futuristic_knob": 0.42, "top_p": 0.85},
            reasoning_effort=None,
            json_schema=None,
        )
        self.assertEqual(payload, {"top_p": 0.85})

    def test_reasoning_effort_added_when_set(self):
        payload: dict = {}
        _apply_sampler_kwargs(
            payload,
            samplers=None,
            reasoning_effort="high",
            json_schema=None,
        )
        self.assertEqual(payload["reasoning_effort"], "high")

    def test_json_schema_wraps_in_response_format_envelope(self):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        payload: dict = {}
        _apply_sampler_kwargs(
            payload,
            samplers=None,
            reasoning_effort=None,
            json_schema=schema,
        )
        self.assertIn("response_format", payload)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], schema)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])


class BuildSamplerOverridesTests(unittest.TestCase):
    def test_skips_unset_fields(self):
        request = SimpleNamespace(
            topP=None, topK=None, minP=None, repeatPenalty=None,
            seed=None, mirostatMode=None, mirostatTau=None, mirostatEta=None,
        )
        self.assertEqual(_build_sampler_overrides(request), {})

    def test_emits_llama_field_names(self):
        # The override dict uses llama-server's snake_case field names
        # (top_p, not topP) so it can be merged directly into the payload.
        request = SimpleNamespace(
            topP=0.9, topK=40, minP=0.05, repeatPenalty=1.1,
            seed=7, mirostatMode=2, mirostatTau=5.0, mirostatEta=0.1,
        )
        overrides = _build_sampler_overrides(request)
        self.assertEqual(overrides["top_p"], 0.9)
        self.assertEqual(overrides["top_k"], 40)
        self.assertEqual(overrides["min_p"], 0.05)
        self.assertEqual(overrides["repeat_penalty"], 1.1)
        self.assertEqual(overrides["seed"], 7)
        self.assertEqual(overrides["mirostat"], 2)
        self.assertEqual(overrides["mirostat_tau"], 5.0)
        self.assertEqual(overrides["mirostat_eta"], 0.1)

    def test_partial_override_keeps_only_set_fields(self):
        request = SimpleNamespace(
            topP=0.9, topK=None, minP=None, repeatPenalty=None,
            seed=42, mirostatMode=None, mirostatTau=None, mirostatEta=None,
        )
        overrides = _build_sampler_overrides(request)
        self.assertEqual(overrides, {"top_p": 0.9, "seed": 42})


if __name__ == "__main__":
    unittest.main()
