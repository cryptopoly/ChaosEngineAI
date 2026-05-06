"""Phase 3.3 tests for the logprobs request field."""

from __future__ import annotations

import unittest

from backend_service.models import GenerateRequest
from backend_service.state import _build_sampler_overrides


class LogprobsRequestTests(unittest.TestCase):
    def test_field_omitted_by_default(self):
        req = GenerateRequest(prompt="test")
        self.assertIsNone(req.logprobs)

    def test_field_accepts_top_k(self):
        req = GenerateRequest(prompt="test", logprobs=5)
        self.assertEqual(req.logprobs, 5)

    def test_field_rejects_zero_or_negative(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="test", logprobs=0)
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="test", logprobs=-1)

    def test_field_rejects_extreme_top_k(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GenerateRequest(prompt="test", logprobs=99)


class SamplerBuilderLogprobsTests(unittest.TestCase):
    def test_omits_logprobs_when_none(self):
        req = GenerateRequest(prompt="test")
        overrides = _build_sampler_overrides(req)
        self.assertNotIn("logprobs", overrides)
        self.assertNotIn("top_logprobs", overrides)

    def test_emits_logprobs_true_and_top_k_when_set(self):
        req = GenerateRequest(prompt="test", logprobs=5)
        overrides = _build_sampler_overrides(req)
        self.assertTrue(overrides.get("logprobs"))
        self.assertEqual(overrides.get("top_logprobs"), 5)

    def test_existing_samplers_are_preserved(self):
        req = GenerateRequest(prompt="test", topP=0.9, logprobs=3)
        overrides = _build_sampler_overrides(req)
        self.assertEqual(overrides.get("top_p"), 0.9)
        self.assertEqual(overrides.get("top_logprobs"), 3)


if __name__ == "__main__":
    unittest.main()
