"""Phase 3.3 follow-up tests for MLX top-k logprob extraction.

The full mlx_worker subprocess can't be exercised in CI (needs MLX +
a loaded model), but the `_extract_top_logprobs` helper is pure Python
+ numpy and exercises the OpenAI-shaped envelope conversion. Test by
constructing a fake GenerationResponse with hand-built logprobs.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np

from backend_service.mlx_worker import _extract_top_logprobs


@dataclass
class _FakeResponse:
    token: int
    logprobs: np.ndarray


class _FakeTokenizer:
    """Map token id → human-readable string for assertions."""

    VOCAB = {
        0: " the",
        1: " quick",
        2: " brown",
        3: " fox",
        4: " jumps",
    }

    def decode(self, token_ids):
        return "".join(self.VOCAB.get(int(tid), f"<{tid}>") for tid in token_ids)


def _make_response(chosen: int, logprobs: list[float]) -> _FakeResponse:
    return _FakeResponse(token=chosen, logprobs=np.array(logprobs, dtype=np.float32))


class TopLogprobsExtractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()

    def test_returns_none_for_zero_top_k(self):
        resp = _make_response(0, [-0.5, -1.0, -2.0])
        self.assertIsNone(_extract_top_logprobs(resp, self.tokenizer, 0))

    def test_returns_none_when_logprobs_missing(self):
        resp = _FakeResponse(token=0, logprobs=None)  # type: ignore[arg-type]
        self.assertIsNone(_extract_top_logprobs(resp, self.tokenizer, 5))

    def test_returns_chosen_token_with_top_k_alts(self):
        # Logprobs with chosen=0 (" the"), top-3 alternatives = 0, 1, 2.
        resp = _make_response(0, [-0.1, -0.5, -0.8, -2.0, -3.5])
        out = _extract_top_logprobs(resp, self.tokenizer, 3)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), 1)
        entry = out[0]
        self.assertEqual(entry["token"], " the")
        self.assertAlmostEqual(entry["logprob"], -0.1, places=5)
        # Alternatives ordered by logprob descending.
        alt_tokens = [a["token"] for a in entry["alternatives"]]
        self.assertEqual(alt_tokens, [" the", " quick", " brown"])
        # Top alternative logprob equals the chosen logprob.
        self.assertAlmostEqual(entry["alternatives"][0]["logprob"], -0.1, places=5)

    def test_top_k_capped_at_vocab_size(self):
        resp = _make_response(0, [-0.1, -0.5])
        out = _extract_top_logprobs(resp, self.tokenizer, 10)
        self.assertEqual(len(out[0]["alternatives"]), 2)

    def test_chosen_token_logprob_matches_array(self):
        # Chose token 3 (logprob -2.0). Top-2 alternatives stay 0, 1.
        resp = _make_response(3, [-0.1, -0.5, -0.8, -2.0, -3.5])
        out = _extract_top_logprobs(resp, self.tokenizer, 2)
        self.assertEqual(out[0]["token"], " fox")
        self.assertAlmostEqual(out[0]["logprob"], -2.0, places=5)

    def test_handles_empty_logprob_array(self):
        resp = _FakeResponse(token=0, logprobs=np.array([], dtype=np.float32))
        self.assertIsNone(_extract_top_logprobs(resp, self.tokenizer, 5))

    def test_handles_2d_array_gracefully(self):
        # mlx-lm normally returns 1D; defensive check that we don't
        # crash on unexpected shapes.
        resp = _FakeResponse(token=0, logprobs=np.array([[-0.1, -0.5]]))
        self.assertIsNone(_extract_top_logprobs(resp, self.tokenizer, 5))

    def test_token_decode_failure_fallback(self):
        class _BadTokenizer:
            def decode(self, _ids):
                raise RuntimeError("bad")

        resp = _make_response(0, [-0.1, -0.5, -0.8])
        out = _extract_top_logprobs(resp, _BadTokenizer(), 2)
        # Decoder failures fall through to empty strings rather than
        # propagating; logprob numbers still surface.
        self.assertIsNotNone(out)
        self.assertEqual(out[0]["token"], "")
        self.assertEqual(out[0]["alternatives"][0]["token"], "")
        self.assertAlmostEqual(out[0]["alternatives"][0]["logprob"], -0.1, places=5)

    def test_logprobs_remain_sane_floats(self):
        resp = _make_response(0, [-0.1, -0.5, -0.8, -2.0])
        out = _extract_top_logprobs(resp, self.tokenizer, 4)
        for alt in out[0]["alternatives"]:
            self.assertTrue(math.isfinite(alt["logprob"]))


if __name__ == "__main__":
    unittest.main()
