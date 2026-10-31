"""Decode speed in the launch preview: measured history first, bandwidth estimate otherwise."""
from __future__ import annotations

import unittest
from unittest import mock

from backend_service.helpers.bandwidth import apple_bandwidth, nvidia_bandwidth
from backend_service.helpers.cache import compute_cache_preview, estimate_decode_tok_s
from backend_service.helpers.model_arch import architecture_from_config
from backend_service.helpers.speed_history import measured_decode_tok_s


class BandwidthTableTests(unittest.TestCase):
    def test_apple_chips_and_binned_variants(self):
        self.assertEqual(apple_bandwidth("Apple M4 Max", 40), 546)
        self.assertEqual(apple_bandwidth("Apple M4 Max", 32), 410)
        self.assertEqual(apple_bandwidth("Apple M4 Pro", 20), 273)
        self.assertEqual(apple_bandwidth("Apple M4", 10), 120)
        self.assertIsNone(apple_bandwidth("Apple M9", 10))

    def test_nvidia_prefers_the_specific_name(self):
        self.assertEqual(nvidia_bandwidth("NVIDIA GeForce RTX 4070 Ti SUPER"), 672)
        self.assertEqual(nvidia_bandwidth("NVIDIA GeForce RTX 4070"), 504)
        self.assertIsNone(nvidia_bandwidth("Tesla T4"))


class EstimateCalibrationTests(unittest.TestCase):
    """Measured medians on an M4 Max (546 GB/s); the estimate stays within 25%."""

    POINTS = [
        # weights GB, active fraction, measured tok/s
        (8.0, 1.0, 52.2),     # Qwen3-4B bf16
        (0.3, 1.0, 346.2),    # Qwen3-0.6B 4-bit
        (8.5, 1.0, 41.5),     # Ternary Bonsai 27B
        (20.4, 0.070, 93.8),  # Qwen3.6-35B-A3B 4-bit (MoE)
        (51.6, 0.149, 52.0),  # Gemma 4 26B-A4B bf16 (MoE)
    ]

    def test_within_a_quarter_of_measured(self):
        for weights, active, measured in self.POINTS:
            estimate = estimate_decode_tok_s(
                weights_gb=weights, active_fraction=active, kv_bytes_per_token_read=0, bandwidth_gbps=546,
            )
            self.assertLess(abs(estimate / measured - 1), 0.25, (weights, active, measured, estimate))

    def test_moe_reads_only_the_active_share(self):
        dense = estimate_decode_tok_s(weights_gb=40, active_fraction=1.0, kv_bytes_per_token_read=0, bandwidth_gbps=546)
        moe = estimate_decode_tok_s(weights_gb=40, active_fraction=0.05, kv_bytes_per_token_read=0, bandwidth_gbps=546)
        self.assertGreater(moe, dense * 4)


class ActiveFractionTests(unittest.TestCase):
    def test_qwen_moe(self):
        arch = architecture_from_config({
            "model_type": "qwen3_moe", "num_hidden_layers": 40, "num_attention_heads": 16, "num_key_value_heads": 2,
            "head_dim": 256, "hidden_size": 2048, "num_experts": 256, "num_experts_per_tok": 8,
            "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512, "vocab_size": 248320,
        })
        self.assertAlmostEqual(arch.active_fraction, 0.07, delta=0.02)  # "35B-A3B"

    def test_dense_is_one(self):
        arch = architecture_from_config({
            "model_type": "qwen3", "num_hidden_layers": 28, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2048, "intermediate_size": 6144,
        })
        self.assertEqual(arch.active_fraction, 1.0)


def _chat(ref, tok_s, *, strategy="native", tokens=200, backend="mlx", spec=False, when="2026-10-01"):
    return {"messages": [{"metrics": {
        "modelRef": ref, "tokS": tok_s, "cacheStrategy": strategy, "completionTokens": tokens,
        "backend": backend, "speculativeDecoding": spec, "generatedAt": when,
    }}]}


class HistoryTests(unittest.TestCase):
    def _measured(self, sessions, runs=(), strategy="native", ref="org/model"):
        return measured_decode_tok_s(sessions, runs, model_path=None, model_repo=ref, strategy=strategy)

    def test_median_of_matching_runs(self):
        sessions = [_chat("org/model", v) for v in (40, 50, 60)]
        self.assertEqual(self._measured(sessions), 50)

    def test_ignores_mock_short_speculative_and_other_strategies(self):
        sessions = [
            _chat("org/model", 48, backend="mock"),
            _chat("org/model", 900, tokens=5),
            _chat("org/model", 120, spec=True),
            _chat("org/model", 10, strategy="turboquant"),
            _chat("org/other", 70),
        ]
        self.assertIsNone(self._measured(sessions))

    def test_needs_two_samples_and_reads_benchmarks_too(self):
        self.assertIsNone(self._measured([_chat("org/model", 50)]))
        runs = [{"modelRef": "org/model", "tokS": 54, "cacheStrategy": "native", "completionTokens": 300,
                 "backend": "mlx", "measuredAt": "2026-10-02"}]
        self.assertEqual(self._measured([_chat("org/model", 50)], runs), 52)


class PreviewSpeedTests(unittest.TestCase):
    def _preview(self, **kwargs):
        with mock.patch("backend_service.helpers.bandwidth.memory_bandwidth", return_value=(546.0, "M4 Max", True)):
            return compute_cache_preview(
                bits=0, num_layers=36, num_heads=32, num_kv_heads=8, hidden_size=4096,
                context_tokens=8192, params_b=4.0, strategy="native", **kwargs,
            )

    def test_measured_speed_wins(self):
        preview = self._preview(weights_gb=8.0, measured_tok_s=61.0)
        self.assertEqual(preview["speedSource"], "measured")
        self.assertEqual(preview["estimatedTokS"], 61.0)

    def test_estimate_scales_with_weights(self):
        small = self._preview(weights_gb=2.0)
        large = self._preview(weights_gb=16.0)
        self.assertEqual(small["speedSource"], "estimate")
        self.assertGreater(small["estimatedTokS"], large["estimatedTokS"] * 3)
        self.assertEqual(small["bandwidthGbps"], 546.0)


if __name__ == "__main__":
    unittest.main()
