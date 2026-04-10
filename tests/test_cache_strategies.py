import unittest

from backend_service.cache_strategies import CacheStrategyRegistry
from backend_service.cache_strategies.native import NativeStrategy
from backend_service.cache_strategies.triattention import TriAttentionStrategy
from backend_service.cache_strategies.rotorquant import RotorQuantStrategy
from backend_service.cache_strategies.megakernel import MegaKernelStrategy


class CacheStrategyRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()

    def test_native_always_available(self):
        native = self.registry.get("native")
        self.assertIsNotNone(native)
        self.assertTrue(native.is_available())

    def test_native_is_default(self):
        default = self.registry.default()
        self.assertEqual(default.strategy_id, "native")

    def test_external_strategies_report_unavailable(self):
        for strategy_id in ("triattention", "rotorquant", "megakernel"):
            strategy = self.registry.get(strategy_id)
            self.assertIsNotNone(strategy, f"Strategy '{strategy_id}' not found in registry")
            self.assertFalse(strategy.is_available(), f"Strategy '{strategy_id}' should not be available")

    def test_available_returns_all_strategies(self):
        available = self.registry.available()
        ids = [s["id"] for s in available]
        self.assertIn("native", ids)
        self.assertIn("triattention", ids)
        self.assertIn("rotorquant", ids)
        self.assertIn("megakernel", ids)

    def test_native_cache_flags(self):
        native = self.registry.get("native")
        flags = native.llama_cpp_cache_flags(0)
        self.assertEqual(flags, ["--cache-type-k", "f16", "--cache-type-v", "f16"])

    def test_native_label(self):
        native = self.registry.get("native")
        self.assertEqual(native.label(0, 0), "Native f16")

    def test_native_estimate_no_compression(self):
        native = self.registry.get("native")
        baseline, optimised = native.estimate_cache_bytes(
            num_layers=32, num_heads=32, hidden_size=4096,
            context_tokens=8192, bits=0, fp16_layers=0,
        )
        self.assertEqual(baseline, optimised)
        self.assertGreater(baseline, 0)

    def test_stub_strategies_have_bit_ranges(self):
        for strategy_id in ("triattention", "rotorquant", "megakernel"):
            strategy = self.registry.get(strategy_id)
            bit_range = strategy.supported_bit_range()
            self.assertIsNotNone(bit_range)
            self.assertEqual(len(bit_range), 2)
            self.assertLessEqual(bit_range[0], bit_range[1])

    def test_stub_strategies_support_fp16_layers(self):
        for strategy_id in ("triattention", "rotorquant", "megakernel"):
            strategy = self.registry.get(strategy_id)
            self.assertTrue(strategy.supports_fp16_layers())

    def test_stub_estimate_compresses(self):
        for strategy_id in ("triattention", "rotorquant", "megakernel"):
            strategy = self.registry.get(strategy_id)
            baseline, optimised = strategy.estimate_cache_bytes(
                num_layers=32, num_heads=32, hidden_size=4096,
                context_tokens=8192, bits=3, fp16_layers=4,
            )
            self.assertLess(optimised, baseline)

    def test_stub_make_mlx_cache_raises(self):
        for strategy_id in ("triattention", "rotorquant", "megakernel"):
            strategy = self.registry.get(strategy_id)
            with self.assertRaises(NotImplementedError):
                strategy.make_mlx_cache(32, 3, 4, False, None)

    def test_native_make_mlx_cache_returns_none(self):
        native = self.registry.get("native")
        result = native.make_mlx_cache(32, 0, 0, False, None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
