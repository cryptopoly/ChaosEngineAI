import unittest
from types import SimpleNamespace
from unittest.mock import patch

from compression import CacheStrategyRegistry
from compression.native import NativeStrategy
from compression.rotorquant import RotorQuantStrategy
from compression.triattention import TriAttentionStrategy
from compression.turboquant import TurboQuantStrategy


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

    def test_external_strategies_registered(self):
        for strategy_id in ("triattention", "rotorquant"):
            strategy = self.registry.get(strategy_id)
            self.assertIsNotNone(strategy, f"Strategy '{strategy_id}' not found in registry")

    def test_available_returns_all_strategies(self):
        available = self.registry.available()
        ids = [s["id"] for s in available]
        self.assertIn("native", ids)
        self.assertIn("rotorquant", ids)
        self.assertIn("triattention", ids)
        self.assertIn("turboquant", ids)
        self.assertEqual(len(ids), 4)

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

    def test_native_make_mlx_cache_returns_none(self):
        native = self.registry.get("native")
        result = native.make_mlx_cache(32, 0, 0, False, None)
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # TriAttention
    # ------------------------------------------------------------------

    def test_triattention_requires_vllm(self):
        tri = self.registry.get("triattention")
        self.assertIsNotNone(tri.supported_bit_range())
        self.assertTrue(tri.supports_fp16_layers())

    def test_triattention_mlx_raises(self):
        tri = self.registry.get("triattention")
        with self.assertRaises(NotImplementedError) as ctx:
            tri.make_mlx_cache(32, 3, 4, False, None)
        self.assertIn("vLLM", str(ctx.exception))

    def test_triattention_llama_raises(self):
        tri = self.registry.get("triattention")
        with self.assertRaises(NotImplementedError) as ctx:
            tri.llama_cpp_cache_flags(3)
        self.assertIn("vLLM", str(ctx.exception))

    def test_triattention_estimate_compresses(self):
        tri = self.registry.get("triattention")
        baseline, optimised = tri.estimate_cache_bytes(
            num_layers=32, num_heads=32, hidden_size=4096,
            context_tokens=8192, bits=3, fp16_layers=4,
        )
        self.assertLess(optimised, baseline)

    # ------------------------------------------------------------------
    # RotorQuant
    # ------------------------------------------------------------------

    def test_rotorquant_bit_range(self):
        rq = self.registry.get("rotorquant")
        self.assertEqual(rq.supported_bit_range(), (3, 4))

    def test_rotorquant_llama_flags(self):
        rq = self.registry.get("rotorquant")
        flags = rq.llama_cpp_cache_flags(3)
        self.assertEqual(flags, ["--cache-type-k", "iso3", "--cache-type-v", "iso3"])
        flags4 = rq.llama_cpp_cache_flags(4)
        self.assertEqual(flags4, ["--cache-type-k", "iso4", "--cache-type-v", "iso4"])

    def test_rotorquant_llama_flags_planar(self):
        rq = self.registry.get("rotorquant")
        flags = rq.llama_cpp_cache_flags_planar(3)
        self.assertEqual(flags, ["--cache-type-k", "planar3", "--cache-type-v", "planar3"])

    def test_rotorquant_mlx_raises_helpful_message(self):
        rq = self.registry.get("rotorquant")
        with self.assertRaises(NotImplementedError) as ctx:
            rq.make_mlx_cache(32, 3, 4, False, None)
        self.assertIn("PyTorch/CUDA", str(ctx.exception))
        self.assertIn("llama.cpp", str(ctx.exception))

    def test_rotorquant_estimate_compresses(self):
        rq = self.registry.get("rotorquant")
        baseline, optimised = rq.estimate_cache_bytes(
            num_layers=32, num_heads=32, hidden_size=4096,
            context_tokens=8192, bits=3, fp16_layers=4,
        )
        self.assertLess(optimised, baseline)

    def test_rotorquant_label(self):
        rq = self.registry.get("rotorquant")
        self.assertEqual(rq.label(3, 4), "Rotor 3-bit 4+4")

    def test_rotorquant_bits_clamped(self):
        rq = self.registry.get("rotorquant")
        # Bits below 3 should clamp to 3
        flags = rq.llama_cpp_cache_flags(1)
        self.assertEqual(flags, ["--cache-type-k", "iso3", "--cache-type-v", "iso3"])
        # Bits above 4 should clamp to 4
        flags = rq.llama_cpp_cache_flags(8)
        self.assertEqual(flags, ["--cache-type-k", "iso4", "--cache-type-v", "iso4"])

    def test_rotorquant_is_available_with_current_turboquant_exports(self):
        rq = RotorQuantStrategy()
        module = SimpleNamespace(TurboQuantMSE=object(), TurboQuantCache=object())
        with patch("compression.rotorquant._load_turboquant_module", return_value=module):
            self.assertTrue(rq.is_available())

    def test_rotorquant_is_unavailable_without_supported_marker(self):
        rq = RotorQuantStrategy()
        with patch("compression.rotorquant._load_turboquant_module", return_value=object()):
            self.assertFalse(rq.is_available())

    # ------------------------------------------------------------------
    # TurboQuant
    # ------------------------------------------------------------------

    def test_turboquant_is_available_when_required_hooks_exist(self):
        tq = TurboQuantStrategy()
        with patch(
            "compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["def make_adaptive_cache():\n    pass", "def apply_patch():\n    pass"],
        ):
            self.assertTrue(tq.is_available())

    def test_turboquant_is_unavailable_without_required_hooks(self):
        tq = TurboQuantStrategy()
        with patch(
            "compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["TurboQuant = PolarQuant"],
        ):
            self.assertFalse(tq.is_available())

    def test_turboquant_mlx_cache_raises_helpful_message_without_hooks(self):
        tq = TurboQuantStrategy()
        with patch(
            "compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["TurboQuant = PolarQuant"],
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                tq.make_mlx_cache(32, 3, 4, False, None)
        self.assertIn("required MLX adapter hooks", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
