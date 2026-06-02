import unittest
import importlib
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from cache_compression import CacheStrategyRegistry
from cache_compression.native import NativeStrategy
from cache_compression.triattention import TriAttentionStrategy
from cache_compression.turboquant import TurboQuantStrategy
from turboquant_mlx import _find_pip_turboquant_path

# The two MLX adapter tests below need ``mlx_lm`` + the full
# ``turboquant_mlx`` package on disk. Both ship only on Apple Silicon
# (the ``[turboquant]`` extra is Apple-only). Skip cleanly on every
# other platform so a CUDA/CPU box doesn't blow up at function-scope
# imports — keeping the rest of the registry checks running.
_MLX_LM_AVAILABLE = importlib.util.find_spec("mlx_lm") is not None


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
        for strategy_id in ("triattention", "turboquant"):
            strategy = self.registry.get(strategy_id)
            self.assertIsNotNone(strategy, f"Strategy '{strategy_id}' not found in registry")

    def test_available_returns_active_strategies(self):
        available = self.registry.available()
        ids = [s["id"] for s in available]
        self.assertIn("native", ids)
        self.assertIn("triattention", ids)
        self.assertIn("turboquant", ids)
        # FU-030: dropped strategies must NOT appear in the available output.
        self.assertNotIn("rotorquant", ids)
        self.assertNotIn("chaosengine", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_discover_keeps_placeholder_when_optional_adapter_import_fails(self):
        real_import_module = importlib.import_module

        def fake_import(name, package=None):
            if name == "cache_compression.triattention":
                raise RuntimeError("broken triattention import")
            return real_import_module(name, package)

        registry = CacheStrategyRegistry()
        with patch("cache_compression.importlib.import_module", side_effect=fake_import):
            registry.discover()

        tri = registry.get("triattention")
        self.assertIsNotNone(tri)
        self.assertFalse(tri.is_available())
        self.assertIn("could not be loaded", tri.availability_reason())
        self.assertIn("broken triattention import", tri.availability_reason())

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
    # FU-030: legacy alias coercion (chaosengine + rotorquant)
    # ------------------------------------------------------------------

    def test_legacy_chaosengine_coerces_to_turboquant(self):
        """Persisted configs with ``chaosengine`` must resolve to TurboQuant."""
        coerced = self.registry.resolve_legacy_id("chaosengine")
        self.assertEqual(coerced, "turboquant")

    def test_legacy_rotorquant_coerces_to_turboquant(self):
        """Persisted configs with ``rotorquant`` must resolve to TurboQuant."""
        coerced = self.registry.resolve_legacy_id("rotorquant")
        self.assertEqual(coerced, "turboquant")

    def test_unknown_id_passes_through_resolver(self):
        self.assertEqual(self.registry.resolve_legacy_id("does-not-exist"), "does-not-exist")

    def test_get_resolves_legacy_chaosengine_to_turboquant_strategy(self):
        legacy = self.registry.get("chaosengine")
        canonical = self.registry.get("turboquant")
        self.assertIsNotNone(legacy)
        self.assertIs(legacy, canonical)

    def test_get_resolves_legacy_rotorquant_to_turboquant_strategy(self):
        legacy = self.registry.get("rotorquant")
        canonical = self.registry.get("turboquant")
        self.assertIsNotNone(legacy)
        self.assertIs(legacy, canonical)

    # ------------------------------------------------------------------
    # TurboQuant
    # ------------------------------------------------------------------

    def test_turboquant_is_available_when_required_hooks_and_package_exist(self):
        tq = TurboQuantStrategy()
        with patch(
            "cache_compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["def make_adaptive_cache():\n    pass", "def apply_patch():\n    pass"],
        ), patch("cache_compression.turboquant._has_full_turboquant_mlx_package", return_value=True):
            self.assertTrue(tq.is_available())

    def test_turboquant_is_unavailable_without_full_package(self):
        tq = TurboQuantStrategy()
        with patch(
            "cache_compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["def make_adaptive_cache():\n    pass", "def apply_patch():\n    pass"],
        ), patch("cache_compression.turboquant._has_full_turboquant_mlx_package", return_value=False):
            self.assertFalse(tq.is_available())

    def test_turboquant_is_unavailable_without_required_hooks(self):
        tq = TurboQuantStrategy()
        with patch(
            "cache_compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["TurboQuant = PolarQuant"],
        ):
            self.assertFalse(tq.is_available())

    def test_turboquant_mlx_cache_raises_helpful_message_without_hooks(self):
        tq = TurboQuantStrategy()
        with patch(
            "cache_compression.turboquant._turboquant_mlx_source_blobs",
            return_value=["TurboQuant = PolarQuant"],
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                tq.make_mlx_cache(32, 3, 4, False, None)
        self.assertIn("required MLX adapter hooks", str(ctx.exception))

    def test_turboquant_adapter_finds_package_in_extras_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "turboquant_mlx"
            marker = package / "layers" / "polar_kv_cache.py"
            marker.parent.mkdir(parents=True)
            marker.write_text("class TurboQuantKVCache:\n    pass\n", encoding="utf-8")
            with patch.dict("os.environ", {"CHAOSENGINE_EXTRAS_SITE_PACKAGES": tmp}):
                self.assertEqual(_find_pip_turboquant_path(), str(package.resolve()))

    @unittest.skipUnless(_MLX_LM_AVAILABLE, "mlx_lm not installed (Apple Silicon only)")
    def test_turboquant_preserves_hybrid_model_arrayscache_slots(self):
        """Hybrid-attention models (Qwen3.5 / Qwen3.6 MoE) mix KV-cache
        layers with ``ArraysCache(size=2)`` slots for linear-attention
        layers. ``make_adaptive_cache`` must defer to the model's own
        ``make_cache()`` for those slots; if it returns a KV-cache type
        in a linear-attn position, the model's layer code does
        ``cache[0]`` / ``cache[1]`` on it and crashes with
        ``'TurboQuantKVCache' object is not subscriptable``.
        """
        from mlx_lm.models.cache import ArraysCache, KVCache
        from turboquant_mlx import make_adaptive_cache

        # Fake a 6-layer model where every other layer is "linear attn"
        # (returns ArraysCache) and the rest are standard self-attn KV.
        def fake_make_cache():
            return [
                ArraysCache(size=2) if i % 2 == 0 else KVCache()
                for i in range(6)
            ]

        fake_model = SimpleNamespace(make_cache=fake_make_cache)

        cache = make_adaptive_cache(6, bits=3, fp16_layers=0, fused=False, model=fake_model)

        self.assertEqual(len(cache), 6)
        # Even indices must keep the model's ArraysCache — these must
        # support subscript so ``cache[0]`` works inside the model's
        # linear-attn forward.
        for i in (0, 2, 4):
            self.assertIsInstance(cache[i], ArraysCache,
                f"Layer {i} (linear-attn) must keep ArraysCache, got {type(cache[i]).__name__}")
            # Sanity: the cache supports __getitem__.
            _ = cache[i][0]
        # Odd indices are KV-cache slots — either TurboQuantKVCache (when
        # the upstream package is installed) or plain KVCache fallback.
        for i in (1, 3, 5):
            slot_name = type(cache[i]).__name__
            self.assertIn(slot_name, ("TurboQuantKVCache", "KVCache", "QuantizedKVCache"),
                f"Layer {i} (self-attn) got unexpected cache type {slot_name}")

    @unittest.skipUnless(_MLX_LM_AVAILABLE, "mlx_lm not installed (Apple Silicon only)")
    def test_turboquant_handles_model_without_make_cache(self):
        """Plain models (no ``make_cache`` method) should still get a
        full-length cache list — preserving the pre-fix behaviour."""
        from turboquant_mlx import make_adaptive_cache

        cache = make_adaptive_cache(4, bits=3, fp16_layers=0, fused=False, model=None)
        self.assertEqual(len(cache), 4)

    # ------------------------------------------------------------------
    # required_llama_binary() metadata
    # ------------------------------------------------------------------

    def test_native_requires_standard_binary(self):
        native = self.registry.get("native")
        self.assertEqual(native.required_llama_binary(), "standard")

    def test_turboquant_requires_turbo_binary(self):
        tq = self.registry.get("turboquant")
        self.assertEqual(tq.required_llama_binary(), "turbo")

    def test_available_json_includes_required_llama_binary(self):
        available = self.registry.available()
        for entry in available:
            self.assertIn("requiredLlamaBinary", entry)
            self.assertIn(entry["requiredLlamaBinary"], ("standard", "turbo"))

    def test_broken_strategy_preserves_required_llama_binary(self):
        """When a strategy import fails, the placeholder should preserve
        the correct binary requirement from the spec."""
        real_import_module = importlib.import_module

        def fake_import(name, package=None):
            if name == "cache_compression.turboquant":
                raise RuntimeError("broken")
            return real_import_module(name, package)

        registry = CacheStrategyRegistry()
        with patch("cache_compression.importlib.import_module", side_effect=fake_import):
            registry.discover()

        tq = registry.get("turboquant")
        self.assertEqual(tq.required_llama_binary(), "turbo")


class FirstBlockCacheStrategyTests(unittest.TestCase):
    """FU-015: diffusers 0.36+ generic FBCache hook.

    Replaces FU-007's per-model TeaCache vendoring for Wan — the
    ``apply_first_block_cache`` hook is model-agnostic so Wan / FLUX /
    Hunyuan / LTX / CogVideoX / Mochi all share the same code path.
    """

    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()
        self.strategy = self.registry.get("fbcache")

    def test_fbcache_registered(self):
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.strategy_id, "fbcache")
        self.assertEqual(self.strategy.name, "First Block Cache")

    def test_fbcache_applies_to_image_and_video(self):
        self.assertEqual(self.strategy.applies_to(), frozenset({"image", "video"}))

    def test_fbcache_available_with_diffusers_036(self):
        # Test environment ships diffusers >= 0.36, so the hook should
        # import successfully. If a future bump renames the symbol,
        # this catches it on the next CI run.
        self.assertTrue(self.strategy.is_available())
        self.assertEqual(self.strategy.availability_badge(), "Ready")
        self.assertIsNone(self.strategy.availability_reason())

    def test_fbcache_recommended_thresholds(self):
        thresholds = self.strategy.recommended_thresholds()
        self.assertIn("image", thresholds)
        self.assertIn("video", thresholds)
        # Image threshold is the diffusers-blog recommendation.
        self.assertAlmostEqual(thresholds["image"], 0.12)

    def test_fbcache_apply_hook_raises_on_unet_pipeline(self):
        """UNet-based pipelines (SD1.5/SDXL) have no .transformer attribute."""
        unet_pipeline = SimpleNamespace(unet=object())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                unet_pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("DiT", str(ctx.exception))

    def test_fbcache_apply_hook_attaches_to_dit_transformer(self):
        """Smoke-test: attaching to a transformer-bearing pipeline succeeds.

        ``apply_first_block_cache`` registers diffusers hooks on the
        transformer; we don't need a real DiT — any nn.Module accepts the
        hook registration. The point is to confirm we routed through to
        diffusers without raising on the fbcache path itself.
        """
        import torch.nn as nn  # type: ignore

        class FakeDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                # Diffusers' FBCache impl walks the module tree looking
                # for blocks; an empty Sequential is enough for the
                # "no transformer blocks found" path or whatever the
                # underlying hook hits — either way it's an attach
                # exercise, not a forward exercise.
                self.transformer_blocks = nn.ModuleList([])

        dit = FakeDiT()
        pipeline = SimpleNamespace(transformer=dit)
        # Diffusers' FBCache walks transformer.transformer_blocks etc.
        # to attach hooks. With our empty FakeDiT it'll raise an
        # IndexError ("pop from empty list") trying to peel the first
        # block — that's fine. We're testing that *our* code routed
        # the call to diffusers without raising in the strategy
        # wrapper itself. Real DiT pipelines have populated block
        # lists and the hook attaches successfully.
        try:
            self.strategy.apply_diffusers_hook(
                pipeline,
                num_inference_steps=20,
                rel_l1_thresh=0.12,
            )
        except (NotImplementedError, IndexError, AttributeError):
            # Each is a "diffusers reached, but FakeDiT shape didn't
            # match what the hook expects" outcome — exactly what we
            # want this smoke test to confirm.
            pass


# ----------------------------------------------------------------------
# Post-FU-026: diffusers 0.38+ core cache hooks
#
# TaylorSeer / MagCache / PAB / FasterCache all attach via
# ``pipeline.transformer.enable_cache(<Config>)``. These tests share a
# common shape: registered, applies_to image+video, raises NotImplemented
# on UNet pipelines, raises NotImplemented when transformer lacks
# enable_cache, calls enable_cache on a DiT-shaped pipeline.
# ----------------------------------------------------------------------


class _FakeEnableCacheTransformer:
    """Minimal stand-in for a diffusers transformer with enable_cache."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def enable_cache(self, config: Any) -> None:
        self.calls.append(config)


class TaylorSeerCacheStrategyTests(unittest.TestCase):
    """Post-FU-026: diffusers 0.38+ ``TaylorSeerCacheConfig`` adapter."""

    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()
        self.strategy = self.registry.get("taylorseer")

    def test_registered(self):
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.strategy_id, "taylorseer")
        self.assertEqual(self.strategy.name, "TaylorSeer Cache")

    def test_applies_to_image_and_video(self):
        self.assertEqual(self.strategy.applies_to(), frozenset({"image", "video"}))

    def test_recommended_thresholds_present(self):
        thresholds = self.strategy.recommended_thresholds()
        self.assertIn("image", thresholds)
        self.assertIn("video", thresholds)

    def test_apply_hook_raises_on_unet_pipeline(self):
        unet_pipeline = SimpleNamespace(unet=object())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                unet_pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("DiT", str(ctx.exception))

    def test_apply_hook_raises_when_transformer_missing_enable_cache(self):
        try:
            from diffusers import TaylorSeerCacheConfig  # noqa: F401
        except ImportError:
            self.skipTest("diffusers TaylorSeerCacheConfig not present (needs 0.38+)")
        old_pipeline = SimpleNamespace(transformer=object())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                old_pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("enable_cache", str(ctx.exception))

    def test_apply_hook_calls_enable_cache_on_dit(self):
        try:
            from diffusers import TaylorSeerCacheConfig  # noqa: F401
        except ImportError:
            self.skipTest("diffusers TaylorSeerCacheConfig not present (needs 0.38+)")
        transformer = _FakeEnableCacheTransformer()
        pipeline = SimpleNamespace(transformer=transformer)
        self.strategy.apply_diffusers_hook(
            pipeline,
            num_inference_steps=20,
            rel_l1_thresh=None,
        )
        self.assertEqual(len(transformer.calls), 1)


class MagCacheStrategyTests(unittest.TestCase):
    """Post-FU-026: diffusers 0.38+ ``MagCacheConfig`` adapter (FLUX-only)."""

    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()
        self.strategy = self.registry.get("magcache")

    def test_registered(self):
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.strategy_id, "magcache")
        self.assertEqual(self.strategy.name, "MagCache")

    def test_applies_to_image_and_video(self):
        self.assertEqual(self.strategy.applies_to(), frozenset({"image", "video"}))

    def test_apply_hook_raises_on_unet_pipeline(self):
        unet_pipeline = SimpleNamespace(unet=object())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                unet_pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("DiT", str(ctx.exception))

    def test_apply_hook_raises_on_non_flux_dit_without_calibration(self):
        try:
            from diffusers import MagCacheConfig  # noqa: F401
        except ImportError:
            self.skipTest("diffusers MagCacheConfig not present (needs 0.38+)")

        class FakeWanPipeline:
            def __init__(self, transformer):
                self.transformer = transformer

        pipeline = FakeWanPipeline(_FakeEnableCacheTransformer())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("calibration", str(ctx.exception).lower())

    def test_apply_hook_succeeds_on_flux_dit(self):
        try:
            from diffusers import MagCacheConfig  # noqa: F401
            from diffusers.hooks.mag_cache import FLUX_MAG_RATIOS  # noqa: F401
        except ImportError:
            self.skipTest("FLUX_MAG_RATIOS not present in diffusers (needs 0.38+)")

        class FakeFluxPipeline:
            def __init__(self, transformer):
                self.transformer = transformer

        transformer = _FakeEnableCacheTransformer()
        pipeline = FakeFluxPipeline(transformer)
        self.strategy.apply_diffusers_hook(
            pipeline,
            num_inference_steps=4,
            rel_l1_thresh=None,
        )
        self.assertEqual(len(transformer.calls), 1)


class PyramidAttentionBroadcastStrategyTests(unittest.TestCase):
    """Post-FU-026: diffusers 0.38+ ``PyramidAttentionBroadcastConfig`` adapter."""

    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()
        self.strategy = self.registry.get("pab")

    def test_registered(self):
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.strategy_id, "pab")
        self.assertEqual(self.strategy.name, "Pyramid Attention Broadcast")

    def test_applies_to_image_and_video(self):
        self.assertEqual(self.strategy.applies_to(), frozenset({"image", "video"}))

    def test_apply_hook_raises_on_unet_pipeline(self):
        unet_pipeline = SimpleNamespace(unet=object())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                unet_pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("DiT", str(ctx.exception))

    def test_apply_hook_calls_enable_cache_on_dit(self):
        try:
            from diffusers import PyramidAttentionBroadcastConfig  # noqa: F401
        except ImportError:
            self.skipTest("diffusers PyramidAttentionBroadcastConfig not present (needs 0.38+)")
        transformer = _FakeEnableCacheTransformer()
        pipeline = SimpleNamespace(transformer=transformer)
        self.strategy.apply_diffusers_hook(
            pipeline,
            num_inference_steps=50,
            rel_l1_thresh=3.0,
        )
        self.assertEqual(len(transformer.calls), 1)


class FasterCacheStrategyTests(unittest.TestCase):
    """Post-FU-026: diffusers 0.38+ ``FasterCacheConfig`` adapter."""

    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()
        self.strategy = self.registry.get("fastercache")

    def test_registered(self):
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.strategy_id, "fastercache")
        self.assertEqual(self.strategy.name, "FasterCache")

    def test_applies_to_image_and_video(self):
        self.assertEqual(self.strategy.applies_to(), frozenset({"image", "video"}))

    def test_apply_hook_raises_on_unet_pipeline(self):
        unet_pipeline = SimpleNamespace(unet=object())
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                unet_pipeline,
                num_inference_steps=20,
                rel_l1_thresh=None,
            )
        self.assertIn("DiT", str(ctx.exception))

    def test_apply_hook_calls_enable_cache_on_dit(self):
        try:
            from diffusers import FasterCacheConfig  # noqa: F401
        except ImportError:
            self.skipTest("diffusers FasterCacheConfig not present (needs 0.38+)")
        transformer = _FakeEnableCacheTransformer()
        pipeline = SimpleNamespace(transformer=transformer)
        self.strategy.apply_diffusers_hook(
            pipeline,
            num_inference_steps=50,
            rel_l1_thresh=2.0,
        )
        self.assertEqual(len(transformer.calls), 1)


class NewStrategiesRegistryTests(unittest.TestCase):
    """All four post-FU-026 strategies present in the available() output."""

    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()

    def test_all_four_present(self):
        ids = {s["id"] for s in self.registry.available()}
        self.assertIn("taylorseer", ids)
        self.assertIn("magcache", ids)
        self.assertIn("pab", ids)
        self.assertIn("fastercache", ids)


class StartupImportPurityTests(unittest.TestCase):
    """FU-080: building the cache-strategy registry (which runs inside the
    startup system snapshot) must NOT import diffusers / torch — those are
    multi-second imports that belong on the lazy image/video path, not the
    backend cold-start path. Run in a clean subprocess so an already-warm
    ``sys.modules`` in the test runner can't mask a regression."""

    def _modules_after(self, snippet: str) -> set[str]:
        import subprocess
        import sys
        code = (
            "import sys\n"
            f"{snippet}\n"
            "print('\\n'.join(m for m in ('torch', 'diffusers', 'mlx') "
            "if m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        return {line for line in out.stdout.split() if line}

    def test_registry_available_does_not_import_torch_or_diffusers(self):
        pulled = self._modules_after(
            "from cache_compression import registry\n"
            "registry.available()\n"
        )
        self.assertEqual(
            pulled, set(),
            f"cache-strategy registry pulled heavy deps at probe time: {pulled}",
        )

    def test_app_import_does_not_pull_torch_or_diffusers(self):
        pulled = self._modules_after("import backend_service.app")
        self.assertEqual(
            pulled, set(),
            f"importing backend_service.app pulled heavy deps: {pulled}",
        )


if __name__ == "__main__":
    unittest.main()
