"""Tests for TeaCache diffusion cache strategy.

Covers: registry registration, availability reporting, diffusion-domain
signaling, forward-patch mapping shape, and the apply_diffusers_hook
contract (NotImplementedError for unsupported pipelines, rejection of
bad thresholds, attribute mutation on the transformer class when a
patch is available).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cache_compression import (
    CacheStrategyRegistry,
    apply_diffusion_cache_strategy,
)
from cache_compression.teacache import (
    TeaCacheStrategy,
    _DEFAULT_REL_L1_THRESH,
    _FORWARD_PATCHES,
    _RECOMMENDED_THRESHOLDS,
)


class TeaCacheRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.registry = CacheStrategyRegistry()
        self.registry.discover()

    def test_teacache_registered(self):
        strategy = self.registry.get("teacache")
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.strategy_id, "teacache")
        self.assertEqual(strategy.name, "TeaCache")

    def test_teacache_in_available_list(self):
        ids = [entry["id"] for entry in self.registry.available()]
        self.assertIn("teacache", ids)

    def test_teacache_applies_to_diffusion(self):
        strategy = self.registry.get("teacache")
        domains = strategy.applies_to()
        self.assertIn("image", domains)
        self.assertIn("video", domains)
        self.assertNotIn("text", domains)

    def test_available_json_includes_applies_to(self):
        entries = self.registry.available()
        for entry in entries:
            self.assertIn("appliesTo", entry)
            self.assertIsInstance(entry["appliesTo"], list)

    def test_text_strategies_default_to_text_domain(self):
        """The default CacheStrategy.applies_to() is {"text"} — every
        pre-existing strategy (native, rotor, tri, turbo, chaos) should
        still report text-only without any code change.
        """
        for sid in ("native", "rotorquant", "triattention", "turboquant", "chaosengine"):
            strategy = self.registry.get(sid)
            self.assertEqual(
                strategy.applies_to(), frozenset({"text"}),
                f"{sid} should default to text-only applies_to()",
            )


class TeaCacheMetadataTests(unittest.TestCase):
    def setUp(self):
        self.strategy = TeaCacheStrategy()

    def test_bit_range_none(self):
        # TeaCache is a float-threshold strategy, not a quantizer.
        self.assertIsNone(self.strategy.supported_bit_range())
        self.assertIsNone(self.strategy.default_bits())
        self.assertFalse(self.strategy.supports_fp16_layers())

    def test_default_threshold_matches_constant(self):
        self.assertEqual(self.strategy.default_rel_l1_thresh(), _DEFAULT_REL_L1_THRESH)
        self.assertEqual(_DEFAULT_REL_L1_THRESH, 0.4)

    def test_recommended_thresholds_shape(self):
        presets = self.strategy.recommended_thresholds()
        self.assertIn("balanced", presets)
        self.assertIn("conservative", presets)
        self.assertIn("aggressive", presets)
        # Each entry is (float, str). Thresholds must be monotonically
        # increasing from conservative → aggressive so the UI slider maps
        # cleanly to speedup tiers.
        thresh_c = presets["conservative"][0]
        thresh_b = presets["balanced"][0]
        thresh_a = presets["aggressive"][0]
        self.assertLess(thresh_c, thresh_b)
        self.assertLess(thresh_b, thresh_a)

    def test_supported_pipeline_classes_matches_patches_dict(self):
        self.assertEqual(
            self.strategy.supported_pipeline_classes(),
            frozenset(_FORWARD_PATCHES.keys()),
        )


class TeaCacheAvailabilityTests(unittest.TestCase):
    def test_scaffold_badge_when_patches_empty(self):
        """While _FORWARD_PATCHES is empty the registry should report a
        warning badge — strategy is wired but no pipeline has a patch."""
        strategy = TeaCacheStrategy()
        with patch("cache_compression.teacache._FORWARD_PATCHES", {}):
            with patch("cache_compression.teacache._diffusers_available", return_value=True):
                self.assertEqual(strategy.availability_badge(), "Scaffold")
                self.assertEqual(strategy.availability_tone(), "warning")
                reason = strategy.availability_reason()
                self.assertIn("scaffold", reason.lower())

    def test_ready_badge_when_patches_present(self):
        strategy = TeaCacheStrategy()
        fake_patches = {"FluxTransformer2DModel": ("m", "f")}
        with patch("cache_compression.teacache._FORWARD_PATCHES", fake_patches):
            with patch("cache_compression.teacache._diffusers_available", return_value=True):
                self.assertEqual(strategy.availability_badge(), "Ready")
                self.assertEqual(strategy.availability_tone(), "ready")
                reason = strategy.availability_reason()
                self.assertIn("FluxTransformer2DModel", reason)

    def test_install_badge_when_diffusers_missing(self):
        strategy = TeaCacheStrategy()
        with patch("cache_compression.teacache._diffusers_available", return_value=False):
            self.assertFalse(strategy.is_available())
            self.assertEqual(strategy.availability_badge(), "Install")
            self.assertEqual(strategy.availability_tone(), "install")
            reason = strategy.availability_reason()
            self.assertIn("diffusers", reason)


class TeaCacheHookTests(unittest.TestCase):
    def setUp(self):
        self.strategy = TeaCacheStrategy()

    def _make_pipeline_with_transformer_class(self, class_name: str):
        # Build a fresh class per test so the attribute writes in
        # apply_diffusers_hook don't leak between cases. The class body
        # doesn't need to be meaningful — we only assert on attributes
        # set after the hook runs.
        klass = type(class_name, (), {})
        transformer = klass()
        pipeline = SimpleNamespace(transformer=transformer)
        return pipeline, klass

    def test_raises_when_pipeline_has_no_transformer(self):
        pipeline = SimpleNamespace()  # no .transformer
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                pipeline, num_inference_steps=25, rel_l1_thresh=0.4,
            )
        self.assertIn("DiT", str(ctx.exception))

    def test_raises_when_transformer_class_unsupported(self):
        pipeline, _ = self._make_pipeline_with_transformer_class("UnknownDiTModel")
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.apply_diffusers_hook(
                pipeline, num_inference_steps=25, rel_l1_thresh=0.4,
            )
        # Error must name the transformer class so the UI can show an
        # actionable message.
        self.assertIn("UnknownDiTModel", str(ctx.exception))

    def test_rejects_zero_threshold(self):
        pipeline, _ = self._make_pipeline_with_transformer_class("FluxTransformer2DModel")
        with self.assertRaises(ValueError):
            self.strategy.apply_diffusers_hook(
                pipeline, num_inference_steps=25, rel_l1_thresh=0,
            )

    def test_rejects_negative_threshold(self):
        pipeline, _ = self._make_pipeline_with_transformer_class("FluxTransformer2DModel")
        with self.assertRaises(ValueError):
            self.strategy.apply_diffusers_hook(
                pipeline, num_inference_steps=25, rel_l1_thresh=-0.1,
            )

    def test_rejects_zero_steps(self):
        pipeline, _ = self._make_pipeline_with_transformer_class("FluxTransformer2DModel")
        with self.assertRaises(ValueError):
            self.strategy.apply_diffusers_hook(
                pipeline, num_inference_steps=0, rel_l1_thresh=0.4,
            )

    def test_patched_pipeline_gets_attributes_and_forward(self):
        """When a fake patch is registered for a pipeline class the hook
        should replace .forward and set all seven TeaCache state attrs."""

        def fake_forward(*args, **kwargs):
            return None

        fake_module = SimpleNamespace(teacache_forward=fake_forward)
        pipeline, klass = self._make_pipeline_with_transformer_class("FakeDiT")
        patches = {"FakeDiT": ("cache_compression.teacache_fake_patch", "teacache_forward")}

        with patch("cache_compression.teacache._FORWARD_PATCHES", patches):
            with patch(
                "cache_compression.teacache.importlib.import_module",
                return_value=fake_module,
            ):
                self.strategy.apply_diffusers_hook(
                    pipeline, num_inference_steps=30, rel_l1_thresh=0.5,
                )

        # Forward replaced on the class (not just the instance) per
        # upstream pattern — all transformers of this class share it.
        self.assertIs(klass.forward, fake_forward)
        # Seven TeaCache state attributes present and correct.
        self.assertTrue(klass.enable_teacache)
        self.assertEqual(klass.cnt, 0)
        self.assertEqual(klass.num_steps, 30)
        self.assertEqual(klass.rel_l1_thresh, 0.5)
        self.assertEqual(klass.accumulated_rel_l1_distance, 0)
        self.assertIsNone(klass.previous_modulated_input)
        self.assertIsNone(klass.previous_residual)

    def test_patched_pipeline_defaults_threshold_when_none(self):
        def fake_forward(*args, **kwargs):
            return None

        fake_module = SimpleNamespace(teacache_forward=fake_forward)
        pipeline, klass = self._make_pipeline_with_transformer_class("FakeDiT2")
        patches = {"FakeDiT2": ("cache_compression.teacache_fake_patch", "teacache_forward")}

        with patch("cache_compression.teacache._FORWARD_PATCHES", patches):
            with patch(
                "cache_compression.teacache.importlib.import_module",
                return_value=fake_module,
            ):
                self.strategy.apply_diffusers_hook(
                    pipeline, num_inference_steps=20, rel_l1_thresh=None,
                )

        self.assertEqual(klass.rel_l1_thresh, _DEFAULT_REL_L1_THRESH)


class TeaCacheTextEngineRefusalTests(unittest.TestCase):
    """TeaCache must not be usable as a text-LLM KV cache. All text hooks
    raise with messages pointing at a compatible alternative."""

    def setUp(self):
        self.strategy = TeaCacheStrategy()

    def test_make_mlx_cache_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.make_mlx_cache(32, 0, 0, False, None)
        self.assertIn("diffusion", str(ctx.exception).lower())

    def test_llama_cpp_cache_flags_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            self.strategy.llama_cpp_cache_flags(0)
        self.assertIn("llama.cpp", str(ctx.exception))

    def test_apply_vllm_patches_is_noop(self):
        # No exception — TeaCache is not a vLLM monkeypatch.
        self.assertIsNone(self.strategy.apply_vllm_patches())

    def test_estimate_cache_bytes_reports_no_compression(self):
        baseline, optimised = self.strategy.estimate_cache_bytes(
            num_layers=32, num_heads=32, hidden_size=4096,
            context_tokens=8192, bits=0, fp16_layers=0,
        )
        # TeaCache isn't a KV compressor — the preview must show no
        # shrink (otherwise the UI would mislead users into expecting
        # lower VRAM use when they get faster generation instead).
        self.assertEqual(baseline, optimised)


class ApplyDiffusionCacheStrategyHelperTests(unittest.TestCase):
    """The shared ``apply_diffusion_cache_strategy`` helper is the entry
    point image_runtime and video_runtime use — must never raise
    NotImplementedError up to the caller, and must skip silently on
    unknown ids or domain mismatch."""

    def test_returns_none_when_strategy_id_falsy(self):
        self.assertIsNone(apply_diffusion_cache_strategy(
            pipeline=SimpleNamespace(),
            strategy_id=None,
            num_inference_steps=25,
            rel_l1_thresh=0.4,
            domain="image",
        ))
        self.assertIsNone(apply_diffusion_cache_strategy(
            pipeline=SimpleNamespace(),
            strategy_id="",
            num_inference_steps=25,
            rel_l1_thresh=0.4,
            domain="image",
        ))

    def test_returns_note_for_unknown_strategy(self):
        note = apply_diffusion_cache_strategy(
            pipeline=SimpleNamespace(),
            strategy_id="nonexistent",
            num_inference_steps=25,
            rel_l1_thresh=0.4,
            domain="image",
        )
        self.assertIsNotNone(note)
        self.assertIn("not found", note.lower())

    def test_returns_note_on_domain_mismatch(self):
        # native is text-only; asking it to apply to image should skip.
        note = apply_diffusion_cache_strategy(
            pipeline=SimpleNamespace(),
            strategy_id="native",
            num_inference_steps=25,
            rel_l1_thresh=0.4,
            domain="image",
        )
        self.assertIsNotNone(note)
        self.assertIn("does not apply", note.lower())

    def test_swallows_not_implemented_from_hook(self):
        # TeaCache hook raises NotImplementedError for unsupported
        # pipelines; the helper must convert that into a note (so the
        # image/video generate path keeps running on the stock pipeline).
        pipeline = SimpleNamespace(transformer=SimpleNamespace())
        note = apply_diffusion_cache_strategy(
            pipeline=pipeline,
            strategy_id="teacache",
            num_inference_steps=25,
            rel_l1_thresh=0.4,
            domain="image",
        )
        self.assertIsNotNone(note)
        self.assertIn("not applied", note.lower())


if __name__ == "__main__":
    unittest.main()
