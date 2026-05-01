"""Tests for the Phase 2.11 model capability resolver.

The resolver maps a loaded-model ref/canonical to a typed
`ModelCapabilities` blob the frontend uses to gate composer features
and render badges. Catalog match wins; a substring heuristic fallback
applies for refs that don't appear in the curated catalog.
"""

import unittest

from backend_service.catalog.capabilities import (
    ModelCapabilities,
    resolve_capabilities,
)


class ResolveCapabilitiesTests(unittest.TestCase):
    def test_returns_empty_when_ref_unknown(self):
        caps = resolve_capabilities("totally-unknown/random-model", None)
        self.assertEqual(caps.tags, ())
        self.assertFalse(caps.supportsVision)
        self.assertFalse(caps.supportsTools)
        self.assertFalse(caps.supportsReasoning)

    def test_catalog_match_promotes_typed_flags(self):
        # Vision flag depends on the runtime confirming mmproj is loaded.
        # Pass vision_enabled=True to simulate the post-mmproj-wiring
        # state; the catalog has both vision and reasoning tags.
        caps = resolve_capabilities("google/gemma-4-E4B-it", None, vision_enabled=True)
        self.assertTrue(caps.supportsVision)
        self.assertTrue(caps.supportsReasoning)
        self.assertIn("vision", caps.tags)

    def test_canonical_repo_lookup_falls_back_when_ref_misses(self):
        caps = resolve_capabilities(
            "mlx-community/gemma-4-12B-it-4bit",
            canonical_repo="google/gemma-4-12B-it",
            vision_enabled=True,
        )
        self.assertTrue(caps.supportsVision)

    def test_heuristic_picks_up_vision_in_ref_name(self):
        caps = resolve_capabilities(
            "custom-org/my-llava-vision-model-7b",
            None,
            vision_enabled=True,
        )
        self.assertTrue(caps.supportsVision)
        self.assertIn("vision", caps.tags)

    def test_heuristic_picks_up_reasoning_for_r1_models(self):
        caps = resolve_capabilities("DeepSeek/DeepSeek-R1-Distill-Qwen-7B", None)
        self.assertTrue(caps.supportsReasoning)

    def test_heuristic_picks_up_coder_models(self):
        caps = resolve_capabilities("Qwen/Qwen3-Coder-Instruct", None)
        self.assertTrue(caps.supportsCoding)

    def test_instruct_models_get_tools_capability(self):
        caps = resolve_capabilities("meta/llama-4-8B-instruct", None)
        self.assertTrue(caps.supportsTools)

    def test_to_dict_preserves_all_fields(self):
        caps = ModelCapabilities(
            supportsVision=True,
            supportsTools=True,
            tags=("vision", "tool-use"),
        )
        d = caps.to_dict()
        self.assertTrue(d["supportsVision"])
        self.assertTrue(d["supportsTools"])
        self.assertFalse(d["supportsReasoning"])
        self.assertEqual(d["tags"], ["vision", "tool-use"])

    def test_family_fallback_picks_up_multilingual(self):
        # Variants don't all carry every family-level tag. When a ref
        # doesn't match a variant directly, the family-level fallback
        # supplies its capability list — Gemma 4 family includes
        # "multilingual", which should propagate.
        caps = resolve_capabilities("custom-org/gemma-4-fork-quant", None)
        self.assertTrue(caps.supportsMultilingual)
        self.assertIn("multilingual", caps.tags)

    def test_none_inputs_return_empty_capabilities(self):
        caps = resolve_capabilities(None, None)
        self.assertEqual(caps.tags, ())
        self.assertFalse(any([
            caps.supportsVision, caps.supportsTools, caps.supportsReasoning,
            caps.supportsCoding, caps.supportsAgents, caps.supportsAudio,
            caps.supportsVideo, caps.supportsMultilingual,
        ]))

    def test_mlx_engine_demotes_vision_even_when_runtime_says_enabled(self):
        # Belt-and-braces: even if a future mmproj-equivalent path on
        # MLX claims vision_enabled=True, the engine demotion still
        # fires because the MLX worker subprocess has no image-carrying
        # code. Re-enable this check only after mlx-vlm is actually wired.
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="mlx",
            vision_enabled=True,
        )
        self.assertFalse(caps.supportsVision)
        self.assertIn("vision", caps.tags)

    def test_turboquant_engine_demotes_vision(self):
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="turboquant",
            vision_enabled=True,
        )
        self.assertFalse(caps.supportsVision)

    def test_llama_cpp_engine_keeps_vision_when_runtime_enabled(self):
        # llama.cpp accepts image_url parts natively when an mmproj is
        # loaded — vision_enabled=True simulates that runtime state.
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="llama.cpp",
            vision_enabled=True,
        )
        self.assertTrue(caps.supportsVision)

    def test_llama_cpp_engine_demotes_vision_when_runtime_disabled(self):
        # Default vision_enabled=False — even on llama.cpp, vision must
        # be demoted until the runtime confirms mmproj is loaded. This
        # is the post-launch fix for the user's "model hallucinates
        # about attached image" report.
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="llama.cpp",
        )
        self.assertFalse(caps.supportsVision)

    def test_engine_unset_demotes_vision_without_runtime_proof(self):
        # Default behaviour (no engine, no vision_enabled) demotes
        # vision — callers must opt in by proving runtime support.
        caps = resolve_capabilities("google/gemma-4-E4B-it", None)
        self.assertFalse(caps.supportsVision)


if __name__ == "__main__":
    unittest.main()
