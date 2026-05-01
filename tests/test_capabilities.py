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
        caps = resolve_capabilities("google/gemma-4-E4B-it", None)
        self.assertTrue(caps.supportsVision)
        self.assertTrue(caps.supportsReasoning)
        self.assertIn("vision", caps.tags)

    def test_canonical_repo_lookup_falls_back_when_ref_misses(self):
        caps = resolve_capabilities(
            "mlx-community/gemma-4-12B-it-4bit",
            canonical_repo="google/gemma-4-12B-it",
        )
        self.assertTrue(caps.supportsVision)

    def test_heuristic_picks_up_vision_in_ref_name(self):
        caps = resolve_capabilities("custom-org/my-llava-vision-model-7b", None)
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

    def test_mlx_engine_demotes_vision(self):
        # Hotfix (2026-05-01): the MLX worker subprocess never wired
        # vision input through, so even when the catalog says a model
        # supports vision the resolver must demote that flag for the
        # MLX engine. Catalog-level "vision" tag stays in `tags` so the
        # UI can still surface "this model would support vision via
        # llama.cpp" later, but the typed flag drives the composer
        # gate that hides the image-attach button today.
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="mlx",
        )
        self.assertFalse(caps.supportsVision)
        self.assertIn("vision", caps.tags)

    def test_turboquant_engine_demotes_vision(self):
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="turboquant",
        )
        self.assertFalse(caps.supportsVision)

    def test_llama_cpp_engine_keeps_vision(self):
        # llama.cpp accepts image_url parts natively when an mmproj is
        # loaded, so vision should remain promoted on this path.
        caps = resolve_capabilities(
            "google/gemma-4-E4B-it",
            None,
            engine="llama.cpp",
        )
        self.assertTrue(caps.supportsVision)

    def test_engine_unset_keeps_catalog_capabilities(self):
        # Default behaviour (no engine specified) preserves the catalog
        # capability list — important for tests / callers that don't
        # know the engine yet.
        caps = resolve_capabilities("google/gemma-4-E4B-it", None)
        self.assertTrue(caps.supportsVision)


if __name__ == "__main__":
    unittest.main()
