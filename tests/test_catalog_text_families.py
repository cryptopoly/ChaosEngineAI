"""Catalog gate for the frontier text families added for the release
(DeepSeek V4, GLM-5, Gemma 4, MiniMax M2). Asserts they parse, carry every
field the discover payload builder reads, and surface in the family payloads
— so a malformed entry can't ship a broken Discover tab.
"""

import unittest

from backend_service.catalog.text_models import MODEL_FAMILIES

_REQUIRED_FAMILY_FIELDS = {
    "id", "name", "provider", "headline", "summary", "description",
    "updatedLabel", "popularityLabel", "likesLabel", "badges", "capabilities",
    "defaultVariantId", "variants", "readme",
}
_REQUIRED_VARIANT_FIELDS = {
    "id", "name", "repo", "link", "paramsB", "sizeGb", "format",
    "quantization", "capabilities", "note", "contextWindow", "launchMode", "backend",
}


class NewTextFamiliesTests(unittest.TestCase):
    def setUp(self):
        self.by_id = {f["id"]: f for f in MODEL_FAMILIES}

    _ALL_NEW_FAMILIES = (
        "deepseek-v4", "glm-5", "gemma-4", "minimax-m2", "ornith-1",
        "kimi-k2-6", "llama-4-scout", "minimax-m3", "mistral-large-3",
        # 2026-09 additions.
        "qwen-3-8", "bonsai", "nemotron-3-5-lightning", "minicpm-5",
    )

    def test_all_new_families_present(self):
        for fid in self._ALL_NEW_FAMILIES:
            self.assertIn(fid, self.by_id, f"{fid} missing from MODEL_FAMILIES")

    def test_new_families_have_required_shape(self):
        for fid in self._ALL_NEW_FAMILIES:
            fam = self.by_id[fid]
            self.assertEqual(_REQUIRED_FAMILY_FIELDS - set(fam), set(), f"{fid} family fields")
            self.assertTrue(fam["variants"], f"{fid} has variants")
            variant_ids = [v["id"] for v in fam["variants"]]
            self.assertIn(fam["defaultVariantId"], variant_ids, f"{fid} default variant valid")
            for v in fam["variants"]:
                self.assertEqual(_REQUIRED_VARIANT_FIELDS - set(v), set(), f"{fid}/{v['id']} variant fields")
                self.assertEqual(v["link"], f"https://huggingface.co/{v['repo']}", f"{fid}/{v['id']} link")
                self.assertIn(v["backend"], ("mlx", "llama.cpp", "vllm"))
                self.assertIn(v["launchMode"], ("direct", "convert"))

    def test_text_only_families_have_no_vision(self):
        # DeepSeek V4 / GLM-5 / MiniMax M2 carry no vision_config in their HF
        # configs — must not advertise vision (broken composer affordance if so).
        # Ornith-1.0 inherits Qwen 3.5's vision_config but DeepReinforce ships it
        # text-only (agentic coding), so it must not advertise vision either.
        # Mistral Large 3 is text-only (the Pixtral line carries Mistral's vision).
        for fid in (
            "deepseek-v4", "glm-5", "minimax-m2", "ornith-1", "mistral-large-3",
            "nemotron-3-5-lightning", "minicpm-5",
        ):
            fam = self.by_id[fid]
            self.assertNotIn("vision", fam["capabilities"], f"{fid} family vision tag")
            for v in fam["variants"]:
                self.assertNotIn("vision", v["capabilities"], f"{fid}/{v['id']} vision tag")

    def test_qwen38_is_multimodal_like_qwen36(self):
        # Qwen3.8-27B ships Qwen3.6-27B's Qwen3_5ForConditionalGeneration
        # config (vision_config included) — same FU-072 vision policy.
        for v in self.by_id["qwen-3-8"]["variants"]:
            self.assertIn("vision", v["capabilities"], v["id"])

    def test_bonsai_lists_only_stock_runtime_formats(self):
        # Bonsai 2 (PrismML fork / mlx-vlm >= 0.7.2), MLX 1-bit (mlx-vlm
        # kernels) and Q2_0-g64 GGUF (llama.cpp >= b9994) can't run on the
        # shipped runtimes — keep them out until they can.
        for v in self.by_id["bonsai"]["variants"]:
            self.assertNotIn("Bonsai-2-", v["repo"], v["id"])
            self.assertNotIn("1bit", v["repo"], v["id"])
            if v["format"] == "GGUF":
                self.assertIn("Q1_0", v["ggufFile"], v["id"])
            else:
                self.assertEqual(v["format"], "MLX", v["id"])
                self.assertIn("2bit", v["repo"], v["id"])

    def test_low_bit_memory_estimate(self):
        # 1-bit / ternary labels used to fall through to the full-precision
        # factor, so Bonsai 27B advertised ~28.6 GB (measured peak: 7.9 GB).
        from backend_service.helpers.model_family_payload import _estimate_runtime_memory_gb

        for v in self.by_id["bonsai"]["variants"]:
            estimate = _estimate_runtime_memory_gb(v["paramsB"], v["quantization"])
            full = _estimate_runtime_memory_gb(v["paramsB"], "BF16")
            self.assertLess(estimate, full, f"{v['id']}: {estimate} GB vs {full} GB full-precision")
        self.assertLess(_estimate_runtime_memory_gb(27.0, "2-bit (ternary)"), 12.0)
        self.assertGreater(
            _estimate_runtime_memory_gb(27.0, "2-bit (ternary)"),
            _estimate_runtime_memory_gb(27.0, "Q1_0 (1-bit)"),
        )

    def test_multi_quant_gguf_rows_in_new_families_are_pinned(self):
        for fid in ("qwen-3-8", "bonsai", "nemotron-3-5-lightning", "minicpm-5"):
            for v in self.by_id[fid]["variants"]:
                if v["format"] == "GGUF":
                    self.assertTrue(v.get("ggufFile"), f"{fid}/{v['id']} needs a ggufFile pin")

    def test_gemma4_carries_vision_capability(self):
        # All Gemma 4 sizes are multimodal (Gemma4ForConditionalGeneration + vision_config).
        fam = self.by_id["gemma-4"]
        self.assertIn("vision", fam["capabilities"])
        for v in fam["variants"]:
            self.assertIn("vision", v["capabilities"], f"gemma-4/{v['id']} missing vision tag")

    def test_multimodal_chat_families_carry_vision(self):
        # Kimi K2.6 (KimiK25ForConditionalGeneration), Llama 4 Scout
        # (Llama4ForConditionalGeneration), and MiniMax M3 (minimax_m3_vl) all
        # ship a functional vision encoder — verified against their live HF
        # config.json (vision_config present, ConditionalGeneration arch).
        for fid in ("kimi-k2-6", "llama-4-scout", "minimax-m3"):
            fam = self.by_id[fid]
            self.assertIn("vision", fam["capabilities"], f"{fid} family missing vision tag")
            for v in fam["variants"]:
                self.assertIn("vision", v["capabilities"], f"{fid}/{v['id']} missing vision tag")

    def test_long_context_chat_families(self):
        # Verify the headline context windows match the live config.json
        # max_position_embeddings: Llama 4 Scout = 10M, MiniMax M3 = 1M.
        for v in self.by_id["llama-4-scout"]["variants"]:
            self.assertEqual(v["contextWindow"], "10M", f"{v['id']} Scout context wrong")
        for v in self.by_id["minimax-m3"]["variants"]:
            self.assertEqual(v["contextWindow"], "1M", f"{v['id']} M3 context wrong")

    def test_gemma4_contexts(self):
        # E2B = 128K, 31B = 256K — verify the catalog reflects the config.json values.
        e2b_variants = [v for v in self.by_id["gemma-4"]["variants"] if "E2B" in v["repo"]]
        b31_variants = [v for v in self.by_id["gemma-4"]["variants"] if "31B" in v["repo"] or "31b" in v["repo"]]
        self.assertTrue(e2b_variants, "no E2B variants found")
        self.assertTrue(b31_variants, "no 31B variants found")
        for v in e2b_variants:
            self.assertEqual(v["contextWindow"], "128K", f"{v['id']} E2B context wrong")
        for v in b31_variants:
            self.assertEqual(v["contextWindow"], "256K", f"{v['id']} 31B context wrong")

    def test_minimax_m27_context(self):
        fam = self.by_id["minimax-m2"]
        for v in fam["variants"]:
            self.assertEqual(v["contextWindow"], "200K", f"minimax-m2/{v['id']} context wrong")

    def test_new_families_surface_in_discover_payloads(self):
        from backend_service.helpers.discovery import _model_family_payloads

        payloads = _model_family_payloads({"totalMemoryGb": 64, "availableMemoryGb": 32}, [])
        ids = {p.get("id") for p in payloads}
        for fid in self._ALL_NEW_FAMILIES:
            self.assertIn(fid, ids, f"{fid} missing from discover payloads")


if __name__ == "__main__":
    unittest.main()
