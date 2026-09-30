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
        "glm-5-3-flash",
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
                self.assertIn(v["backend"], ("mlx", "llama.cpp", "vllm", "tensorfold"))
                self.assertIn(v["launchMode"], ("direct", "convert"))

    def test_text_only_families_have_no_vision(self):
        # DeepSeek V4 / GLM-5 / MiniMax M2 carry no vision_config in their HF
        # configs — must not advertise vision (broken composer affordance if so).
        # Ornith-1.0 inherits Qwen 3.5's vision_config but DeepReinforce ships it
        # text-only (agentic coding), so it must not advertise vision either.
        # Mistral Large 3 is text-only (the Pixtral line carries Mistral's vision).
        for fid in (
            "deepseek-v4", "glm-5", "minimax-m2", "ornith-1", "mistral-large-3",
            "nemotron-3-5-lightning", "minicpm-5", "glm-5-3-flash",
        ):
            fam = self.by_id[fid]
            self.assertNotIn("vision", fam["capabilities"], f"{fid} family vision tag")
            for v in fam["variants"]:
                self.assertNotIn("vision", v["capabilities"], f"{fid}/{v['id']} vision tag")

    def test_qwen38_is_multimodal_like_qwen36(self):
        # Qwen3.8-27B ships Qwen3.6-27B's Qwen3_5ForConditionalGeneration
        # config (vision_config included) — same FU-072 vision policy.
        # Flash Next is a different (Qwen4-preview) architecture whose vision
        # support is unverified, so only the 27B rows are held to this.
        for v in self.by_id["qwen-3-8"]["variants"]:
            if "Flash-Next" in v["repo"]:
                self.assertNotIn("vision", v["capabilities"], v["id"])
                continue
            self.assertIn("vision", v["capabilities"], v["id"])

    def test_bonsai_lists_only_stock_runtime_formats(self):
        # Bonsai 2 (PrismML fork / mlx-vlm >= 0.7.2), MLX 1-bit (mlx-vlm
        # kernels) and Q2_0-g64 GGUF (llama.cpp >= b9994) can't run on the
        # shipped runtimes — keep them out until they can. The one exception
        # is Bonsai 2's MLX pack, which the TensorFold engine reads.
        for v in self.by_id["bonsai"]["variants"]:
            if "Bonsai-2-" in v["repo"]:
                self.assertEqual(v["backend"], "tensorfold", v["id"])
                self.assertEqual(v["format"], "MLX", v["id"])
                continue
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

    def _variants(self):
        return {v["repo"]: v for f in MODEL_FAMILIES for v in f["variants"]}

    def test_tensorfold_rows_match_the_engine_registry(self):
        # The catalog and ``inference/_tensorfold.py`` describe the same set of
        # checkpoints: every one the registry calls exclusive must be listed with
        # the tensorfold backend (nothing else can load it), and a tensorfold
        # row must name a checkpoint the registry knows.
        from backend_service.inference._tensorfold import exclusive_repos, supported_repos

        variants = self._variants()
        for repo in exclusive_repos():
            self.assertIn(repo, variants, f"{repo} is exclusive to TensorFold but not in the catalog")
            self.assertEqual(variants[repo]["backend"], "tensorfold", repo)
        for repo, variant in variants.items():
            if variant["backend"] == "tensorfold":
                self.assertIn(repo, supported_repos(), f"{repo} is a tensorfold row the engine does not serve")

    def test_every_tensorfold_tested_checkpoint_is_listed(self):
        from backend_service.inference._tensorfold import supported_repos

        variants = self._variants()
        for repo in supported_repos():
            self.assertIn(repo, variants, f"{repo} is TensorFold-tested but has no catalog row")

    def test_non_exclusive_tensorfold_checkpoints_keep_a_standard_runtime(self):
        # Native / community conversions also load on stock MLX, so their rows
        # must not claim the tensorfold backend (which would hide them from a
        # Mac without TensorFold installed).
        from backend_service.inference._tensorfold import exclusive_repos, supported_repos

        variants = self._variants()
        for repo in set(supported_repos()) - set(exclusive_repos()):
            self.assertEqual(variants[repo]["backend"], "mlx", repo)

    def test_rows_with_unknown_parameter_counts_state_their_memory(self):
        # paramsB 0 means "unknown"; the parameter-count formula would then
        # advertise ~1.6 GB for a 113 GB checkpoint.
        from backend_service.helpers.discovery import _model_family_payloads

        payloads = _model_family_payloads({"totalMemoryGb": 64, "availableMemoryGb": 32}, [])
        by_repo = {v["repo"]: v for f in payloads for v in f["variants"]}
        for repo, variant in self._variants().items():
            if variant["paramsB"] == 0:
                self.assertGreater(variant.get("estimatedMemoryGb", 0), variant["sizeGb"], repo)
                self.assertEqual(by_repo[repo]["estimatedMemoryGb"], variant["estimatedMemoryGb"], repo)
        # an explicit figure wins over the formula; rows without one still use it
        self.assertEqual(by_repo["Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"]["estimatedMemoryGb"], 118.0)
        self.assertEqual(by_repo["mlx-community/Qwen3.8-27B-4bit"]["estimatedMemoryGb"], 21.0)

    def test_tensorfold_rows_state_their_memory_requirement(self):
        from backend_service.inference._tensorfold import min_memory_gb_by_repo

        variants = self._variants()
        for repo, gb in min_memory_gb_by_repo().items():
            note = variants[repo]["note"]
            self.assertIn(f"{gb} GB", note, f"{repo} note should state the {gb} GB Mac requirement")

    def test_new_families_surface_in_discover_payloads(self):
        from backend_service.helpers.discovery import _model_family_payloads

        payloads = _model_family_payloads({"totalMemoryGb": 64, "availableMemoryGb": 32}, [])
        ids = {p.get("id") for p in payloads}
        for fid in self._ALL_NEW_FAMILIES:
            self.assertIn(fid, ids, f"{fid} missing from discover payloads")


if __name__ == "__main__":
    unittest.main()
