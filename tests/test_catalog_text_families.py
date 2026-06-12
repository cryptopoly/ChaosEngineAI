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

    _ALL_NEW_FAMILIES = ("deepseek-v4", "glm-5", "gemma-4", "minimax-m2")

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
        for fid in ("deepseek-v4", "glm-5", "minimax-m2"):
            fam = self.by_id[fid]
            self.assertNotIn("vision", fam["capabilities"], f"{fid} family vision tag")
            for v in fam["variants"]:
                self.assertNotIn("vision", v["capabilities"], f"{fid}/{v['id']} vision tag")

    def test_gemma4_carries_vision_capability(self):
        # All Gemma 4 sizes are multimodal (Gemma4ForConditionalGeneration + vision_config).
        fam = self.by_id["gemma-4"]
        self.assertIn("vision", fam["capabilities"])
        for v in fam["variants"]:
            self.assertIn("vision", v["capabilities"], f"gemma-4/{v['id']} missing vision tag")

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
