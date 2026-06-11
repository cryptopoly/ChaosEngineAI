"""Catalog gate for the frontier text families added for the release
(DeepSeek V4, GLM-5). Asserts they parse, carry every field the discover
payload builder reads, and surface in the family payloads — so a malformed
entry can't ship a broken Discover tab.
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

    def test_deepseek_v4_and_glm5_present(self):
        self.assertIn("deepseek-v4", self.by_id)
        self.assertIn("glm-5", self.by_id)

    def test_new_families_have_required_shape(self):
        for fid in ("deepseek-v4", "glm-5"):
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

    def test_new_families_are_text_only_no_vision(self):
        # DeepSeek V4 + GLM-5 configs carry no vision_config — the catalog
        # must not advertise vision (would render a broken composer affordance).
        for fid in ("deepseek-v4", "glm-5"):
            fam = self.by_id[fid]
            self.assertNotIn("vision", fam["capabilities"], f"{fid} family vision tag")
            for v in fam["variants"]:
                self.assertNotIn("vision", v["capabilities"], f"{fid}/{v['id']} vision tag")

    def test_new_families_surface_in_discover_payloads(self):
        from backend_service.helpers.discovery import _model_family_payloads

        payloads = _model_family_payloads({"totalMemoryGb": 64, "availableMemoryGb": 32}, [])
        ids = {p.get("id") for p in payloads}
        self.assertIn("deepseek-v4", ids)
        self.assertIn("glm-5", ids)


if __name__ == "__main__":
    unittest.main()
