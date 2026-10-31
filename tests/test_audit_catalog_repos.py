"""Pure parts of ``scripts/audit-catalog-repos.py`` (the Hub calls are not exercised)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit-catalog-repos.py"
    spec = importlib.util.spec_from_file_location("audit_catalog_repos", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_catalog_repos"] = module
    spec.loader.exec_module(module)
    return module


audit = _load_script()


class CollectRefsTests(unittest.TestCase):
    def test_text_rows_pin_a_file_in_their_own_repo(self) -> None:
        refs = audit.collect_refs({"text": [{"variants": [{"id": "t", "repo": "o/gguf", "ggufFile": "m-Q4_K_M.gguf"}]}]})
        self.assertEqual(refs, [audit.Ref("text", "t", "o/gguf", "m-Q4_K_M.gguf", "repo")])

    def test_image_rows_report_every_companion_repo(self) -> None:
        variant = {
            "id": "v",
            "repo": "o/base",
            "ggufRepo": "o/gguf",
            "ggufFile": "m.gguf",
            "loraRepo": "o/lora",
            "loraFile": "l.safetensors",
            "sdcppAux": {"--vae": {"repo": "o/vae", "file": "ae.safetensors"}},
        }
        found = {(r.repo, r.filename, r.role) for r in audit.collect_refs({"image": [{"variants": [variant]}]})}
        self.assertEqual(
            found,
            {
                ("o/base", None, "repo"),
                ("o/gguf", "m.gguf", "gguf"),
                ("o/lora", "l.safetensors", "lora"),
                ("o/vae", "ae.safetensors", "sdcppAux --vae"),
            },
        )

    def test_distill_weights_check_both_expert_files(self) -> None:
        variant = {
            "id": "v",
            "repo": "o/base",
            "distillTransformerRepo": "o/distill",
            "distillTransformerHighNoiseFile": "high.safetensors",
            "distillTransformerLowNoiseFile": "low.safetensors",
        }
        files = {r.filename for r in audit.collect_refs({"video": [{"variants": [variant]}]}) if r.repo == "o/distill"}
        self.assertEqual(files, {"high.safetensors", "low.safetensors"})

    def test_rows_without_a_repo_add_nothing(self) -> None:
        self.assertEqual(audit.collect_refs({"image": [{"variants": [{"id": "v"}]}]}), [])


class FindProblemsTests(unittest.TestCase):
    def test_unreachable_repos_and_missing_files_are_reported_separately(self) -> None:
        refs = [
            audit.Ref("image", "a", "o/gone", "x.gguf", "gguf"),
            audit.Ref("image", "b", "o/live", "have.gguf", "gguf"),
            audit.Ref("image", "c", "o/live", "lost.gguf", "gguf"),
            audit.Ref("image", "d", "o/live", None, "repo"),
        ]
        dead, missing = audit.find_problems(refs, {"o/gone": None, "o/live": {"have.gguf"}})
        self.assertEqual(list(dead), ["o/gone"])
        self.assertEqual([r.variant for r in missing], ["c"])

    def test_a_clean_catalog_has_no_problems(self) -> None:
        refs = [audit.Ref("text", "t", "o/r", "f.gguf", "repo")]
        self.assertEqual(audit.find_problems(refs, {"o/r": {"f.gguf"}}), ({}, []))


if __name__ == "__main__":
    unittest.main()
