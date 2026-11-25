"""Video upscale catalog rows: their fields, fit through ``helpers.fit``, and the release audit of unpinned weights."""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

from backend_service.catalog import video_upscale_models as catalog
from backend_service.helpers.fit import variant_fit
from backend_service.helpers.hardware_class import HardwareClass


def _audit():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit-catalog-repos.py"
    spec = importlib.util.spec_from_file_location("audit_catalog_repos_upscale", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_catalog_repos_upscale"] = module
    spec.loader.exec_module(module)
    return module


CUDA_24 = HardwareClass("cuda", "cuda-24", "NVIDIA, 24 GB", 24.0, 24.0)
CUDA_4 = HardwareClass("cuda", "cuda-4", "NVIDIA, 4 GB", 4.0, 4.0)
MAC_16 = HardwareClass("apple", "apple-16", "Apple Silicon, 16 GB", 16.0, 10.7)


class CatalogTests(unittest.TestCase):
    def test_rows(self):
        ids = [row["id"] for row in catalog.VIDEO_UPSCALE_MODELS]
        self.assertEqual(ids, ["realesr-general-x4v3", "realesrgan-x4plus"])
        self.assertEqual(catalog.find_model(catalog.DEFAULT_MODEL_ID)["default"], True)
        self.assertIsNone(catalog.find_model("nope"))
        for row in catalog.VIDEO_UPSCALE_MODELS:
            self.assertEqual(row["licence"]["name"], "BSD-3-Clause")
            self.assertTrue(row["url"].startswith("https://github.com/xinntao/Real-ESRGAN/releases/download/"))
            self.assertTrue(row["url"].endswith("/" + row["file"]))
            self.assertGreater(row["sizeBytes"], 1_000_000)
            self.assertEqual(row["scale"], 4)
            self.assertEqual(row["footprintBasis"], "estimated")  # the memory numbers are not measured
            self.assertIn("Estimate", row["speedNote"])
            if row["sha256"] is not None:
                self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            else:
                self.assertIn("Not pinned", row["sha256Todo"])

    def test_fit_uses_the_shared_gate(self):
        for row in catalog.VIDEO_UPSCALE_MODELS:
            fit = variant_fit(row, "video", CUDA_24)
            self.assertEqual(fit["status"], "fits")
            self.assertEqual(fit["basis"], "estimated")
            self.assertIn("catalog estimate", fit["reason"])
        small = variant_fit(catalog.find_model("realesrgan-x4plus"), "video", CUDA_4)
        self.assertEqual(small["status"], "wont-fit")
        self.assertEqual(variant_fit(catalog.find_model("realesr-general-x4v3"), "video", MAC_16)["needGb"], 3.0)

    def test_catalog_with_fit_copies_rows(self):
        rows = catalog.catalog_with_fit(CUDA_24)
        self.assertTrue(all("fit" in row for row in rows))
        self.assertTrue(all("fit" not in row for row in catalog.VIDEO_UPSCALE_MODELS))


class ReleaseAuditTests(unittest.TestCase):
    def test_audit_reports_rows_without_a_hash(self):
        audit = _audit()
        rows = [{"id": "a", "url": "https://x/a.pth", "sha256": None},
                {"id": "b", "url": "https://x/b.pth", "sha256": "0" * 64}, {"id": "c"}]
        self.assertEqual(audit.find_unpinned_files(rows), ["a: https://x/a.pth"])

    def test_the_release_audit_fails_while_a_weight_is_unpinned(self):
        # TODO(FU): pin the two Real-ESRGAN hashes after the first download on the 4090; until then this list is
        # non-empty and ``scripts/audit-catalog-repos.py`` exits 1, which blocks a release by design.
        audit = _audit()
        flagged = audit.find_unpinned_files(catalog.VIDEO_UPSCALE_MODELS)
        self.assertEqual(len(flagged), len(catalog.unpinned_rows()))
        for line in flagged:
            self.assertTrue(re.match(r"^[a-z0-9-]+: https://github\.com/", line))


if __name__ == "__main__":
    unittest.main()
