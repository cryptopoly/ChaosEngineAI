"""Image / video catalog listing: no false "installed", no duplicates, rows in their own family."""
from __future__ import annotations

import collections
import unittest
from unittest import mock

from backend_service.catalog.image_models import IMAGE_MODEL_FAMILIES
from backend_service.catalog.video_models import VIDEO_MODEL_FAMILIES
from backend_service.helpers import image_validation as iv


class LibraryNameMatchTests(unittest.TestCase):
    def test_a_longer_repo_name_does_not_mark_a_shorter_one_installed(self):
        # Qwen/Qwen-Image-2.1 on disk made Qwen/Qwen-Image count as installed.
        library = [{"name": "Qwen/Qwen-Image-2.1", "path": "/x/models--Qwen--Qwen-Image-2.1"}]
        variant = {"id": "Qwen/Qwen-Image", "name": "Qwen-Image", "repo": "Qwen/Qwen-Image"}
        with mock.patch.object(iv, "_image_repo_runtime_ready", return_value=False), \
                mock.patch.object(iv, "validate_local_diffusers_snapshot", return_value=None):
            self.assertFalse(iv._image_variant_available_locally(variant, library))

    def test_the_same_repo_still_matches(self):
        library = [{"name": "Qwen/Qwen-Image", "path": "/x/models--Qwen--Qwen-Image"}]
        variant = {"id": "Qwen/Qwen-Image", "name": "Qwen-Image", "repo": "Qwen/Qwen-Image"}
        with mock.patch.object(iv, "_image_repo_runtime_ready", return_value=False), \
                mock.patch.object(iv, "_candidate_model_dirs", return_value=["/x"]), \
                mock.patch.object(iv, "validate_local_diffusers_snapshot", return_value=None):
            self.assertTrue(iv._image_variant_available_locally(variant, library))


class CatalogShapeTests(unittest.TestCase):
    def _check(self, families):
        ids = collections.Counter(v["id"] for f in families for v in f["variants"])
        names = collections.Counter(v["name"] for f in families for v in f["variants"])
        self.assertEqual([k for k, c in ids.items() if c > 1], [])
        self.assertEqual([k for k, c in names.items() if c > 1], [])
        for family in families:
            for variant in family["variants"]:
                self.assertEqual(variant.get("familyId", family["id"]), family["id"], variant["id"])
                self.assertNotIn("(planned)", variant.get("runtime", ""), variant["id"])

    def test_image_catalog(self):
        self._check(IMAGE_MODEL_FAMILIES)

    def test_video_catalog(self):
        self._check(VIDEO_MODEL_FAMILIES)


if __name__ == "__main__":
    unittest.main()
