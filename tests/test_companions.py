"""Companion downloads and pinned files for models that share a repo.

Iris-3B keeps four checkpoints in one 42 GB repo and reads its text encoder
from a Qwen repo; Nemotron 3.5 Bolt's drafter lives in a repo of its own.
Without these, the Iris download button pulled 42 GB and the first run then
fetched the text encoder silently.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers import image_validation
from backend_service.helpers.companions import companion_downloads
from backend_service.helpers.images import _image_repo_allow_patterns

_IRIS = "speridlabs/iris-3b"


class CompanionDownloadsTests(unittest.TestCase):
    def test_iris_brings_its_text_encoder(self) -> None:
        repos = [repo for repo, _patterns in companion_downloads(_IRIS)]
        self.assertEqual(repos, ["Qwen/Qwen3-VL-4B-Instruct"])
        patterns = companion_downloads(_IRIS)[0][1]
        self.assertIn("model-*.safetensors", patterns)

    def test_bolt_brings_nvidias_drafter(self) -> None:
        self.assertEqual(
            companion_downloads("nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB"),
            [("nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash", ["Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf"])],
        )

    def test_other_repos_bring_nothing(self) -> None:
        self.assertEqual(companion_downloads("Qwen/Qwen-Image-2.1"), [])
        self.assertEqual(companion_downloads("someone/else"), [])


class PinnedImageFilesTests(unittest.TestCase):
    def test_iris_downloads_its_turbo_folder_not_the_42_gb_repo(self) -> None:
        patterns = _image_repo_allow_patterns(_IRIS)
        self.assertEqual(patterns[0], "turbo/model.safetensors")
        self.assertNotIn("depth/*", patterns)
        self.assertFalse(any(p.startswith(("depth", "upscaler")) for p in patterns))

    def test_installed_means_every_pinned_file_is_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            with mock.patch.object(image_validation, "_hf_repo_snapshot_dir", return_value=snapshot):
                self.assertFalse(image_validation._image_repo_runtime_ready(_IRIS))
                self.assertIn("turbo/model.safetensors", image_validation._image_download_validation_error(_IRIS))
                (snapshot / "turbo").mkdir()
                (snapshot / "turbo" / "model.safetensors").write_bytes(b"0")
                self.assertTrue(image_validation._image_repo_runtime_ready(_IRIS))
                self.assertIsNone(image_validation._image_download_validation_error(_IRIS))

    def test_repos_without_pins_keep_their_old_checks(self) -> None:
        self.assertEqual(image_validation._pinned_download_files("Qwen/Qwen-Image-2.1"), [])


if __name__ == "__main__":
    unittest.main()
