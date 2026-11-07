"""Music downloads report bytes to the shared tracker, companion repo included (Stable Audio's T5 encoder)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.state import ChaosEngineState

GB = 1024 ** 3
SOUND = "stabilityai/stable-audio-open-small"
TURBO = "ACE-Step/Ace-Step1.5"


class MusicDownloadProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.state = ChaosEngineState(
            settings_path=tmp / "settings.json",
            benchmarks_path=tmp / "benchmarks.json",
            chat_sessions_path=tmp / "chats.json",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _start(self, repo: str, files_by_repo: dict[str, list[dict]], cached_bytes) -> dict:
        class NoThread:
            def __init__(self, *, target=None, daemon=None):
                pass

            def start(self):
                pass

        with mock.patch("backend_service.helpers.huggingface._hf_repo_downloaded_bytes", side_effect=cached_bytes), \
             mock.patch("backend_service.helpers.huggingface._hf_repo_preflight_size_gb", return_value=6.0), \
             mock.patch("backend_service.helpers.huggingface._hub_repo_files", side_effect=lambda r: {"files": files_by_repo.get(r, [])}), \
             mock.patch("backend_service.state.downloads.threading.Thread", side_effect=NoThread):
            return dict(self.state.start_download(repo))

    def test_total_includes_the_companion_and_progress_counts_its_bytes(self) -> None:
        files = {
            SOUND: [
                {"path": "model.safetensors", "sizeBytes": int(1.7 * GB)},
                {"path": "unused.ckpt", "sizeBytes": 2 * GB},
            ],
            "google-t5/t5-base": [
                {"path": "model.safetensors", "sizeBytes": int(0.9 * GB)},
                {"path": "flax_model.msgpack", "sizeBytes": 5 * GB},
            ],
        }
        row = self._start(SOUND, files, lambda repo, patterns=None: GB if repo == SOUND else GB // 2)
        self.assertAlmostEqual(row["totalGb"], 2.6, places=1)
        self.assertAlmostEqual(row["downloadedGb"], 1.5, places=1)
        self.assertAlmostEqual(row["progress"], 1.5 / 2.6, places=1)

    def test_a_row_without_companions_sizes_as_before(self) -> None:
        row = self._start(TURBO, {}, lambda repo, patterns=None: 0)
        self.assertEqual(row["totalGb"], 6.0)


if __name__ == "__main__":
    unittest.main()
