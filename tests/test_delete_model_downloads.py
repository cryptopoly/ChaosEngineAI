"""Deleting a model forgets its finished download record (it would keep the row 'installed')."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend_service.state.lifecycle import delete_model_path


def _state(root: Path, downloads: dict) -> SimpleNamespace:
    return SimpleNamespace(
        _lock=threading.RLock(),
        settings={"modelDirectories": [{"path": str(root), "enabled": True}]},
        runtime=SimpleNamespace(loaded_model=None),
        add_log=lambda *args, **kwargs: None,
        _library=lambda force=False: [],
        _downloads=downloads,
    )


class DeleteModelDownloadRecordTests(unittest.TestCase):
    def test_completed_record_of_a_deleted_hf_cache_folder_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "models--org--model-a"
            folder.mkdir()
            (folder / "weights.bin").write_bytes(b"x")
            downloads = {
                "org/model-a": {"state": "completed"},
                "org/model-b": {"state": "completed"},
                "org/model-c": {"state": "downloading"},
            }
            delete_model_path(_state(root, downloads), str(folder))
            self.assertNotIn("org/model-a", downloads)
            self.assertIn("org/model-b", downloads)
            self.assertIn("org/model-c", downloads)

    def test_an_active_download_of_the_same_repo_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "models--org--model-a"
            folder.mkdir()
            downloads = {"org/model-a": {"state": "downloading"}}
            delete_model_path(_state(root, downloads), str(folder))
            self.assertIn("org/model-a", downloads)


if __name__ == "__main__":
    unittest.main()
