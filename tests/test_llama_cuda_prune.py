"""Pruning older CUDA llama-server tag folders after an install of the pinned tag."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers import llama_cuda


def _make(root: Path, name: str, tag: str | None) -> Path:
    folder = root / name
    folder.mkdir()
    (folder / "llama-server.exe").write_bytes(b"x")
    (folder / "ggml.dll").write_bytes(b"y")
    if tag is not None:
        (folder / "installed.json").write_text(json.dumps({"tag": tag}), encoding="utf-8")
    return folder


class PruneOldInstallsTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        patch = mock.patch.object(llama_cuda, "_folder_in_use", return_value=False)
        patch.start()
        self.addCleanup(patch.stop)

    def test_removes_old_tags_and_keeps_the_current_one(self) -> None:
        _make(self.root, "b100", "b100")
        _make(self.root, "b200", "b200")
        _make(self.root, "b300", "b300")
        removed = llama_cuda.prune_old_installs("b300", root=self.root)
        self.assertEqual(sorted(removed), ["b100", "b200"])
        self.assertEqual([p.name for p in self.root.iterdir()], ["b300"])

    def test_keeps_a_folder_without_a_marker_and_other_files(self) -> None:
        stranger = _make(self.root, "my-own-build", None)
        staging = _make(self.root, ".staging-abc", "b100")
        note = self.root / "readme.txt"
        note.write_text("keep", encoding="utf-8")
        _make(self.root, "b100", "b100")
        llama_cuda.prune_old_installs("b300", root=self.root)
        self.assertTrue(stranger.is_dir())
        self.assertTrue(staging.is_dir())
        self.assertTrue(note.is_file())
        self.assertFalse((self.root / "b100").exists())

    def test_keeps_a_corrupt_marker(self) -> None:
        folder = _make(self.root, "b100", None)
        (folder / "installed.json").write_text("{not json", encoding="utf-8")
        logs: list[str] = []
        self.assertEqual(llama_cuda.prune_old_installs("b300", logs.append, root=self.root), [])
        self.assertTrue(folder.is_dir())

    def test_a_running_server_keeps_its_folder(self) -> None:
        _make(self.root, "b100", "b100")
        _make(self.root, "b200", "b200")
        with mock.patch.object(llama_cuda, "_folder_in_use", side_effect=lambda f: f.name == "b100"):
            removed = llama_cuda.prune_old_installs("b300", root=self.root)
        self.assertEqual(removed, ["b200"])
        self.assertTrue((self.root / "b100").is_dir())

    def test_a_locked_file_is_logged_and_skipped(self) -> None:
        _make(self.root, "b100", "b100")
        _make(self.root, "b200", "b200")
        real = llama_cuda.shutil.rmtree

        def fake(path, *args, **kwargs):
            if Path(path).name == "b100":
                raise PermissionError("locked by another process")
            return real(path, *args, **kwargs)

        logs: list[str] = []
        with mock.patch.object(llama_cuda.shutil, "rmtree", side_effect=fake):
            removed = llama_cuda.prune_old_installs("b300", logs.append, root=self.root)
        self.assertEqual(removed, ["b200"])
        self.assertTrue((self.root / "b100").is_dir())
        self.assertTrue(any("b100" in line and "left in place" in line for line in logs))

    def test_missing_root_is_fine(self) -> None:
        self.assertEqual(llama_cuda.prune_old_installs("b300", root=self.root / "nope"), [])


if __name__ == "__main__":
    unittest.main()
