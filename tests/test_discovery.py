"""Tests for the os.scandir-based discovery walker."""
import os
import tempfile
import unittest
from pathlib import Path

from backend_service.helpers.discovery import _du_size_gb, _path_size_bytes


class PathSizeBytesTests(unittest.TestCase):
    def test_returns_zero_for_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            self.assertEqual(_path_size_bytes(missing), 0)

    def test_returns_file_size_for_regular_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.gguf"
            target.write_bytes(b"x" * 1024)
            self.assertEqual(_path_size_bytes(target), 1024)

    def test_aggregates_nested_directory_sizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshots"
            (root / "rev-a" / "blobs").mkdir(parents=True)
            (root / "rev-b" / "blobs").mkdir(parents=True)
            (root / "rev-a" / "blobs" / "weight.bin").write_bytes(b"a" * 2048)
            (root / "rev-b" / "blobs" / "weight.bin").write_bytes(b"b" * 4096)
            (root / "config.json").write_text("{}", encoding="utf-8")
            total = _path_size_bytes(root)
            self.assertEqual(total, 2048 + 4096 + 2)

    def test_hf_blob_and_its_snapshot_link_count_once_when_scandir_has_no_inodes(self):
        # Windows: os.scandir's cached stat reports st_ino/st_dev 0 for plain files but real ids for a followed
        # symlink, so a Hugging Face blob and the snapshot link to it were both counted (RTX 4090 run: every
        # cached model showed twice its size). Simulate that scandir on any OS.
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "models--org--name"
            (repo / "blobs").mkdir(parents=True)
            (repo / "snapshots" / "rev").mkdir(parents=True)
            blob = repo / "blobs" / "abc123"
            blob.write_bytes(b"w" * 4096)
            try:
                os.symlink(str(blob), str(repo / "snapshots" / "rev" / "model.bin"))
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not allowed here")

            real_scandir = os.scandir

            class ZeroInodeEntry:
                def __init__(self, entry):
                    self._entry = entry
                    self.path, self.name = entry.path, entry.name

                def is_dir(self, follow_symlinks=True):
                    return self._entry.is_dir(follow_symlinks=follow_symlinks)

                def is_symlink(self):
                    return self._entry.is_symlink()

                def stat(self, follow_symlinks=True):
                    st = self._entry.stat(follow_symlinks=follow_symlinks)
                    if self._entry.is_symlink():
                        return st
                    return os.stat_result((st.st_mode, 0, 0, st.st_nlink, st.st_uid, st.st_gid, st.st_size,
                                           int(st.st_atime), int(st.st_mtime), int(st.st_ctime)))

            class Wrapper:
                def __init__(self, path):
                    self._it = real_scandir(path)

                def __enter__(self):
                    return (ZeroInodeEntry(e) for e in self._it)

                def __exit__(self, *exc):
                    self._it.close()

            with mock.patch("backend_service.helpers.discovery.os.scandir", Wrapper):
                self.assertEqual(_path_size_bytes(repo), 4096)

    def test_skips_symlink_loops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "loop"
            root.mkdir()
            (root / "real.bin").write_bytes(b"x" * 512)
            try:
                os.symlink(str(root), str(root / "self"))
            except (OSError, NotImplementedError):
                self.skipTest("symlink not supported on this platform")
            total = _path_size_bytes(root)
            self.assertGreaterEqual(total, 512)
            self.assertLess(total, 1024)

    def test_follows_snapshot_symlinks_to_blob_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshots" / "rev-a"
            blobs = root / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            blob = blobs / "sha256"
            blob.write_bytes(b"x" * 2048)
            try:
                os.symlink(str(blob), str(snapshot / "model.safetensors"))
            except (OSError, NotImplementedError):
                self.skipTest("symlink not supported on this platform")
            self.assertEqual(_path_size_bytes(snapshot), 2048)

    def test_dedupes_hardlinks_via_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blobs").mkdir()
            (root / "snapshots").mkdir()
            blob = root / "blobs" / "sha256"
            blob.write_bytes(b"x" * 1024)
            try:
                os.link(str(blob), str(root / "snapshots" / "model.bin"))
            except (OSError, NotImplementedError):
                self.skipTest("hardlink not supported on this platform")
            total = _path_size_bytes(root)
            self.assertEqual(total, 1024)


class DuSizeGbTests(unittest.TestCase):
    def test_file_size_in_gb(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "model.gguf"
            target.write_bytes(b"x" * (1024 * 1024 * 100))
            size = _du_size_gb(target)
            self.assertGreater(size, 0.09)
            self.assertLess(size, 0.11)

    def test_directory_size_in_gb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model-dir"
            root.mkdir()
            for idx in range(5):
                (root / f"shard-{idx}.bin").write_bytes(b"x" * (1024 * 1024 * 20))
            size = _du_size_gb(root)
            self.assertGreater(size, 0.09)
            self.assertLess(size, 0.11)


if __name__ == "__main__":
    unittest.main()
