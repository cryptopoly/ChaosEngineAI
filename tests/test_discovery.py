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
