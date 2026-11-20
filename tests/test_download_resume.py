"""FU-275: a paused Hub download resumes from its partial file, and the progress meter ignores dead partials."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from backend_service.helpers.hf_cache_paths import _incomplete_sizes, prune_stale_partials
from backend_service.hf_resume_script import HF_RESUME_PATCH

MB = 1024 * 1024


def _install_patch(payload: bytes):
    """Run the subprocess patch text against a stub ``file_download`` module serving ``payload``."""
    calls: dict[str, object] = {"orig": 0}
    fd = types.ModuleType("huggingface_hub.file_download")

    def orig(*args, **kwargs):
        calls["orig"] += 1

    def http_get(url, f, *, headers=None, expected_size=None, resume_size=0, tqdm_class=None):
        calls["resume_size"] = resume_size
        f.write(payload[resume_size:])

    def chmod_and_move(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes())
        Path(src).unlink()

    fd._download_to_tmp_and_move = orig
    fd.http_get = http_get
    fd._chmod_and_move = chmod_and_move
    hub = types.ModuleType("huggingface_hub")
    hub.file_download = fd
    with mock.patch.dict(sys.modules, {"huggingface_hub": hub, "huggingface_hub.file_download": fd}):
        exec(HF_RESUME_PATCH, {})
    return fd, calls


class ResumePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.blobs = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.payload = bytes(range(256)) * (50 * MB // 256)
        self.sha = hashlib.sha256(self.payload).hexdigest()

    def _run(self, fd, size: int | None = None):
        dest = self.blobs / self.sha
        fd._download_to_tmp_and_move(
            self.blobs / "etag1.incomplete", dest, "https://example.invalid/f", {}, len(self.payload),
            "f.bin", False, "etag1", None, None,
        )
        return dest

    def test_resumes_from_the_largest_partial_minus_the_safety_margin(self) -> None:
        fd, calls = _install_patch(self.payload)
        (self.blobs / "etag1.aaaa1111.incomplete").write_bytes(self.payload[: 40 * MB])
        (self.blobs / "etag1.bbbb2222.incomplete").write_bytes(self.payload[: 10 * MB])
        dest = self._run(fd)
        self.assertEqual(calls["orig"], 0)
        self.assertEqual(calls["resume_size"], 40 * MB - 32 * MB)
        self.assertEqual(dest.read_bytes(), self.payload)
        self.assertEqual(list(self.blobs.glob("*.incomplete")), [])

    def test_fresh_download_uses_the_original_function(self) -> None:
        fd, calls = _install_patch(self.payload)
        self._run(fd)
        self.assertEqual(calls["orig"], 1)

    def test_small_partial_is_dropped_and_restarted(self) -> None:
        fd, calls = _install_patch(self.payload)
        (self.blobs / "etag1.aaaa1111.incomplete").write_bytes(b"x" * MB)
        self._run(fd)
        self.assertEqual(calls["orig"], 1)
        self.assertEqual(list(self.blobs.glob("*.incomplete")), [])

    def test_corrupt_prefix_fails_the_checksum_and_is_removed(self) -> None:
        fd, calls = _install_patch(self.payload)
        (self.blobs / "etag1.aaaa1111.incomplete").write_bytes(b"\xff" * (40 * MB))
        with self.assertRaises(OSError):
            self._run(fd)
        self.assertEqual(list(self.blobs.glob("*.incomplete")), [])
        self.assertFalse((self.blobs / self.sha).exists())


class PartialAccountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = Path(self._tmp.name)
        (self.cache / "blobs").mkdir()

    def _partial(self, name: str, size: int) -> Path:
        path = self.cache / "blobs" / name
        path.write_bytes(b"0" * size)
        return path

    def test_meter_ignores_partials_of_finished_blobs(self) -> None:
        self._partial("done.1.incomplete", 500)
        (self.cache / "blobs" / "done").write_bytes(b"0" * 10)
        self._partial("open.1.incomplete", 300)
        self._partial("open.2.incomplete", 200)
        every, live = _incomplete_sizes(self.cache)
        self.assertEqual(live, 300)
        self.assertEqual(every, 500)

    def test_prune_keeps_only_the_largest_partial_of_an_unfinished_blob(self) -> None:
        self._partial("done.1.incomplete", 500)
        (self.cache / "blobs" / "done").write_bytes(b"0")
        big = self._partial("open.1.incomplete", 300)
        self._partial("open.2.incomplete", 200)
        freed = prune_stale_partials(self.cache)
        self.assertEqual(freed, 700)
        self.assertEqual([p.name for p in (self.cache / "blobs").glob("*.incomplete")], [big.name])


if __name__ == "__main__":
    unittest.main()
