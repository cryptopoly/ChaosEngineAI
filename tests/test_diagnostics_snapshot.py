import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from backend_service.routes import diagnostics as diag


class RunSectionsTests(unittest.TestCase):
    def test_slow_section_is_bounded_and_noted(self):
        release = threading.Event()

        def slow():
            release.wait(10)
            return {"ok": True}

        start = time.time()
        out = diag._run_sections({"fast": lambda: {"ok": 1}, "slow": slow, "bad": lambda: 1 / 0}, 0.5)
        elapsed = time.time() - start
        release.set()
        self.assertLess(elapsed, 3)
        self.assertEqual(out["fast"], {"ok": 1})
        self.assertIn("timed out", out["slow"]["error"])
        self.assertIn("ZeroDivisionError", out["bad"]["error"])

    def test_sections_run_in_parallel(self):
        start = time.time()
        diag._run_sections({f"s{i}": (lambda: time.sleep(0.4)) for i in range(5)}, 5)
        self.assertLess(time.time() - start, 1.5)


class DirSizeCacheTests(unittest.TestCase):
    def setUp(self):
        diag._dir_size_state.clear()

    def test_slow_walk_returns_note_then_cached_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.bin").write_bytes(b"x" * 100)
            gate = threading.Event()
            real = diag._dir_size

            def slow(path):
                gate.wait(5)
                return real(path)

            with mock.patch.object(diag, "_dir_size", slow), mock.patch.object(diag, "_DIR_SIZE_WAIT_SECONDS", 0.2):
                start = time.time()
                size, note = diag._cached_dir_size(Path(tmp))
                self.assertLess(time.time() - start, 2)
                self.assertIsNone(size)
                self.assertIn("background", note)
                gate.set()
                diag._dir_size_state[str(Path(tmp))]["event"].wait(3)
                size, note = diag._cached_dir_size(Path(tmp))
                self.assertEqual((size, note), (100, None))

    def test_dir_size_counts_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "d").mkdir()
            (Path(tmp) / "d" / "a").write_bytes(b"12345")
            (Path(tmp) / "b").write_bytes(b"123")
            self.assertEqual(diag._dir_size(Path(tmp)), 8)


if __name__ == "__main__":
    unittest.main()
