"""Freed system memory goes back to the OS after Unload on Linux."""
import sys
import unittest
from unittest import mock

from backend_service.helpers import malloc_release


class ReleaseFreedMemoryTests(unittest.TestCase):
    def test_off_linux_it_does_nothing(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.assertFalse(malloc_release.release_freed_memory())

    def test_a_libc_without_malloc_trim_is_not_an_error(self):
        with mock.patch.object(sys, "platform", "linux"), mock.patch("ctypes.CDLL", side_effect=OSError("no libc")):
            self.assertFalse(malloc_release.release_freed_memory())

    def test_it_calls_malloc_trim_with_zero(self):
        libc = mock.Mock()
        libc.malloc_trim.return_value = 1
        with mock.patch.object(sys, "platform", "linux"), mock.patch("ctypes.CDLL", return_value=libc):
            self.assertTrue(malloc_release.release_freed_memory())
        libc.malloc_trim.assert_called_once_with(0)


class PipelineLoadKwargsTests(unittest.TestCase):
    """Windows loads diffusers pipelines without the memory map (Wan 2.2 TI2V-5B: 46.8 GB private -> 28.5 GB)."""

    def test_windows_disables_the_memory_map(self):
        with mock.patch.object(sys, "platform", "win32"):
            self.assertEqual(malloc_release.pipeline_load_kwargs(), {"disable_mmap": True})

    def test_other_platforms_keep_the_default(self):
        for platform in ("linux", "darwin"):
            with mock.patch.object(sys, "platform", platform):
                self.assertEqual(malloc_release.pipeline_load_kwargs(), {})

    def test_image_and_video_runtimes_pass_it_to_the_pipeline_load(self):
        from pathlib import Path
        import re

        root = Path(__file__).resolve().parents[1] / "backend_service"
        for name in ("image_runtime", "video_runtime"):
            source = (root / name / "__init__.py").read_text(encoding="utf-8")
            call = re.search(r"\.from_pretrained\(\s*local_path,[^)]*\)", source)
            self.assertIsNotNone(call, name)
            self.assertIn("**pipeline_load_kwargs()", call.group(0), name)


if __name__ == "__main__":
    unittest.main()
