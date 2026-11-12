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


class AllocatorEnvTests(unittest.TestCase):
    """mimalloc (torch on Windows) must give back memory freed into segments of threads that have exited."""

    def test_windows_sets_purge_and_abandoned_segment_options(self):
        env: dict[str, str] = {}
        applied = malloc_release.apply_allocator_env(env, platform="win32")
        self.assertEqual(env["MIMALLOC_ABANDONED_RECLAIM_ON_FREE"], "1")
        self.assertEqual(env["MIMALLOC_PURGE_DELAY"], "0")
        self.assertEqual(env["MIMALLOC_RESET_DELAY"], "0")
        self.assertEqual(applied, env)

    def test_a_user_value_wins(self):
        env = {"MIMALLOC_ABANDONED_RECLAIM_ON_FREE": "0"}
        applied = malloc_release.apply_allocator_env(env, platform="win32")
        self.assertEqual(env["MIMALLOC_ABANDONED_RECLAIM_ON_FREE"], "0")
        self.assertNotIn("MIMALLOC_ABANDONED_RECLAIM_ON_FREE", applied)

    def test_other_platforms_are_left_alone(self):
        for platform in ("linux", "darwin"):
            env: dict[str, str] = {}
            self.assertEqual(malloc_release.apply_allocator_env(env, platform=platform), {})
            self.assertEqual(env, {})

    def test_the_package_applies_it_before_torch_loads(self):
        import os
        import subprocess

        env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
        code = ("import os, sys, backend_service; print('torch' in sys.modules, "
                "os.environ.get('MIMALLOC_ABANDONED_RECLAIM_ON_FREE'), os.environ.get('MIMALLOC_PURGE_DELAY'))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True).stdout.split()
        self.assertEqual(out[0], "False")
        self.assertEqual(out[1:], ["1", "0"] if sys.platform == "win32" else ["None", "None"])


# Real measurement (Windows with torch only): tensors made in a thread that has exited and freed from another thread,
# as Unload frees weights that loader threads allocated. The process must drop back near its baseline.
_ABANDONED_PROBE = """
import gc, sys, threading, time
import backend_service  # applies the allocator settings before torch loads
import psutil, torch
proc = psutil.Process()
keep = []
def load():
    for _ in range(40):
        keep.append(torch.ones(24 * 2 ** 20, dtype=torch.uint8))
base = proc.memory_info().private
t = threading.Thread(target=load); t.start(); t.join()
loaded = proc.memory_info().private
keep.clear(); gc.collect(); time.sleep(1)
print(base, loaded, proc.memory_info().private)
"""


@unittest.skipUnless(sys.platform == "win32", "torch links mimalloc on Windows only")
class AbandonedSegmentMeasurementTests(unittest.TestCase):
    def test_memory_freed_from_another_thread_goes_back_to_the_os(self):
        import importlib.util
        import os
        import subprocess

        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("psutil") is None:
            self.skipTest("torch or psutil not installed")
        env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
        result = subprocess.run([sys.executable, "-c", _ABANDONED_PROBE], capture_output=True, text=True, env=env,
                                timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        base, loaded, after = (int(v) for v in result.stdout.split()[-3:])
        grown = loaded - base
        self.assertGreater(grown, 512 * 2 ** 20)  # the probe really allocated about 1 GB
        self.assertLess(after - base, 0.2 * grown, f"kept {(after - base) / 2 ** 30:.2f} GB of {grown / 2 ** 30:.2f} GB")


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
