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


class ModelThreadTests(unittest.TestCase):
    """Load, generate and unload share one thread on Windows, so freed weights go back to the OS."""

    def _thread(self):
        return malloc_release.ModelThread("test-model")

    def test_calls_run_on_one_thread_that_is_not_the_callers(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor

        model = self._thread()
        with mock.patch.object(malloc_release.ModelThread, "enabled", return_value=True):
            with ThreadPoolExecutor(max_workers=3) as callers:
                idents = list(callers.map(lambda _: model.run(threading.get_ident), range(6)))
        self.assertEqual(len(set(idents)), 1)
        self.assertNotEqual(idents[0], threading.get_ident())

    def test_a_nested_call_runs_inline(self):
        import threading

        model = self._thread()
        with mock.patch.object(malloc_release.ModelThread, "enabled", return_value=True):
            outer, inner = model.run(lambda: (threading.get_ident(), model.run(threading.get_ident)))
        self.assertEqual(outer, inner)

    def test_results_arguments_and_exceptions_pass_through(self):
        model = self._thread()
        with mock.patch.object(malloc_release.ModelThread, "enabled", return_value=True):
            self.assertEqual(model.run(lambda a, b=0: a + b, 2, b=3), 5)
            with self.assertRaises(KeyError):
                model.run(lambda: {}["missing"])

    def test_context_variables_travel_with_the_call(self):
        import contextvars

        var: contextvars.ContextVar[str] = contextvars.ContextVar("owner", default="app")
        model = self._thread()
        with mock.patch.object(malloc_release.ModelThread, "enabled", return_value=True):
            token = var.set("agent")
            try:
                self.assertEqual(model.run(var.get), "agent")
            finally:
                var.reset(token)

    def test_off_windows_or_switched_off_it_calls_directly(self):
        import threading

        self.assertFalse(malloc_release.ModelThread.enabled(platform="linux", environ={}))
        self.assertFalse(malloc_release.ModelThread.enabled(platform="darwin", environ={}))
        self.assertFalse(malloc_release.ModelThread.enabled(platform="win32", environ={"CHAOSENGINE_MODEL_THREAD": "0"}))
        self.assertTrue(malloc_release.ModelThread.enabled(platform="win32", environ={}))
        with mock.patch.object(malloc_release.ModelThread, "enabled", return_value=False):
            self.assertEqual(self._thread().run(threading.get_ident), threading.get_ident())

    def test_image_and_video_managers_route_the_pipeline_calls_through_it(self):
        from pathlib import Path
        import re

        root = Path(__file__).resolve().parents[1] / "backend_service"
        image = (root / "image_runtime" / "__init__.py").read_text(encoding="utf-8")
        for call in ("_MODEL_THREAD.run(self._diffusers.generate", "_MODEL_THREAD.run(self._diffusers.unload",
                     "_MODEL_THREAD.run(self._diffusers.preload"):
            self.assertIn(call, image)
        self.assertIsNone(re.search(r"self\._diffusers\.(generate|unload|preload)\(", image))
        video = (root / "video_runtime" / "__init__.py").read_text(encoding="utf-8")
        for call in ("_MODEL_THREAD.run(self._engine.generate", "_MODEL_THREAD.run(self._engine.unload",
                     "_MODEL_THREAD.run(self._engine.preload"):
            self.assertIn(call, video)


# Real measurement (Windows with torch only): weights allocated on one live worker thread and freed on another, as a
# request worker loads a pipeline and another serves Unload. Without the model thread the memory stays committed.
_LIVE_THREAD_PROBE = """
import gc, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import backend_service
import psutil, torch
from backend_service.helpers.malloc_release import ModelThread
proc = psutil.Process()
model = ModelThread("probe")
keep = []
def load():
    for _ in range(40):
        keep.append(torch.ones(24 * 2 ** 20, dtype=torch.uint8))
def clear():
    keep.clear(); gc.collect()
loader, unloader = ThreadPoolExecutor(max_workers=1), ThreadPoolExecutor(max_workers=1)
base = proc.memory_info().private
loader.submit(model.run, load).result()
loaded = proc.memory_info().private
unloader.submit(model.run, clear).result()
time.sleep(1)
print(base, loaded, proc.memory_info().private)
"""


@unittest.skipUnless(sys.platform == "win32", "torch links mimalloc on Windows only")
class LiveThreadMeasurementTests(unittest.TestCase):
    def _probe(self, model_thread: str):
        import importlib.util
        import os
        import subprocess

        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("psutil") is None:
            self.skipTest("torch or psutil not installed")
        env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
        env["CHAOSENGINE_MODEL_THREAD"] = model_thread
        result = subprocess.run([sys.executable, "-c", _LIVE_THREAD_PROBE], capture_output=True, text=True, env=env,
                                timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        base, loaded, after = (int(v) for v in result.stdout.split()[-3:])
        grown = loaded - base
        self.assertGreater(grown, 512 * 2 ** 20)  # the probe really allocated about 1 GB
        return grown, after - base

    def test_memory_comes_back_when_load_and_unload_share_the_model_thread(self):
        grown, kept = self._probe("1")
        self.assertLess(kept, 0.2 * grown, f"kept {kept / 2 ** 30:.2f} GB of {grown / 2 ** 30:.2f} GB")


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
