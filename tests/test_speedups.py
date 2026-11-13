import os
import unittest
from types import SimpleNamespace
from unittest import mock

from backend_service.helpers import speedups


class SpeedupsTests(unittest.TestCase):
    def test_nothing_happens_unless_requested(self):
        calls = []
        vae = SimpleNamespace(enable_tiling=lambda: calls.append("tile"))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHAOSENGINE_SPEEDUPS", None)
            self.assertEqual(speedups.apply_speedups(SimpleNamespace(vae=vae)), [])
        self.assertEqual(calls, [])

    def test_vae_modes_are_switched_on_by_name(self):
        calls = []
        vae = SimpleNamespace(enable_tiling=lambda: calls.append("tile"), enable_slicing=lambda: calls.append("slice"))
        with mock.patch.dict(os.environ, {"CHAOSENGINE_SPEEDUPS": "vae_tiling, vae_slicing"}):
            notes = speedups.apply_speedups(SimpleNamespace(vae=vae))
        self.assertEqual(calls, ["tile", "slice"])
        self.assertEqual(notes, ["VAE tiling on", "VAE slicing on"])

    def test_compile_reports_a_missing_compiler_instead_of_failing_the_first_step(self):
        compiled = []
        transformer = SimpleNamespace(compile_repeated_blocks=lambda **kw: compiled.append(kw))
        with mock.patch.dict(os.environ, {"CHAOSENGINE_SPEEDUPS": "compile"}), \
                mock.patch.object(speedups, "compile_blocker", return_value="no C compiler"):
            notes = speedups.apply_speedups(SimpleNamespace(transformer=transformer))
        self.assertEqual(compiled, [])
        self.assertIn("no C compiler", notes[0])

    def test_compile_runs_when_nothing_blocks_it(self):
        compiled = []
        transformer = SimpleNamespace(compile_repeated_blocks=lambda **kw: compiled.append(kw))
        with mock.patch.dict(os.environ, {"CHAOSENGINE_SPEEDUPS": "compile"}), \
                mock.patch.object(speedups, "compile_blocker", return_value=None):
            notes = speedups.apply_speedups(SimpleNamespace(transformer=transformer))
        self.assertEqual(compiled, [{"fullgraph": True}])
        self.assertIn("torch.compile on", notes[0])


class LinuxCompileBlockerTests(unittest.TestCase):
    """torch.compile on Linux needs a C compiler and Python.h for Triton's launcher."""

    def _blocker(self, *, compiler: bool, headers: bool, cc: str = ""):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            if headers:
                (Path(tmp) / "Python.h").write_text("", encoding="utf-8")
            env = {"CC": cc, "C_INCLUDE_PATH": ""}
            with mock.patch.dict(os.environ, env, clear=False), \
                    mock.patch("sys.platform", "linux"), \
                    mock.patch("importlib.util.find_spec", return_value=object()), \
                    mock.patch("shutil.which", return_value="/usr/bin/gcc" if compiler else None), \
                    mock.patch("sysconfig.get_paths", return_value={"include": tmp if headers else str(Path(tmp) / "none")}):
                return speedups.compile_blocker()

    def test_a_linux_box_without_a_compiler_is_told_so(self):
        self.assertIn("no C compiler", self._blocker(compiler=False, headers=True))

    def test_a_linux_box_without_python_headers_is_told_so(self):
        self.assertIn("python3-dev", self._blocker(compiler=True, headers=False))

    def test_a_linux_box_with_both_is_not_blocked(self):
        self.assertIsNone(self._blocker(compiler=True, headers=True))


if __name__ == "__main__":
    unittest.main()
