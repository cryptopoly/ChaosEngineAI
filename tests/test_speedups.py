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


if __name__ == "__main__":
    unittest.main()
