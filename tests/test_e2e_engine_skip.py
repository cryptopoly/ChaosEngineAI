"""The unusable-settings E2E check is the MLX lane's: it skips when the load landed on another engine."""

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_suite():
    spec = importlib.util.spec_from_file_location("e2e_suite_engine_skip", SCRIPTS / "e2e_test_suite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e2e_suite_engine_skip"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class NonMlxEngineSkipTests(unittest.TestCase):
    def test_mlx_load_runs_the_check(self):
        self.assertIsNone(_load_suite()._non_mlx_engine_skip({"state": "loaded", "engine": "mlx"}))

    def test_other_engines_skip_with_a_reason(self):
        suite = _load_suite()
        for engine in ("tensorfold-native", "tensorfold", "llama.cpp", None):
            reason = suite._non_mlx_engine_skip({"state": "loaded", "engine": engine})
            self.assertIn("not MLX", reason or "")

    def test_non_dict_response_skips(self):
        self.assertIsNotNone(_load_suite()._non_mlx_engine_skip(None))


if __name__ == "__main__":
    unittest.main()
