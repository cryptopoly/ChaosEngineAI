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


class GatedRowCheckTests(unittest.TestCase):
    def test_token_present_skips_even_without_a_reason(self):
        verdict = _load_suite()._gated_row_blocked_check({"accessBlockedReason": None, "hasLocalData": False}, token_present=True)
        self.assertEqual(verdict[0], "skip")
        self.assertIn("token", verdict[1])

    def test_no_token_and_no_reason_fails(self):
        verdict = _load_suite()._gated_row_blocked_check({"accessBlockedReason": None, "hasLocalData": False}, token_present=False)
        self.assertEqual(verdict[0], "fail")

    def test_no_token_with_a_reason_skips(self):
        verdict = _load_suite()._gated_row_blocked_check({"accessBlockedReason": "gated, add a token"}, token_present=False)
        self.assertEqual(verdict[0], "skip")
        self.assertIn("gated", verdict[1])

    def test_token_probe_reads_env_then_settings_flag_only(self):
        from unittest import mock

        suite = _load_suite()
        with mock.patch.dict("os.environ", {"HF_TOKEN": "x"}):
            self.assertTrue(suite._hf_token_present())
        body = b'{"hasHuggingFaceToken": true, "huggingFaceToken": "masked"}'
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(suite, "_voice_http", return_value=(200, body)):
            self.assertTrue(suite._hf_token_present())
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(suite, "_voice_http", return_value=(200, b'{"hasHuggingFaceToken": false}')):
            self.assertFalse(suite._hf_token_present())
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(suite, "_voice_http", side_effect=OSError):
            self.assertFalse(suite._hf_token_present())


if __name__ == "__main__":
    unittest.main()
