"""The E2E checks that take "the smallest GGUF" skip a repo that holds only a drafter (FU-611, Windows gate 2026-11-23)."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_suite():
    spec = importlib.util.spec_from_file_location("e2e_suite_gguf_pick", SCRIPTS / "e2e_test_suite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e2e_suite_gguf_pick"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class HoldsModelGgufTests(unittest.TestCase):
    def setUp(self):
        self.suite = _load_suite()
        self.tmp = Path(tempfile.mkdtemp())

    def test_a_drafter_only_repo_is_not_a_model(self):
        snap = self.tmp / "models--unsloth--gemma-4-12B-it-GGUF" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "mtp-gemma-4-12b-it.gguf").write_bytes(b"GGUF")
        self.assertFalse(self.suite._holds_model_gguf(snap.parents[1]))
        self.assertFalse(self.suite._holds_model_gguf(snap / "mtp-gemma-4-12b-it.gguf"))

    def test_a_repo_with_the_model_beside_its_drafter_is_a_model(self):
        snap = self.tmp / "models--org--Qwen3-GGUF" / "snapshots" / "abc"
        snap.mkdir(parents=True)
        (snap / "mtp-qwen3.gguf").write_bytes(b"GGUF")
        (snap / "mmproj-qwen3.gguf").write_bytes(b"GGUF")
        (snap / "Qwen3-0.6B-Q8_0.gguf").write_bytes(b"GGUF")
        self.assertTrue(self.suite._holds_model_gguf(snap.parents[1]))
        self.assertTrue(self.suite._holds_model_gguf(snap / "Qwen3-0.6B-Q8_0.gguf"))


if __name__ == "__main__":
    unittest.main()
