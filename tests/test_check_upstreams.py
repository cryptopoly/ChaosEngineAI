"""scripts/check-upstreams.py: pins are found in the repo, rows report behind or current, and it never blocks."""

import importlib.util
import sys
import unittest
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("check_upstreams", Path(__file__).resolve().parent.parent / "scripts" / "check-upstreams.py")
cu = importlib.util.module_from_spec(_SPEC)
sys.modules["check_upstreams"] = cu  # dataclasses look the module up by name
_SPEC.loader.exec_module(cu)


class PinTests(unittest.TestCase):
    def test_every_pin_is_found_in_the_repo(self):
        self.assertRegex(cu.read_pin("scripts/update-llama-cpp.sh", r'LLAMA_CPP_REF="\$\{LLAMA_CPP_REF:-(b\d+)\}"') or "", r"^b\d+$")
        self.assertRegex(cu.read_pin("backend_service/inference/_tensorfold_native.py", r'^NATIVE_VERSION = "([\d.]+)"') or "", r"^\d+\.\d+\.\d+$")
        self.assertRegex(cu.read_pin("scripts/install-ltx2-mlx.sh", r'LTX2_MLX_REF="\$\{LTX2_MLX_REF:-([0-9a-f]{40})\}"') or "", r"^[0-9a-f]{40}$")
        self.assertTrue(cu.read_pin("scripts/install-mflux.sh", r'MFLUX_VERSION="\$\{MFLUX_VERSION:-([\d.]+)\}"'))
        self.assertTrue(cu.read_pin("backend_service/music_env.py", r'^ACE_REF = "([0-9a-f]{40})"'))
        self.assertTrue(cu.read_pin("backend_service/voice_env.py", r'^CHATTERBOX_REF = "([0-9a-f]{7,40})"'))
        self.assertTrue(cu.read_pin("pyproject.toml", r"triattention\.git@([0-9a-f]{40})"))
        self.assertTrue(cu.read_pin("pyproject.toml", r"dflash-mlx\.git@([0-9a-f]{40})"))
        self.assertTrue(cu.read_pin("e2e/package.json", r'"@playwright/test": "([\d.]+)"'))


class RowTests(unittest.TestCase):
    def test_states(self):
        self.assertEqual(cu.row("x", "1.2.6", "1.2.7").state, "behind")
        self.assertEqual(cu.row("x", "1.2.7", "1.2.7").state, "current")
        self.assertEqual(cu.row("x", "1.2.7", None).state, "unknown")
        self.assertEqual(cu.row("x", None, "1.2.7").state, "unknown")
        self.assertEqual(cu.row("x", "abc12345", "abc12345ffff", same=cu.sha_same).state, "current")

    def test_version_sort_is_numeric(self):
        self.assertEqual(sorted(["v1.2.10", "v1.2.9", "v1.10.0"], key=cu.version_key)[-1], "v1.10.0")
        self.assertEqual(sorted(["v1.2.10", "v1.2.9"], key=cu.version_key)[-1], "v1.2.10")

    def test_it_never_blocks(self):
        original = cu.collect
        cu.collect = lambda: [cu.Row("a", "1", "2", "behind"), cu.Row("b", "?", "?", "unknown")]
        try:
            self.assertEqual(cu.main([]), 0)
        finally:
            cu.collect = original


if __name__ == "__main__":
    unittest.main()
