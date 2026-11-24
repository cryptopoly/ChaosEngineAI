"""The llama-server install hint names a way that exists on the machine (Homebrew only on a Mac)."""

import unittest
from unittest import mock

from backend_service.inference import binaries


class LlamaInstallHintTests(unittest.TestCase):
    def test_mac_gets_homebrew(self):
        with mock.patch.object(binaries.sys, "platform", "darwin"):
            self.assertEqual(binaries.llama_install_hint(), "brew install llama.cpp")

    def test_windows_and_linux_point_to_setup(self):
        for platform in ("win32", "linux"):
            with self.subTest(platform=platform), mock.patch.object(binaries.sys, "platform", platform):
                hint = binaries.llama_install_hint()
                self.assertNotIn("brew", hint)
                self.assertIn("Setup", hint)


if __name__ == "__main__":
    unittest.main()
