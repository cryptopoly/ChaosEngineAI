"""llama-server ``--version`` parsing for the Setup status route (old and b12649+ formats)."""

import subprocess
import unittest
from pathlib import Path
from unittest import mock

from backend_service.routes.setup import llama_server


def _proc(stderr: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=stderr)


class VersionStringTests(unittest.TestCase):
    def test_old_format(self):
        with mock.patch.object(llama_server.subprocess, "run", return_value=_proc("version: 12441 (164ab018f)\n")):
            self.assertEqual(llama_server._version_string(Path("x")), ("12441 (164ab018f)", "12441"))

    def test_b12649_format(self):
        blob = "version: 1.0.0-dev (build 12649, commit 1f8baae28)\nbuilt with Clang 22.1.8 for Windows x86_64\n"
        with mock.patch.object(llama_server.subprocess, "run", return_value=_proc(blob)):
            self.assertEqual(llama_server._version_string(Path("x")), ("12649 (1f8baae28)", "12649"))


if __name__ == "__main__":
    unittest.main()
