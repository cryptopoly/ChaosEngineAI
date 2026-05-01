"""Pin the persistent GPU extras path on Windows / Linux / macOS.

The desktop installer is configured to leave this directory alone on
uninstall (``src-tauri/installer.nsh`` on Windows; macOS uninstall is
``rm /Applications/ChaosEngineAI.app`` which doesn't touch
``~/Library/Application Support``). If the path computed by
``_extras_site_packages`` ever drifts from what the installer hooks
expect, the uninstall safety net breaks.

The tests below pin both halves of the contract — the parent directory
and the ABI tag layout — so any future move is loud.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from backend_service.routes import setup as setup_routes


class ExtrasSitePackagesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patcher = mock.patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        # Drop the override knob — the explicit env path is for tests
        # that pin a custom location, not for the cross-OS shape check.
        os.environ.pop("CHAOSENGINE_EXTRAS_SITE_PACKAGES", None)

    def tearDown(self) -> None:
        self._env_patcher.stop()

    def test_path_includes_chaosengine_extras_namespace(self) -> None:
        path = setup_routes._extras_site_packages()
        self.assertIsNotNone(path)
        assert path is not None  # type narrow
        parts = path.parts
        # 'ChaosEngineAI/extras/cp{maj}{min}/site-packages' suffix.
        # The tree above (LOCALAPPDATA / Library/Application Support /
        # XDG_DATA_HOME) is platform-specific; we only assert the tail.
        self.assertEqual(parts[-4], "ChaosEngineAI")
        self.assertEqual(parts[-3], "extras")
        self.assertTrue(parts[-2].startswith("cp"))
        self.assertEqual(parts[-1], "site-packages")

    def test_python_abi_tag_matches_runtime(self) -> None:
        path = setup_routes._extras_site_packages()
        assert path is not None
        expected_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        self.assertEqual(path.parts[-2], expected_tag)

    def test_env_override_wins(self) -> None:
        override = "/tmp/chaosengine-extras-override"
        os.environ["CHAOSENGINE_EXTRAS_SITE_PACKAGES"] = override
        path = setup_routes._extras_site_packages()
        self.assertEqual(path, Path(override))


if __name__ == "__main__":
    unittest.main()
