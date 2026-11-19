"""A package without ``__version__`` still reports the version its metadata names (Nunchaku 1.2.x)."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from backend_service.inference import accelerators


class SafeVersionTests(unittest.TestCase):
    def test_metadata_fills_in_when_the_module_has_no_version(self) -> None:
        fake = types.ModuleType("fake_accel")
        with mock.patch.dict(sys.modules, {"fake_accel": fake}), \
             mock.patch.object(accelerators, "_spec_exists", return_value=True), \
             mock.patch("importlib.metadata.version", return_value="1.2.1+cu12.8torch2.11"):
            self.assertEqual(accelerators._safe_version("fake_accel"), "1.2.1+cu12.8torch2.11")

    def test_the_module_version_wins_and_a_missing_package_stays_none(self) -> None:
        fake = types.ModuleType("fake_accel")
        fake.__version__ = "9.9"
        with mock.patch.dict(sys.modules, {"fake_accel": fake}), mock.patch.object(accelerators, "_spec_exists", return_value=True):
            self.assertEqual(accelerators._safe_version("fake_accel"), "9.9")
        bare = types.ModuleType("fake_accel")
        from importlib import metadata

        with mock.patch.dict(sys.modules, {"fake_accel": bare}), \
             mock.patch.object(accelerators, "_spec_exists", return_value=True), \
             mock.patch("importlib.metadata.version", side_effect=metadata.PackageNotFoundError("x")):
            self.assertIsNone(accelerators._safe_version("fake_accel"))


if __name__ == "__main__":
    unittest.main()
