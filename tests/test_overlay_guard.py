"""Overlay copies of the bundled MLX core that no longer match are moved aside."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backend_service import runtime_paths


def _install(site: Path, name: str, version: str, files: list[str]) -> None:
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    record = []
    for relative in files:
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(version, encoding="utf-8")
        record.append(f"{relative},,")
    record.append(f"{info.name}/METADATA,,")
    (info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")


class QuarantineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.bundled = base / "bundle" / "site-packages"
        self.extras = base / "extras" / "cp312" / "site-packages"
        _install(self.bundled, "mlx", "0.32.4", ["mlx/__init__.py", "mlx/core.so"])
        _install(self.bundled, "mlx-metal", "0.32.4", ["mlx/lib/libmlx.dylib"])
        _install(self.bundled, "mlx-lm", "0.32.2", ["mlx_lm/__init__.py"])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_a_partial_old_overlay_is_moved_aside(self) -> None:
        # mlx-whisper's install brought mlx 0.31 without mlx-lm.
        _install(self.extras, "mlx", "0.31.2", ["mlx/__init__.py", "mlx/core.so"])
        _install(self.extras, "mlx-metal", "0.31.2", ["mlx/lib/libmlx.dylib"])
        _install(self.extras, "mlx-whisper", "0.4.3", ["mlx_whisper/__init__.py"])

        moved = runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled])

        self.assertEqual(sorted(moved), ["mlx 0.31.2", "mlx-metal 0.31.2"])
        self.assertFalse((self.extras / "mlx" / "core.so").exists())
        self.assertFalse((self.extras / "mlx" / "lib" / "libmlx.dylib").exists())
        self.assertTrue((self.extras / "mlx_whisper" / "__init__.py").exists())
        stale = self.extras.parent / "stale-core"
        self.assertTrue((stale / "mlx-0.31.2" / "mlx" / "core.so").exists())
        self.assertFalse(any(self.extras.glob("mlx-0.31.2.dist-info")))
        self.assertFalse((self.extras / "mlx").exists())  # emptied folders go too

    def test_matching_versions_and_unrelated_packages_stay(self) -> None:
        _install(self.extras, "mlx", "0.32.4", ["mlx/__init__.py"])
        _install(self.extras, "mlx-vlm", "0.4.4", ["mlx_vlm/__init__.py"])

        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), [])
        self.assertTrue((self.extras / "mlx" / "__init__.py").exists())

    def test_only_packaged_builds_touch_the_overlay(self) -> None:
        _install(self.extras, "mlx", "0.31.2", ["mlx/__init__.py"])
        with mock.patch.dict(os.environ, {"CHAOSENGINE_EXTRAS_SITE_PACKAGES": str(self.extras)}, clear=False):
            os.environ.pop("CHAOSENGINE_EMBEDDED_RUNTIME", None)
            self.assertEqual(runtime_paths.guard_overlay_against_bundle(), [])
        self.assertTrue((self.extras / "mlx" / "__init__.py").exists())


if __name__ == "__main__":
    unittest.main()
