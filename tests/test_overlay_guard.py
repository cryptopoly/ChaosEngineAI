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

    def test_a_mixed_huggingface_hub_overlay_is_moved_whole(self) -> None:
        # Two pip --target runs left 1.12.2 and 2.2.0 dist-infos over one set of files;
        # transformers then sees hub 1.12.2 and refuses to import.
        _install(self.bundled, "huggingface-hub", "2.2.0", ["huggingface_hub/__init__.py"])
        _install(self.extras, "huggingface-hub", "1.12.2", ["huggingface_hub/__init__.py", "huggingface_hub/old.py"])
        _install(self.extras, "huggingface-hub", "2.2.0", ["huggingface_hub/__init__.py", "huggingface_hub/new.py"])
        (self.extras / "huggingface_hub" / "__pycache__").mkdir()
        (self.extras / "huggingface_hub" / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")

        moved = runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled])

        self.assertEqual(sorted(moved), ["huggingface-hub 1.12.2", "huggingface-hub 2.2.0"])
        self.assertFalse((self.extras / "huggingface_hub").exists())
        self.assertFalse(any(self.extras.glob("huggingface_hub-*.dist-info")))

    def test_a_clean_lockstep_package_stays(self) -> None:
        _install(self.bundled, "transformers", "5.20.2", ["transformers/__init__.py"])
        _install(self.extras, "transformers", "5.20.2", ["transformers/__init__.py"])

        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), [])
        self.assertTrue((self.extras / "transformers" / "__init__.py").exists())

    def test_an_older_lockstep_package_is_moved_and_the_bundle_runs(self) -> None:
        _install(self.bundled, "transformers", "5.20.2", ["transformers/__init__.py"])
        _install(self.extras, "transformers", "5.7.0", ["transformers/__init__.py"])

        moved = runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled])

        self.assertEqual(moved, ["transformers 5.7.0"])
        self.assertFalse((self.extras / "transformers").exists())

    def test_matching_versions_and_unrelated_packages_stay(self) -> None:
        _install(self.extras, "mlx", "0.32.4", ["mlx/__init__.py"])
        _install(self.extras, "mlx-vlm", "0.4.4", ["mlx_vlm/__init__.py"])

        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), [])
        self.assertTrue((self.extras / "mlx" / "__init__.py").exists())

    def test_a_half_replaced_copy_of_a_bundled_package_is_moved_aside(self) -> None:
        # Windows: pip could not replace psutil's loaded .pyd, so psutil/ kept 3 of its files and the
        # backend failed to import it at the next start.
        files = ["psutil/__init__.py", "psutil/_common.py", "psutil/_pswindows.py", "psutil/_psutil_windows.pyd"]
        _install(self.bundled, "psutil", "7.2.2", files)
        _install(self.extras, "psutil", "7.2.2", files)
        (self.extras / "psutil" / "_common.py").unlink()

        moved = runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled])

        self.assertEqual(moved, ["psutil 7.2.2"])
        self.assertFalse((self.extras / "psutil").exists())
        self.assertFalse(any(self.extras.glob("psutil-*.dist-info")))
        self.assertTrue((self.extras.parent / "stale-core" / "psutil-7.2.2" / "psutil" / "__init__.py").exists())

    def test_moving_the_same_version_twice_does_not_stop_the_tidy_up(self) -> None:
        # A GPU-runtime install brought huggingface-hub 2.4.0 back after it had been moved once; the second
        # move clashed with the first ("Destination path ... already exists") and aborted the whole guard,
        # so a broken pydantic_core behind it was never moved and the backend did not start.
        _install(self.bundled, "huggingface-hub", "2.2.0", ["huggingface_hub/__init__.py"])
        _install(self.bundled, "pydantic-core", "2.41.5", ["pydantic_core/__init__.py", "pydantic_core/core_schema.py"])
        _install(self.extras, "huggingface-hub", "2.4.0", ["huggingface_hub/__init__.py"])
        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), ["huggingface-hub 2.4.0"])
        _install(self.extras, "huggingface-hub", "2.4.0", ["huggingface_hub/__init__.py"])
        _install(self.extras, "pydantic-core", "2.50.4", ["pydantic_core/__init__.py", "pydantic_core/core_schema.py"])
        (self.extras / "pydantic_core" / "core_schema.py").unlink()

        moved = runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled])

        self.assertEqual(sorted(moved), ["huggingface-hub 2.4.0", "pydantic-core 2.50.4"])
        stale = self.extras.parent / "stale-core"
        self.assertTrue((stale / "huggingface-hub-2.4.0.2" / "huggingface_hub" / "__init__.py").exists())
        self.assertFalse((self.extras / "pydantic_core").exists())

    def test_an_overlay_pydantic_goes_with_its_core(self) -> None:
        # pydantic 2.14.3 in the overlay demanded pydantic-core 2.50.4 once the broken core had been moved:
        # "The installed pydantic-core version (2.41.5) is incompatible with the current pydantic version".
        _install(self.bundled, "pydantic", "2.12.5", ["pydantic/__init__.py"])
        _install(self.bundled, "pydantic-core", "2.41.5", ["pydantic_core/__init__.py"])
        _install(self.extras, "pydantic", "2.14.3", ["pydantic/__init__.py"])
        _install(self.extras, "pydantic-core", "2.50.4", ["pydantic_core/__init__.py"])

        moved = runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled])

        self.assertEqual(sorted(moved), ["pydantic 2.14.3", "pydantic-core 2.50.4"])
        self.assertFalse((self.extras / "pydantic").exists())
        self.assertFalse((self.extras / "pydantic_core").exists())

    def test_a_same_version_native_copy_of_a_bundled_package_is_moved_aside(self) -> None:
        # The backend loaded the overlay's yaml/_yaml.pyd and psutil/_psutil_windows.pyd, so a GPU-runtime
        # install could not replace them (WinError 5) and six packages failed.
        files = ["yaml/__init__.py", "yaml/_yaml.cp312-win_amd64.pyd"]
        _install(self.bundled, "pyyaml", "6.0.3", files)
        _install(self.extras, "pyyaml", "6.0.3", files)

        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), ["pyyaml 6.0.3"])
        self.assertFalse((self.extras / "yaml").exists())

    def test_a_different_version_native_copy_stays(self) -> None:
        files = ["yaml/__init__.py", "yaml/_yaml.cp312-win_amd64.pyd"]
        _install(self.bundled, "pyyaml", "6.0.2", files)
        _install(self.extras, "pyyaml", "6.0.3", files)

        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), [])
        self.assertTrue((self.extras / "yaml" / "__init__.py").exists())

    def test_a_whole_copy_of_a_bundled_package_and_unbundled_broken_ones_stay(self) -> None:
        files = ["psutil/__init__.py", "psutil/_common.py"]
        _install(self.bundled, "psutil", "7.2.2", files)
        _install(self.extras, "psutil", "7.2.2", files)
        _install(self.extras, "diffusers", "0.42.1", ["diffusers/__init__.py", "diffusers/pipelines.py"])
        (self.extras / "diffusers" / "pipelines.py").unlink()  # not in the bundle: not ours to move

        self.assertEqual(runtime_paths.quarantine_stale_core_packages(self.extras, [self.bundled]), [])
        self.assertTrue((self.extras / "psutil" / "_common.py").exists())
        self.assertTrue((self.extras / "diffusers" / "__init__.py").exists())

    def test_only_packaged_builds_touch_the_overlay(self) -> None:
        _install(self.extras, "mlx", "0.31.2", ["mlx/__init__.py"])
        with mock.patch.dict(os.environ, {"CHAOSENGINE_EXTRAS_SITE_PACKAGES": str(self.extras)}, clear=False):
            os.environ.pop("CHAOSENGINE_EMBEDDED_RUNTIME", None)
            self.assertEqual(runtime_paths.guard_overlay_against_bundle(), [])
        self.assertTrue((self.extras / "mlx" / "__init__.py").exists())


class GuardRunsFirstTests(unittest.TestCase):
    def test_app_tidies_the_overlay_before_importing_any_third_party_package(self) -> None:
        # A broken overlay pydantic_core crashed `from fastapi import ...` at app.py line 13, before the guard.
        import ast

        source = Path(runtime_paths.__file__).with_name("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        guard_line = next(
            node.lineno for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "guard_overlay_against_bundle"
        )
        stdlib = {"__future__", "os", "ipaddress", "secrets", "signal", "time", "uuid", "datetime", "pathlib", "typing"}
        for node in tree.body:
            if node.lineno >= guard_line or not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = [node.module] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            for name in names:
                top = (name or "").split(".")[0]
                self.assertTrue(
                    top in stdlib or name == "backend_service.runtime_paths",
                    f"app.py imports {name} before guard_overlay_against_bundle() (line {node.lineno})",
                )


if __name__ == "__main__":
    unittest.main()
