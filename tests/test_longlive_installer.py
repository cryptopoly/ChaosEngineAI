"""Unit tests for ``backend_service.longlive_installer``.

The real install is a ~10 minute network-heavy process, so we pin the
pre-flight behaviour (platform guards, missing git, venv path layout)
rather than running it end-to-end.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from backend_service import longlive_installer
from backend_service.longlive_installer import (
    INSTALL_PHASES,
    LongLiveInstallError,
    _DROPPED_REQUIREMENTS,
    _filter_requirements,
    _venv_python_path,
    install,
    main,
)


class VenvPathTests(unittest.TestCase):
    # ``os.name`` is shared with pathlib, so patching it inside the ``with``
    # block would flip ``Path(...)`` to return PosixPath on Windows hosts
    # and blow up at construction time. Build the input path first, patch
    # after, so only the function-under-test sees the patched value.

    def test_windows_uses_scripts_python_exe(self):
        venv_dir = Path("venv-root")
        with mock.patch.object(longlive_installer.os, "name", "nt"):
            path = _venv_python_path(venv_dir)
        self.assertEqual(path.parts[-2:], ("Scripts", "python.exe"))

    def test_posix_uses_bin_python(self):
        venv_dir = Path("venv-root")
        with mock.patch.object(longlive_installer.os, "name", "posix"):
            path = _venv_python_path(venv_dir)
        self.assertEqual(path.parts[-2:], ("bin", "python"))


class PreflightTests(unittest.TestCase):
    def test_install_rejects_macos(self):
        with mock.patch("platform.system", return_value="Darwin"):
            with self.assertRaises(LongLiveInstallError) as ctx:
                install()
        self.assertIn("macOS", str(ctx.exception))

    def test_install_rejects_missing_git(self):
        with TemporaryDirectory() as tmp, \
             mock.patch("platform.system", return_value="Linux"), \
             mock.patch("shutil.which", return_value=None):
            with self.assertRaises(LongLiveInstallError) as ctx:
                install(root=Path(tmp))
        self.assertIn("git", str(ctx.exception).lower())


class InstallerModuleIsInvocableTests(unittest.TestCase):
    def test_has_main_entrypoint(self):
        # The setup route calls ``python -m backend_service.longlive_installer``,
        # which requires both a callable ``main`` and an ``if __name__ ==
        # "__main__"`` guard. Regression guard — if either goes away the
        # install button silently starts a no-op subprocess.
        self.assertTrue(callable(getattr(longlive_installer, "main", None)))
        module_source = Path(longlive_installer.__file__).read_text(
            encoding="utf-8"
        )
        self.assertIn('if __name__ == "__main__"', module_source)

    def test_help_flag_exits_zero_without_installing(self):
        # Regression guard: ``--help`` must never trigger ``install()`` —
        # otherwise a casual smoke test or an accidental shell invocation
        # kicks off a 10-minute CUDA-specific install.
        with mock.patch.object(longlive_installer, "install") as mock_install:
            self.assertEqual(main(["--help"]), 0)
            self.assertEqual(main(["-h"]), 0)
        mock_install.assert_not_called()

    def test_unknown_args_exit_nonzero_without_installing(self):
        with mock.patch.object(longlive_installer, "install") as mock_install:
            self.assertEqual(main(["--bogus"]), 2)
        mock_install.assert_not_called()


class FilterRequirementsTests(unittest.TestCase):
    """Regression tests for the requirements.txt filter.

    The upstream LongLive ``requirements.txt`` lists ``nvidia-tensorrt``
    alongside ``nvidia-pyindex``. Pip resolves all deps before installing
    any, hits the PyPI placeholder for ``nvidia-tensorrt``, and aborts the
    entire install — the symptom is "LongLive install hangs" reports on
    Windows. The filter strips these (and ``pycuda``, which needs CUDA
    toolkit headers Windows users don't typically have) before the pip
    call. Inference works without all three.
    """

    def test_drops_known_breaking_requirements(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "requirements.txt"
            src.write_text(
                "opencv-python>=4.9.0.80\n"
                "diffusers==0.31.0\n"
                "nvidia-pyindex\n"
                "nvidia-tensorrt\n"
                "pycuda\n"
                "transformers>=4.49.0\n",
                encoding="utf-8",
            )
            dst = Path(tmp) / "requirements.filtered.txt"
            dropped = _filter_requirements(src, dst)
            kept_lines = [line.strip() for line in dst.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(
            sorted(d.strip() for d in dropped),
            ["nvidia-pyindex", "nvidia-tensorrt", "pycuda"],
        )
        self.assertIn("opencv-python>=4.9.0.80", kept_lines)
        self.assertIn("diffusers==0.31.0", kept_lines)
        self.assertIn("transformers>=4.49.0", kept_lines)
        for dropped_name in _DROPPED_REQUIREMENTS:
            self.assertFalse(
                any(dropped_name in line for line in kept_lines),
                f"{dropped_name} should not appear in kept requirements",
            )

    def test_strips_version_specifiers_when_matching(self):
        # If upstream tightens the pin, we still need to drop the line.
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "requirements.txt"
            src.write_text(
                "nvidia-tensorrt>=8.6.1\n"
                "pycuda==2024.1\n"
                "diffusers~=0.31\n",
                encoding="utf-8",
            )
            dst = Path(tmp) / "requirements.filtered.txt"
            dropped = _filter_requirements(src, dst)
            kept_lines = [line.strip() for line in dst.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(dropped), 2)
        self.assertIn("diffusers~=0.31", kept_lines)

    def test_preserves_blank_lines_and_comments(self):
        with TemporaryDirectory() as tmp:
            src = Path(tmp) / "requirements.txt"
            src.write_text(
                "# Core deps\n"
                "diffusers==0.31.0\n"
                "\n"
                "# CUDA export pipeline (we drop these)\n"
                "nvidia-tensorrt\n",
                encoding="utf-8",
            )
            dst = Path(tmp) / "requirements.filtered.txt"
            _filter_requirements(src, dst)
            contents = dst.read_text(encoding="utf-8")
        self.assertIn("# Core deps", contents)
        self.assertIn("# CUDA export pipeline", contents)


class ProgressCallbackTests(unittest.TestCase):
    """The async job worker in ``setup.py`` relies on the ``progress``
    callback to drive the InstallLogPanel. If a phase stops emitting an
    event, the UI counter goes silent. Pin the contract.
    """

    def test_install_phases_constant_lists_known_phases(self):
        # The job worker maps each phase to a friendly label and uses the
        # tuple's length for ``packageTotal``. Adding/removing without
        # updating the labels in routes/setup.py would silently mis-label
        # the progress counter.
        self.assertEqual(
            INSTALL_PHASES,
            (
                "clone", "venv", "pip-upgrade", "pip-requirements",
                "flash-attn", "pip-hub", "weights-longlive",
                "weights-wan", "marker",
            ),
        )

    def test_install_signature_accepts_progress_callback(self):
        # Smoke check: the installer must keep accepting ``progress=`` as a
        # kwarg or the worker's wiring breaks at call time.
        import inspect
        params = inspect.signature(install).parameters
        self.assertIn("progress", params)
        self.assertIsNotNone(params["progress"].default)


if __name__ == "__main__":
    unittest.main()
