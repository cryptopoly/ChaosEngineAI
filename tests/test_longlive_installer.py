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
    LongLiveInstallError,
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


if __name__ == "__main__":
    unittest.main()
