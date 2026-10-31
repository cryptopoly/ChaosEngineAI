"""install-mflux.sh must keep a working install when an update fails."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install-mflux.sh"

# A stand-in interpreter: passes the version probe, builds a venv whose pip fails
# (or succeeds) on demand, so the script's own move-aside and restore logic runs.
FAKE_PYTHON = """#!/bin/bash
if [[ "$1" == "-c" ]]; then
  case "$2" in
    *platform.machine*) exit 0 ;;
    *version_info*) echo "3.12"; exit 0 ;;
  esac
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
  mkdir -p "$3/bin"
  cat > "$3/bin/pip" <<'PIP'
#!/bin/bash
[[ "$1" == "install" && "$*" == *"mflux=="* && -n "$FAKE_PIP_FAIL" ]] && { echo "boom" >&2; exit 1; }
[[ "$1" == "show" ]] && { echo "Version: 0.24.0"; exit 0; }
exit 0
PIP
  printf '#!/bin/bash\\nexit 0\\n' > "$3/bin/python"
  printf '#!/bin/bash\\nexit 0\\n' > "$3/bin/mflux-generate"
  chmod +x "$3/bin/pip" "$3/bin/python" "$3/bin/mflux-generate"
  echo "new" > "$3/marker"
  exit 0
fi
exit 0
"""


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("bash"), "the installer is macOS only")
class InstallMfluxRestoreTests(unittest.TestCase):
    def _run(self, home: Path, fail_pip: bool) -> subprocess.CompletedProcess:
        fake = home / "fake-python"
        fake.write_text(FAKE_PYTHON)
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        env = {**os.environ, "HOME": str(home), "PYTHON": str(fake)}
        if fail_pip:
            env["FAKE_PIP_FAIL"] = "1"
        return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)

    def _old_install(self, home: Path) -> Path:
        venv = home / ".chaosengine" / "mflux-venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "marker").write_text("old")
        version = home / ".chaosengine" / "bin" / "mflux.version"
        version.parent.mkdir(parents=True)
        version.write_text("0.23.0\n")
        return venv

    def test_a_failed_update_puts_the_previous_install_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            venv = self._old_install(home)
            result = self._run(home, fail_pip=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((venv / "marker").read_text(), "old")
            self.assertEqual((home / ".chaosengine" / "bin" / "mflux.version").read_text(), "0.23.0\n")
            self.assertFalse((home / ".chaosengine" / "mflux-venv.previous").exists())
            self.assertIn("previous mflux install was put back", result.stdout)

    def test_a_good_update_replaces_it_and_leaves_no_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            venv = self._old_install(home)
            result = self._run(home, fail_pip=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((venv / "marker").read_text().strip(), "new")
            self.assertEqual((home / ".chaosengine" / "bin" / "mflux.version").read_text().splitlines()[0], "0.24.0")
            self.assertFalse((home / ".chaosengine" / "mflux-venv.previous").exists())
            self.assertFalse((home / ".chaosengine" / "bin" / "mflux.version.previous").exists())

    def test_a_first_install_that_fails_leaves_nothing_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            result = self._run(home, fail_pip=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((home / ".chaosengine" / "bin" / "mflux.version").exists())


if __name__ == "__main__":
    unittest.main()
