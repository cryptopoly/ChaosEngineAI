"""scripts/check-version-sync.mjs: the app version must agree in every file and with the release tag.

The updater compares the running app's version with latest.json, which the release
workflow takes from the tag. v0.9.4 shipped files named 0.9.3, so an install that
was already current kept being offered the same release.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-version-sync.mjs"
FILES = (
    "package.json",
    "package-lock.json",
    "pyproject.toml",
    "src-tauri/Cargo.toml",
    "src-tauri/Cargo.lock",
    "src-tauri/tauri.conf.json",
)


def _run(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", str(SCRIPT), "--root", str(root), *extra], capture_output=True, text=True, timeout=30
    )


@unittest.skipUnless(shutil.which("node"), "needs node")
class VersionSyncTests(unittest.TestCase):
    def _copy(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="ce-version-"))
        for rel in FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, root / rel)
        return root

    def _version(self) -> str:
        return json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text())["version"]

    def test_the_repository_agrees_with_itself(self) -> None:
        done = _run(self._copy())
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_the_matching_tag_passes_and_another_fails(self) -> None:
        root = self._copy()
        self.assertEqual(_run(root, "--tag", f"v{self._version()}").returncode, 0)
        self.assertEqual(_run(root, "--tag", "v0.0.1").returncode, 1)

    def test_a_stale_lock_file_fails(self) -> None:
        root = self._copy()
        lock = root / "src-tauri" / "Cargo.lock"
        name = (root / "src-tauri" / "Cargo.toml").read_text().split('name = "', 1)[1].split('"', 1)[0]
        lock.write_text(lock.read_text().replace(f'name = "{name}"\nversion = "{self._version()}"', f'name = "{name}"\nversion = "0.0.1"'))
        done = _run(root)
        self.assertEqual(done.returncode, 1)
        self.assertIn("Cargo.lock=0.0.1", done.stderr)

    def test_a_stale_tauri_conf_fails(self) -> None:
        root = self._copy()
        conf = root / "src-tauri" / "tauri.conf.json"
        conf.write_text(conf.read_text().replace(f'"version": "{self._version()}"', '"version": "0.0.1"', 1))
        self.assertEqual(_run(root).returncode, 1)


if __name__ == "__main__":
    unittest.main()
