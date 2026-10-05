"""The Setup tab's installer scripts must be staged into the packaged backend.

``routes/setup/{mflux,mtplx,tensorfold}.py`` run ``bash <backend>/scripts/<name>``,
resolved next to ``backend_service``. ``scripts/stage-runtime.mjs`` copied only the
Python packages, so the release bundle had no ``scripts`` directory and every one
of those Install buttons failed with "No such file" (found in the v0.9.3 bundle).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _staged_scripts() -> set[str]:
    source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
    match = re.search(r"const BUNDLED_INSTALL_SCRIPTS = \[(.*?)\];", source, re.S)
    assert match, "stage-runtime.mjs lost its BUNDLED_INSTALL_SCRIPTS list"
    return set(re.findall(r'"([^"]+\.sh)"', match.group(1)))


def _scripts_the_routes_run() -> set[str]:
    found: set[str] = set()
    for route in (ROOT / "backend_service" / "routes" / "setup").glob("*.py"):
        text = route.read_text(encoding="utf-8")
        found.update(re.findall(r'parents\[3\]\s*/\s*"scripts"\s*/\s*"([^"]+\.sh)"', text))
    return found


class InstallScriptStagingTests(unittest.TestCase):
    def test_the_routes_still_use_the_pattern_this_test_reads(self) -> None:
        self.assertGreaterEqual(
            _scripts_the_routes_run(),
            {"install-mflux.sh", "install-mtplx.sh", "install-tensorfold.sh"},
        )

    def test_every_script_a_route_runs_is_staged(self) -> None:
        missing = _scripts_the_routes_run() - _staged_scripts()
        self.assertEqual(missing, set(), f"routes run scripts the bundle does not stage: {sorted(missing)}")

    def test_every_staged_script_exists(self) -> None:
        for name in _staged_scripts():
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
