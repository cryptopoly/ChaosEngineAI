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


class BundledLlamaCppTests(unittest.TestCase):
    def test_the_bundled_llama_cpp_tag_matches_the_update_script_pin(self) -> None:
        stage = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        update = (ROOT / "scripts" / "update-llama-cpp.sh").read_text(encoding="utf-8")
        staged = re.search(r'LLAMA_CPP_TAG = process\.env\.CHAOSENGINE_LLAMA_CPP_TAG \|\| "(b\d+)"', stage)
        pinned = re.search(r'LLAMA_CPP_REF="\$\{LLAMA_CPP_REF:-(b\d+)\}"', update)
        self.assertTrue(staged and pinned, "could not read one of the llama.cpp pins")
        self.assertEqual(staged.group(1), pinned.group(1))

    def test_the_release_is_fetched_by_tag_not_as_latest(self) -> None:
        stage = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        self.assertNotIn("releases/latest", stage)
        self.assertIn("releases/tags/", stage)


if __name__ == "__main__":
    unittest.main()


class ReleaseConstraintsTests(unittest.TestCase):
    """Release builds install the versions the app is tested with, not "newest today"."""

    def test_every_release_venv_install_uses_the_constraints(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
        installs = [line for line in workflow.splitlines() if re.search(r"pip install (?!--upgrade pip)", line)]
        self.assertTrue(installs)
        for line in installs:
            self.assertIn("-c release-constraints.txt", line, line.strip())

    def test_the_bundle_step_applies_them_too(self) -> None:
        stage = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        self.assertIn('"-c", releaseConstraintsPath', stage)

    def test_core_packages_are_pinned(self) -> None:
        pins = {
            line.split("==", 1)[0].lower().replace("_", "-")
            for line in (ROOT / "release-constraints.txt").read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.startswith("#")
        }
        for name in ("mlx", "mlx-lm", "huggingface-hub", "transformers", "fastapi", "uvicorn"):
            self.assertIn(name, pins)
