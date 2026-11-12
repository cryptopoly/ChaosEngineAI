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
            {"install-mflux.sh", "install-ltx2-mlx.sh", "install-mtplx.sh", "install-tensorfold.sh"},
        )

    def test_every_script_a_route_runs_is_staged(self) -> None:
        missing = _scripts_the_routes_run() - _staged_scripts()
        self.assertEqual(missing, set(), f"routes run scripts the bundle does not stage: {sorted(missing)}")

    def test_every_staged_script_exists(self) -> None:
        for name in _staged_scripts():
            self.assertTrue((ROOT / "scripts" / name).is_file(), name)


class BundledCliTests(unittest.TestCase):
    def test_the_cli_is_staged_for_the_agent_bridge_relay(self) -> None:
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        match = re.search(r"const BUNDLED_CLI_SCRIPTS = \[(.*?)\];", source, re.S)
        self.assertTrue(match, "stage-runtime.mjs lost its BUNDLED_CLI_SCRIPTS list")
        self.assertIn('"chaosengine-cli"', match.group(1))
        self.assertTrue((ROOT / "scripts" / "chaosengine-cli").is_file())


class BundledAgentGuideTests(unittest.TestCase):
    def test_the_agent_guide_is_staged_at_the_backend_root(self) -> None:
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        match = re.search(r"const BUNDLED_ROOT_DOCS = \[(.*?)\];", source, re.S)
        self.assertTrue(match, "stage-runtime.mjs lost its BUNDLED_ROOT_DOCS list")
        self.assertIn('"AGENTS.md"', match.group(1))
        self.assertIn("...BUNDLED_ROOT_DOCS", source)
        self.assertTrue((ROOT / "AGENTS.md").is_file())


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


class WindowsArchiveNameTests(unittest.TestCase):
    def test_the_runtime_archive_is_named_without_a_drive_letter(self) -> None:
        # GNU tar (Git Bash's, first on PATH there) treats "C:\\...\\runtime.tar.gz" given to -f as host "C";
        # a local Windows release build failed with "Cannot connect to C: resolve failed".
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        call = re.search(r'execFileSync\("tar", \["-czf", (.+?), "-C"', source)
        self.assertTrue(call, "stage-runtime.mjs no longer creates the archive with tar -czf")
        self.assertEqual(call.group(1).strip(), "path.relative(workspaceRoot, archiveDest)")
        body = source[call.start():call.start() + 300]
        self.assertIn("cwd: workspaceRoot", body)


class WindowsSymlinkTests(unittest.TestCase):
    def test_windows_runtime_copy_follows_symlinks(self) -> None:
        # The Windows runner's python3.exe is a symlink to C:/hostedtoolcache/...;
        # shipped verbatim, the archive could not be unpacked and the app never started.
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        body = source[source.index("function copyTree"):source.index("function copyFile")]
        self.assertIn("dereference: onWindows", body)
        self.assertIn("verbatimSymlinks: !onWindows", body)


class ReleaseBuildScriptTests(unittest.TestCase):
    """build.sh must stage a clean pinned venv, never the dev .venv."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (ROOT / "build.sh").read_text(encoding="utf-8")

    def test_uses_dedicated_release_venv_and_exports_it(self) -> None:
        self.assertIn('RELEASE_VENV="$SCRIPT_DIR/.venv-release"', self.script)
        self.assertRegex(self.script, r'export CHAOSENGINE_EMBED_PYTHON_BIN="\$RELEASE_PY"')

    def test_dev_venv_is_not_touched(self) -> None:
        self.assertNotRegex(self.script, r"(^|[^-\w])\.venv/(bin|Scripts)/")
        self.assertNotRegex(self.script, r"-m venv \.venv(\s|$)")

    def test_every_pip_install_is_constrained(self) -> None:
        installs = [l for l in self.script.splitlines() if "pip install" in l and '"$RELEASE_PY"' in l]
        self.assertTrue(installs)
        for line in installs:
            if "--upgrade pip" in line:
                continue
            self.assertIn("-c release-constraints.txt", line)

    def test_release_venv_is_ignored_by_git(self) -> None:
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".venv-release/", [l.strip() for l in ignore])

    def test_gpu_bundle_keeps_its_warning(self) -> None:
        self.assertIn("CHAOSENGINE_BUNDLE_GPU", self.script)
        self.assertIn("WARNING: CHAOSENGINE_BUNDLE_GPU=1", self.script)

    def test_stage_runtime_guards_the_nsis_archive_limit(self) -> None:
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        self.assertIn("1.9 * 1024 ** 3", source)
        self.assertIn("assertArchiveFitsInstaller(archiveDest)", source)

    def test_archive_limit_constant_is_declared_before_main_runs(self) -> None:
        # main() is called at module level; a const declared after that call is
        # still in its temporal dead zone when main() reads it (v0.10.0 build crash).
        source = (ROOT / "scripts" / "stage-runtime.mjs").read_text(encoding="utf-8")
        self.assertLess(
            source.index("const NSIS_ARCHIVE_LIMIT_BYTES"),
            source.index("\nmain();"),
        )
