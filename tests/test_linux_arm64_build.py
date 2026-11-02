"""Linux ARM64 (DGX Spark class) build pieces: CI job, runtime staging, updater manifest, llama.cpp build."""
from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class WorkflowMatrixTests(unittest.TestCase):
    def test_release_and_build_workflows_have_an_arm_linux_job(self) -> None:
        for workflow in (".github/workflows/release.yml", ".github/workflows/build.yml"):
            text = _read(workflow)
            self.assertRegex(
                text,
                r"os: ubuntu-22\.04-arm\s+platform: linux\s+arch: aarch64",
                workflow,
            )


class StageRuntimeTests(unittest.TestCase):
    def test_arm_linux_stages_the_cpu_llama_build_from_the_real_asset_names(self) -> None:
        source = _read("scripts/stage-runtime.mjs")
        self.assertIn('if (plat === "linux" && arch === "arm64") return "linux-arm64";', source)
        match = re.search(r'"linux-arm64": \[\s*/(.+?)/i,', source, re.S)
        self.assertIsNotNone(match)
        pattern = re.compile(match.group(1).replace("\\.", "."), re.I)
        # Names published with llama.cpp b11981.
        self.assertTrue(pattern.match("llama-b11981-bin-ubuntu-arm64.tar.gz"))
        for other in (
            "llama-b11981-bin-ubuntu-cuda-13.4-arm64.tar.gz",
            "llama-b11981-bin-linux-arm64-snapdragon.tar.gz",
            "llama-b11981-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "cudart-llama-b11981-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        ):
            self.assertFalse(pattern.match(other), other)


class UpdaterManifestTests(unittest.TestCase):
    def test_arm_assets_are_optional_and_use_both_arch_spellings(self) -> None:
        source = _read("scripts/generate-updater-manifest.mjs")
        for expected in ("_aarch64.AppImage", "_arm64.deb", 'platforms["linux-aarch64"]', "linux-aarch64-${bundle.kind}"):
            self.assertIn(expected, source)
        # A release without ARM assets must not list them as missing.
        self.assertIn("if (!assetName) continue;", source)


class LlamaBuildScriptTests(unittest.TestCase):
    def test_script_parses_and_picks_cuda_only_without_metal(self) -> None:
        script = ROOT / "scripts" / "update-llama-cpp.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text(encoding="utf-8")
        self.assertIn("-DGGML_METAL=ON", text)
        self.assertIn("-DGGML_CUDA=ON", text)
        self.assertLess(text.index("-DGGML_METAL=ON"), text.index("-DGGML_CUDA=ON"))
        self.assertIn("command -v nvcc", text)


if __name__ == "__main__":
    unittest.main()
