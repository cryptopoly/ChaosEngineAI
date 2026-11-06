#!/usr/bin/env python3
"""Check that every package a Linux build installs has a wheel for a target platform.

Runs ``pip install --dry-run --only-binary=:all: --platform ...`` from any machine,
so a Mac can prove that Linux ARM64 (DGX Spark) has a wheel for each pinned
package without ARM hardware. Nothing is installed; pip downloads metadata only.

Groups:
  core   the release venv (release.yml "Create Python venv (Linux)"); a failure exits 1
  voice  the portable voice pack (voice_env.py), CUDA libraries included
  gpu    torch / diffusers / transformers at the release pins (Setup's GPU runtime)
  vllm   vLLM at the app's floor

Usage:
  .venv/bin/python scripts/check-platform-wheels.py                # linux aarch64, Python 3.12
  .venv/bin/python scripts/check-platform-wheels.py --arch x86_64
  .venv/bin/python scripts/check-platform-wheels.py --groups core voice
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "release-constraints.txt"

# Mirrors the Linux venv step in .github/workflows/release.yml.
CORE = ["fastapi", "psutil", "uvicorn", "pypdf>=6.21.4", "python-multipart", "huggingface_hub", "babel"]
GPU = ["torch", "diffusers", "transformers", "accelerate"]
VLLM = ["vllm>=0.28.0"]


def _voice_packages() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from backend_service import voice_env  # noqa: PLC0415 - script-local import

    return [*voice_env._PORTABLE_PACKAGES, *voice_env._CUDA_PACKAGES]


def platform_tags(arch: str) -> list[str]:
    # pip matches --platform tags exactly, so list every manylinux level a
    # glibc 2.35 system (Ubuntu 22.04, the build baseline) can load.
    return [f"manylinux_2_{minor}_{arch}" for minor in range(35, 16, -1)] + [f"manylinux2014_{arch}"]


def check(packages: list[str], arch: str, python: str, constrained: bool) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as target:
        cmd = [
            sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
            "--quiet", "--disable-pip-version-check", "--target", target,
            "--only-binary=:all:", "--implementation", "cp", "--python-version", python,
        ]
        for tag in platform_tags(arch):
            cmd += ["--platform", tag]
        if constrained:
            cmd += ["-c", str(CONSTRAINTS)]
        proc = subprocess.run([*cmd, *packages], capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    lines = [line for line in (proc.stderr or proc.stdout).splitlines() if line.strip()]
    errors = [line for line in lines if "ERROR" in line or "No matching" in line]
    return False, (errors or lines or ["pip failed"])[0].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="aarch64", choices=("aarch64", "x86_64"))
    parser.add_argument("--python", default="3.12", help="target Python version (the bundle uses 3.12)")
    parser.add_argument("--groups", nargs="+", default=["core", "voice", "gpu", "vllm"],
                        choices=("core", "voice", "gpu", "vllm"))
    args = parser.parse_args()

    groups = {
        "core": (CORE, True),
        "voice": (_voice_packages(), False),
        "gpu": (GPU, True),
        "vllm": (VLLM, False),
    }
    print(f"Wheels for Linux {args.arch}, CPython {args.python}")
    core_ok = True
    for name in args.groups:
        packages, constrained = groups[name]
        ok, detail = check(packages, args.arch, args.python, constrained)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:6} ({len(packages)} packages{', release pins' if constrained else ''})"
              + ("" if ok else f": {detail}"))
        if not ok:
            # Narrow a failing group down to the packages that have no wheel.
            for package in packages:
                single_ok, single = check([package], args.arch, args.python, constrained)
                if not single_ok:
                    print(f"         no wheel: {package}: {single}")
        if name == "core" and not ok:
            core_ok = False
    return 0 if core_ok else 1


if __name__ == "__main__":
    sys.exit(main())
