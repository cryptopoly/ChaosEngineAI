"""NVIDIA CUDA build of llama-server for Linux (x86_64 and ARM64, e.g. DGX Spark).

The release bundle stages a portable llama-server (Vulkan on x86_64, CPU on
ARM64). This module downloads ggml-org's prebuilt CUDA build of the same pinned
tag, plus its CUDA runtime archive, into ``~/.chaosengine/bin/llama-cuda/<tag>/``.
The binaries carry ``RUNPATH $ORIGIN``, so the CUDA runtime libraries placed next
to them are found without ``LD_LIBRARY_PATH``; only ``libcuda.so.1`` comes from
the NVIDIA driver. The install is used only when this host's ``llama-server
--list-devices`` reports a CUDA device, so a machine without a working NVIDIA
driver keeps the bundled build.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

# Must match LLAMA_CPP_TAG in scripts/stage-runtime.mjs (a test checks).
LLAMA_CPP_TAG = "b11981"

INSTALL_ROOT = Path.home() / ".chaosengine" / "bin" / "llama-cuda"
_MARKER = "installed.json"
_DOWNLOAD_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"
# CUDA 13 needs driver 580 or newer and drops GPUs older than Turing.
_CUDA13_MIN_DRIVER = 580

_usable_cache: dict[str, str | None] = {}


def host_arch() -> str | None:
    """llama.cpp's asset arch for this Linux host, or None when unsupported."""
    if not sys.platform.startswith("linux"):
        return None
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return None


def nvidia_driver_major() -> int | None:
    """Major version of the NVIDIA driver, or None when nvidia-smi is missing or fails."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = (proc.stdout or "").strip().splitlines()
    try:
        return int(first[0].split(".")[0]) if proc.returncode == 0 and first else None
    except ValueError:
        return None


def cuda_version_for(arch: str, driver_major: int | None) -> str:
    """CUDA build to fetch: 13.4 by default; 12.8 on x86_64 with a pre-580 driver."""
    if arch == "x64" and driver_major is not None and driver_major < _CUDA13_MIN_DRIVER:
        return "12.8"
    return "13.4"


def asset_names(arch: str, cuda: str, tag: str = LLAMA_CPP_TAG) -> tuple[str, str]:
    """(llama.cpp build, CUDA runtime) archive names published with ``tag``."""
    suffix = f"bin-ubuntu-cuda-{cuda}-{arch}.tar.gz"
    return f"llama-{tag}-{suffix}", f"cudart-llama-{tag}-{suffix}"


def install_dir(tag: str = LLAMA_CPP_TAG) -> Path:
    return INSTALL_ROOT / tag


def installed_binary(tag: str = LLAMA_CPP_TAG) -> Path | None:
    """The installed CUDA llama-server for ``tag``, or None."""
    target = install_dir(tag)
    binary = target / "llama-server"
    if (target / _MARKER).is_file() and binary.is_file() and os.access(binary, os.X_OK):
        return binary
    return None


def installed_info(tag: str = LLAMA_CPP_TAG) -> dict | None:
    marker = install_dir(tag) / _MARKER
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def lists_cuda_device(binary: Path) -> bool:
    """True when ``binary --list-devices`` reports at least one CUDA device."""
    try:
        proc = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    return proc.returncode == 0 and any(
        line.strip().startswith("CUDA") for line in output.splitlines()
    )


def usable_binary() -> str | None:
    """The installed CUDA llama-server when this host can run it on a GPU, else None.

    Off when ``CHAOSENGINE_LLAMA_CUDA=0``. The device check runs once per
    installed binary and is cached for the life of the process.
    """
    if os.environ.get("CHAOSENGINE_LLAMA_CUDA") == "0":
        return None
    binary = installed_binary()
    if binary is None:
        return None
    key = f"{binary}:{binary.stat().st_mtime_ns}"
    if key not in _usable_cache:
        _usable_cache[key] = str(binary) if lists_cuda_device(binary) else None
    return _usable_cache[key]


def clear_cache() -> None:
    _usable_cache.clear()


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract a tar.gz into ``dest``, refusing members that would land outside it."""
    root = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive member escapes the install folder: {member.name}")
            if member.islnk() or member.issym():
                link = (target.parent / member.linkname).resolve()
                if root not in link.parents:
                    raise ValueError(f"archive link escapes the install folder: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"unsupported archive member: {member.name}")
        tar.extractall(root)  # noqa: S202 - every member checked above


def _flatten_into(source: Path, dest: Path) -> None:
    """Move every file and symlink under ``source`` into ``dest`` (no subfolders)."""
    for path in sorted(source.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        target = dest / path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        path.rename(target)


def _download(url: str, dest: Path, progress: Callable[[int, int | None], None]) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI"})
    with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:  # noqa: S310 - fixed https host
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            progress(done, total_bytes)


def install(
    log: Callable[[str], None] = lambda _msg: None,
    progress: Callable[[str, int, int | None], None] = lambda *_args: None,
    tag: str = LLAMA_CPP_TAG,
) -> Path:
    """Download and unpack the CUDA build for this host. Returns the llama-server path.

    Raises RuntimeError when the host is not Linux x86_64 / ARM64. The new
    folder replaces an older install of the same tag only after it is complete.
    """
    arch = host_arch()
    if arch is None:
        raise RuntimeError("The CUDA build of llama-server is for Linux x86_64 and ARM64 only.")
    driver = nvidia_driver_major()
    cuda = cuda_version_for(arch, driver)
    names = asset_names(arch, cuda, tag)
    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=INSTALL_ROOT, prefix=".staging-") as tmp:
        work = Path(tmp)
        unpacked = work / "unpacked"
        flat = work / "flat"
        unpacked.mkdir()
        flat.mkdir()
        for name in names:
            url = f"{_DOWNLOAD_BASE}/{tag}/{name}"
            log(f"Downloading {name}")
            archive = work / name
            _download(url, archive, lambda done, total, n=name: progress(n, done, total))
            log(f"Unpacking {name}")
            _safe_extract(archive, unpacked)
            archive.unlink()
        _flatten_into(unpacked, flat)
        binary = flat / "llama-server"
        if not binary.is_file():
            raise RuntimeError("The downloaded archive has no llama-server.")
        binary.chmod(binary.stat().st_mode | 0o111)
        (flat / _MARKER).write_text(
            json.dumps({"tag": tag, "arch": arch, "cuda": cuda, "assets": list(names)}),
            encoding="utf-8",
        )
        final = install_dir(tag)
        old = work / "previous"
        if final.exists():
            final.rename(old)
        flat.rename(final)
    clear_cache()
    log(f"Installed CUDA {cuda} llama-server ({tag}) in {final}")
    return final / "llama-server"
