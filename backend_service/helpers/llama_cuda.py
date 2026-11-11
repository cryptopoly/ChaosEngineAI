"""NVIDIA CUDA build of llama-server for Linux (x86_64, ARM64) and Windows (x64, ARM64).

The release bundle stages a portable llama-server (Vulkan on x86_64, CPU on
ARM64). This module downloads ggml-org's prebuilt CUDA build of the same pinned
tag, plus its CUDA runtime archive, into ``~/.chaosengine/bin/llama-cuda/<tag>/``.
On Linux the binaries carry ``RUNPATH $ORIGIN``; on Windows the DLLs sit next to
``llama-server.exe``, which is where the loader looks first. Either way the CUDA
runtime libraries are found without touching ``LD_LIBRARY_PATH`` or ``PATH``, and
only the driver library (``libcuda.so.1`` / ``nvcuda.dll``) comes from the NVIDIA
driver. The install is used only when this host's ``llama-server --list-devices``
reports a CUDA device, so a machine without a working NVIDIA driver keeps the
bundled build. Archives are checked against the sha256 published with the release.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

# Must match LLAMA_CPP_TAG in scripts/stage-runtime.mjs (a test checks).
LLAMA_CPP_TAG = "b12292"

INSTALL_ROOT = Path.home() / ".chaosengine" / "bin" / "llama-cuda"
_MARKER = "installed.json"
_DOWNLOAD_BASE = "https://github.com/ggml-org/llama.cpp/releases/download"
# CUDA 13 needs driver 580 or newer and drops GPUs older than Turing.
_CUDA13_MIN_DRIVER = 580

_usable_cache: dict[str, str | None] = {}


def host_os() -> str | None:
    """``"linux"`` or ``"windows"`` when llama.cpp publishes CUDA builds for it, else None."""
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return None


def host_arch() -> str | None:
    """llama.cpp's asset arch for this host (Linux or Windows), or None when unsupported."""
    if host_os() is None:
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


_DRIVER_CUDA_RE = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")


def nvidia_driver_cuda() -> tuple[int, int] | None:
    """Newest CUDA version the installed driver supports, as ``(major, minor)``, or None.

    Read from the ``nvidia-smi`` header, which says ``CUDA Version: 13.1`` on older drivers and
    ``CUDA UMD Version: 13.5`` on newer ones (621.14). There is no ``--query-gpu`` field for it.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        proc = subprocess.run([nvidia_smi], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _DRIVER_CUDA_RE.search(proc.stdout or "") if proc.returncode == 0 else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_version_for(arch: str, driver_major: int | None) -> str:
    """Linux CUDA build to fetch: 13.5 by default; 12.8 on x86_64 with a pre-580 driver."""
    if arch == "x64" and driver_major is not None and driver_major < _CUDA13_MIN_DRIVER:
        return "12.8"
    return "13.5"


# Windows builds published by ggml-org: CUDA 12.4 (x64) and 13.5 (x64, ARM64). A CUDA 13
# runtime needs a driver that reports CUDA 13 or newer; CUDA 12.4 needs one reporting 12.4+.
_WIN_CUDA12_MIN = (12, 4)


def windows_cuda_version(
    arch: str, driver_cuda: tuple[int, int] | None, driver_major: int | None = None
) -> str | None:
    """Windows build for this driver ("13.5" or "12.4"), or None when the driver is too old.

    ``driver_cuda`` is the version ``nvidia-smi`` reports. When it cannot be read the driver
    major decides (580 and newer is CUDA 13, 551 and newer is CUDA 12.4).
    """
    if driver_cuda is None and driver_major is not None:
        if driver_major >= _CUDA13_MIN_DRIVER:
            driver_cuda = (13, 0)
        elif driver_major >= 551:
            driver_cuda = (12, 4)
        else:
            driver_cuda = (0, 0)
    if driver_cuda is None:
        return None
    if driver_cuda[0] >= 13:
        return "13.5"
    if arch == "x64" and driver_cuda >= _WIN_CUDA12_MIN:
        return "12.4"
    return None


def asset_names(arch: str, cuda: str, tag: str = LLAMA_CPP_TAG, os_name: str | None = None) -> tuple[str, str]:
    """(llama.cpp build, CUDA runtime) archive names published with ``tag``."""
    if (os_name or host_os()) == "windows":
        suffix = f"bin-win-cuda-{cuda}-{arch}.zip"
        return f"llama-{tag}-{suffix}", f"cudart-llama-{suffix}"
    suffix = f"bin-ubuntu-cuda-{cuda}-{arch}.tar.gz"
    return f"llama-{tag}-{suffix}", f"cudart-llama-{tag}-{suffix}"


# sha256 of the Windows archives of the pinned tag, from the release's published digests.
# Other tags are looked up from the release at install time (see ``release_digests``).
_PINNED_SHA256: dict[str, str] = {
    "llama-b12292-bin-win-cuda-12.4-x64.zip": "078a43924a90f5d93c77727743f36b8cfa5da59795a8f2651c09d1b734e87088",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    "llama-b12292-bin-win-cuda-13.5-x64.zip": "f173b246ab8d7e55c13024fdf53192d5b6a0c333042c37724c38d4c333cb92be",
    "cudart-llama-bin-win-cuda-13.5-x64.zip": "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
    "llama-b12292-bin-win-cuda-13.5-arm64.zip": "276f93b8e3f6faf67d05c466a301666be8dc4ef2124f259ff3b85bb89c8f30d6",
    "cudart-llama-bin-win-cuda-13.5-arm64.zip": "f89e7b2457f627d052dd99f42205ddfd01964454809754cdc649cffc74320dca",
}


def release_digests(tag: str) -> dict[str, str]:
    """``{asset name: sha256}`` the GitHub release publishes for ``tag`` (empty when unreachable)."""
    url = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "ChaosEngineAI", "Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - fixed https host
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return {}
    digests: dict[str, str] = {}
    for asset in payload.get("assets") or []:
        digest = str(asset.get("digest") or "")
        if digest.startswith("sha256:") and asset.get("name"):
            digests[str(asset["name"])] = digest.split(":", 1)[1].lower()
    return digests


def expected_sha256(name: str, tag: str, fetch: Callable[[str], dict[str, str]] = release_digests) -> str | None:
    """The published sha256 for ``name`` (pinned table for the pinned tag, else the release), or None."""
    pinned = _PINNED_SHA256.get(name) if tag == LLAMA_CPP_TAG else None
    return pinned or fetch(tag).get(name)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def server_name() -> str:
    return "llama-server.exe" if host_os() == "windows" else "llama-server"


def plan(tag: str = LLAMA_CPP_TAG) -> dict:
    """What this host can install: ``{"ok", "state", "message", "cuda", ...}``.

    ``state`` is ``unsupported`` (no published build for this OS or CPU), ``no-driver``
    (nvidia-smi missing or failing), ``driver-too-old`` or ``ready``. Nothing here names a
    card: the driver version alone decides, and the CUDA build itself reports the devices.
    """
    os_name, arch = host_os(), host_arch()
    base = {"os": os_name, "arch": arch, "cuda": None, "driverMajor": None, "driverCuda": None}
    if os_name is None or arch is None:
        return {**base, "ok": False, "state": "unsupported",
                "message": "A CUDA build of llama-server is published for Linux and Windows on x86_64 or ARM64 only."}
    driver = nvidia_driver_major()
    if driver is None:
        return {**base, "ok": False, "state": "no-driver",
                "message": "No NVIDIA driver found (nvidia-smi did not answer). The bundled build stays in use."}
    driver_cuda = nvidia_driver_cuda()
    shown = f"{driver_cuda[0]}.{driver_cuda[1]}" if driver_cuda else None
    base.update(driverMajor=driver, driverCuda=shown)
    cuda = windows_cuda_version(arch, driver_cuda, driver) if os_name == "windows" else cuda_version_for(arch, driver)
    if cuda is None:
        have = f"CUDA {shown}" if shown else f"driver {driver}"
        return {**base, "ok": False, "state": "driver-too-old",
                "message": f"The NVIDIA driver is too old ({have}). llama.cpp's CUDA builds need CUDA 12.4 or newer "
                           "(driver 551 or newer on Windows); update the driver, then install again."}
    return {**base, "ok": True, "state": "ready", "cuda": cuda, "message": ""}


def install_dir(tag: str = LLAMA_CPP_TAG) -> Path:
    return INSTALL_ROOT / tag


def installed_binary(tag: str = LLAMA_CPP_TAG) -> Path | None:
    """The installed CUDA llama-server for ``tag``, or None."""
    target = install_dir(tag)
    binary = target / server_name()
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


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip into ``dest``, refusing members that would land outside it."""
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (root / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive member escapes the install folder: {info.filename}")
        zf.extractall(root)  # noqa: S202 - every member checked above


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        _safe_extract_zip(archive, dest)
    else:
        _safe_extract(archive, dest)


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


_MIN_FREE_BYTES = 3 * (1 << 30)


def install(
    log: Callable[[str], None] = lambda _msg: None,
    progress: Callable[[str, int, int | None], None] = lambda *_args: None,
    tag: str = LLAMA_CPP_TAG,
) -> Path:
    """Download and unpack the CUDA build for this host. Returns the llama-server path.

    Raises RuntimeError when the host has no published build, no NVIDIA driver or a driver
    that is too old, when an archive does not match its published sha256, or when the disk
    is nearly full. The new folder replaces an older install of the same tag only after it is
    complete.
    """
    chosen = plan(tag)
    if not chosen["ok"]:
        raise RuntimeError(chosen["message"])
    arch, cuda, os_name = chosen["arch"], chosen["cuda"], chosen["os"]
    names = asset_names(arch, cuda, tag, os_name)
    INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(INSTALL_ROOT).free
    if free < _MIN_FREE_BYTES:
        raise RuntimeError(f"Not enough disk space: {free / (1 << 30):.1f} GB free, the install needs about 3 GB.")
    checksums: dict[str, str] = {}
    verified: dict[str, bool] = {}
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
            expected = expected_sha256(name, tag)
            actual = sha256_of(archive)
            if expected is None:
                log(f"No published sha256 for {name}; recorded {actual[:12]} without a comparison")
                verified[name] = False
            elif actual != expected:
                raise RuntimeError(
                    f"{name} does not match its published sha256 (got {actual[:12]}, expected {expected[:12]}); nothing was installed."
                )
            else:
                log(f"Verified sha256 of {name}")
                verified[name] = True
            checksums[name] = actual
            log(f"Unpacking {name}")
            _extract(archive, unpacked)
            archive.unlink()
        _flatten_into(unpacked, flat)
        binary = flat / server_name()
        if not binary.is_file():
            raise RuntimeError("The downloaded archive has no llama-server.")
        if os_name != "windows":
            binary.chmod(binary.stat().st_mode | 0o111)
        (flat / _MARKER).write_text(
            json.dumps({
                "tag": tag, "os": os_name, "arch": arch, "cuda": cuda, "assets": list(names),
                "sha256": checksums, "verified": verified,
                "driverMajor": chosen["driverMajor"], "driverCuda": chosen["driverCuda"],
                "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }),
            encoding="utf-8",
        )
        final = install_dir(tag)
        old = work / "previous"
        if final.exists():
            try:
                final.rename(old)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not replace the installed build ({exc}). Unload the GGUF model, then install again."
                ) from exc
        flat.rename(final)
    clear_cache()
    log(f"Installed CUDA {cuda} llama-server ({tag}) in {final}")
    return final / server_name()
