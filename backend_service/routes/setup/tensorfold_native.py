"""TensorFold 1.x native engine: install and status endpoints (CUDA hosts).

No Python venv and no bash: a Python thread downloads the release archive for
this platform, checks its sha256 against the published ``.sha256`` file, unpacks
it into ``~/.chaosengine/tensorfold-native/<version>/`` and runs
``tensorfold-native capabilities --json`` before it records the install in
``install.json``. Windows has no release archive (1.2.7 ships Linux and macOS
only), so there the install either adopts a binary the user built from source
(``{"binaryPath": "..."}``) or downloads a self-hosted archive named by
``CHAOSENGINE_TENSORFOLD_WINDOWS_URL`` and ``..._SHA256``.

``GET /api/setup/tensorfold-native-status`` returns the capability gate (GPU,
driver, free VRAM, the binary's own report) and the model registry, so the
Setup page offers the engine where it works and says why where it does not.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend_service.inference import _tensorfold_native as native

router = APIRouter()

_PHASES = ("preflight", "downloading", "verifying", "unpacking", "checking")
_TIMEOUT_SECONDS = 60


@dataclass
class _Job:
    phase: str = "idle"
    message: str = ""
    percent: float = 0.0
    target_dir: str | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0
    done: bool = False
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        index = _PHASES.index(self.phase) + 1 if self.phase in _PHASES else 0
        return {
            "id": "tensorfold-native-install",
            "phase": self.phase,
            "message": self.message,
            "packageCurrent": self.message or None,
            "packageIndex": index,
            "packageTotal": len(_PHASES),
            "percent": round(self.percent, 1),
            "targetDir": self.target_dir,
            "error": self.error,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "attempts": list(self.attempts),
            "done": self.done,
            "extras": [],
        }


_JOB = _Job()
_JOB_LOCK = threading.Lock()


class InstallError(RuntimeError):
    """A refusal or failure the user can read."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, dest: Path) -> None:
    """Unpack a ``.tar.gz`` or ``.zip`` into ``dest``, refusing any member that leaves it."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    def checked(name: str) -> Path:
        target = (root / name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise InstallError(f"The archive holds a path outside its folder ({name}); refusing to unpack it.") from exc
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                checked(member.filename)
            bundle.extractall(root)
        return
    try:
        with tarfile.open(archive, "r:*") as bundle:
            for member in bundle.getmembers():
                checked(member.name)
                if member.issym() or member.islnk():
                    link = (Path(member.name).parent / member.linkname).as_posix()
                    checked(link if not os.path.isabs(member.linkname) else member.linkname)
                if member.isdev():
                    raise InstallError(f"The archive holds a device file ({member.name}); refusing to unpack it.")
            bundle.extractall(root, filter="data") if hasattr(tarfile, "data_filter") else bundle.extractall(root)  # noqa: S202 - members checked above
    except tarfile.TarError as exc:
        raise InstallError(f"The archive could not be read: {exc}") from exc


def _download(url: str, target: Path, progress: Callable[[float], None]) -> None:
    from backend_service.helpers.hub_http import ssl_context

    request = urllib.request.Request(url, headers={"User-Agent": "chaosengine-installer"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS, context=ssl_context()) as response, target.open("wb") as out:  # noqa: S310 - https only, checked by the caller
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                progress(done / total)


def _fetch_text(url: str) -> str:
    from backend_service.helpers.hub_http import ssl_context

    request = urllib.request.Request(url, headers={"User-Agent": "chaosengine-installer"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS, context=ssl_context()) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def record_install(*, version: str, binary: Path, platform: str, sha256: str | None, source: str) -> None:
    root = native.install_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "binary": binary.resolve().relative_to(root.resolve()).as_posix(),
        "platform": platform,
        "sha256": sha256,
        "source": source,
        "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    temp = native.install_record_path().with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(native.install_record_path())


def finish_install(binary: Path, *, platform: str, sha256: str | None, source: str, fallback_version: str) -> native.NativeCapabilities:
    """Make the binary executable, check it answers, record it. Raises ``InstallError``."""
    if os.name != "nt":
        binary.chmod(binary.stat().st_mode | 0o111)
    caps = native.probe_capabilities(binary)
    if caps is None:
        raise InstallError(
            f"{binary.name} was unpacked but did not answer 'capabilities --json'. It may need a newer driver "
            "or a different CPU architecture; nothing was recorded."
        )
    record_install(version=caps.version or fallback_version, binary=binary, platform=platform, sha256=sha256, source=source)
    return caps


def adopt_binary(source: Path, *, platform: str) -> tuple[Path, native.NativeCapabilities]:
    """Copy a binary the user built from source into the data folder and record it."""
    if source.name not in ("tensorfold-native", "tensorfold-native.exe") or not source.is_file():
        raise InstallError("Pick the tensorfold-native executable (tensorfold-native.exe on Windows).")
    digest = sha256_file(source)
    probe = native.probe_capabilities(source)
    if probe is None:
        raise InstallError(f"{source} did not answer 'capabilities --json'; it is not a usable TensorFold binary.")
    version = probe.version or "local"
    dest_dir = native.install_root() / f"{version}-{digest[:8]}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    shutil.copy2(source, dest)
    if sha256_file(dest) != digest:
        raise InstallError("The copied binary does not match the original; check the disk and try again.")
    _copy_grammar_library(source, dest_dir)
    caps = finish_install(dest, platform=platform, sha256=digest, source=f"adopted:{source}", fallback_version=version)
    return dest, caps


def _copy_grammar_library(source: Path, dest_dir: Path) -> None:
    """Bring the structured-output library a source build keeps beside or above ``bin`` along with the binary.

    ``zig build grammar`` installs ``tfgrammar`` to ``native/lib``; the server opens ``../lib`` and then the
    folder of the executable, and the data folder holds the executable alone, so the library is copied beside it.
    """
    for root in (source.parent, source.parent / "lib", source.parent.parent / "lib"):
        if not root.is_dir():
            continue
        for found in root.glob("*tfgrammar*"):
            if found.is_file() and found.suffix.lower() in (".dll", ".so", ".dylib"):
                shutil.copy2(found, dest_dir / found.name)
                return


def install_from_asset(asset: native.ReleaseAsset, progress: Callable[[str, float, str], None]) -> tuple[Path, native.NativeCapabilities]:
    if not asset.url.startswith("https://"):
        raise InstallError("Refusing a non-https download URL.")
    root = native.install_root()
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="download-", dir=root))
    try:
        archive = work / asset.name
        progress("downloading", 0.0, f"Downloading {asset.name}")
        _download(asset.url, archive, lambda frac: progress("downloading", frac * 70.0, f"Downloading {asset.name}"))
        progress("verifying", 72.0, "Checking sha256")
        expected = asset.sha256
        if expected is None and asset.sha256_url:
            expected = native.parse_sha256_file(_fetch_text(asset.sha256_url), asset.name)
        if not expected:
            raise InstallError("No sha256 is published for this archive; refusing an unverified download.")
        actual = sha256_file(archive)
        if actual != expected:
            raise InstallError(f"sha256 mismatch for {asset.name}: expected {expected}, got {actual}. The download was discarded.")
        progress("unpacking", 80.0, "Unpacking")
        version_dir = root / f"{native.NATIVE_VERSION}-{asset.platform}"
        if version_dir.exists():
            shutil.rmtree(version_dir)
        safe_extract(archive, version_dir)
        binary = native.locate_binary_in(version_dir)
        if binary is None:
            raise InstallError("The archive holds no tensorfold-native binary.")
        progress("checking", 92.0, "Checking the binary")
        caps = finish_install(binary, platform=asset.platform, sha256=actual, source=asset.url, fallback_version=native.NATIVE_VERSION)
        return binary, caps
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _refresh_capabilities() -> None:
    try:
        from backend_service.inference.capabilities import refresh_install_detection

        refresh_install_detection()
    except Exception:  # noqa: BLE001 - best effort; the next probe picks it up
        pass


def _worker(binary_path: str | None) -> None:
    job = _JOB

    def progress(phase: str, percent: float, message: str) -> None:
        job.phase, job.percent, job.message = phase, percent, message

    try:
        key = native.platform_key()
        if key is None:
            raise InstallError("TensorFold ships no build for this operating system or CPU.")
        gpu = native.read_gpu()
        gate = native.evaluate_gate(gpu, None, key=key, asset_available=True, binary_present=True)
        if not gate.offered:
            raise InstallError(gate.message)
        if binary_path:
            progress("checking", 50.0, "Checking the binary")
            binary, caps = adopt_binary(Path(binary_path).expanduser(), platform=key)
        else:
            asset = native.release_asset(key)
            if asset is None:
                raise InstallError(
                    "TensorFold publishes no Windows build. Build tensorfold-native from source for your card and choose "
                    f"the executable here, or set {native.WINDOWS_ARCHIVE_URL_ENV} and {native.WINDOWS_ARCHIVE_SHA_ENV} "
                    "to a self-hosted archive. For structured output also run `zig build grammar -Dxgrammar=DIR` (xgrammar v0.2.8); "
                    "tfgrammar.dll is picked up from the build's native/lib folder."
                )
            binary, caps = install_from_asset(asset, progress)
        post = native.evaluate_gate(gpu, caps, key=key, asset_available=True, binary_present=True)
        job.attempts.append({"phase": "checking", "package": "gate", "ok": post.offered, "output": post.message})
        if not post.offered:
            raise InstallError(post.message)
        job.phase, job.percent, job.done = "done", 100.0, True
        job.message = f"TensorFold {caps.version or ''} ready at {binary.parent}".replace("  ", " ")
        _refresh_capabilities()
    except InstallError as exc:
        job.phase, job.error, job.done = "error", str(exc), True
    except Exception as exc:  # noqa: BLE001 - a network or disk error becomes the job's message
        job.phase, job.error, job.done = "error", f"{type(exc).__name__}: {exc}", True
    finally:
        job.finished_at = time.time()


def start_install(binary_path: str | None = None) -> dict[str, Any]:
    with _JOB_LOCK:
        if _JOB.phase not in ("idle", "done", "error"):
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        _JOB.phase = "preflight"
        _JOB.message = "Checking this machine"
        _JOB.started_at = time.time()
        _JOB.target_dir = str(native.install_root())
    threading.Thread(target=_worker, args=(binary_path,), daemon=True).start()
    return _JOB.to_dict()


def job_state() -> dict[str, Any]:
    return _JOB.to_dict()


def native_status() -> dict[str, Any]:
    """The capability gate plus registry, for the Setup page."""
    key = native.platform_key()
    binary = native.find_binary()
    caps = native.probe_capabilities(binary) if binary else None
    gpu = native.read_gpu() if key != "macos-arm64" else None
    asset = native.release_asset(key)
    smallest = min((native.estimate_vram_gb(m, m.ctx_ref, drafts=False) for m in native.NATIVE_MODELS), default=None)
    gate = native.evaluate_gate(
        gpu, caps, key=key, asset_available=asset is not None, binary_present=binary is not None, smallest_model_gb=smallest
    )
    record = native.read_install_record() or {}
    free_gb = gpu.free_mb / 1000.0 if gpu else None
    return {
        "installed": binary is not None,
        "version": record.get("version") or (caps.version if caps else None),
        "binary": str(binary) if binary else None,
        "platform": key,
        "archive": asset.name if asset else None,
        "archiveAvailable": asset is not None,
        "gate": gate.to_dict(),
        "gpu": (
            {
                "name": gpu.name,
                "computeCapability": gpu.compute_capability,
                "driver": gpu.driver_version,
                "cuda": gpu.cuda_version,
                "totalMb": gpu.total_mb,
                "freeMb": gpu.free_mb,
            }
            if gpu
            else None
        ),
        "chip": caps.chip if caps else None,
        "backends": list(caps.backends) if caps else [],
        "grammar": bool(binary) and _grammar_present(binary),
        "models": native.supported_models(caps, free_gb=free_gb),
        "needsBinaryPath": key == "windows-x86_64" and asset is None,
    }


def _grammar_present(binary: Path) -> bool:
    from backend_service.inference.tensorfold_native_engine import grammar_library_present

    return grammar_library_present(binary)


class InstallNativeRequest(BaseModel):
    # A tensorfold-native executable to adopt (the Windows source-build path).
    binaryPath: str | None = None


@router.get("/api/setup/tensorfold-native-status")
def tensorfold_native_status() -> dict[str, Any]:
    return native_status()


@router.post("/api/setup/install-tensorfold-native")
def start_tensorfold_native_install(body: InstallNativeRequest | None = None) -> dict[str, Any]:
    path = (body.binaryPath or "").strip() if body else ""
    if path and not Path(path).expanduser().is_file():
        raise HTTPException(status_code=400, detail=f"No such file: {path}")
    return start_install(path or None)


@router.get("/api/setup/install-tensorfold-native/status")
def tensorfold_native_install_status() -> dict[str, Any]:
    return job_state()
