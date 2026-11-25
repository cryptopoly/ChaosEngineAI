"""The quality upscale add-on (FlashVSR+ Turbo): its own venv, the pinned upstream code and the Turbo weights.

Opt-in (D0145), never at startup: only when the user presses Install in the Upscale tab
(``routes/setup/video_upscale_quality.py``). NVIDIA on Windows or Linux x86_64 only; elsewhere nothing is created or
downloaded and the reason is returned. Layout under ``~/.chaosengine/addons/video-upscale-quality``:

* ``venv/``: Python 3.12 (the backend's own interpreter, through ``voice_env._base_python`` so a packaged app's venv
  survives an update), torch 2.7.1+cu128, ``triton-windows`` 3.3.1.post21 on Windows (the official triton comes with
  torch on Linux) and the packages pinned in ``quality_requirements.txt``, wheels only.
* ``code/FlashVSR_Plus/``: the upstream repository at ``FLASHVSR_REF``, from GitHub's commit archive. The archive's
  bytes are not stable across GitHub's zip builds, so the check is a digest of the unpacked tree (``tree_digest``:
  every member's path and SHA-256), pinned in ``CODE_TREE_SHA256``. Members that would land outside the folder are
  refused before anything is written.
* ``weights/``: the Turbo files of the catalogue row (``catalog/video_upscale_models.py``), from the Hugging Face repo
  at its pinned revision, each through a ``.part`` file that resumes after a cancel and is kept only when its size
  and SHA-256 match the row.
* ``addon.json``: the marker, written last, after an import check in the venv that also asks torch for the GPU.

The upstream code is read, not run, at install (``pyproject`` has no dependencies and no build hooks; every
``torch.load`` passes ``weights_only=True``; spike section 2). Licences: FlashVSR+ code and weights Apache-2.0,
Wan2.1 Apache-2.0, triton-windows MIT, torch BSD-3-Clause.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from backend_service.catalog.video_upscale_models import QUALITY_MODEL_ID, find_quality_model, hf_file_url

FLASHVSR_REPO = "https://github.com/OpenImagingLab/FlashVSR_Plus"
FLASHVSR_REF = "b515ffc6dba57fdc9ad84c66cadd87d05f27590a"
CODE_ARCHIVE = f"{FLASHVSR_REPO}/archive/{FLASHVSR_REF}.zip"
# ``tree_digest`` of the archive's members (74 files, 12,993,743 bytes, computed 2026-11-25 from the commit archive).
CODE_TREE_SHA256 = "45e456a8fa68173c523ca7d156eb896557eef113ba7393c908178c084fc136f9"
CODE_ARCHIVE_MAX_BYTES = 64 * 1024 * 1024  # the archive is 8.9 MB
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION = "2.7.1+cu128"
REQUIREMENTS = Path(__file__).with_name("quality_requirements.txt")
ENGINE_VERSION = f"flashvsr-plus {FLASHVSR_REF[:7]}"
_SCHEMA = 1
# Downloads: the torch wheel 3.3 GB (Linux also pulls about 3 GB of NVIDIA libraries), other wheels 0.1 GB, weights
# 7.0 GB. Installed: the venv is about 6 GB on Windows. Free space asked for before anything starts:
DOWNLOAD_GB = {"Windows": 10.4, "Linux": 13.5}
MIN_FREE_GB = 20.0
_PIP_TIMEOUT_S = 3600
_CHECK_TIMEOUT_S = 600

_CHECK_CODE = (
    "import json, sys\n"
    "code = sys.argv[1]\n"
    "sys.path.insert(0, code); sys.path.insert(0, code + '/operators/flashvsr_attn')\n"
    "import torch, triton, numpy, cv2, einops, imageio, imageio_ffmpeg, tqdm, PIL\n"
    "from safetensors.torch import load_file\n"
    "import flashvsr_attn\n"
    "assert torch.__version__ == {torch!r}, 'torch ' + torch.__version__\n"
    "assert torch.cuda.is_available(), 'torch ' + torch.__version__ + ' sees no CUDA GPU'\n"
    "from inference import InferenceRequest, build_inference_pipeline\n"
    "from inference.loading import ModelPaths\n"
    "from inference.presets import stream_config_for_mode\n"
    "print(json.dumps({{'torch': torch.__version__, 'cuda': torch.version.cuda, 'triton': triton.__version__, "
    "'gpu': torch.cuda.get_device_name(0), 'python': '.'.join(map(str, sys.version_info[:3]))}}))\n"
).format(torch=TORCH_VERSION)


# ---------------------------------------------------------------------------
# where things live
# ---------------------------------------------------------------------------


def addon_dir() -> Path:
    return Path.home() / ".chaosengine" / "addons" / "video-upscale-quality"


def venv_dir() -> Path:
    return addon_dir() / "venv"


def venv_python() -> Path:
    if os.name == "nt":
        return venv_dir() / "Scripts" / "python.exe"
    return venv_dir() / "bin" / "python"


def code_dir() -> Path:
    return addon_dir() / "code" / "FlashVSR_Plus"


def weights_dir() -> Path:
    return addon_dir() / "weights"


def _marker_path() -> Path:
    return addon_dir() / "addon.json"


def row() -> dict[str, Any]:
    found = find_quality_model(QUALITY_MODEL_ID)
    assert found is not None
    return found


def weight_path(entry: dict[str, Any]) -> Path:
    """Where one of the row's files goes; the catalogue path must stay inside ``weights/``."""
    rel = PurePosixPath(str(entry["path"]))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Refused weight path {entry['path']!r}")
    return weights_dir().joinpath(*rel.parts)


def model_paths() -> dict[str, str]:
    """The four files the pipeline loads (``inference.loading.ModelPaths`` fields)."""
    return {"dit": str(weights_dir() / "turbo" / "dit.safetensors"),
            "lq_projection": str(weights_dir() / "turbo" / "lq_proj_in.safetensors"),
            "context": str(code_dir() / "prompt_tensor" / "posi_prompt.pth"),
            "tiny_decoder": str(weights_dir() / "TCDecoder.ckpt")}


# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------


def _nvidia_present() -> bool:
    from backend_service.helpers.gpu import nvidia_gpu_present  # noqa: PLC0415

    return nvidia_gpu_present()


def lane(system: str | None = None, machine: str | None = None, nvidia: bool | None = None) -> str:
    """``cuda`` on Windows or Linux x86_64 with an NVIDIA GPU, else ``unsupported`` (Macs, CPU-only, Linux aarch64:
    torch's cu128 Windows and x86_64 wheels are the ones tested)."""
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system in ("Windows", "Linux") and machine in ("amd64", "x86_64"):
        return "cuda" if (_nvidia_present() if nvidia is None else nvidia) else "unsupported"
    return "unsupported"


UNSUPPORTED_REASON = ("The quality upscaler needs an NVIDIA graphics card on Windows or Linux; nothing is installed or "
                      "downloaded on this computer.")


def unavailable_reason() -> str | None:
    return None if lane() == "cuda" else UNSUPPORTED_REASON


def _python_supported(version: tuple[int, ...] | None = None) -> bool:
    return tuple((version or sys.version_info)[:2]) == (3, 12)


# ---------------------------------------------------------------------------
# the code archive
# ---------------------------------------------------------------------------


def _member_parts(name: str, top: str) -> tuple[str, ...] | None:
    """The path of an archive member below the top folder, or None for the folder itself. Raises on anything that
    could land outside the target (absolute, drive letters, ``..``, backslashes, another top folder)."""
    if "\\" in name or ":" in name:
        raise ValueError(f"Refused archive member {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != top:
        raise ValueError(f"Refused archive member {name!r}")
    return tuple(path.parts[1:]) or None


def _top_folder() -> str:
    return f"FlashVSR_Plus-{FLASHVSR_REF}"


def tree_digest(files: dict[str, bytes] | list[tuple[str, str]]) -> str:
    """SHA-256 over ``"<path>\\0<sha256 of the content>\\n"`` for every file, sorted by path. Takes the contents by
    path, or ``(path, sha256)`` pairs."""
    pairs = sorted((path, hashlib.sha256(data).hexdigest()) for path, data in files.items()) if isinstance(
        files, dict) else sorted(files)
    digest = hashlib.sha256()
    for path, sha in pairs:
        digest.update(f"{path}\0{sha}\n".encode())
    return digest.hexdigest()


def archive_members(data: bytes) -> dict[str, bytes]:
    """The files of the commit archive by their path below the top folder; raises ``ValueError`` on a refused one."""
    top = _top_folder()
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            parts = _member_parts(info.filename, top)
            if parts is None or info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:  # a symlink member: never written as a link
                raise ValueError(f"Refused archive member {info.filename!r} (a link)")
            out["/".join(parts)] = archive.read(info)
    return out


def unpack_code(data: bytes, expected: str = CODE_TREE_SHA256) -> None:
    """Check the archive's tree digest, then write it to ``code/FlashVSR_Plus`` (via a ``.partial`` folder)."""
    members = archive_members(data)
    seen = tree_digest(members)
    if seen != expected:
        raise RuntimeError(f"The FlashVSR+ code archive does not match its pinned digest (got {seen[:12]}…); "
                           "nothing was kept.")
    target = code_dir()
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    for rel, content in members.items():
        path = partial.joinpath(*rel.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    (target / ".chaosengine-ref").write_text(FLASHVSR_REF, encoding="utf-8")


def code_installed() -> bool:
    try:
        return (code_dir() / ".chaosengine-ref").read_text(encoding="utf-8").strip() == FLASHVSR_REF
    except OSError:
        return False


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


class InstallCancelled(Exception):
    pass


def _hash_file(path: Path, digest: Any, cancel: threading.Event | None = None) -> int:
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            if cancel is not None and cancel.is_set():
                raise InstallCancelled("Install cancelled.")
            digest.update(block)
            size += len(block)
    return size


def download_file(url: str, target: Path, size: int, sha256: str, *, opener: Callable[..., Any] | None = None,
                  on_bytes: Callable[[int], None] | None = None, cancel: threading.Event | None = None) -> None:
    """Fetch ``url`` to ``target`` through ``target.part``, resuming a part left by a cancel (HTTP Range). The file is
    kept only when its byte count and SHA-256 match; otherwise the part is removed and ``RuntimeError`` raised."""
    import urllib.request  # noqa: PLC0415

    opener = opener or urllib.request.urlopen
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    have = 0
    if part.is_file() and 0 < part.stat().st_size < size:
        have = _hash_file(part, digest, cancel)
    else:
        part.unlink(missing_ok=True)
    if on_bytes is not None:
        on_bytes(have)
    request = urllib.request.Request(url, headers={"User-Agent": "chaosengine-addon", **(
        {"Range": f"bytes={have}-"} if have else {})})
    try:
        with opener(request, timeout=60) as response:
            if have and getattr(response, "status", 200) != 206:  # the server ignored the range: start again
                digest, have = hashlib.sha256(), 0
                part.unlink(missing_ok=True)
                if on_bytes is not None:
                    on_bytes(0)
            with part.open("ab") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    if cancel is not None and cancel.is_set():
                        raise InstallCancelled("Install cancelled.")
                    out.write(block)
                    digest.update(block)
                    have += len(block)
                    if on_bytes is not None:
                        on_bytes(have)
    except OSError as exc:  # the part stays for a resume
        raise RuntimeError(f"Could not download {target.name} ({exc}).") from exc
    if have != size or digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        what = f"{have} bytes, expected {size}" if have != size else "a different SHA-256 than the pinned one"
        raise RuntimeError(f"{target.name} came down as {what}; nothing was kept.")
    os.replace(part, target)


def _verified() -> set[str]:
    marker = read_marker() or {}
    return set(marker.get("weightsVerified") or []) | set(_read_json(addon_dir() / "weights.json").get("verified", []))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _remember_verified(sha256: str) -> None:
    path = addon_dir() / "weights.json"
    data = _read_json(path)
    data["verified"] = sorted(set(data.get("verified", [])) | {sha256})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def weight_present(entry: dict[str, Any]) -> bool:
    """The file is there at its size (cheap; the hash was checked when it was written)."""
    path = weight_path(entry)
    return path.is_file() and path.stat().st_size == int(entry["sizeBytes"])


def weights_installed() -> bool:
    return all(weight_present(entry) for entry in row()["files"])


def ensure_weight(entry: dict[str, Any], *, opener: Callable[..., Any] | None = None,
                  on_bytes: Callable[[int], None] | None = None, cancel: threading.Event | None = None) -> str:
    """Make one weight file present and verified: ``kept`` (already verified), ``checked`` (a file of the right size
    that was not verified yet, e.g. copied in by hand, hashed now) or ``downloaded``."""
    path = weight_path(entry)
    if weight_present(entry):
        if entry["sha256"] in _verified():
            if on_bytes is not None:
                on_bytes(int(entry["sizeBytes"]))
            return "kept"
        digest = hashlib.sha256()
        _hash_file(path, digest, cancel)
        if digest.hexdigest() == entry["sha256"]:
            _remember_verified(entry["sha256"])
            if on_bytes is not None:
                on_bytes(int(entry["sizeBytes"]))
            return "checked"
        path.unlink()  # same size, wrong content: fetch it again
    download_file(hf_file_url(row(), str(entry["path"])), path, int(entry["sizeBytes"]), str(entry["sha256"]),
                  opener=opener, on_bytes=on_bytes, cancel=cancel)
    _remember_verified(entry["sha256"])
    return "downloaded"


# ---------------------------------------------------------------------------
# the plan and the marker
# ---------------------------------------------------------------------------


def pip_command(python: str | None = None) -> list[str]:
    return [python or str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
            "-r", str(REQUIREMENTS), "--extra-index-url", TORCH_INDEX]


def check_command(python: str | None = None) -> list[str]:
    return [python or str(venv_python()), "-I", "-c", _CHECK_CODE, str(code_dir())]


def install_plan(base_python: str = "python") -> list[dict[str, Any]]:
    """What an install does, in order (the job runs these; tests read them)."""
    entries = row()["files"]
    return [
        {"phase": "venv", "command": [base_python, "-m", "venv", str(venv_dir())]},
        {"phase": "packages", "command": pip_command()},
        {"phase": "code", "url": CODE_ARCHIVE, "treeSha256": CODE_TREE_SHA256, "target": str(code_dir())},
        {"phase": "weights", "files": [{"url": hf_file_url(row(), e["path"]), "target": str(weight_path(e)),
                                        "sizeBytes": e["sizeBytes"], "sha256": e["sha256"]} for e in entries]},
        {"phase": "check", "command": check_command()},
    ]


def read_marker() -> dict[str, Any] | None:
    data = _read_json(_marker_path())
    return data or None


def installed() -> bool:
    """A verified install for this commit and these weights, whose venv still exists."""
    marker = read_marker()
    return bool(
        marker
        and marker.get("schema") == _SCHEMA
        and marker.get("ref") == FLASHVSR_REF
        and marker.get("torch") == TORCH_VERSION
        and marker.get("revision") == row()["revision"]
        and venv_python().exists()
        and code_installed()
        and weights_installed()
    )


def _write_marker(versions: dict[str, Any]) -> None:
    _marker_path().write_text(json.dumps({
        "schema": _SCHEMA, "ref": FLASHVSR_REF, "revision": row()["revision"], "model": QUALITY_MODEL_ID,
        "weightsVerified": sorted(e["sha256"] for e in row()["files"]),
        "installedAt": time.strftime("%Y-%m-%dT%H:%M:%S"), **{k: str(v) for k, v in versions.items()},
    }, indent=1), encoding="utf-8")


def disk_mb() -> float:
    root = addon_dir()
    if not root.is_dir():
        return 0.0
    return round(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) / 1e6, 1)


# ---------------------------------------------------------------------------
# the install job
# ---------------------------------------------------------------------------

PHASES = ("preflight", "venv", "packages", "code", "weights", "check")


@dataclass
class _Job:
    phase: str = "idle"  # idle | <PHASES> | done | error | cancelled
    error: str | None = None
    bytes_done: int = 0
    bytes_total: int = 0
    log: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    cancel: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None

    def running(self) -> bool:
        return self.phase not in ("idle", "done", "error", "cancelled")

    def to_dict(self) -> dict[str, Any]:
        index = PHASES.index(self.phase) + 1 if self.phase in PHASES else 0
        return {"phase": self.phase, "phaseIndex": index, "phaseTotal": len(PHASES), "error": self.error,
                "running": self.running(), "done": self.phase in ("done", "error", "cancelled"),
                "bytesDone": self.bytes_done, "bytesTotal": self.bytes_total, "log": self.log[-30:],
                "startedAt": self.started_at, "finishedAt": self.finished_at}


_JOB = _Job()
_LOCK = threading.Lock()


def job_status() -> dict[str, Any]:
    return _JOB.to_dict()


def status() -> dict[str, Any]:
    reason = unavailable_reason()
    item = row()
    return {"available": reason is None, "unavailableReason": reason, "installed": installed(),
            "folder": str(addon_dir()), "diskMb": disk_mb(), "ref": FLASHVSR_REF, "revision": item["revision"],
            "downloadGb": DOWNLOAD_GB.get(platform.system(), DOWNLOAD_GB["Linux"]), "weightsGb": item["sizeGb"],
            "minFreeGb": MIN_FREE_GB, "job": job_status()}


def start_install(runner: Callable[[], None] | None = None) -> dict[str, Any]:
    with _LOCK:
        if _JOB.running():
            return _JOB.to_dict()
        _JOB.__init__()  # type: ignore[misc]
        reason = unavailable_reason()
        if reason:  # not this computer: say why and start nothing (no venv, no thread, no download)
            _JOB.phase, _JOB.error, _JOB.finished_at = "error", reason, time.time()
            return _JOB.to_dict()
        _JOB.phase, _JOB.started_at = "preflight", time.time()
    threading.Thread(target=runner or _run_install, name="chaosengine-upscale-quality-install", daemon=True).start()
    return _JOB.to_dict()


def cancel_install() -> dict[str, Any]:
    _JOB.cancel.set()
    process = _JOB.process
    if process is not None and process.poll() is None:
        process.terminate()
    return _JOB.to_dict()


def _child_env() -> dict[str, str]:
    from backend_service.inference._utils import _isolated_child_env  # noqa: PLC0415

    return _isolated_child_env({"PYTHONUNBUFFERED": "1", "PIP_NO_INPUT": "1"})


def _run(command: list[str], timeout: float = _PIP_TIMEOUT_S) -> str:
    """One install command, its output streamed into the job log; returns the output."""
    if _JOB.cancel.is_set():
        raise InstallCancelled("Install cancelled.")
    _JOB.log.append("$ " + " ".join("<code>" if "\n" in part else part for part in command[:8]))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="replace", env=_child_env())
    _JOB.process = process
    lines: list[str] = []
    deadline = time.time() + timeout
    for line in process.stdout or []:
        line = line.rstrip()
        if line:
            lines.append(line)
            _JOB.log.append(line[:300])
            del _JOB.log[:-200]
        if time.time() > deadline:
            process.kill()
    process.wait()
    _JOB.process = None
    if _JOB.cancel.is_set():
        raise InstallCancelled("Install cancelled.")
    if process.returncode != 0:
        tail = next((l for l in reversed(lines) if l.strip()), str(process.returncode))
        raise RuntimeError(f"`{' '.join(command[1:4])} …` failed: {tail[:240]}")
    return "\n".join(lines)


def _fetch_archive(url: str, opener: Callable[..., Any] | None = None) -> bytes:
    import urllib.request  # noqa: PLC0415

    opener = opener or urllib.request.urlopen
    try:
        with opener(urllib.request.Request(url, headers={"User-Agent": "chaosengine-addon"}), timeout=60) as response:
            data = response.read(CODE_ARCHIVE_MAX_BYTES + 1)
    except OSError as exc:
        raise RuntimeError(f"Could not download the FlashVSR+ code ({exc}).") from exc
    if len(data) > CODE_ARCHIVE_MAX_BYTES:
        raise RuntimeError("The FlashVSR+ code archive is far larger than expected; nothing was kept.")
    return data


def _versions_from(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict):
            return found
    raise RuntimeError("The quality upscaler check printed no versions.")


def _run_install(opener: Callable[..., Any] | None = None) -> None:
    try:
        reason = unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        if not _python_supported():
            raise RuntimeError("The quality upscaler needs Python 3.12 for its own environment; the app runs on "
                               f"{sys.version_info[0]}.{sys.version_info[1]}.")
        root = addon_dir()
        root.mkdir(parents=True, exist_ok=True)
        if not installed():
            free_gb = shutil.disk_usage(root).free / 1e9
            missing_gb = sum(int(e["sizeBytes"]) for e in row()["files"] if not weight_present(e)) / 1e9
            want = MIN_FREE_GB if not venv_python().exists() else missing_gb + 1.0
            if free_gb < want:
                raise RuntimeError(f"Only {free_gb:.1f} GB free in {root.anchor or root}; the quality upscaler "
                                   f"needs about {want:.0f} GB.")

        _JOB.phase = "venv"
        if venv_dir().exists() and not venv_python().exists():
            if not (venv_dir() / "pyvenv.cfg").is_file() and any(venv_dir().iterdir()):
                raise RuntimeError(f"{venv_dir()} is not a Python environment, so it was left alone.")
            shutil.rmtree(venv_dir(), ignore_errors=True)  # a broken earlier attempt
        if not venv_python().exists():
            from backend_service.voice_env import _base_python  # noqa: PLC0415

            _run([_base_python(), "-m", "venv", str(venv_dir())])

        _JOB.phase = "packages"
        _run(pip_command())  # pip skips what is already there, so a resumed install is quick

        _JOB.phase = "code"
        if not code_installed():
            unpack_code(_fetch_archive(CODE_ARCHIVE, opener))

        _JOB.phase = "weights"
        entries = row()["files"]
        _JOB.bytes_total = sum(int(e["sizeBytes"]) for e in entries)
        base = 0
        for entry in entries:
            def on_bytes(count: int, base: int = base) -> None:
                _JOB.bytes_done = base + count

            result = ensure_weight(entry, opener=opener, on_bytes=on_bytes, cancel=_JOB.cancel)
            _JOB.log.append(f"{entry['path']}: {result}")
            base += int(entry["sizeBytes"])
        _JOB.bytes_done = _JOB.bytes_total

        _JOB.phase = "check"
        _write_marker(_versions_from(_run(check_command(), timeout=_CHECK_TIMEOUT_S)))
        _JOB.phase = "done"
    except InstallCancelled as exc:
        _JOB.phase, _JOB.error = "cancelled", str(exc)
    except Exception as exc:  # noqa: BLE001 - shown to the user with the log
        _JOB.phase, _JOB.error = ("cancelled", "Install cancelled.") if _JOB.cancel.is_set() else ("error", str(exc))
    finally:
        _JOB.finished_at = time.time()
        _JOB.process = None


NOT_INSTALLED = ("The quality upscaler is not installed. Install it from the Upscale tab (quality tier), then try "
                 "again.")
