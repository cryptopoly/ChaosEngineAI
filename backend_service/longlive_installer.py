"""Python port of the LongLive installer.

Replaces the legacy ``scripts/install-longlive.sh`` so Windows sidecars
can run the install without needing Git Bash or WSL. macOS is still
rejected up front — LongLive is CUDA-only, so even on Darwin the bash
script did nothing but exit 2. Linux and Windows both drive through the
same Python path now.

Invocable two ways:
    * In-process: ``from backend_service.longlive_installer import install``
    * As a module: ``python -m backend_service.longlive_installer`` (used
      by ``/api/setup/install-system-package`` so the long-running install
      stays out of the sidecar process).
"""

from __future__ import annotations

import datetime
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from backend_service.longlive_engine import resolve_install


LONGLIVE_REPO = "https://github.com/NVlabs/LongLive.git"
LONGLIVE_REF = os.environ.get("LONGLIVE_REF", "main")
LONGLIVE_HF_REPO = "Efficient-Large-Model/LongLive-1.3B"
WAN_HF_REPO = "Wan-AI/Wan2.1-T2V-1.3B"

# Requirements we strip from upstream's ``requirements.txt`` before handing
# it to pip. These are TensorRT / ONNX export pipeline deps used by the
# upstream demo's deployment scripts — none of them are needed for inference
# via our subprocess wrapper, and they actively break Windows installs:
#
# - ``nvidia-pyindex`` + ``nvidia-tensorrt``: the latter is a placeholder on
#   PyPI that errors out (``RuntimeError: This package is hosted on NVIDIA
#   Python Package Index``). Even if you set up nvidia-pyindex first, the
#   real ``nvidia-tensorrt`` ships only a Linux wheel — no Windows support.
#   Pip resolves all deps before installing any, so the placeholder error
#   aborts the entire install before a single useful package downloads.
#   This is the root cause of every "LongLive install hangs" report on
#   Windows.
#
# - ``pycuda``: needs CUDA toolkit headers + a working host C++ compiler
#   (Visual Studio Build Tools on Windows). Most users don't have those,
#   and LongLive doesn't import it from any inference path.
#
# Stripping these turns a guaranteed-fail install into a working one. We
# log the dropped lines so the user can see what was filtered and we don't
# silently mask a future upstream change.
_DROPPED_REQUIREMENTS: tuple[str, ...] = (
    "nvidia-pyindex",
    "nvidia-tensorrt",
    "pycuda",
)


class LongLiveInstallError(RuntimeError):
    """Raised when the install cannot proceed (missing git, macOS, etc.)."""


def _filter_requirements(source: Path, target: Path) -> list[str]:
    """Copy ``source`` to ``target`` minus the lines naming a dropped req.

    Returns the list of dropped lines (with their original whitespace) so
    the installer can log what was filtered. Comparison ignores version
    specifiers and inline comments — ``nvidia-tensorrt>=8.6 ; sys_platform``
    matches the bare ``nvidia-tensorrt`` entry in ``_DROPPED_REQUIREMENTS``.
    """
    dropped: list[str] = []
    kept: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            kept.append(raw_line)
            continue
        # Strip extras and version specifiers: ``foo[bar]==1.2 ; cond`` -> ``foo``
        name = line.split(";", 1)[0].split("[", 1)[0]
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
            if sep in name:
                name = name.split(sep, 1)[0]
                break
        name = name.strip().lower()
        if name in _DROPPED_REQUIREMENTS:
            dropped.append(raw_line)
            continue
        kept.append(raw_line)
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dropped


def _venv_python_path(venv_dir: Path) -> Path:
    # Windows venvs use ``Scripts/python.exe``; POSIX uses ``bin/python``.
    # The native path layout matters because we hand this path straight
    # to ``subprocess`` without a shell.
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# Ordered list of user-visible install phases. The async job worker uses
# this to drive a progress counter; the CLI install path ignores it.
# Adding a phase? Update both the order here AND the matching
# ``progress(...)`` calls inside ``install()`` so the percentages match.
INSTALL_PHASES: tuple[str, ...] = (
    "clone",            # git clone or update
    "venv",             # python -m venv
    "pip-upgrade",      # pip / setuptools / wheel
    "pip-requirements", # filtered requirements.txt
    "flash-attn",       # optional, may skip
    "pip-hub",          # huggingface-hub
    "weights-longlive", # snapshot of LongLive checkpoints (~5 GB)
    "weights-wan",      # snapshot of Wan2.1 T2V base (~3 GB)
    "marker",           # ready.marker (last)
)


def _noop_progress(_event: dict[str, object]) -> None:
    """Default progress sink for the CLI path.

    The async job in ``setup.py`` overrides this with a writer that
    updates ``_LONGLIVE_JOB``. Out-of-band stdout/stderr still goes
    through the ``logger`` callback unchanged.
    """


def _run(
    cmd: Sequence[str | Path],
    logger: Callable[[str], None],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(str(c) for c in cmd)
    logger(f"$ {printable}")
    result = subprocess.run(
        [str(c) for c in cmd],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        logger(result.stdout.rstrip())
    if check and result.returncode != 0:
        raise LongLiveInstallError(
            f"Command failed with exit code {result.returncode}: {printable}"
        )
    return result


def _snapshot_download(
    venv_python: Path,
    repo_id: str,
    local_dir: Path,
    logger: Callable[[str], None],
) -> None:
    # Run snapshot_download inside the install's own venv so the freshly
    # installed huggingface_hub is the one doing the fetch — the sidecar
    # Python might be on a different, older hub version.
    script = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download("
        f"repo_id={repo_id!r}, "
        f"local_dir={str(local_dir)!r}, "
        "local_dir_use_symlinks=False)"
    )
    _run([venv_python, "-c", script], logger)


def install(
    root: Path | None = None,
    logger: Callable[[str], None] = print,
    python_executable: str | None = None,
    progress: Callable[[dict[str, object]], None] = _noop_progress,
) -> None:
    """Run the full LongLive install into ``root`` (or the default path).

    Raises ``LongLiveInstallError`` on missing prereqs, clone/pip failures,
    or an HF download error. flash-attn is best-effort — its failure is
    logged as a warning but does not abort the install.

    ``progress`` receives structured phase events for the async job
    worker. Each event is a ``dict`` with at least ``{"phase": <name>,
    "ok": bool}`` and optionally ``output`` for terminal display. The
    CLI path uses a no-op sink — only the human-readable ``logger``
    output is shown there.
    """
    system = platform.system()
    if system == "Darwin":
        raise LongLiveInstallError(
            "LongLive requires CUDA; macOS is not supported. "
            "Install on a Windows or Linux machine with a recent NVIDIA GPU."
        )

    if shutil.which("git") is None:
        raise LongLiveInstallError(
            "git not found on PATH. Install git and retry."
        )

    host_python = python_executable or sys.executable

    info = resolve_install(root)
    target = info.root
    repo_dir = info.repo_dir
    venv_dir = target / "venv"
    weights_dir = info.weights_dir
    wan_dir = info.wan_base_dir

    target.mkdir(parents=True, exist_ok=True)
    logger(f"==> LongLive install target: {target}")

    # ── Phase: clone / update repo ──────────────────────────────────
    if (repo_dir / ".git").is_dir():
        logger("==> updating existing checkout")
        try:
            _run(["git", "-C", repo_dir, "fetch", "--all", "--prune"], logger)
            _run(["git", "-C", repo_dir, "checkout", LONGLIVE_REF], logger)
            _run(
                ["git", "-C", repo_dir, "reset", "--hard", f"origin/{LONGLIVE_REF}"],
                logger,
            )
        except LongLiveInstallError:
            progress({"phase": "clone", "ok": False})
            raise
    else:
        logger(f"==> cloning {LONGLIVE_REPO} ({LONGLIVE_REF})")
        try:
            _run(
                [
                    "git", "clone", "--depth", "1",
                    "--branch", LONGLIVE_REF,
                    LONGLIVE_REPO, repo_dir,
                ],
                logger,
            )
        except LongLiveInstallError:
            progress({"phase": "clone", "ok": False})
            raise
    progress({"phase": "clone", "ok": True})

    # ── Phase: venv ─────────────────────────────────────────────────
    if not venv_dir.is_dir():
        logger(f"==> creating venv at {venv_dir}")
        try:
            _run([host_python, "-m", "venv", venv_dir], logger)
        except LongLiveInstallError:
            progress({"phase": "venv", "ok": False})
            raise

    venv_python = _venv_python_path(venv_dir)
    if not venv_python.exists():
        progress({"phase": "venv", "ok": False})
        raise LongLiveInstallError(
            f"venv python not found at {venv_python} after ``-m venv`` — "
            "the host interpreter may be an embeddable distribution without "
            "venv support. Install a full Python 3.10+ and retry."
        )
    progress({"phase": "venv", "ok": True})

    # ── Phase: pip-upgrade ──────────────────────────────────────────
    logger("==> upgrading pip")
    try:
        _run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            logger,
        )
    except LongLiveInstallError:
        progress({"phase": "pip-upgrade", "ok": False})
        raise
    progress({"phase": "pip-upgrade", "ok": True})

    # ── Phase: pip-requirements (filtered) ──────────────────────────
    logger("==> installing LongLive requirements")
    filtered_requirements = target / "requirements.filtered.txt"
    dropped = _filter_requirements(repo_dir / "requirements.txt", filtered_requirements)
    if dropped:
        logger(
            "==> dropped TensorRT/pycuda export deps (not needed for inference, "
            "would break Windows install): " + ", ".join(d.strip() for d in dropped)
        )
    try:
        _run(
            [venv_python, "-m", "pip", "install", "-r", filtered_requirements],
            logger,
        )
    except LongLiveInstallError:
        progress({"phase": "pip-requirements", "ok": False})
        raise
    progress({"phase": "pip-requirements", "ok": True})

    # ── Phase: flash-attn (optional) ────────────────────────────────
    # flash-attn requires CUDA headers + a long build, and LongLive falls
    # back to slower attention kernels when it's missing. Probe first so
    # we don't reinstall on re-runs.
    probe = subprocess.run(
        [str(venv_python), "-c", "import flash_attn"],
        capture_output=True,
    )
    if probe.returncode != 0:
        logger("==> installing flash-attn (optional, may take several minutes)")
        attn = _run(
            [venv_python, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            logger,
            check=False,
        )
        if attn.returncode != 0:
            logger("warning: flash-attn install failed — LongLive will run but slower")
            progress({"phase": "flash-attn", "ok": False, "skipped": True})
        else:
            progress({"phase": "flash-attn", "ok": True})
    else:
        logger("==> flash-attn already importable, skipping build")
        progress({"phase": "flash-attn", "ok": True, "skipped": True})

    # ── Phase: pip-hub ──────────────────────────────────────────────
    try:
        _run([venv_python, "-m", "pip", "install", "huggingface-hub"], logger)
    except LongLiveInstallError:
        progress({"phase": "pip-hub", "ok": False})
        raise
    progress({"phase": "pip-hub", "ok": True})

    # ── Phase: weights-longlive ─────────────────────────────────────
    weights_dir.mkdir(parents=True, exist_ok=True)
    logger(f"==> downloading LongLive checkpoints from {LONGLIVE_HF_REPO}")
    try:
        _snapshot_download(venv_python, LONGLIVE_HF_REPO, weights_dir, logger)
    except LongLiveInstallError:
        progress({"phase": "weights-longlive", "ok": False})
        raise
    progress({"phase": "weights-longlive", "ok": True})

    # ── Phase: weights-wan ──────────────────────────────────────────
    wan_dir.mkdir(parents=True, exist_ok=True)
    logger(f"==> downloading Wan 2.1 T2V 1.3B base from {WAN_HF_REPO}")
    try:
        _snapshot_download(venv_python, WAN_HF_REPO, wan_dir, logger)
    except LongLiveInstallError:
        progress({"phase": "weights-wan", "ok": False})
        raise
    progress({"phase": "weights-wan", "ok": True})

    # ── Phase: marker ───────────────────────────────────────────────
    # The marker is what ``LongLiveInstallInfo.ready`` keys off, so write
    # it last — a crash midway through leaves ``ready`` as False and the
    # Studio will re-prompt for install rather than silently mis-running.
    commit = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    marker_text = (
        f"repo_commit={commit}\n"
        f"repo_ref={LONGLIVE_REF}\n"
        f"installed_at={datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"longlive_root={target}\n"
    )
    info.marker.write_text(marker_text, encoding="utf-8")
    progress({"phase": "marker", "ok": True})

    logger("")
    logger("==> LongLive install complete")
    logger(f"Repo:     {repo_dir}")
    logger(f"Venv:     {venv_dir}")
    logger(f"Weights:  {weights_dir}")
    logger(f"Wan base: {wan_dir}")
    logger(f"Marker:   {info.marker}")


_USAGE = (
    "usage: python -m backend_service.longlive_installer [-h|--help]\n"
    "\n"
    "Installs NVlabs/LongLive into an isolated venv at\n"
    "$CHAOSENGINE_LONGLIVE_ROOT (default ~/.chaosengine/longlive).\n"
    "CUDA only — macOS is rejected. Takes 5–15 minutes.\n"
    "\n"
    "Called by the /api/setup/install-system-package route when the\n"
    "Studio or Video Discover install button is clicked.\n"
)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if any(a in ("-h", "--help") for a in args):
        print(_USAGE)
        return 0
    if args:
        print(f"error: unexpected arguments: {args}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        install()
    except LongLiveInstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface the raw error to the user
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
