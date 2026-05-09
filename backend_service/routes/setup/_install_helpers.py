"""Shared pip-install helpers for the Setup routes.

The CUDA torch installer and the GPU bundle installer both walk the same
pip-target install dance (purge, write, verify) so the primitives live here
as a private module under ``backend_service.routes.setup``.

Public via the package ``__init__`` re-exports for backwards-compatibility
with tests that still patch ``backend_service.routes.setup._<name>`` paths.

Extracted from ``routes/setup/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend_service.runtime_paths import extras_site_packages


_CUDA_TORCH_INDEXES: list[str] = [
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu121",
    "https://download.pytorch.org/whl/nightly/cu128",
]


def _read_python_version(python: str) -> str | None:
    """Return e.g. ``3.13.2`` for the given Python interpreter, or ``None``."""
    try:
        result = subprocess.run(
            [python, "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _site_packages_for(python_executable: str) -> Path | None:
    """Return the site-packages directory for the given interpreter, or None."""
    try:
        result = subprocess.run(
            [
                python_executable, "-c",
                "import sysconfig; print(sysconfig.get_paths().get('purelib') or sysconfig.get_paths().get('platlib') or '')",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    path = (result.stdout or "").strip()
    return Path(path) if path else None


def _purge_broken_distributions(site_packages: Path) -> list[str]:
    """Delete ``~*`` stub directories pip leaves behind after an interrupted install.

    On Windows, pip atomically renames the old version of a package to ``~<name>``
    before unpacking the new one. If the process is killed mid-install (antivirus,
    a file lock, Ctrl-C) the stub is left behind. Subsequent ``pip install`` runs
    then print ``WARNING: Ignoring invalid distribution ~arkupsafe`` forever and
    sometimes refuse to heal the tree. Removing these stubs is cheap and safe —
    they contain no authoritative data.
    """
    if not site_packages.is_dir():
        return []
    removed: list[str] = []
    for entry in site_packages.iterdir():
        if not entry.name.startswith("~"):
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            if not entry.exists():
                removed.append(entry.name)
        except OSError:
            continue
    return removed


def _purge_stale_torch_from_extras(extras_dir: Path) -> list[str]:
    """Remove torch and its NVIDIA runtime deps from the extras dir.

    Reported failure mode: extras contained ``torch-2.6.0+cu124.dist-info``
    from an earlier CUDA install plus a ``torch-2.11.0+cpu`` folder from a
    later clobber. Python's importer couldn't resolve either cleanly, so
    ``import torch`` raised ``ModuleNotFoundError`` even though files were
    on disk. Wiping the family before a reinstall forces a known-clean
    slate.

    Matches by directory/file name prefix:
      - exactly ``torch`` (the package folder)
      - anything starting with ``torch-`` (dist-info, partial installs)
      - anything starting with ``nvidia_`` or ``nvidia-`` (CUDA runtime deps)

    Does NOT match sibling packages like ``torchvision`` or ``torchaudio`` —
    they start with ``torchv``/``torcha``, not ``torch-``, so the prefix
    check leaves them alone.
    """
    if not extras_dir.is_dir():
        return []
    removed: list[str] = []
    for entry in extras_dir.iterdir():
        name = entry.name
        lower = name.lower()
        is_torch = name == "torch" or lower.startswith("torch-")
        is_nvidia = lower.startswith("nvidia_") or lower.startswith("nvidia-")
        if not (is_torch or is_nvidia):
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            if not entry.exists():
                removed.append(name)
        except OSError:
            continue
    return removed


def _find_installed_torch_version(extras_dir: Path) -> str | None:
    """Return the torch version recorded in its dist-info METADATA, if any.

    Used after a successful CUDA torch install so we can pin torch in a
    constraints file for the subsequent gpu-bundle packages, preventing
    pip's resolver from silently swapping the CUDA wheel for a CPU one
    while installing diffusers/transformers/etc. from default PyPI.
    """
    if not extras_dir.is_dir():
        return None
    for entry in extras_dir.iterdir():
        lower = entry.name.lower()
        if not (lower.startswith("torch-") and lower.endswith(".dist-info")):
            continue
        metadata = entry / "METADATA"
        if not metadata.is_file():
            continue
        try:
            text = metadata.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip() or None
    return None


def _is_cuda_torch_version(torch_version: str | None) -> bool:
    """Return True for PyTorch wheels with a CUDA local-version tag."""
    return bool(torch_version and "+cu" in torch_version.lower())


def _write_torch_constraint(extras_dir: Path, torch_version: str) -> Path:
    """Pin torch in a constraints.txt so follow-up installs can't swap it.

    Without this pin, ``pip install diffusers --target extras/`` could let
    pip's resolver pull a newer torch from default PyPI (which ships only
    the CPU wheel) — silently replacing the CUDA wheel we just installed.

    The local-version segment (``+cu124``, ``+cpu``, ...) is stripped from
    the pin: ``torch==2.6.0`` matches the installed ``2.6.0+cu124`` per
    PEP 440 (a public-only specifier ignores local segments on candidates),
    but ``torch==2.6.0+cu124`` is unsatisfiable from default PyPI — no
    ``+cu124`` wheel exists there, so follow-up installs (accelerate,
    bitsandbytes, ...) bail with::

        ResolutionImpossible: ... accelerate depends on torch>=2.0.0 ...
        The user requested (constraint) torch==2.6.0+cu124

    even though the installed CUDA wheel obviously satisfies ``>=2.0.0``.
    """
    base_version = torch_version.split("+", 1)[0]
    path = extras_dir / ".chaosengine-torch-constraints.txt"
    path.write_text(f"torch=={base_version}\n", encoding="utf-8")
    return path


def _all_attempts_lack_wheel(attempts: list[dict[str, Any]]) -> bool:
    """True when pip reported 'No matching distribution' for every attempt.

    This is the signature of a Python version PyTorch doesn't ship wheels
    for (either too old or too new) — the fix is a different Python, not
    a different CUDA index. We surface that specifically to the UI so
    the user doesn't keep retrying.
    """
    if not attempts:
        return False
    for attempt in attempts:
        if attempt.get("ok"):
            return False
        text = (attempt.get("output") or "").lower()
        if "no matching distribution" not in text and "from versions: none" not in text:
            return False
    return True


def _extras_site_packages() -> Path | None:
    """Resolve the user-persistent extras site-packages dir.

    The Tauri shell sets ``CHAOSENGINE_EXTRAS_SITE_PACKAGES`` to a path
    namespaced by Python ``major.minor`` (e.g.
    ``~/Library/Application Support/ChaosEngineAI/extras/cp312/site-packages``)
    so wheels compiled against one Python ABI can't shadow a different
    interpreter on the next launch — that bit users in v0.7.0-rc.4 when
    a switch from cp311 to cp312 left a dead pydantic_core wheel in
    place. When the backend runs outside Tauri (``python -m backend_service``
    for dev / tests) we fall back to a predictable default that uses the
    *current* interpreter's tag.
    """
    return extras_site_packages()


def _run_pip_install(
    python: str,
    spec: str,
    target: Path,
    index_url: str | None,
    extra_flags: list[str],
) -> tuple[bool, str]:
    """Run ``pip install --target`` and return (ok, captured_output).

    Uses ``--upgrade`` so re-installs pick up newer versions and
    ``--target`` so we never touch the bundled site-packages (avoids the
    classic Windows WinError 5 from overwriting a loaded .pyd).

    Two defensive flags + an env tweak prevent the "CUDA torch silently
    swapped for CPU torch" failure mode that shipped in v0.7.2:

      * ``--upgrade-strategy=only-if-needed`` keeps pip from eagerly
        upgrading transitive deps. Without it, ``pip install accelerate
        --upgrade`` would consider torch a candidate for upgrade and pull
        the latest matching wheel from default PyPI — which is the CPU
        wheel, clobbering the CUDA wheel installed in step 1.

      * ``PYTHONPATH=<extras>`` on the pip child env lets pip's resolver
        see packages already installed in the extras tree. With ``--target``
        alone, pip only checks the bundled venv site-packages for "already
        installed" — and since the venv is empty (we install everything to
        extras), pip thinks torch is missing and resolves it fresh from
        PyPI. With extras on PYTHONPATH, pip reads the dist-info we just
        wrote and skips the reinstall.
    """
    cmd = [
        python, "-m", "pip", "install",
        "--disable-pip-version-check",
        "--upgrade",
        "--upgrade-strategy", "only-if-needed",
        "--target", str(target),
        *extra_flags,
    ]
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.append(spec)

    # Pip reads dist-info from sys.path to detect already-installed
    # packages. ``--target`` writes there but doesn't add it to sys.path,
    # so we splice extras onto PYTHONPATH for the child process. Pip
    # never imports the package code itself (just reads METADATA), so this
    # is safe even for native-wheel deps like torch / numpy.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, f"pip install {spec} timed out after 30 minutes"
    except OSError as exc:
        return False, f"pip install {spec}: {exc}"
    output = ((result.stdout or "") + ("\n" + result.stderr if result.stderr else "")).strip()
    return result.returncode == 0, output
