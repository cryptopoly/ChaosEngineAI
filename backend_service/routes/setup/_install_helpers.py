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


def _cleanup_mlx_video_shadow_metadata(extras_dir: Path) -> list[str]:
    """Remove stale PyPI ``mlx-video`` dist-info folders from ``--target``.

    Blaizzy's generator package and the unrelated PyPI preprocessing package
    share the normalized project name ``mlx-video``. pip's ``--target`` mode
    can leave both ``mlx_video-*.dist-info`` folders behind after a forced git
    reinstall, which makes version/provenance checks ambiguous even when the
    importable package directory was correctly overwritten.
    """
    removed: list[str] = []
    if not extras_dir.exists():
        return removed
    for dist_info in extras_dir.glob("mlx_video-*.dist-info"):
        metadata_path = dist_info / "METADATA"
        try:
            metadata = metadata_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            metadata = ""
        if "github.com/Blaizzy/mlx-video" in metadata:
            continue
        shutil.rmtree(dist_info, ignore_errors=True)
        removed.append(dist_info.name)
    return removed


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


# Packages with C-extensions linked against torch's C++ ABI. Their wheels
# break on torch minor/major bumps because the ABI symbols change between
# torch releases — a bitsandbytes wheel compiled against torch 2.6 raises
# ``undefined symbol`` on torch 2.7 import. ``--force-reinstall`` is the
# fix: pip then picks the wheel whose metadata matches the newly-installed
# torch. Patch bumps (2.6.0 → 2.6.1) keep the ABI stable, so the rebuild
# is only triggered on minor / major upgrades.
_TORCH_ABI_DEPENDENT_PACKAGES: tuple[str, ...] = (
    "bitsandbytes",
    "torchao",
    "nunchaku",
    "sageattention",
)


def _extract_cuda_tag(torch_version: str | None) -> str | None:
    """Return the ``cu124``-style suffix from a torch version, or ``None``.

    PyTorch wheels stamp a local-version segment after ``+`` that names
    the CUDA flavour they were built against — ``2.6.0+cu124``,
    ``2.6.0+cu128``, ``2.6.0+cpu``. The CUDA download index URL is
    ``https://download.pytorch.org/whl/<tag>``, so extracting the tag
    lets us query the *same* index the user is currently on (an upgrade
    query should never jump CUDA flavours without the user's say-so —
    cu128 wheels need a newer driver than cu121).
    """
    if not torch_version or "+" not in torch_version:
        return None
    _base, _, suffix = torch_version.partition("+")
    tag = suffix.lower().split(".", 1)[0].split("-", 1)[0]
    if tag.startswith("cu") and tag[2:].isdigit():
        return tag
    return None


def _index_url_for_cuda_tag(tag: str | None) -> str | None:
    """Return the PyTorch download index URL for a ``cu124``-style tag."""
    if not tag:
        return None
    tag = tag.lower()
    if not tag.startswith("cu") or not tag[2:].isdigit():
        return None
    return f"https://download.pytorch.org/whl/{tag}"


def _parse_version_triple(version: str) -> tuple[int, int, int] | None:
    """Return ``(major, minor, patch)`` for a PEP 440 version, or ``None``.

    Ignores local-version segments (``+cu124``) and pre-release tags
    (``rc1``, ``dev0``) — torch's wheels are uniformly ``X.Y.Z+local`` so
    a simple split is good enough. PEP 440 ordering is more complex than
    this in general but for the patch/minor/major classification we only
    care about the numeric triple.
    """
    base = version.split("+", 1)[0].split("-", 1)[0]
    parts = base.split(".")
    if len(parts) < 2:
        return None
    try:
        # Strip non-digit suffixes on each segment (e.g. "0rc1" → "0").
        # Crucially we keep only the LEADING digits — "0rc1" must parse
        # to 0, not 01, otherwise "2.6.0rc1" would sort as 2.6.1 which
        # is a different (newer) version.
        def _leading_int(part: str) -> int:
            digits: list[str] = []
            for ch in part:
                if not ch.isdigit():
                    break
                digits.append(ch)
            return int("".join(digits)) if digits else 0

        return (
            _leading_int(parts[0]),
            _leading_int(parts[1]),
            _leading_int(parts[2]) if len(parts) > 2 else 0,
        )
    except ValueError:
        return None


def _classify_torch_upgrade(current: str, latest: str) -> str | None:
    """Classify the diff between two torch versions.

    Returns one of ``"patch"`` / ``"minor"`` / ``"major"`` when ``latest``
    is strictly newer than ``current``, ``None`` otherwise. ``None`` means
    "no upgrade to offer" — the caller should not render an upgrade pill.
    """
    cur = _parse_version_triple(current)
    lat = _parse_version_triple(latest)
    if cur is None or lat is None:
        return None
    if lat <= cur:
        return None
    if lat[0] != cur[0]:
        return "major"
    if lat[1] != cur[1]:
        return "minor"
    return "patch"


def _query_latest_torch_version(python: str, index_url: str, timeout: int = 30) -> str | None:
    """Query the PyTorch download index for the newest ``torch`` version.

    Uses ``pip index versions torch --index-url <url>`` — pip lists the
    output as ``torch (X.Y.Z+local)`` on the first line, with
    ``Available versions: ...`` underneath. The ``pip index`` subcommand
    is officially "experimental" but stable since pip 21.2 (2021) and
    used widely in CI. Returns ``None`` on any failure (network error,
    pip API drift, empty output) so callers degrade gracefully into "no
    upgrade detected" rather than surfacing a confusing error.
    """
    cmd = [
        python, "-m", "pip", "index", "versions", "torch",
        "--index-url", index_url, "--disable-pip-version-check",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout or ""
    # Preferred shape: ``torch (2.6.0+cu124)`` on a line by itself.
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()
        if lower.startswith("torch (") and stripped.endswith(")"):
            inner = stripped[len("torch ("):-1].strip()
            return inner or None
    # Fallback: ``Available versions: X.Y.Z+local, ...`` — the first
    # entry is the newest by pip convention.
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        lower = stripped.lower()
        if lower.startswith("available versions:"):
            payload = stripped.split(":", 1)[1].strip()
            first = payload.split(",", 1)[0].strip()
            return first or None
    return None


def _abi_dependents_present(extras_dir: Path) -> list[str]:
    """Return the ABI-dependent packages currently installed in extras.

    Scans for either the importable package directory (``bitsandbytes/``)
    or a matching ``dist-info`` folder (``bitsandbytes-0.43.0.dist-info``)
    — pip leaves at least one of those behind for every ``--target``
    install. We do NOT walk PYTHONPATH or check the bundled venv because
    those wheels are read-only / pre-built and not subject to our rebuild.
    """
    if not extras_dir.is_dir():
        return []
    present: list[str] = []
    children = list(extras_dir.iterdir()) if extras_dir.is_dir() else []
    child_names_lower = [child.name.lower() for child in children]
    for name in _TORCH_ABI_DEPENDENT_PACKAGES:
        # Top-level package directory (``bitsandbytes/`` or
        # ``sage_attention/`` — pip normalises hyphens to underscores).
        candidates = (name.lower(), name.lower().replace("-", "_"))
        found = False
        for candidate in candidates:
            if candidate in child_names_lower:
                found = True
                break
        if not found:
            # dist-info folder fallback.
            for child_name in child_names_lower:
                if child_name.startswith(f"{name.lower()}-") and child_name.endswith(".dist-info"):
                    found = True
                    break
                if child_name.startswith(f"{name.lower().replace('-', '_')}-") and child_name.endswith(".dist-info"):
                    found = True
                    break
        if found:
            present.append(name)
    return present


def _move_torch_to_rollback(extras_dir: Path, current_version: str) -> Path | None:
    """Move existing torch + nvidia_* dirs to a sibling rollback directory.

    Returns the rollback path on success, ``None`` if there was nothing
    to move or the move itself failed. Unlike
    :func:`_purge_stale_torch_from_extras`, the wheel files are preserved
    on disk so a failed upgrade can be restored without re-downloading
    the 2.5 GB CUDA torch wheel. Caller responsibilities:

      * Delete the rollback dir after the upgrade verifies successfully
        (or pass it to :func:`_cleanup_old_torch_rollbacks` to keep the
        N most recent as safety nets).
      * Call :func:`_restore_torch_from_rollback` if verification fails.

    The rollback dir name is dot-prefixed (``.torch-rollback-<version>``)
    so Python's import machinery skips it — it's a stash, not a package.
    """
    if not extras_dir.is_dir():
        return None
    safe_version = "".join(
        ch if ch.isalnum() or ch in (".", "-", "_") else "_"
        for ch in current_version
    )
    rollback = extras_dir / f".torch-rollback-{safe_version}"
    if rollback.exists():
        # Stale rollback from a prior aborted upgrade — clear it before
        # we move new files in. If we can't clear it we still try a
        # numeric suffix so we don't lose either set of files.
        shutil.rmtree(rollback, ignore_errors=True)
        if rollback.exists():
            suffix = 1
            while (extras_dir / f"{rollback.name}.{suffix}").exists() and suffix < 100:
                suffix += 1
            rollback = extras_dir / f"{rollback.name}.{suffix}"
    try:
        rollback.mkdir()
    except OSError:
        return None
    moved = 0
    for entry in extras_dir.iterdir():
        name = entry.name
        lower = name.lower()
        if name == rollback.name:
            continue
        is_torch = name == "torch" or lower.startswith("torch-")
        is_nvidia = lower.startswith("nvidia_") or lower.startswith("nvidia-")
        if not (is_torch or is_nvidia):
            continue
        try:
            shutil.move(str(entry), str(rollback / name))
            moved += 1
        except OSError:
            continue
    if moved == 0:
        # Nothing actually moved — clean up the empty rollback dir so we
        # don't leave a stub behind.
        try:
            rollback.rmdir()
        except OSError:
            pass
        return None
    return rollback


def _restore_torch_from_rollback(extras_dir: Path, rollback_path: Path) -> bool:
    """Restore torch + nvidia_* from a rollback dir.

    Used when an upgrade fails verification. Wipes any half-installed
    new torch from ``extras_dir`` first, then moves the rollback contents
    back. Removes the now-empty rollback dir. Returns ``True`` on full
    success, ``False`` when any move failed (the rollback dir is kept so
    a user can recover manually).
    """
    if not rollback_path.is_dir():
        return False
    # Wipe any half-installed new torch first — we're about to write the
    # old one back over the same paths and a partial new wheel would
    # leave a Frankenstein dist-info mix that ``import torch`` can't
    # reconcile.
    try:
        _purge_stale_torch_from_extras(extras_dir)
    except OSError:
        pass
    ok = True
    for entry in list(rollback_path.iterdir()):
        try:
            shutil.move(str(entry), str(extras_dir / entry.name))
        except OSError:
            ok = False
    if ok:
        try:
            rollback_path.rmdir()
        except OSError:
            # Non-fatal: rollback dir lingers but extras has the right
            # files. ``_cleanup_old_torch_rollbacks`` will reap it later.
            pass
    return ok


def _cleanup_old_torch_rollbacks(extras_dir: Path, keep: int = 1) -> list[str]:
    """Remove old ``.torch-rollback-*`` dirs, keeping the most recent ``keep``.

    Each rollback is ~2.5 GB on disk; keeping one as a "oh no, the new
    torch broke my generation" safety net is worth the cost, keeping ten
    is not. Sort by mtime descending so the freshest rollback survives;
    older ones get reaped.
    """
    if not extras_dir.is_dir() or keep < 0:
        return []
    candidates: list[Path] = []
    for entry in extras_dir.iterdir():
        if entry.is_dir() and entry.name.startswith(".torch-rollback-"):
            candidates.append(entry)
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    removed: list[str] = []
    for path in candidates[keep:]:
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            removed.append(path.name)
    return removed


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
