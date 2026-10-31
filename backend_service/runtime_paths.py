from __future__ import annotations

import os
import sys
from pathlib import Path


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _user_data_base() -> Path:
    home = Path.home()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    if sys.platform == "darwin":
        return home / "Library" / "Application Support"
    return Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")


def extras_site_packages() -> Path | None:
    """Primary persistent target for runtime-installed Python packages."""
    env_path = os.environ.get("CHAOSENGINE_EXTRAS_SITE_PACKAGES")
    if env_path:
        return Path(env_path)
    return _user_data_base() / "ChaosEngineAI" / "extras" / _python_tag() / "site-packages"


def extras_site_package_candidates() -> list[Path]:
    """Import-path candidates for persisted runtime packages.

    The canonical path is ``ChaosEngineAI/extras``. The ``com.chaosengineai.desktop``
    candidate covers early desktop builds that used Tauri's bundle identifier
    as their app-data root.
    """
    candidates: list[Path] = []
    primary = extras_site_packages()
    if primary is not None:
        candidates.append(primary)

    base = _user_data_base()
    tag = _python_tag()
    legacy = base / "com.chaosengineai.desktop" / "extras" / tag / "site-packages"
    candidates.append(legacy)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def ensure_extras_on_sys_path() -> list[Path]:
    """Make persisted runtime dirs importable without shadowing repo-local shims.

    The Tauri launcher normally exports ``PYTHONPATH`` before Python starts.
    Doing it again inside the backend makes direct/dev launches and older
    launchers converge on the same package resolution.

    ``sys.path`` entries are *appended* rather than inserted near the top so
    repo-local adapter packages (notably ``turboquant_mlx``, which acts as a
    thin shim around the upstream ``turboquant-mlx-full`` install in extras)
    keep import authority. Without this, prepending extras pulls in the raw
    upstream package directly and the shim's exported helpers
    (``_find_pip_turboquant_path``, ``make_adaptive_cache``, ``apply_patch``)
    become unreachable, breaking both the cache-strategy adapter at runtime
    and ``tests/test_cache_strategies.py`` during pytest collection.
    """
    existing_candidates = [path for path in extras_site_package_candidates() if path.is_dir()]
    if existing_candidates:
        pythonpath_entries = [
            entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
        ]
        pythonpath_keys = {
            os.path.normcase(os.path.abspath(entry))
            for entry in pythonpath_entries
        }
        prepend_entries: list[str] = []
        for path in existing_candidates:
            key = os.path.normcase(os.path.abspath(path))
            if key in pythonpath_keys:
                continue
            prepend_entries.append(str(path))
            pythonpath_keys.add(key)
        if prepend_entries:
            os.environ["PYTHONPATH"] = os.pathsep.join(prepend_entries + pythonpath_entries)

    sys_path_keys = {
        os.path.normcase(os.path.abspath(entry))
        for entry in sys.path
        if entry
    }
    inserted: list[Path] = []
    for path in existing_candidates:
        key = os.path.normcase(os.path.abspath(path))
        if key in sys_path_keys:
            continue
        sys.path.append(str(path))
        sys_path_keys.add(key)
        inserted.append(path)
    return inserted


# Packages the release bundles itself and that must match each other. An
# overlay copy of a different version shadows the bundled one, because the
# overlay sits ahead of the bundled site-packages: after an app update that
# moves the MLX series, an overlay left by an older Setup install (mlx-whisper
# brings mlx without mlx-lm) would run the new mlx-lm on the old mlx.
_BUNDLED_CORE = ("mlx", "mlx-metal", "mlx-lm")

# Packages transformers / mlx-lm import in lock-step with each other. Successive
# ``pip install --target`` runs into the overlay layer several versions of these
# on top of one another (huggingface-hub 1.12 files under a 2.2 dist-info), so
# the overlay is not just stale, it is a mix. Unless the overlay holds exactly
# the bundled version, every copy of the package goes and the bundled one runs.
_LOCKSTEP_CORE = ("huggingface-hub", "transformers", "tokenizers", "safetensors", "hf-xet")


def _normalise(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _dist_infos(directory: Path) -> dict[str, list[tuple[str, Path]]]:
    """``{normalised name: [(version, dist-info dir), ...]}`` for one site dir."""
    found: dict[str, list[tuple[str, Path]]] = {}
    try:
        entries = list(directory.glob("*.dist-info"))
    except OSError:
        return found
    for info in entries:
        name = version = None
        try:
            for line in (info / "METADATA").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("Name:") and name is None:
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("Version:") and version is None:
                    version = line.split(":", 1)[1].strip()
                if name and version:
                    break
        except OSError:
            continue
        if name and version:
            found.setdefault(_normalise(name), []).append((version, info))
    return found


def quarantine_stale_core_packages(extras_dir: Path, bundled_dirs: list[Path]) -> list[str]:
    """Move overlay copies of the bundled MLX core that differ from the bundle.

    Files are moved (not deleted) to ``<extras parent>/stale-core/<name>-<version>``
    using each package's RECORD, so a package that shares a directory with
    another (mlx-metal installs into ``mlx/``) moves exactly its own files.
    Returns ``["mlx 0.31.2", ...]`` for what was moved.
    """
    import shutil

    if not extras_dir.is_dir():
        return []
    bundled: dict[str, str] = {}
    for directory in bundled_dirs:
        for name, versions in _dist_infos(directory).items():
            bundled.setdefault(name, versions[0][0])
    overlay = _dist_infos(extras_dir)
    root = extras_dir.resolve()
    moved: list[str] = []
    emptied: set[Path] = set()
    for name in _BUNDLED_CORE + _LOCKSTEP_CORE:
        want = bundled.get(name)
        if want is None:
            continue
        copies = overlay.get(name, [])
        # A lock-step package that is clean (one copy, the bundled version) stays;
        # anything else is moved whole, including a copy that matches the bundle.
        move_all = name in _LOCKSTEP_CORE and (len(copies) > 1 or any(v != want for v, _ in copies))
        for version, info in copies:
            if version == want and not move_all:
                continue
            target = extras_dir.parent / "stale-core" / f"{name}-{version}"
            try:
                record = (info / "RECORD").read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                record = []
            for line in record:
                relative = line.split(",", 1)[0].strip()
                if not relative:
                    continue
                source = (extras_dir / relative).resolve()
                if root not in source.parents or not source.is_file():
                    continue  # paths outside the overlay (scripts) stay put
                destination = target / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                emptied.add(source.parent)
            if info.exists():
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(info), str(target / info.name))
            moved.append(f"{name} {version}")
    # Drop the folders those moves left empty (deepest first), stopping at
    # the overlay root; rmdir refuses any folder that still holds files.
    for folder in sorted(emptied, key=lambda p: len(p.parts), reverse=True):
        while folder != root and root in folder.parents:
            try:
                leftovers = [child for child in folder.iterdir() if child.name != "__pycache__"]
                if not leftovers:
                    shutil.rmtree(folder / "__pycache__", ignore_errors=True)
            except OSError:
                break
            try:
                folder.rmdir()
            except OSError:
                break
            folder = folder.parent
    return moved


def guard_overlay_against_bundle() -> list[str]:
    """Packaged builds: apply :func:`quarantine_stale_core_packages` at startup."""
    if os.environ.get("CHAOSENGINE_EMBEDDED_RUNTIME") != "1":
        return []
    extras = extras_site_packages()
    if extras is None or not extras.is_dir():
        return []
    extras_key = os.path.normcase(os.path.realpath(extras))
    bundled = [
        Path(entry) for entry in sys.path
        if entry and os.path.isdir(entry) and os.path.normcase(os.path.realpath(entry)) != extras_key
    ]
    try:
        moved = quarantine_stale_core_packages(extras, bundled)
    except OSError as exc:
        print(f"[chaosengine] could not tidy the extras overlay: {exc}", file=sys.stderr)
        return []
    if moved:
        print(
            "[chaosengine] moved overlay packages that no longer match this release to "
            f"{extras.parent / 'stale-core'}: {', '.join(moved)}",
            file=sys.stderr,
        )
    return moved
