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
