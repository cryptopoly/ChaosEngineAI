"""MLX conversion helpers — supported-arch probe + path / size utilities.

Pre-flight checks for ``RuntimeController.convert_model``: peek a HF repo's
``model_type`` without downloading weights, list the architectures the
installed mlx-lm version supports, suggest the closest supported variant
when there isn't an exact match, and a couple of small filesystem helpers
used to display conversion targets.

Extracted from ``inference/__init__.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from backend_service.inference._constants import WORKSPACE_ROOT


_MLX_LM_SUPPORTED_CACHE: tuple[str | None, frozenset[str] | None] = (None, None)


def _mlx_lm_supported_model_types(python_executable: str) -> frozenset[str] | None:
    """List of model_type strings supported by the installed mlx-lm version.

    We enumerate `mlx_lm.models.<module>` files and return the set of
    module names. mlx_lm's `_get_classes` does a direct
    `importlib.import_module(f"mlx_lm.models.{model_type}")` so matching
    module name is the correct compatibility check.

    Cached per python_executable. Returns None on any failure (meaning
    "we couldn't check, don't block the conversion").
    """
    global _MLX_LM_SUPPORTED_CACHE
    cached_key, cached_val = _MLX_LM_SUPPORTED_CACHE
    if cached_key == python_executable and cached_val is not None:
        return cached_val

    probe = (
        "import os, pkgutil, json, importlib.util;"
        "spec = importlib.util.find_spec('mlx_lm.models');"
        "paths = spec.submodule_search_locations if spec else [];"
        "names = sorted({m.name for p in paths for m in pkgutil.iter_modules([p]) if not m.name.startswith('_') and not m.ispkg});"
        "print(json.dumps(names))"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return None
        names = json.loads(result.stdout.strip())
        if not isinstance(names, list):
            return None
        supported = frozenset(n for n in names if isinstance(n, str))
        _MLX_LM_SUPPORTED_CACHE = (python_executable, supported)
        return supported
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None


def _peek_hf_model_type(
    hf_path_arg: str | None,
    *,
    convert_env: dict[str, str] | None = None,
) -> str | None:
    """Read `config.json.model_type` without downloading any weights.

    - Local directory: read config.json directly from disk.
    - HF repo id: use huggingface_hub.hf_hub_download for JUST config.json
      (few KB). Honours HF_TOKEN / HUGGING_FACE_HUB_TOKEN for gated repos.
    - HF cache directory (models--owner--name/snapshots/<rev>/config.json):
      walk the snapshot dir.

    Returns None on any failure — callers must treat None as "could not
    pre-flight, proceed optimistically".
    """
    if not hf_path_arg:
        return None

    def _read(p: Path) -> str | None:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        mt = data.get("model_type")
        if isinstance(mt, str) and mt.strip():
            return mt.strip()
        return None

    candidate = Path(hf_path_arg)
    if candidate.exists():
        if candidate.is_file() and candidate.name == "config.json":
            return _read(candidate)
        if candidate.is_dir():
            direct = candidate / "config.json"
            if direct.is_file():
                return _read(direct)
            # HF cache layout: models--owner--name/snapshots/<rev>/config.json
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                for rev in sorted(snapshots.iterdir(), reverse=True):
                    cfg = rev / "config.json"
                    if cfg.is_file():
                        return _read(cfg)
        return None

    # Remote HF repo id — pull just config.json.
    if "/" not in hf_path_arg:
        return None
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None
    env = dict(convert_env or os.environ)
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    try:
        cfg_path = hf_hub_download(
            repo_id=hf_path_arg,
            filename="config.json",
            token=token,
        )
        return _read(Path(cfg_path))
    except Exception:
        return None


def _nearest_supported_arch(requested: str, supported: frozenset[str]) -> str | None:
    """Suggest the closest supported architecture for a given model_type.

    Extracts the base family (letters only) and picks the highest-numbered
    supported variant of that family, preferring plain names (no suffixes
    like _text / _moe / _vl) so "gemma4" → "gemma3" not "gemma3n".
    """
    if not requested or not supported:
        return None
    base_match = re.match(r"^([a-z]+)", requested.lower())
    if not base_match:
        return None
    base = base_match.group(1)
    candidates = [s for s in supported if re.match(rf"^{base}\d*$", s.lower())]
    if not candidates:
        # Fall back to any family member (including suffixed variants).
        candidates = [s for s in supported if s.lower().startswith(base)]
    if not candidates:
        return None

    def _score(name: str) -> tuple[int, int, str]:
        m = re.search(r"(\d+)", name)
        num = int(m.group(1)) if m else 0
        # Prefer plain names (no underscore suffix) over variant names.
        plain = 1 if "_" not in name else 0
        return (plain, num, name)

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _default_conversion_output(source_label: str) -> str:
    home_models = Path.home() / "Models"
    home_models.mkdir(parents=True, exist_ok=True)
    base = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in source_label)
    base = base.strip("-_") or "model"
    candidate = home_models / f"{base}-mlx"
    suffix = 2
    while candidate.exists():
        candidate = home_models / f"{base}-mlx-{suffix}"
        suffix += 1
    return str(candidate)


def _bytes_to_gb(value: int | float) -> float:
    """Convert a byte count to gigabytes, rounded to 2 decimal places."""
    try:
        return round(float(value) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return 0.0


def _path_size_bytes(path: str | Path | None) -> int:
    if path is None:
        return 0

    target = Path(path).expanduser()
    if not target.exists():
        return 0
    if target.is_file():
        try:
            return int(target.stat().st_size)
        except OSError:
            return 0

    total = 0
    try:
        for child in target.rglob("*"):
            if child.is_file():
                total += int(child.stat().st_size)
    except OSError:
        return total
    return total
