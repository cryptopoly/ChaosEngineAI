"""Snapshot integrity checks — sharded safetensors + GGUF directory probes.

Pure filesystem helpers used during discovery and at the boundary of
``/api/models/files`` to detect interrupted downloads (missing safetensors
shards, ``*.gguf.part`` files with no main weights, mmproj-only GGUF
directories) and produce surface-ready error messages.

Extracted from ``backend_service/helpers/discovery.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.discovery`` so existing
imports keep working.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_SHARDED_WEIGHT_RE = re.compile(
    r"(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.(?P<suffix>safetensors|bin)$",
    re.IGNORECASE,
)


def _incomplete_sharded_weight_reason(path: Path) -> str | None:
    try:
        files = [entry.name for entry in path.iterdir() if entry.is_file()]
    except OSError:
        return None

    shard_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in files:
        match = _SHARDED_WEIGHT_RE.match(filename)
        if not match:
            continue
        key = (match.group("prefix"), match.group("suffix").lower())
        expected_total = int(match.group("total"))
        shard_index = int(match.group("index"))
        group = shard_groups.setdefault(key, {"expected_total": expected_total, "present": set()})
        group["expected_total"] = max(int(group["expected_total"]), expected_total)
        group["present"].add(shard_index)

    for (_prefix, suffix), group in shard_groups.items():
        expected_total = int(group["expected_total"])
        present = set(group["present"])
        if expected_total <= 1:
            continue
        missing = [index for index in range(1, expected_total + 1) if index not in present]
        if missing:
            sample = ", ".join(f"{index:05d}" for index in missing[:3])
            more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            return (
                f"Incomplete sharded {suffix} checkpoint: found {len(present)} of {expected_total} shard files. "
                f"Missing shards include {sample}{more}."
            )
    return None


def _incomplete_gguf_directory_reason(path: Path) -> str | None:
    try:
        gguf_files = [entry for entry in path.rglob("*.gguf") if entry.is_file()]
        part_files = [entry for entry in path.rglob("*.gguf.part") if entry.is_file()]
    except OSError:
        return None

    main_files = [entry for entry in gguf_files if "mmproj" not in entry.name.lower()]
    if main_files:
        return None
    if part_files:
        sample = ", ".join(entry.name for entry in part_files[:2])
        more = f" (+{len(part_files) - 2} more)" if len(part_files) > 2 else ""
        return (
            f"GGUF download is incomplete: main model weights are still downloading "
            f"({sample}{more})."
        )
    if gguf_files:
        return "GGUF directory only contains a vision projector (mmproj) and no main model weights."
    return None


def _list_weight_files(raw_path: str) -> dict[str, Any]:
    """Inspect a model path and list its weight files.

    Used by the frontend picker to let users choose a specific .gguf when a
    directory contains multiple weights. Mirrors ``_resolve_gguf_path`` logic
    for GGUF directories.
    """
    target = Path(os.path.expanduser(raw_path or "")).expanduser()
    if not target.exists():
        return {
            "path": str(target),
            "format": "unknown",
            "files": [],
            "broken": True,
            "brokenReason": "Path does not exist",
        }

    def _gb(p: Path) -> float:
        try:
            return round(p.stat().st_size / (1024 ** 3), 2)
        except OSError:
            return 0.0

    # Single file
    if target.is_file():
        suffix = target.suffix.lower()
        if suffix == ".gguf":
            fmt = "GGUF"
        elif suffix == ".safetensors":
            fmt = "Transformers"
        else:
            fmt = "unknown"
        return {
            "path": str(target),
            "format": fmt,
            "files": [
                {
                    "name": target.name,
                    "path": str(target),
                    "sizeGb": _gb(target),
                    "role": "main",
                }
            ],
            "broken": False,
            "brokenReason": None,
        }

    # Directory
    ggufs = sorted(target.rglob("*.gguf"), key=lambda f: f.stat().st_size, reverse=True)
    gguf_partials = sorted(target.rglob("*.gguf.part"))
    if ggufs or gguf_partials:
        broken_reason = _incomplete_gguf_directory_reason(target)
        files = []
        for f in ggufs:
            is_mmproj = "mmproj" in f.name.lower()
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "sizeGb": _gb(f),
                    "role": "mmproj" if is_mmproj else "main",
                }
            )
        for f in gguf_partials:
            files.append(
                {
                    "name": f.name,
                    "path": str(f),
                    "sizeGb": _gb(f),
                    "role": "partial",
                }
            )
        return {
            "path": str(target),
            "format": "GGUF",
            "files": files,
            "broken": broken_reason is not None,
            "brokenReason": broken_reason,
        }

    safetensors = sorted(target.glob("*.safetensors"))
    if safetensors:
        shard_reason = _incomplete_sharded_weight_reason(target)
        files = [
            {
                "name": f.name,
                "path": str(f),
                "sizeGb": _gb(f),
                "role": "main",
            }
            for f in safetensors
        ]
        has_mlx = any(f.name == "model.safetensors" for f in safetensors) or (target / "model.safetensors").exists()
        fmt = "MLX" if has_mlx and not (target / "model.safetensors.index.json").exists() else "Transformers"
        return {
            "path": str(target),
            "format": fmt,
            "files": files,
            "broken": shard_reason is not None,
            "brokenReason": shard_reason,
        }

    # No weights found
    return {
        "path": str(target),
        "format": "unknown",
        "files": [],
        "broken": True,
        "brokenReason": "No .gguf or .safetensors weights found",
    }
