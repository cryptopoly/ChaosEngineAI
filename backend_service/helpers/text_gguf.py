"""Pinned GGUF files for text-catalog variants.

Community GGUF repos usually ship several quants side by side — Q4_K_M,
Q6_K, Q8_0 and often a BF16 — so a whole-repo snapshot of a 27B model is
~70 GB and of a frontier MoE runs to terabytes. A text variant that sets
``ggufFile`` names the one file it means (the first shard, for split
files): downloads fetch only that quant plus any vision projector, and the
loader prefers it over the "largest .gguf in the directory" fallback,
which would otherwise pick the BF16 / Q8_0 sitting next to it.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Iterable

_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def pinned_gguf_file(repo_id: str | None) -> str | None:
    """The ``ggufFile`` a text-catalog variant pins for ``repo_id``, if any."""
    if not repo_id:
        return None
    from backend_service.catalog.text_models import MODEL_FAMILIES  # noqa: PLC0415

    for family in MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant.get("repo") == repo_id and variant.get("ggufFile"):
                return str(variant["ggufFile"])
    return None


def gguf_file_pattern(gguf_file: str) -> str:
    """Glob matching every shard of a (possibly split) GGUF file."""
    match = _SHARD_RE.search(gguf_file)
    if match is None:
        return gguf_file
    return f"{gguf_file[: match.start()]}-*-of-{match.group(2)}.gguf"


def text_repo_allow_patterns(repo_id: str) -> list[str] | None:
    """``snapshot_download`` allowlist for a pinned text GGUF repo.

    ``None`` for unpinned repos so the caller downloads them in full.
    ``*mmproj*.gguf`` keeps the vision projector the llama.cpp engine
    wires via ``--mmproj`` (FU-072).
    """
    gguf_file = pinned_gguf_file(repo_id)
    if gguf_file is None:
        return None
    return [gguf_file_pattern(gguf_file), "*mmproj*.gguf", "*.md", "LICENSE*"]


def matched_size_bytes(files: Iterable[dict], patterns: list[str]) -> int:
    """Sum ``sizeBytes`` of the Hub file entries an allowlist would fetch."""
    total = 0
    for entry in files:
        path = str(entry.get("path") or "")
        if any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            total += int(entry.get("sizeBytes") or 0)
    return total


def repo_id_from_path(path: Path) -> str | None:
    """Recover ``owner/name`` from an HF-cache dir (``models--owner--name``)
    or an ``owner/name`` model-directory layout."""
    for part in reversed(path.parts):
        if part.startswith("models--"):
            owner, _, name = part[len("models--"):].partition("--")
            if owner and name:
                return f"{owner}/{name}"
    if len(path.parts) >= 2:
        return f"{path.parts[-2]}/{path.parts[-1]}"
    return None


def pick_model_gguf(directory: Path, model_files: list[Path]) -> Path:
    """Choose the GGUF to load from ``model_files`` (largest first).

    The catalog pin wins when it is on disk. Otherwise keep the largest
    file, but step back to its first shard — llama.cpp opens split GGUFs
    from ``-00001-of-``, and the largest shard is rarely that one.
    """
    pinned = pinned_gguf_file(repo_id_from_path(directory))
    if pinned:
        pattern = gguf_file_pattern(Path(pinned).name)
        matches = sorted(f for f in model_files if fnmatch.fnmatch(f.name, pattern))
        if matches:
            return matches[0]
    chosen = model_files[0]
    match = _SHARD_RE.search(chosen.name)
    if match and match.group(1) != "00001":
        first = chosen.with_name(f"{chosen.name[: match.start()]}-00001-of-{match.group(2)}.gguf")
        if first.exists():
            return first
    return chosen
