"""What a model path holds on disk: GGUF weights, safetensors weights, or neither.

Engine routing asks this module instead of reading the path or repo name,
so a folder a user calls ``my-gguf-notes-mlx`` still loads on MLX and a
``.gguf`` file inside a folder called ``mlx`` still loads on llama.cpp.
A folder holding both (``config.json`` with safetensors next to a stray
``.gguf``) counts as a safetensors checkpoint; an explicit ``llama.cpp``
backend still loads its GGUF. Only a Hub repo id with nothing on disk
falls back to its name, since there is nothing else to read offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GGUF_MAGIC = b"GGUF"

WeightsFormat = Literal["gguf", "safetensors", "pytorch"]

_SAFETENSORS_SUFFIXES = {".safetensors", ".npz"}
_PYTORCH_SUFFIXES = {".bin", ".pt", ".pth"}
# A model folder holds tens of files; a cap keeps a mistaken pick of a big
# folder (a home directory, a dataset) from walking the whole disk.
_SCAN_FILE_LIMIT = 20_000
_SCAN_DEPTH_LIMIT = 6
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


@dataclass(frozen=True)
class WeightsScan:
    gguf: tuple[str, ...]
    safetensors: tuple[str, ...]
    pytorch: tuple[str, ...]
    has_config: bool
    names: tuple[str, ...]


def has_gguf_magic(path: Path) -> bool | None:
    """True when the file starts with the GGUF magic, None when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == GGUF_MAGIC
    except OSError:
        return None


def scan_weights(directory: Path) -> WeightsScan:
    """List the weight files under ``directory`` (HF cache snapshots included)."""
    gguf: list[str] = []
    safetensors: list[str] = []
    pytorch: list[str] = []
    names: list[str] = []
    has_config = False
    seen_dirs: set[str] = set()
    root_depth = len(Path(directory).parts)
    count = 0
    for current, dirnames, filenames in os.walk(directory, followlinks=True):
        try:
            real = os.path.realpath(current)
        except OSError:
            real = current
        if real in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real)
        # The HF cache keeps content-addressed copies under blobs/ with no
        # file names; the snapshot links name them.
        dirnames[:] = sorted(d for d in dirnames if d not in {"blobs", ".git"} and not d.startswith(".cache"))
        if len(Path(current).parts) - root_depth >= _SCAN_DEPTH_LIMIT:
            dirnames[:] = []
        for name in sorted(filenames):
            count += 1
            if count > _SCAN_FILE_LIMIT:
                break
            names.append(name)
            suffix = Path(name).suffix.lower()
            if suffix == ".gguf":
                gguf.append(name)
            elif suffix in _SAFETENSORS_SUFFIXES:
                safetensors.append(name)
            elif suffix in _PYTORCH_SUFFIXES:
                pytorch.append(name)
            elif name == "config.json":
                has_config = True
        if count > _SCAN_FILE_LIMIT:
            break
    return WeightsScan(tuple(gguf), tuple(safetensors), tuple(pytorch), has_config, tuple(names))


def weights_format(path: str | Path) -> WeightsFormat | None:
    """The weights a file or folder holds, read from its files, never its name.

    A folder with ``config.json`` and safetensors (or MLX ``.npz``) weights
    is a safetensors checkpoint even when a ``.gguf`` sits next to them. A
    folder whose weights are ``.gguf`` files (projectors and drafters count)
    is GGUF. ``None`` means no weights at all.
    """
    target = Path(os.path.expanduser(str(path)))
    try:
        if target.is_file():
            suffix = target.suffix.lower()
            if suffix == ".gguf":
                return "gguf"
            if suffix in _SAFETENSORS_SUFFIXES:
                return "safetensors"
            if suffix in _PYTORCH_SUFFIXES:
                return "pytorch"
            return None
        if not target.is_dir():
            return None
    except OSError:
        return None
    scan = scan_weights(target)
    if scan.safetensors and scan.has_config:
        return "safetensors"
    if scan.gguf:
        return "gguf"
    if scan.safetensors:
        return "safetensors"
    if scan.pytorch:
        return "pytorch"
    return None


def describe_contents(path: str | Path, limit: int = 8) -> str:
    """Short list of the files under ``path`` for an error message."""
    target = Path(os.path.expanduser(str(path)))
    try:
        if target.is_file():
            return target.name
        if not target.is_dir():
            return "nothing (the path does not exist)"
    except OSError:
        return "nothing readable"
    names = scan_weights(target).names
    if not names:
        return "no files"
    shown = ", ".join(names[:limit])
    more = len(names) - limit
    return f"{shown} and {more} more" if more > 0 else shown


def _cached_repo_dir(repo_id: str) -> Path | None:
    from backend_service.helpers.hf_cache_paths import _hf_repo_cache_dir

    try:
        cache_dir = _hf_repo_cache_dir(repo_id)
        return cache_dir if (cache_dir / "snapshots").is_dir() else None
    except (OSError, ValueError):
        return None


def looks_like_gguf(value: str | None) -> bool:
    """True when ``value`` names GGUF weights.

    A path on disk is judged by its files. A ``.gguf`` file name that is not
    on disk (a remote file) is GGUF. A Hub repo id is judged by its Hugging
    Face cache copy when there is one, and only otherwise by its name.
    """
    if not value:
        return False
    text = str(value).strip()
    if not text:
        return False
    candidate = Path(os.path.expanduser(text))
    try:
        on_disk = candidate.exists()
    except OSError:
        on_disk = False
    if on_disk:
        return weights_format(candidate) == "gguf"
    if candidate.suffix.lower() == ".gguf":
        return True
    if _REPO_ID_RE.match(text):
        cached = _cached_repo_dir(text)
        if cached is not None:
            found = weights_format(cached)
            if found is not None:
                return found == "gguf"
        return "gguf" in text.lower()
    return False
