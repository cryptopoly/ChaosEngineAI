"""Leave out weight shards that no index file uses when downloading a repo.

Some repos keep an old set of shards next to the current one: LTX-2.3
distilled ships ``model-*-of-00008`` (33 GB, unused) beside the
``model-*-of-00018`` set its ``model.safetensors.index.json`` lists, so a
plain snapshot fetched 38 GB the model never reads. Only files with
shard-style names (``-00001-of-00018.safetensors``) in a folder that has an
index are candidates; single files such as diffusers' ``*.fp16.safetensors``
variants are never dropped.
"""
from __future__ import annotations

import fnmatch
import posixpath
import re
from collections import defaultdict
from typing import Any, Iterable

_SHARD_NAME = re.compile(r"-\d{5}-of-\d{5}\.safetensors$")
_INDEX_SUFFIX = ".safetensors.index.json"


def unused_shard_files(paths: Iterable[str], indexes: dict[str, dict[str, Any]]) -> set[str]:
    """Shard files in an indexed folder that none of that folder's indexes reference."""
    referenced: dict[str, set[str]] = defaultdict(set)
    for index_path, data in indexes.items():
        folder = posixpath.dirname(index_path)
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        for name in (weight_map or {}).values():
            if isinstance(name, str):
                referenced[folder].add(posixpath.join(folder, name) if folder else name)
    unused: set[str] = set()
    for path in paths:
        folder = posixpath.dirname(path)
        if folder in referenced and _SHARD_NAME.search(path) and path not in referenced[folder]:
            unused.add(path)
    return unused


def _list_hub_files(repo: str) -> dict[str, int]:
    """Every file in the repo with its size (the app's own listing is top-level only)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo, files_metadata=True, timeout=10)
    return {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}


def _literal_pattern(path: str) -> str:
    """An fnmatch pattern that matches exactly ``path``."""
    return re.sub(r"([\[\]*?])", r"[\1]", path)


def pruned_allow_patterns(
    repo: str,
    allow_patterns: list[str] | None,
    *,
    list_files=None,
    fetch_json=None,
) -> tuple[list[str], int, int] | None:
    """``(allow_patterns, kept_bytes, skipped_bytes)`` without unused shards.

    None when nothing would be skipped or anything could not be read, so the
    caller keeps its original patterns. ``list_files`` / ``fetch_json`` are
    injectable for tests.
    """
    if list_files is None:
        list_files = _list_hub_files
    if fetch_json is None:
        from backend_service.helpers.hub_http import fetch_hub_json as fetch_json

    sizes = list_files(repo)
    wanted = [
        path for path in sizes
        if allow_patterns is None or any(fnmatch.fnmatch(path, pattern) for pattern in allow_patterns)
    ]
    index_paths = [path for path in wanted if path.endswith(_INDEX_SUFFIX)]
    if not index_paths:
        return None
    indexes: dict[str, dict[str, Any]] = {}
    for path in index_paths:
        data = fetch_json(repo, path)
        if not isinstance(data, dict):
            return None  # can't tell what is used; download everything as before
        indexes[path] = data
    unused = unused_shard_files(wanted, indexes)
    if not unused:
        return None
    kept = [path for path in wanted if path not in unused]
    return (
        [_literal_pattern(path) for path in kept],
        sum(sizes[path] for path in kept),
        sum(sizes[path] for path in unused),
    )
