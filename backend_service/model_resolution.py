"""Which repo a local model is: read from where it lives, never taken on a client's word.

A Hugging Face cache folder (``models--org--name``) names its own repo. A path
outside the cache names a repo only by an exact folder name, or by the exact
weight file a catalogue row pins. Repo hints a client sends (``canonicalRepo``,
a ``modelRef`` naming a catalogue row) are checked against that and dropped
when they disagree, so a request can never route one model as another (L-16).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ID = re.compile(r"[A-Za-z0-9][\w.\-]*/[\w.\-]+")
# ModelScope stores a repo's dots as three underscores in its folder names.
_MODELSCOPE_DOT = re.compile(r"(?<=[A-Za-z0-9])___(?=[A-Za-z0-9])")
_WEIGHT_SUFFIXES = {".gguf", ".safetensors", ".npz", ".bin"}


def _clean_ref(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def infer_hf_repo_from_local_path(path: str | None) -> str | None:
    cleaned = _clean_ref(path)
    if cleaned is None:
        return None
    try:
        parts = Path(cleaned).expanduser().parts
    except Exception:
        return None
    for part in parts:
        if not part.startswith("models--"):
            continue
        repo = part[len("models--"):].replace("--", "/").strip("/")
        if repo:
            return repo
    return None


def looks_like_repo_id(value: str | None) -> bool:
    """``owner/name`` and nothing else: not a filesystem path, not a URL."""
    cleaned = _clean_ref(value)
    return bool(cleaned and _REPO_ID.fullmatch(cleaned))


def names_a_location(path: str | None) -> bool:
    """True when ``path`` can say which model it is: it is on disk, or it is a Hugging Face cache path."""
    cleaned = _clean_ref(path)
    if cleaned is None:
        return False
    if infer_hf_repo_from_local_path(cleaned) is not None:
        return True
    try:
        return Path(cleaned).expanduser().exists()
    except OSError:
        return False


def path_names_repo(path: str, repo: str, *, pinned_file: str | None = None) -> bool:
    """True when ``path``, by where it is, is the model ``repo``.

    A Hugging Face cache path names exactly one repo. Anywhere else the
    folder must be named exactly the repo (``name``, ``org--name`` or
    ``org_name``, case ignored), or the weight file must be exactly the file
    a catalogue row pins. Shared words ("qwen", "4bit") never count.
    """
    derived = infer_hf_repo_from_local_path(path)
    if derived is not None:
        return derived.lower() == repo.strip().lower()
    candidate = Path(path).expanduser()
    file_name: str | None = None
    folder = candidate
    if candidate.suffix.lower() in _WEIGHT_SUFFIXES and not candidate.is_dir():
        file_name = candidate.name
        folder = candidate.parent
    if pinned_file:
        if file_name is not None and file_name == pinned_file:
            return True
        if file_name is None and (folder / pinned_file).is_file():
            return True
    wanted = repo.strip().lower()
    folder_name = _MODELSCOPE_DOT.sub(".", folder.name).lower()
    return folder_name in {wanted.rsplit("/", 1)[-1], wanted.replace("/", "--"), wanted.replace("/", "_")}


def resolve_dflash_target_ref(
    *,
    canonical_repo: str | None,
    path: str | None,
    model_ref: str | None,
) -> str | None:
    return _clean_ref(canonical_repo) or infer_hf_repo_from_local_path(path) or _clean_ref(model_ref)
