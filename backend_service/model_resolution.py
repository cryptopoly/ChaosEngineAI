from __future__ import annotations

from pathlib import Path


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


def resolve_dflash_target_ref(
    *,
    canonical_repo: str | None,
    path: str | None,
    model_ref: str | None,
) -> str | None:
    return _clean_ref(canonical_repo) or infer_hf_repo_from_local_path(path) or _clean_ref(model_ref)

