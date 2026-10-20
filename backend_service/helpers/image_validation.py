"""Image repo validation + friendly download-error helpers.

Pure helpers used at the boundary of the image download flow:
- "is this a known image repo?" predicates
- "is the local snapshot loadable?" check
- friendly translation of HF gated-repo errors so the UI can tell users to
  request access instead of just surfacing a 401.

Extracted from ``backend_service/helpers/images.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.images`` so existing
``from backend_service.helpers.images import _image_download_validation_error``
imports keep working.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend_service.catalog import IMAGE_MODEL_FAMILIES
from backend_service.helpers.discovery import _candidate_model_dirs
from backend_service.helpers.huggingface import _hf_repo_snapshot_dir
from backend_service.image_runtime import validate_local_diffusers_snapshot


def _is_image_repo(repo_id: str) -> bool:
    return any(
        str(variant.get("repo") or "") == repo_id
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
    )


def _sdcpp_gguf_files(repo_id: str) -> list[str]:
    """GGUF files of the stable-diffusion.cpp variants that live in ``repo_id``.

    Those repos hold bare GGUF files, not a diffusers pipeline, so "installed"
    means one of the files is on disk.
    """
    return [
        str(variant["ggufFile"])
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if variant.get("engine") == "sdcpp"
        and variant.get("ggufFile")
        and str(variant.get("repo") or "") == repo_id
        and str(variant.get("ggufRepo") or "") == repo_id
    ]


def _image_repo_runtime_ready(repo_id: str) -> bool:
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    gguf_files = _sdcpp_gguf_files(repo_id)
    if gguf_files:
        return any((Path(snapshot_dir) / name).is_file() for name in gguf_files)
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id) is None


def _image_variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    repo = str(variant.get("repo") or "")
    if variant.get("engine") == "sdcpp" and variant.get("ggufFile") and str(variant.get("ggufRepo") or "") == repo:
        snapshot_dir = _hf_repo_snapshot_dir(repo)
        return snapshot_dir is not None and (Path(snapshot_dir) / str(variant["ggufFile"])).is_file()
    if repo and _image_repo_runtime_ready(repo):
        return True

    candidates = {
        str(variant.get("repo") or "").lower(),
        str(variant.get("name") or "").lower(),
        str(variant.get("id") or "").lower(),
    }
    compact_candidates = {candidate.split("/")[-1] for candidate in candidates if candidate}
    for item in library:
        name = str(item.get("name") or "").lower()
        if not (
            name in candidates
            or any(candidate and candidate in name for candidate in candidates)
            or any(candidate and candidate in name for candidate in compact_candidates)
        ):
            continue
        item_path = Path(str(item.get("path") or "")).expanduser()
        for directory in _candidate_model_dirs(item_path):
            if validate_local_diffusers_snapshot(directory) is None:
                return True
    return False


def _image_download_validation_error(repo_id: str) -> str | None:
    if not _is_image_repo(repo_id):
        return None
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return (
            f"Download did not produce a local snapshot for {repo_id}. "
            "Retry the download and make sure the backend can access Hugging Face."
        )
    gguf_files = _sdcpp_gguf_files(repo_id)
    if gguf_files:
        if any((Path(snapshot_dir) / name).is_file() for name in gguf_files):
            return None
        return f"Download finished but none of {', '.join(gguf_files)} is in the snapshot of {repo_id}."
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id)


def _friendly_image_download_error(repo_id: str, error: str) -> str:
    if not _is_image_repo(repo_id):
        return error
    lowered = error.lower()
    if (
        "cannot access gated repo" in lowered
        or "gated repo" in lowered
        or "authorized list" in lowered
        or ("access to model" in lowered and "restricted" in lowered)
    ):
        return (
            f"{repo_id} is gated on Hugging Face. Your account or token is not approved for this model yet. "
            f"Open https://huggingface.co/{repo_id}, request or accept access, add a read-enabled HF_TOKEN in Settings, then retry."
        )
    return error
