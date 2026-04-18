"""Video model helpers: variant lookup, install detection, payload shaping.

Mirrors ``helpers/images.py`` so the routes for ``/api/video/*`` can drop in
alongside the image routes without a new mental model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend_service.catalog import VIDEO_MODEL_FAMILIES
from backend_service.helpers.huggingface import _hf_repo_snapshot_dir
from backend_service.image_runtime import validate_local_diffusers_snapshot


def _video_model_payloads(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the catalog families enriched with per-variant availability.

    We deliberately don't hit Hugging Face live metadata from here — the image
    version does for download counts, etc. We can bolt that on later if the
    discover UX needs it. For now each variant just knows whether its local
    snapshot is ready to load.
    """
    families: list[dict[str, Any]] = []
    for family in VIDEO_MODEL_FAMILIES:
        variants: list[dict[str, Any]] = []
        for variant in family["variants"]:
            enriched = dict(variant)
            repo = str(enriched.get("repo") or "")
            enriched["availableLocally"] = _video_repo_runtime_ready(repo) if repo else False
            enriched["hasLocalData"] = enriched["availableLocally"] or _video_repo_has_any_local_data(repo)
            enriched["familyName"] = family["name"]
            variants.append(enriched)
        payload = dict(family)
        payload["variants"] = variants
        families.append(payload)
    return families


def _find_video_variant(model_id: str) -> dict[str, Any] | None:
    for family in VIDEO_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == model_id:
                return variant
    return None


def _find_video_variant_by_repo(repo: str) -> dict[str, Any] | None:
    for family in VIDEO_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["repo"] == repo:
                return variant
    return None


def _is_video_repo(repo_id: str) -> bool:
    return any(
        str(variant.get("repo") or "") == repo_id
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
    )


def _video_repo_runtime_ready(repo_id: str) -> bool:
    """True if the local snapshot is complete enough to load via diffusers."""
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id) is None


def _video_repo_has_any_local_data(repo_id: str) -> bool:
    """True if we have a partial or complete snapshot on disk.

    Distinct from ``_video_repo_runtime_ready`` — this is the softer signal used
    to tell the UI "something downloaded for this repo" even if it's incomplete.
    """
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    root = Path(snapshot_dir)
    if not root.exists():
        return False
    try:
        return any(
            candidate.is_file() or candidate.is_symlink()
            for candidate in root.iterdir()
            if not candidate.name.startswith(".")
        )
    except OSError:
        return False


def _video_variant_available_locally(variant: dict[str, Any]) -> bool:
    repo = str(variant.get("repo") or "")
    if not repo:
        return False
    return _video_repo_runtime_ready(repo)


def _video_download_repo_ids() -> set[str]:
    return {
        str(variant.get("repo") or "")
        for family in VIDEO_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    }


def _video_download_validation_error(repo_id: str) -> str | None:
    if not _is_video_repo(repo_id):
        return None
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return (
            f"Download did not produce a local snapshot for {repo_id}. "
            "Retry the download and make sure the backend can access Hugging Face."
        )
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id)
