"""Video model helpers: variant lookup, install detection, payload shaping,
output CRUD.

Mirrors ``helpers/images.py`` so the routes for ``/api/video/*`` can drop in
alongside the image routes without a new mental model.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
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


# ---- Video output CRUD ----
#
# Video artifacts differ from image artifacts in one important way: an mp4 is
# the real deliverable and there's no cheap "preview" we can embed inline. The
# frontend loads the file directly via a dedicated ``/file`` endpoint rather
# than getting a base64 data URL in the list payload.


def _video_output_directory(video_outputs_dir: Path, created_at: str | None = None) -> Path:
    day_label = (created_at or datetime.utcnow().isoformat())[:10]
    output_dir = video_outputs_dir / day_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _hydrate_video_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "")
    model_name = str(payload.get("modelName") or payload.get("modelId") or "Video model")
    return {
        "artifactId": str(payload.get("artifactId") or ""),
        "modelId": str(payload.get("modelId") or ""),
        "modelName": model_name,
        "prompt": prompt,
        "negativePrompt": str(payload.get("negativePrompt") or ""),
        "width": int(payload.get("width") or 768),
        "height": int(payload.get("height") or 512),
        "numFrames": int(payload.get("numFrames") or 0),
        "fps": int(payload.get("fps") or 24),
        "steps": int(payload.get("steps") or 0),
        "guidance": float(payload.get("guidance") or 0.0),
        "seed": int(payload.get("seed") or 0),
        "createdAt": str(
            payload.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        ),
        "durationSeconds": float(payload.get("durationSeconds") or 0.0),
        "clipDurationSeconds": float(payload.get("clipDurationSeconds") or 0.0),
        "videoPath": str(payload.get("videoPath") or "") or None,
        "metadataPath": str(payload.get("metadataPath") or "") or None,
        "videoMimeType": str(payload.get("videoMimeType") or "video/mp4"),
        "videoExtension": str(payload.get("videoExtension") or "mp4"),
        "runtimeLabel": str(payload.get("runtimeLabel") or ""),
        "runtimeNote": str(payload.get("runtimeNote") or "") or None,
    }


def _save_video_artifact(artifact: dict[str, Any], video_outputs_dir: Path) -> dict[str, Any]:
    created_at = str(
        artifact.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    )
    output_dir = _video_output_directory(video_outputs_dir, created_at)
    artifact_id = str(artifact["artifactId"])
    extension = str(artifact.get("videoExtension") or "mp4").lstrip(".")
    video_path = output_dir / f"{artifact_id}.{extension}"
    metadata_path = output_dir / f"{artifact_id}.json"

    video_bytes = artifact.get("videoBytes")
    if isinstance(video_bytes, str):
        video_bytes = base64.b64decode(video_bytes.encode("ascii"))
    if isinstance(video_bytes, (bytes, bytearray)):
        video_path.write_bytes(bytes(video_bytes))
    else:
        raise ValueError(
            "Cannot persist video artifact: no raw bytes supplied. "
            "Pass `videoBytes` as bytes from the generation pipeline."
        )

    persisted = {
        **artifact,
        "videoPath": str(video_path),
        "metadataPath": str(metadata_path),
    }
    metadata_payload = {
        key: value
        for key, value in persisted.items()
        if key not in {"videoBytes", "videoMimeType", "videoExtension"}
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return _hydrate_video_artifact(persisted)


def _load_video_outputs(video_outputs_dir: Path) -> list[dict[str, Any]]:
    if not video_outputs_dir.exists():
        return []
    outputs: list[dict[str, Any]] = []
    for metadata_path in video_outputs_dir.rglob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        outputs.append(_hydrate_video_artifact({**payload, "metadataPath": str(metadata_path)}))
    outputs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return outputs


def _find_video_output(artifact_id: str, video_outputs_dir: Path) -> dict[str, Any] | None:
    for output in _load_video_outputs(video_outputs_dir):
        if output.get("artifactId") == artifact_id:
            return output
    return None


def _delete_video_output(artifact_id: str, video_outputs_dir: Path) -> bool:
    if not video_outputs_dir.exists():
        return False
    found = False
    for metadata_path in video_outputs_dir.rglob(f"{artifact_id}.json"):
        found = True
        video_path = metadata_path.with_suffix(".mp4")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("videoPath"):
                video_path = Path(str(payload["videoPath"]))
        except (OSError, json.JSONDecodeError):
            pass
        try:
            metadata_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass
    return found
