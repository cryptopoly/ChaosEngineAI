"""Video artifact storage helpers — daily-folder layout + JSON sidecars.

Mirror of ``helpers/image_artifacts.py`` for the video gallery so both
runtimes write through the same shape. Generated mp4 / webm bytes land
under ``video_outputs/<YYYY-MM-DD>/<artifact_id>.<ext>`` with a sibling
``<artifact_id>.json`` carrying metadata.

Extracted from ``backend_service/helpers/video.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.video`` so existing imports
keep working.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any


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
