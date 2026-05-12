"""Image artifact storage + placeholder rendering helpers.

Daily-folder layout for generated images, JSON-sidecar metadata, hydrate /
load / find / delete CRUD over the on-disk gallery, plus the SVG
placeholder renderer used when the runtime can't produce a real preview
(no diffusers, mock engine, or pre-load probe).

Extracted from ``backend_service/helpers/images.py`` as part of the
v0.8.0 refactor. Re-exported from ``helpers.images`` so existing
``from backend_service.helpers.images import _save_image_artifact``-style
imports keep working.
"""

from __future__ import annotations

import base64
import json
import urllib.parse
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any


def _stable_image_hash(value: str) -> int:
    acc = 0
    for index, char in enumerate(value):
        acc = (acc + ord(char) * (index + 17)) % 0xFFFFFF
    return acc


def _placeholder_image_data_url(prompt: str, model_name: str, width: int, height: int, seed: int) -> str:
    hash_value = _stable_image_hash(f"{model_name}:{prompt}:{seed}")
    hue_a = hash_value % 360
    hue_b = (hash_value * 7) % 360
    accent_x = 90 + (hash_value % 240)
    accent_y = 80 + ((hash_value >> 3) % 200)
    safe_prompt = html_escape((prompt.strip() or "Generated image preview")[:72])
    safe_model_name = html_escape(model_name)
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="hsl({hue_a} 72% 58%)" />
          <stop offset="100%" stop-color="hsl({hue_b} 68% 46%)" />
        </linearGradient>
      </defs>
      <rect width="{width}" height="{height}" rx="28" fill="url(#bg)" />
      <circle cx="{accent_x}" cy="{accent_y}" r="{max(42, round(width * 0.12))}" fill="rgba(255,255,255,0.18)" />
      <circle cx="{width - accent_x}" cy="{height - accent_y}" r="{max(36, round(width * 0.09))}" fill="rgba(8,12,20,0.18)" />
      <rect x="28" y="{height - 136}" width="{max(240, width - 56)}" height="108" rx="24" fill="rgba(11,15,22,0.38)" stroke="rgba(255,255,255,0.14)" />
      <text x="52" y="{height - 90}" fill="white" font-size="28" font-family="SF Pro Display, Inter, sans-serif" font-weight="700">{safe_model_name}</text>
      <text x="52" y="{height - 52}" fill="rgba(255,255,255,0.88)" font-size="19" font-family="SF Pro Text, Inter, sans-serif">{safe_prompt}</text>
    </svg>
    """.strip()
    return f"data:image/svg+xml;charset=utf-8,{urllib.parse.quote(svg)}"


def _image_output_directory(image_outputs_dir: Path, created_at: str | None = None) -> Path:
    day_label = (created_at or datetime.utcnow().isoformat())[:10]
    output_dir = image_outputs_dir / day_label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _preview_data_url_from_image_path(image_path: str | None) -> str:
    if not image_path:
        return ""
    path = Path(image_path)
    if not path.exists():
        return ""
    suffix = path.suffix.lower()
    try:
        if suffix == ".svg":
            return f"data:image/svg+xml;charset=utf-8,{urllib.parse.quote(path.read_text(encoding='utf-8'))}"
        mime_type = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except OSError:
        return ""


def _hydrate_image_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    prompt = str(payload.get("prompt") or "")
    model_name = str(payload.get("modelName") or payload.get("modelId") or "Image model")
    width = int(payload.get("width") or 1024)
    height = int(payload.get("height") or 1024)
    seed = int(payload.get("seed") or 0)
    image_path = str(payload.get("imagePath") or "")
    metadata_path = str(payload.get("metadataPath") or "")
    preview_url = str(payload.get("previewUrl") or "").strip()
    if not preview_url:
        preview_url = _preview_data_url_from_image_path(image_path) or _placeholder_image_data_url(prompt, model_name, width, height, seed)
    return {
        "artifactId": str(payload.get("artifactId") or ""),
        "modelId": str(payload.get("modelId") or ""),
        "modelName": model_name,
        "prompt": prompt,
        "negativePrompt": str(payload.get("negativePrompt") or ""),
        "width": width,
        "height": height,
        "steps": int(payload.get("steps") or 24),
        "guidance": float(payload.get("guidance") or 5.5),
        "seed": seed,
        "createdAt": str(payload.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
        "durationSeconds": float(payload.get("durationSeconds") or 0.0),
        "previewUrl": preview_url,
        "imagePath": image_path or None,
        "metadataPath": metadata_path or None,
        "runtimeLabel": str(payload.get("runtimeLabel") or ""),
        "runtimeNote": str(payload.get("runtimeNote") or "") or None,
        "qualityPreset": str(payload.get("qualityPreset") or "") or None,
        "draftMode": bool(payload.get("draftMode")),
    }


def _save_image_artifact(artifact: dict[str, Any], image_outputs_dir: Path) -> dict[str, Any]:
    created_at = str(artifact.get("createdAt") or datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    output_dir = _image_output_directory(image_outputs_dir, created_at)
    artifact_id = str(artifact["artifactId"])
    extension = str(artifact.get("imageExtension") or "").lstrip(".")
    preview_url = str(artifact.get("previewUrl") or "")
    if not extension:
        extension = "svg" if preview_url.startswith("data:image/svg+xml") else "png"
    image_path = output_dir / f"{artifact_id}.{extension}"
    metadata_path = output_dir / f"{artifact_id}.json"
    image_bytes = artifact.get("imageBytes")
    if isinstance(image_bytes, str):
        image_bytes = base64.b64decode(image_bytes.encode("ascii"))

    if isinstance(image_bytes, (bytes, bytearray)):
        image_path.write_bytes(bytes(image_bytes))
    elif preview_url.startswith("data:image/svg+xml"):
        image_path.write_text(
            urllib.parse.unquote(preview_url.split(",", 1)[1]),
            encoding="utf-8",
        )
    elif ";base64," in preview_url:
        encoded = preview_url.split(";base64,", 1)[1]
        image_path.write_bytes(base64.b64decode(encoded.encode("ascii")))
    else:
        image_path.write_text("", encoding="utf-8")

    persisted = {
        **artifact,
        "imagePath": str(image_path),
        "metadataPath": str(metadata_path),
    }
    metadata_payload = {
        key: value
        for key, value in persisted.items()
        if key not in {"imageBytes", "imageMimeType", "imageExtension", "previewUrl"}
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return _hydrate_image_artifact(persisted)


def _load_image_outputs(image_outputs_dir: Path) -> list[dict[str, Any]]:
    if not image_outputs_dir.exists():
        return []
    outputs: list[dict[str, Any]] = []
    for metadata_path in image_outputs_dir.rglob("*.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        outputs.append(_hydrate_image_artifact({**payload, "metadataPath": str(metadata_path)}))
    outputs.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return outputs


def _find_image_output(artifact_id: str, image_outputs_dir: Path) -> dict[str, Any] | None:
    for output in _load_image_outputs(image_outputs_dir):
        if output.get("artifactId") == artifact_id:
            return output
    return None


def _delete_image_output(artifact_id: str, image_outputs_dir: Path) -> bool:
    found = False
    for metadata_path in image_outputs_dir.rglob(f"{artifact_id}.json") if image_outputs_dir.exists() else []:
        found = True
        image_path = metadata_path.with_suffix(".svg")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("imagePath"):
                image_path = Path(str(payload["imagePath"]))
        except (OSError, json.JSONDecodeError):
            pass
        try:
            metadata_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            pass
    return found
