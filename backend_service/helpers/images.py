"""Image model helpers: variant lookup, metadata, generation artifacts, output CRUD."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Any

from backend_service.catalog import IMAGE_MODEL_FAMILIES, LATEST_IMAGE_TRACKED_SEEDS
from backend_service.helpers.formatting import _bytes_to_gb
from backend_service.helpers.huggingface import (
    _classify_hub_file,
    _format_hf_updated_label,
    _hf_number_label,
    _hf_repo_snapshot_dir,
    _parse_iso_datetime,
)
from backend_service.helpers.discovery import _candidate_model_dirs
from backend_service.image_runtime import validate_local_diffusers_snapshot


_IMAGE_DISCOVER_METADATA_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_IMAGE_DISCOVER_METADATA_TTL_SECONDS = 6 * 60 * 60
_LATEST_IMAGE_MODELS_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_LATEST_IMAGE_MODELS_TTL_SECONDS = 3 * 60 * 60


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


def _image_model_payloads(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repo_metadata: dict[str, dict[str, Any]] = {}
    repos = sorted({
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    })
    if repos:
        with ThreadPoolExecutor(max_workers=min(4, len(repos))) as executor:
            future_map = {
                executor.submit(_image_repo_live_metadata, repo): repo
                for repo in repos
            }
            try:
                for future in as_completed(future_map, timeout=8):
                    repo = future_map[future]
                    try:
                        repo_metadata[repo] = future.result(timeout=2)
                    except Exception:
                        repo_metadata[repo] = {
                            "metadataWarning": "Live Hugging Face metadata is temporarily unavailable. Showing curated defaults.",
                        }
            except TimeoutError:
                pass  # Return whatever we have so far; missing repos get curated defaults

    families: list[dict[str, Any]] = []
    for family in IMAGE_MODEL_FAMILIES:
        variants = [
            {
                **variant,
                **repo_metadata.get(str(variant.get("repo") or ""), {}),
                "source": "curated",
                "familyName": family.get("name"),
                "availableLocally": _image_variant_available_locally(variant, library),
                "hasLocalData": _hf_repo_snapshot_dir(str(variant.get("repo") or "")) is not None,
            }
            for variant in family["variants"]
        ]
        families.append(
            {
                **family,
                "updatedLabel": _best_image_family_updated_label(family, variants),
                "variants": variants,
            }
        )
    return families


def _find_image_variant(model_id: str) -> dict[str, Any] | None:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == model_id:
                return variant
    return None


def _find_image_variant_by_repo(repo: str) -> dict[str, Any] | None:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["repo"] == repo:
                return variant
    return None


def _image_repo_live_metadata(repo_id: str) -> dict[str, Any]:
    now = time.time()
    cached = _IMAGE_DISCOVER_METADATA_CACHE.get(repo_id)
    if cached is not None:
        cached_at, payload = cached
        if (now - cached_at) < _IMAGE_DISCOVER_METADATA_TTL_SECONDS:
            return payload

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    payload: dict[str, Any]
    try:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{encoded_repo}?blobs=true"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode())

        total_bytes = 0
        weight_bytes = 0
        for sibling in data.get("siblings") or []:
            if not isinstance(sibling, dict):
                continue
            path = str(sibling.get("rfilename") or "")
            if not path:
                continue
            lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
            size_bytes = sibling.get("size") or lfs.get("size") or 0
            try:
                size_int = int(size_bytes)
            except (TypeError, ValueError):
                size_int = 0
            total_bytes += size_int
            if _classify_hub_file(path) == "weight":
                weight_bytes += size_int

        card = data.get("cardData") or {}
        license_value = str(card.get("license") or "").strip() or None if isinstance(card, dict) else None
        downloads = int(data.get("downloads") or 0)
        likes = int(data.get("likes") or 0)
        last_modified = str(data.get("lastModified") or "").strip() or None
        payload = {
            "downloads": downloads,
            "likes": likes,
            "downloadsLabel": _hf_number_label(downloads, "downloads") if downloads > 0 else None,
            "likesLabel": _hf_number_label(likes, "likes") if likes > 0 else None,
            "lastModified": last_modified,
            "updatedLabel": _format_hf_updated_label(last_modified),
            "license": license_value,
            "gated": bool(data.get("gated")),
            "pipelineTag": str(data.get("pipeline_tag") or "").strip() or None,
            "repoSizeBytes": total_bytes or None,
            "repoSizeGb": _bytes_to_gb(total_bytes) if total_bytes > 0 else None,
            "coreWeightsBytes": weight_bytes or None,
            "coreWeightsGb": _bytes_to_gb(weight_bytes) if weight_bytes > 0 else None,
            "metadataWarning": None,
        }
    except urllib.error.HTTPError as exc:
        status = getattr(exc, "code", None)
        payload = {
            "metadataWarning": (
                f"Live Hugging Face metadata is temporarily unavailable (HTTP {status}). Showing curated defaults."
                if status is not None
                else "Live Hugging Face metadata is temporarily unavailable. Showing curated defaults."
            ),
        }
    except (OSError, json.JSONDecodeError):
        payload = {
            "metadataWarning": "Live Hugging Face metadata is temporarily unavailable. Showing curated defaults.",
        }

    _IMAGE_DISCOVER_METADATA_CACHE[repo_id] = (now, payload)
    return payload


def _best_image_family_updated_label(family: dict[str, Any], variants: list[dict[str, Any]]) -> str:
    best_dt: datetime | None = None
    best_label: str | None = None
    for variant in variants:
        last_modified = _parse_iso_datetime(str(variant.get("lastModified") or "") or None)
        if last_modified is None:
            continue
        if best_dt is None or last_modified > best_dt:
            best_dt = last_modified
            best_label = str(variant.get("updatedLabel") or "") or None
    return best_label or str(family.get("updatedLabel") or "Curated")


def _image_task_support_from_metadata(pipeline_tag: str | None, tags: list[str]) -> list[str]:
    pipeline = str(pipeline_tag or "").lower()
    lowered_tags = {str(tag).lower() for tag in tags}
    tasks: list[str] = []
    if (
        pipeline == "text-to-image"
        or "text-to-image" in lowered_tags
        or "image-generation" in lowered_tags
    ):
        tasks.append("txt2img")
    if (
        pipeline == "image-to-image"
        or "image-to-image" in lowered_tags
        or "image-edit" in lowered_tags
        or "editing" in lowered_tags
    ):
        tasks.append("img2img")
    if pipeline == "inpainting" or "inpainting" in lowered_tags or "inpaint" in lowered_tags:
        tasks.append("inpaint")
    return tasks or ["txt2img"]


def _image_recommended_resolution(repo_id: str, pipeline_tag: str | None, tags: list[str]) -> str:
    lowered = repo_id.lower()
    lowered_tags = {str(tag).lower() for tag in tags}
    if "2048" in lowered or "2k" in lowered_tags or "hunyuanimage-2.1" in lowered:
        return "2048x2048"
    if "768" in lowered:
        return "768x768"
    if "512" in lowered:
        return "512x512"
    if "1024" in lowered or "sdxl" in lowered or "flux" in lowered or "sana" in lowered:
        return "1024x1024"
    if str(pipeline_tag or "").lower() == "text-to-image":
        return "1024x1024"
    return "Unknown"


def _image_discover_style_tags(tags: list[str]) -> list[str]:
    preferred = {
        "photoreal",
        "illustration",
        "anime",
        "general",
        "fast",
        "detailed",
        "turbo",
        "distilled",
        "edit",
        "inpaint",
        "flux",
        "sana",
        "qwenimage",
        "hidream",
    }
    seen: list[str] = []
    for tag in tags:
        lowered = str(tag).lower()
        if lowered in preferred and lowered not in seen:
            seen.append(lowered)
    return seen[:4]


def _tracked_latest_seed_payloads(library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for seed in LATEST_IMAGE_TRACKED_SEEDS:
        repo_id = str(seed.get("repo") or "")
        if not repo_id:
            continue
        payloads.append(
            {
                "id": repo_id,
                "familyId": "latest",
                "familyName": "Latest Releases",
                "name": seed.get("name") or repo_id.split("/", 1)[-1],
                "provider": seed.get("provider") or (repo_id.split("/", 1)[0] if "/" in repo_id else "Community"),
                "repo": repo_id,
                "link": f"https://huggingface.co/{repo_id}",
                "runtime": "Tracked diffusers candidate",
                "styleTags": list(seed.get("styleTags") or []),
                "taskSupport": list(seed.get("taskSupport") or ["txt2img"]),
                "sizeGb": float(seed.get("sizeGb") or 0.0),
                "recommendedResolution": str(seed.get("recommendedResolution") or "Unknown"),
                "note": str(
                    seed.get("note")
                    or "Tracked latest image repo surfaced by ChaosEngineAI when the live latest lane is sparse."
                ),
                "availableLocally": _image_repo_runtime_ready(repo_id),
                "estimatedGenerationSeconds": None,
                "downloads": None,
                "likes": None,
                "downloadsLabel": None,
                "likesLabel": None,
                "lastModified": None,
                "updatedLabel": str(seed.get("updatedLabel") or "Tracked latest"),
                "license": seed.get("license"),
                "gated": seed.get("gated"),
                "pipelineTag": seed.get("pipelineTag"),
                "repoSizeBytes": None,
                "repoSizeGb": None,
                "coreWeightsBytes": None,
                "coreWeightsGb": None,
                "metadataWarning": "Showing ChaosEngineAI tracked latest defaults until live Hugging Face metadata is available.",
                "source": "latest",
            }
        )
    return payloads


def _is_latest_image_candidate(model: dict[str, Any], curated_repos: set[str]) -> bool:
    model_id = str(model.get("id") or "")
    if not model_id or model_id in curated_repos:
        return False
    lowered = model_id.lower()
    excluded_fragments = (
        "-lora",
        "controlnet",
        "ip-adapter",
        "tensorrt",
        "_amdgpu",
        "onnx",
        "instruct-pix2pix",
    )
    if any(fragment in lowered for fragment in excluded_fragments):
        return False

    tags = {str(tag).lower() for tag in (model.get("tags") or [])}
    pipeline_tag = str(model.get("pipeline_tag") or "").lower()
    allowed_orgs = {
        "black-forest-labs",
        "stabilityai",
        "qwen",
        "hidream-ai",
        "zai-org",
        "efficient-large-model",
        "hunyuanvideo-community",
        "tencent-hunyuan",
        "thudm",
    }
    provider = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    if provider and provider not in allowed_orgs:
        return False

    if "diffusers" not in tags:
        return False
    image_pipelines = {"text-to-image", "image-to-image", "inpainting"}
    if pipeline_tag in image_pipelines:
        return True
    if {"text-to-image", "image-generation", "image-to-image", "inpainting", "inpaint"} & tags:
        return True
    return False


def _latest_image_model_payloads(library: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    global _LATEST_IMAGE_MODELS_CACHE

    curated_repos = {
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    }

    now = time.time()
    cached_entries = _LATEST_IMAGE_MODELS_CACHE
    if cached_entries is not None and (now - cached_entries[0]) < _LATEST_IMAGE_MODELS_TTL_SECONDS:
        latest = cached_entries[1]
        return [
            {
                **entry,
                "availableLocally": _image_repo_runtime_ready(str(entry.get("repo") or "")),
            }
            for entry in latest
        ]

    try:
        params = urllib.parse.urlencode({
            "filter": "diffusers",
            "sort": "modified",
            "direction": "-1",
            "limit": "48",
            "full": "true",
        })
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        if cached_entries is not None:
            latest = cached_entries[1]
            return [
                {
                    **entry,
                    "availableLocally": _image_repo_runtime_ready(str(entry.get("repo") or "")),
                }
                for entry in latest
            ]
        return _tracked_latest_seed_payloads(library)[:limit]

    candidates: list[dict[str, Any]] = []
    for model in data:
        if not isinstance(model, dict) or not _is_latest_image_candidate(model, curated_repos):
            continue
        model_id = str(model.get("id") or "")
        provider = model_id.split("/", 1)[0] if "/" in model_id else "Community"
        tags = [str(tag) for tag in (model.get("tags") or [])]
        pipeline_tag = str(model.get("pipeline_tag") or "").strip() or None
        metadata = _image_repo_live_metadata(model_id)
        candidates.append({
            "id": model_id,
            "familyId": "latest",
            "familyName": "Latest Releases",
            "name": model_id.split("/", 1)[-1],
            "provider": provider,
            "repo": model_id,
            "link": f"https://huggingface.co/{model_id}",
            "runtime": "Diffusers candidate",
            "styleTags": _image_discover_style_tags(tags),
            "taskSupport": _image_task_support_from_metadata(pipeline_tag, tags),
            "sizeGb": float(metadata.get("coreWeightsGb") or metadata.get("repoSizeGb") or 0.0),
            "recommendedResolution": _image_recommended_resolution(model_id, pipeline_tag, tags),
            "note": (
                "Latest official diffusers-compatible image model tracked by ChaosEngineAI. "
                "Review details on Hugging Face before treating it as a fully curated Studio default."
            ),
            "availableLocally": _image_repo_runtime_ready(model_id),
            "estimatedGenerationSeconds": None,
            "downloads": metadata.get("downloads"),
            "likes": metadata.get("likes"),
            "downloadsLabel": metadata.get("downloadsLabel"),
            "likesLabel": metadata.get("likesLabel"),
            "lastModified": metadata.get("lastModified"),
            "updatedLabel": metadata.get("updatedLabel"),
            "license": metadata.get("license"),
            "gated": bool(metadata.get("gated")) if metadata.get("gated") is not None else None,
            "pipelineTag": metadata.get("pipelineTag") or pipeline_tag,
            "repoSizeBytes": metadata.get("repoSizeBytes"),
            "repoSizeGb": metadata.get("repoSizeGb"),
            "coreWeightsBytes": metadata.get("coreWeightsBytes"),
            "coreWeightsGb": metadata.get("coreWeightsGb"),
            "metadataWarning": metadata.get("metadataWarning"),
            "source": "latest",
        })

    candidates.sort(
        key=lambda entry: (
            _parse_iso_datetime(str(entry.get("lastModified") or "") or None) or datetime.min.replace(tzinfo=timezone.utc),
            int(entry.get("downloads") or 0),
            int(entry.get("likes") or 0),
        ),
        reverse=True,
    )
    seen_repos = {str(entry.get("repo") or "") for entry in candidates}
    for fallback in _tracked_latest_seed_payloads(library):
        repo_id = str(fallback.get("repo") or "")
        if repo_id in seen_repos:
            continue
        candidates.append(fallback)
        seen_repos.add(repo_id)

    latest = candidates[:limit]
    _LATEST_IMAGE_MODELS_CACHE = (now, latest)
    return latest


def _is_image_repo(repo_id: str) -> bool:
    return any(
        str(variant.get("repo") or "") == repo_id
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
    )


def _image_repo_runtime_ready(repo_id: str) -> bool:
    snapshot_dir = _hf_repo_snapshot_dir(repo_id)
    if snapshot_dir is None:
        return False
    return validate_local_diffusers_snapshot(snapshot_dir, repo_id) is None


def _image_variant_available_locally(variant: dict[str, Any], library: list[dict[str, Any]]) -> bool:
    repo = str(variant.get("repo") or "")
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


def _image_download_repo_ids() -> set[str]:
    repos = {
        str(variant.get("repo") or "")
        for family in IMAGE_MODEL_FAMILIES
        for variant in family["variants"]
        if str(variant.get("repo") or "")
    }
    repos.update(
        str(seed.get("repo") or "")
        for seed in LATEST_IMAGE_TRACKED_SEEDS
        if str(seed.get("repo") or "")
    )
    cached_entries = _LATEST_IMAGE_MODELS_CACHE
    if cached_entries is not None:
        repos.update(
            str(entry.get("repo") or "")
            for entry in cached_entries[1]
            if str(entry.get("repo") or "")
        )
    return repos


# ---- Image output CRUD ----

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
