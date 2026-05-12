"""Image model helpers: variant lookup, metadata, generation artifacts, output CRUD."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend_service.catalog import IMAGE_MODEL_FAMILIES, LATEST_IMAGE_TRACKED_SEEDS
from backend_service.helpers.formatting import _bytes_to_gb
from backend_service.helpers.huggingface import (
    _classify_hub_file,
    _format_hf_updated_label,
    _format_release_label,
    _hf_number_label,
    _hf_repo_snapshot_dir,
    _hf_token_cache_key,
    _hf_token_value,
    _parse_iso_datetime,
)
from backend_service.helpers.discovery import _candidate_model_dirs, _path_size_bytes
from backend_service.helpers.image_artifacts import (
    _delete_image_output,
    _find_image_output,
    _hydrate_image_artifact,
    _image_output_directory,
    _load_image_outputs,
    _placeholder_image_data_url,
    _preview_data_url_from_image_path,
    _save_image_artifact,
    _stable_image_hash,
)
from backend_service.helpers.image_validation import (
    _friendly_image_download_error,
    _image_download_validation_error,
    _image_repo_runtime_ready,
    _image_variant_available_locally,
    _is_image_repo,
)
from backend_service.helpers.platform_filter import (
    filter_mlx_only_families,
    is_apple_silicon,
)
from backend_service.image_runtime import validate_local_diffusers_snapshot


_IMAGE_DISCOVER_METADATA_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_IMAGE_DISCOVER_METADATA_TTL_SECONDS = 6 * 60 * 60
_LATEST_IMAGE_MODELS_CACHE: tuple[float, str, list[dict[str, Any]]] | None = None
_LATEST_IMAGE_MODELS_TTL_SECONDS = 3 * 60 * 60

# Cache keyed by (path, mtime_ns) — we recompute only when the snapshot dir
# actually changes. A fresh os.stat() is cheap enough to do per payload call.
_SNAPSHOT_SIZE_CACHE: dict[tuple[str, int], int] = {}


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed > 0:
        return parsed
    return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed > 0:
        return parsed
    return None


def _image_seed_size_metadata(seed: dict[str, Any]) -> tuple[float, float | None, float | None]:
    catalog_size_gb = _positive_float(seed.get("sizeGb"))
    core_weights_gb = _positive_float(seed.get("coreWeightsGb")) or catalog_size_gb
    repo_size_gb = _positive_float(seed.get("repoSizeGb"))
    size_gb = core_weights_gb or repo_size_gb or catalog_size_gb or 0.0
    return float(size_gb), core_weights_gb, repo_size_gb


def _tracked_seed_for_repo(repo_id: str) -> dict[str, Any] | None:
    for seed in LATEST_IMAGE_TRACKED_SEEDS:
        if str(seed.get("repo") or "") == repo_id:
            return seed
    return None


def _clear_image_discover_caches() -> None:
    global _LATEST_IMAGE_MODELS_CACHE
    _IMAGE_DISCOVER_METADATA_CACHE.clear()
    _LATEST_IMAGE_MODELS_CACHE = None


def _snapshot_on_disk_bytes(snapshot_dir: Path | None) -> int | None:
    """Walk the HF snapshot dir and return its true on-disk byte size.

    Delegates to ``_path_size_bytes`` which dedupes by inode, so HF's
    ``snapshots/<commit>/ -> blobs/<hash>`` symlink farm counts each blob
    exactly once. Returns ``None`` when the path is missing or empty so
    callers can distinguish "not on disk" from "zero bytes".
    """
    if snapshot_dir is None:
        return None
    try:
        stat_result = snapshot_dir.stat()
    except OSError:
        return None
    cache_key = (str(snapshot_dir), stat_result.st_mtime_ns)
    cached = _SNAPSHOT_SIZE_CACHE.get(cache_key)
    if cached is not None:
        return cached or None
    total = _path_size_bytes(snapshot_dir)
    _SNAPSHOT_SIZE_CACHE[cache_key] = total
    return total or None


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
        variants = []
        for variant in family["variants"]:
            repo_id = str(variant.get("repo") or "")
            snapshot_dir = _hf_repo_snapshot_dir(repo_id) if repo_id else None
            live_metadata = repo_metadata.get(repo_id, {})
            curated_release_date = str(variant.get("releaseDate") or "").strip() or None
            curated_release_label = _format_release_label(curated_release_date)
            release_label = curated_release_label or live_metadata.get("releaseLabel")
            on_disk_bytes = _snapshot_on_disk_bytes(snapshot_dir)
            variants.append(
                {
                    **variant,
                    **live_metadata,
                    "source": "curated",
                    "familyName": family.get("name"),
                    "availableLocally": _image_variant_available_locally(variant, library),
                    "hasLocalData": snapshot_dir is not None,
                    "localPath": str(snapshot_dir) if snapshot_dir else None,
                    "releaseDate": curated_release_date,
                    "releaseLabel": release_label,
                    "onDiskBytes": on_disk_bytes,
                    "onDiskGb": _bytes_to_gb(on_disk_bytes) if on_disk_bytes else None,
                }
            )
        families.append(
            {
                **family,
                "updatedLabel": _best_image_family_updated_label(family, variants),
                "variants": variants,
            }
        )
    return filter_mlx_only_families(families, on_apple_silicon=is_apple_silicon())


def _find_image_variant(model_id: str) -> dict[str, Any] | None:
    # Search curated families first
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["id"] == model_id:
                return variant
    # Search tracked latest seeds (their id == repo)
    for seed in LATEST_IMAGE_TRACKED_SEEDS:
        repo = str(seed.get("repo") or "")
        if repo == model_id:
            size_gb, core_weights_gb, repo_size_gb = _image_seed_size_metadata(seed)
            return {
                "id": repo,
                "repo": repo,
                "name": seed.get("name") or repo.split("/", 1)[-1],
                "provider": seed.get("provider") or "Community",
                "sizeGb": size_gb,
                "runtimeFootprintGb": seed.get("runtimeFootprintGb"),
                "runtimeFootprintMpsGb": seed.get("runtimeFootprintMpsGb"),
                "runtimeFootprintCudaGb": seed.get("runtimeFootprintCudaGb"),
                "runtimeFootprintCpuGb": seed.get("runtimeFootprintCpuGb"),
                "coreWeightsGb": core_weights_gb,
                "repoSizeGb": repo_size_gb,
                "styleTags": list(seed.get("styleTags") or []),
                "taskSupport": list(seed.get("taskSupport") or ["txt2img"]),
                "recommendedResolution": seed.get("recommendedResolution") or "1024x1024",
            }
    return None


def _find_image_variant_by_repo(repo: str) -> dict[str, Any] | None:
    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if variant["repo"] == repo:
                return variant
    # Search tracked latest seeds
    for seed in LATEST_IMAGE_TRACKED_SEEDS:
        seed_repo = str(seed.get("repo") or "")
        if seed_repo == repo:
            size_gb, core_weights_gb, repo_size_gb = _image_seed_size_metadata(seed)
            return {
                "id": seed_repo,
                "repo": seed_repo,
                "name": seed.get("name") or seed_repo.split("/", 1)[-1],
                "provider": seed.get("provider") or "Community",
                "sizeGb": size_gb,
                "runtimeFootprintGb": seed.get("runtimeFootprintGb"),
                "runtimeFootprintMpsGb": seed.get("runtimeFootprintMpsGb"),
                "runtimeFootprintCudaGb": seed.get("runtimeFootprintCudaGb"),
                "runtimeFootprintCpuGb": seed.get("runtimeFootprintCpuGb"),
                "coreWeightsGb": core_weights_gb,
                "repoSizeGb": repo_size_gb,
                "styleTags": list(seed.get("styleTags") or []),
                "taskSupport": list(seed.get("taskSupport") or ["txt2img"]),
                "recommendedResolution": seed.get("recommendedResolution") or "1024x1024",
            }
    return None


def _image_repo_live_metadata(repo_id: str) -> dict[str, Any]:
    now = time.time()
    cache_key = (repo_id, _hf_token_cache_key())
    cached = _IMAGE_DISCOVER_METADATA_CACHE.get(cache_key)
    if cached is not None:
        cached_at, payload = cached
        if (now - cached_at) < _IMAGE_DISCOVER_METADATA_TTL_SECONDS:
            return payload

    token = _hf_token_value()
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
        used_storage_bytes = _positive_int(data.get("usedStorage"))
        for sibling in data.get("siblings") or []:
            if not isinstance(sibling, dict):
                continue
            path = str(sibling.get("rfilename") or "")
            if not path:
                continue
            lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
            size_int = _positive_int(sibling.get("size")) or _positive_int(lfs.get("size")) or 0
            total_bytes += size_int
            if _classify_hub_file(path) == "weight":
                weight_bytes += size_int
        if total_bytes <= 0 and used_storage_bytes is not None:
            total_bytes = used_storage_bytes

        card = data.get("cardData") or {}
        license_value = str(card.get("license") or "").strip() or None if isinstance(card, dict) else None
        downloads = int(data.get("downloads") or 0)
        likes = int(data.get("likes") or 0)
        last_modified = str(data.get("lastModified") or "").strip() or None
        created_at = str(data.get("createdAt") or "").strip() or None
        payload = {
            "downloads": downloads,
            "likes": likes,
            "downloadsLabel": _hf_number_label(downloads, "downloads") if downloads > 0 else None,
            "likesLabel": _hf_number_label(likes, "likes") if likes > 0 else None,
            "lastModified": last_modified,
            "updatedLabel": _format_hf_updated_label(last_modified),
            "createdAt": created_at,
            "releaseLabel": _format_release_label(created_at),
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

    _IMAGE_DISCOVER_METADATA_CACHE[cache_key] = (now, payload)
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
        release_date = str(seed.get("releaseDate") or "").strip() or None
        snapshot_dir = _hf_repo_snapshot_dir(repo_id)
        on_disk_bytes = _snapshot_on_disk_bytes(snapshot_dir)
        size_gb, core_weights_gb, repo_size_gb = _image_seed_size_metadata(seed)
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
                "sizeGb": size_gb,
                "runtimeFootprintGb": seed.get("runtimeFootprintGb"),
                "runtimeFootprintMpsGb": seed.get("runtimeFootprintMpsGb"),
                "runtimeFootprintCudaGb": seed.get("runtimeFootprintCudaGb"),
                "runtimeFootprintCpuGb": seed.get("runtimeFootprintCpuGb"),
                "recommendedResolution": str(seed.get("recommendedResolution") or "Unknown"),
                "note": str(
                    seed.get("note")
                    or "Tracked latest image repo surfaced by ChaosEngineAI when the live latest lane is sparse."
                ),
                "availableLocally": _image_repo_runtime_ready(repo_id),
                "hasLocalData": snapshot_dir is not None,
                "localPath": str(snapshot_dir) if snapshot_dir else None,
                "onDiskBytes": on_disk_bytes,
                "onDiskGb": _bytes_to_gb(on_disk_bytes) if on_disk_bytes else None,
                "estimatedGenerationSeconds": None,
                "downloads": None,
                "likes": None,
                "downloadsLabel": None,
                "likesLabel": None,
                "lastModified": None,
                "updatedLabel": str(seed.get("updatedLabel") or "Tracked latest"),
                "createdAt": None,
                "releaseDate": release_date,
                "releaseLabel": _format_release_label(release_date),
                "license": seed.get("license"),
                "gated": seed.get("gated"),
                "pipelineTag": seed.get("pipelineTag"),
                "repoSizeBytes": None,
                "repoSizeGb": repo_size_gb,
                "coreWeightsBytes": None,
                "coreWeightsGb": core_weights_gb,
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
        "_lora",
        "lora-",
        "controlnet",
        "ip-adapter",
        "adapter",
        "tensorrt",
        "_amdgpu",
        "onnx",
        "embedding",
        "instruct-pix2pix",
    )
    if any(fragment in lowered for fragment in excluded_fragments):
        return False

    tags = {str(tag).lower() for tag in (model.get("tags") or [])}
    pipeline_tag = str(model.get("pipeline_tag") or "").lower()
    excluded_tags = {
        "lora",
        "controlnet",
        "adapter",
        "adapters",
        "textual-inversion",
        "embedding",
        "embeddings",
        "onnx",
    }
    if tags & excluded_tags:
        return False

    trusted_providers = {
        "black-forest-labs",
        "baidu",
        "stabilityai",
        "qwen",
        "hidream-ai",
        "zai-org",
        "tongyi-mai",
        "nucleusai",
        "efficient-large-model",
        "hunyuanvideo-community",
        "tencent-hunyuan",
        "thudm",
        "diffusers",
    }
    provider = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
    try:
        downloads = int(model.get("downloads") or 0)
    except (TypeError, ValueError):
        downloads = 0
    try:
        likes = int(model.get("likes") or 0)
    except (TypeError, ValueError):
        likes = 0
    if provider and provider not in trusted_providers and downloads < 1000 and likes < 25:
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
    token_cache_key = _hf_token_cache_key()
    cached_entries = _LATEST_IMAGE_MODELS_CACHE
    if (
        cached_entries is not None
        and cached_entries[1] == token_cache_key
        and (now - cached_entries[0]) < _LATEST_IMAGE_MODELS_TTL_SECONDS
    ):
        latest = cached_entries[2]
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
            "sort": "createdAt",
            "direction": "-1",
            "limit": "96",
            "full": "true",
        })
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        token = _hf_token_value()
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        if cached_entries is not None and cached_entries[1] == token_cache_key:
            latest = cached_entries[2]
            return [
                {
                    **entry,
                    "availableLocally": _image_repo_runtime_ready(str(entry.get("repo") or "")),
                }
                for entry in latest
            ]
        return _tracked_latest_seed_payloads(library)[:limit]

    accepted_models: list[dict[str, Any]] = []
    for model in data:
        if not isinstance(model, dict) or not _is_latest_image_candidate(model, curated_repos):
            continue
        accepted_models.append(model)
        if len(accepted_models) >= max(limit * 2, limit):
            break

    candidates: list[dict[str, Any]] = []
    for model in accepted_models:
        model_id = str(model.get("id") or "")
        provider = model_id.split("/", 1)[0] if "/" in model_id else "Community"
        tags = [str(tag) for tag in (model.get("tags") or [])]
        pipeline_tag = str(model.get("pipeline_tag") or "").strip() or None
        metadata = _image_repo_live_metadata(model_id)
        snapshot_dir = _hf_repo_snapshot_dir(model_id)
        on_disk_bytes = _snapshot_on_disk_bytes(snapshot_dir)
        on_disk_gb = _bytes_to_gb(on_disk_bytes) if on_disk_bytes else None
        tracked_seed = _tracked_seed_for_repo(model_id)
        fallback_size_gb, fallback_core_weights_gb, fallback_repo_size_gb = (
            _image_seed_size_metadata(tracked_seed)
            if tracked_seed is not None
            else (0.0, None, None)
        )
        core_weights_gb = _positive_float(metadata.get("coreWeightsGb")) or fallback_core_weights_gb
        repo_size_gb = _positive_float(metadata.get("repoSizeGb")) or fallback_repo_size_gb
        size_gb = (
            _positive_float(metadata.get("coreWeightsGb"))
            or _positive_float(metadata.get("repoSizeGb"))
            or _positive_float(on_disk_gb)
            or _positive_float(fallback_size_gb)
            or 0.0
        )
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
            "sizeGb": size_gb,
            "recommendedResolution": _image_recommended_resolution(model_id, pipeline_tag, tags),
            "note": (
                "Latest official diffusers-compatible image model tracked by ChaosEngineAI. "
                "Review details on Hugging Face before treating it as a fully curated Studio default."
            ),
            "availableLocally": _image_repo_runtime_ready(model_id),
            "hasLocalData": snapshot_dir is not None,
            "localPath": str(snapshot_dir) if snapshot_dir else None,
            "onDiskBytes": on_disk_bytes,
            "onDiskGb": on_disk_gb,
            "estimatedGenerationSeconds": None,
            "downloads": metadata.get("downloads"),
            "likes": metadata.get("likes"),
            "downloadsLabel": metadata.get("downloadsLabel"),
            "likesLabel": metadata.get("likesLabel"),
            "lastModified": metadata.get("lastModified"),
            "updatedLabel": metadata.get("updatedLabel"),
            "createdAt": metadata.get("createdAt"),
            "releaseLabel": metadata.get("releaseLabel"),
            "license": metadata.get("license"),
            "gated": bool(metadata.get("gated")) if metadata.get("gated") is not None else None,
            "pipelineTag": metadata.get("pipelineTag") or pipeline_tag,
            "repoSizeBytes": metadata.get("repoSizeBytes"),
            "repoSizeGb": repo_size_gb,
            "coreWeightsBytes": metadata.get("coreWeightsBytes"),
            "coreWeightsGb": core_weights_gb,
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
    _LATEST_IMAGE_MODELS_CACHE = (now, token_cache_key, latest)
    return latest


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
            for entry in cached_entries[2]
            if str(entry.get("repo") or "")
        )
    return repos


# Diffusers image pipelines (FLUX, SD3.5, SDXL, Sana, HiDream, Qwen-Image, ...)
# always load from the per-component folder layout at the snapshot root. Many
# repos also ship a legacy single-file checkpoint (e.g. ``flux1-schnell.safetensors``
# in ``black-forest-labs/FLUX.1-schnell``) for single-file loaders — ~24 GB of
# duplicate weights the diffusers pipeline never touches. Without an allowlist
# ``snapshot_download`` pulls both copies, so a 23 GB model lands on disk as
# 57+ GB. Mirrors ``_VIDEO_DIFFUSERS_ALLOW_PATTERNS`` in ``helpers/video.py``.
_IMAGE_DIFFUSERS_ALLOW_PATTERNS: list[str] = [
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "text_encoder_2/**",
    "text_encoder_3/**",
    "tokenizer/**",
    "tokenizer_2/**",
    "tokenizer_3/**",
    "transformer/**",
    "transformer_2/**",
    "unet/**",
    "vae/**",
    "feature_extractor/**",
    "image_encoder/**",
    "safety_checker/**",
    "*.md",
    "LICENSE*",
]


def _image_repo_allow_patterns(repo_id: str) -> list[str] | None:
    """Patterns to pass to ``snapshot_download`` for an image repo.

    Returns ``None`` for repos that aren't known curated or tracked image
    models so arbitrary Discover hub results still download in full. Returning
    ``None`` (not an empty list) signals the caller to omit ``allow_patterns``
    entirely — an empty list would match nothing and download zero files.
    """
    if not repo_id:
        return None
    known = _image_download_repo_ids()
    if repo_id not in known:
        return None
    return list(_IMAGE_DIFFUSERS_ALLOW_PATTERNS)


# ---- Image output CRUD ----
