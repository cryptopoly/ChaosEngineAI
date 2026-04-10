"""HuggingFace Hub integration: search, file listing, repo metadata."""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend_service.catalog import MODEL_FAMILIES, IMAGE_MODEL_FAMILIES
from backend_service.helpers.formatting import _bytes_to_gb
from backend_service.helpers.discovery import _path_size_bytes


_HF_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+/[a-zA-Z0-9_.\-]+$")
_HUB_FILE_CACHE: dict[str, dict[str, Any]] = {}


def _search_huggingface_hub(query: str, library: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for models matching the query."""
    try:
        params = urllib.parse.urlencode({
            "search": query,
            "limit": str(limit),
            "sort": "downloads",
            "direction": "-1",
            "filter": "text-generation",
        })
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for model in data:
        model_id = str(model.get("id") or "")
        if not model_id:
            continue
        tags = model.get("tags") or []
        tag_set = {t.lower() for t in tags}

        # Determine format
        is_gguf = "gguf" in tag_set
        is_mlx = any("mlx" in t for t in tag_set)
        fmt = "GGUF" if is_gguf else "MLX" if is_mlx else "Transformers"
        launch_mode = "direct" if is_gguf else "convert"
        backend = "llama.cpp" if is_gguf else "mlx"

        # Check local availability
        name_lower = model_id.lower()
        available_locally = any(
            name_lower in str(item.get("name", "")).lower()
            or name_lower in str(item.get("path", "")).lower()
            for item in library
        )

        # Extract author/org
        parts = model_id.split("/", 1)
        provider = parts[0] if len(parts) > 1 else "Community"

        downloads = model.get("downloads") or 0
        likes = model.get("likes") or 0

        results.append({
            "id": model_id,
            "repo": model_id,
            "name": parts[-1] if parts else model_id,
            "provider": provider,
            "link": f"https://huggingface.co/{model_id}",
            "format": fmt,
            "tags": tags,
            "downloads": downloads,
            "likes": likes,
            "downloadsLabel": f"{downloads:,} downloads",
            "likesLabel": f"{likes:,} likes",
            "availableLocally": available_locally,
            "launchMode": launch_mode,
            "backend": backend,
        })

    return results


def _classify_hub_file(name: str) -> str:
    lowered = name.lower()
    if lowered.endswith((".gguf", ".safetensors", ".bin", ".pt", ".pth")):
        if "mmproj" in lowered:
            return "vision_projector"
        return "weight"
    if lowered in {"config.json", "generation_config.json"}:
        return "config"
    if "tokenizer" in lowered or lowered in {"vocab.json", "merges.txt", "special_tokens_map.json"}:
        return "tokenizer"
    if lowered.startswith("readme") or lowered.endswith(".md"):
        return "readme"
    if lowered.endswith((".jinja", ".chat_template")):
        return "template"
    return "other"


def _hub_repo_file_payload(
    repo_id: str,
    files: list[dict[str, Any]],
    *,
    total_bytes: int | None = None,
    license_value: str | None = None,
    tags: list[str] | None = None,
    pipeline_tag: str | None = None,
    last_modified: str | None = None,
    warning: str | None = None,
) -> dict[str, Any]:
    files.sort(key=lambda entry: (-int(entry.get("sizeBytes") or 0), str(entry.get("path") or "")))
    effective_total = total_bytes if total_bytes is not None else sum(int(entry.get("sizeBytes") or 0) for entry in files)
    return {
        "repo": repo_id,
        "files": files,
        "totalSizeBytes": int(effective_total),
        "totalSizeGb": _bytes_to_gb(effective_total),
        "license": license_value,
        "tags": tags or [],
        "pipelineTag": pipeline_tag,
        "lastModified": last_modified,
        "warning": warning,
    }


def _hub_repo_files(repo_id: str) -> dict[str, Any]:
    """Return file list + metadata for a Hugging Face repo.

    Uses the public REST API so transient failures on the heavier tree
    endpoint do not make Discover look broken. Honours HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN for gated repos and degrades to a non-fatal
    warning on transient upstream 5xx errors.
    """
    cached = _HUB_FILE_CACHE.get(repo_id)
    if cached is not None:
        return cached

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        encoded_repo = urllib.parse.quote(repo_id, safe="/")
        url = f"https://huggingface.co/api/models/{encoded_repo}?blobs=true"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        status = getattr(exc, "code", None)
        if status in (401, 403):
            raise RuntimeError(
                f"Hugging Face refused access to {repo_id} (HTTP {status}). "
                f"Set HF_TOKEN in Settings."
            ) from exc
        if status == 404:
            raise RuntimeError(f"Hugging Face repository not found: {repo_id}") from exc
        if status is not None and status >= 500:
            return _hub_repo_file_payload(
                repo_id,
                [],
                warning=(
                    f"Hugging Face file preview is temporarily unavailable (HTTP {status}). "
                    "You can still download this repo."
                ),
            )
        raise RuntimeError(f"Hugging Face request failed: {exc}") from exc
    except (OSError, json.JSONDecodeError):
        return _hub_repo_file_payload(
            repo_id,
            [],
            warning=(
                "Hugging Face file preview is temporarily unavailable right now. "
                "You can still download this repo."
            ),
        )

    all_files: list[dict[str, Any]] = []
    display_files: list[dict[str, Any]] = []
    total_bytes = 0
    for sibling in data.get("siblings") or []:
        path = str(sibling.get("rfilename") or "")
        if not path:
            continue
        lfs = sibling.get("lfs") if isinstance(sibling.get("lfs"), dict) else {}
        size_bytes = sibling.get("size") or lfs.get("size") or 0
        try:
            size_int = int(size_bytes)
        except (TypeError, ValueError):
            size_int = 0
        record = {
            "path": path,
            "sizeBytes": size_int,
            "sizeGb": _bytes_to_gb(size_int),
            "kind": _classify_hub_file(path),
        }
        total_bytes += size_int
        all_files.append(record)
        if "/" not in path:
            display_files.append(record)

    if not display_files:
        display_files = all_files[:40]

    card = data.get("cardData") or {}
    license_value = card.get("license") if isinstance(card, dict) else None
    payload = _hub_repo_file_payload(
        repo_id,
        display_files,
        total_bytes=total_bytes,
        license_value=license_value,
        tags=list(data.get("tags") or []),
        pipeline_tag=data.get("pipeline_tag"),
        last_modified=data.get("lastModified"),
    )
    _HUB_FILE_CACHE[repo_id] = payload
    return payload


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _format_hf_updated_label(value: str | None) -> str | None:
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    now = datetime.now(timezone.utc)
    month_label = parsed.strftime("%b")
    if parsed.year == now.year:
        return f"Updated {month_label} {parsed.day}"
    return f"Updated {month_label} {parsed.day}, {parsed.year}"


def _hf_number_label(value: int, noun: str) -> str:
    return f"{value:,} {noun}"


def _hf_hub_cache_root() -> Path:
    explicit = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(os.path.expanduser(explicit)).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(os.path.expanduser(hf_home)).expanduser() / "hub"
    # Use huggingface_hub's own cache constant when available -- it handles
    # platform differences (Windows uses LOCALAPPDATA or userprofile).
    try:
        from huggingface_hub import constants as _hf_constants
        return Path(_hf_constants.HF_HUB_CACHE)
    except Exception:
        pass
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cache_dir(repo_id: str) -> Path:
    return _hf_hub_cache_root() / f"models--{repo_id.replace('/', '--')}"


def _hf_repo_downloaded_bytes(repo_id: str) -> int:
    cache_dir = _hf_repo_cache_dir(repo_id)
    try:
        if not cache_dir.exists():
            return 0
    except OSError:
        return 0
    try:
        return _path_size_bytes(cache_dir)
    except OSError:
        return 0


def _hf_repo_snapshot_dir(repo_id: str) -> Path | None:
    cache_dir = _hf_repo_cache_dir(repo_id)
    snapshots_dir = cache_dir / "snapshots"
    ref_path = cache_dir / "refs" / "main"
    try:
        if ref_path.exists():
            revision = ref_path.read_text(encoding="utf-8").strip()
            if revision:
                candidate = snapshots_dir / revision
                if candidate.exists():
                    return candidate
    except OSError:
        pass

    try:
        snapshots = sorted(
            [candidate for candidate in snapshots_dir.iterdir() if candidate.is_dir()],
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return None
    return snapshots[0] if snapshots else None


def _known_repo_size_gb(repo_id: str) -> float | None:
    cached = _HUB_FILE_CACHE.get(repo_id)
    if cached is not None:
        cached_total = cached.get("totalSizeGb")
        if isinstance(cached_total, (int, float)) and cached_total > 0:
            return float(cached_total)

    for family in MODEL_FAMILIES:
        for variant in family["variants"]:
            if str(variant.get("repo") or "") != repo_id:
                continue
            try:
                size_gb = float(variant.get("sizeGb") or 0)
            except (TypeError, ValueError):
                size_gb = 0.0
            if size_gb > 0:
                return size_gb

    for family in IMAGE_MODEL_FAMILIES:
        for variant in family["variants"]:
            if str(variant.get("repo") or "") != repo_id:
                continue
            try:
                size_gb = float(variant.get("sizeGb") or 0)
            except (TypeError, ValueError):
                size_gb = 0.0
            if size_gb > 0:
                return size_gb

    return None
