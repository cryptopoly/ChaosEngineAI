"""HuggingFace Hub integration: search, file listing, repo metadata."""
from __future__ import annotations

import json
import os
import re
import urllib.error
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
_DISCOVER_SEARCH_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_DISCOVER_SEARCH_ALPHA_NUM_RE = re.compile(r"([a-z])(\d)|(\d)([a-z])")
_TEXT_DISCOVER_PIPELINES = {
    "text-generation",
    "image-text-to-text",
    "text2text-generation",
    "conversational",
    "visual-question-answering",
}


def _model_is_text_discover_candidate(model: dict[str, Any]) -> bool:
    pipeline_tag = str(model.get("pipeline_tag") or "").strip().lower()
    tags = [str(tag).strip().lower() for tag in (model.get("tags") or []) if str(tag).strip()]
    if pipeline_tag in _TEXT_DISCOVER_PIPELINES:
        return True
    if any(tag in _TEXT_DISCOVER_PIPELINES for tag in tags):
        return True
    capability_markers = {
        "conversational",
        "text-generation",
        "image-text-to-text",
        "tool-use",
        "vision-language",
        "vqa",
        "gguf",
        "mlx",
    }
    return any(tag in capability_markers for tag in tags)


def _normalize_hub_search_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return ""
    normalized = _DISCOVER_SEARCH_ALPHA_NUM_RE.sub(
        lambda match: f"{match.group(1) or match.group(3)} {match.group(2) or match.group(4)}",
        lowered,
    )
    normalized = _DISCOVER_SEARCH_PUNCT_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _hub_search_tokens(query: str) -> list[str]:
    normalized = _normalize_hub_search_text(query)
    return normalized.split() if normalized else []


def _hub_model_matches_query(model: dict[str, Any], query: str) -> bool:
    tokens = _hub_search_tokens(query)
    if not tokens:
        return True
    model_id = str(model.get("id") or "")
    provider = model_id.split("/", 1)[0] if "/" in model_id else model_id
    fragments = [
        model_id,
        provider,
        str(model.get("pipeline_tag") or ""),
        *(str(tag or "") for tag in (model.get("tags") or [])),
    ]
    haystack = _normalize_hub_search_text(" ".join(fragment for fragment in fragments if fragment))
    return all(token in haystack for token in tokens)


def _search_huggingface_hub(query: str, library: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for models matching the query."""
    try:
        params = urllib.parse.urlencode({
            "search": query,
            "limit": str(max(limit * 5, 60)),
            "sort": "modified",
            "direction": "-1",
            "full": "true",
        })
        url = f"https://huggingface.co/api/models?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for model in data:
        if not _model_is_text_discover_candidate(model):
            continue
        if not _hub_model_matches_query(model, query):
            continue
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
        last_modified = str(model.get("lastModified") or "").strip() or None

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
            "lastModified": last_modified,
            "updatedLabel": _format_hf_updated_label(last_modified),
            "availableLocally": available_locally,
            "launchMode": launch_mode,
            "backend": backend,
        })

    results.sort(
        key=lambda entry: (
            _parse_iso_datetime(str(entry.get("lastModified") or "") or None) or datetime.min.replace(tzinfo=timezone.utc),
            int(entry.get("downloads") or 0),
            int(entry.get("likes") or 0),
        ),
        reverse=True,
    )
    return results[:limit]


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


def _condense_hf_error(error: str) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    ignored_prefixes = (
        "traceback",
        "file ",
        "raise ",
        "for more information",
        "for more details",
    )
    ignored_substrings = (
        "userwarning",
        "warnings.warn",
        "httpstatuserror:",
        "repositorynotfounderror:",
    )
    for line in reversed(lines):
        lowered = line.lower()
        if lowered.startswith(ignored_prefixes):
            continue
        if any(fragment in lowered for fragment in ignored_substrings):
            continue
        return line
    return lines[-1] if lines else str(error).strip()


def _friendly_hf_download_error(repo_id: str, error: str) -> str:
    lowered = str(error).lower()
    if (
        "repository not found" in lowered
        or "repo not found" in lowered
        or "404 client error" in lowered
    ):
        return (
            f"{repo_id} was not found on Hugging Face. "
            "This repo may have moved or the catalog entry may be stale."
        )
    if (
        "refused access" in lowered
        or "http 401" in lowered
        or "http 403" in lowered
        or "invalid username or password" in lowered
        or "authentication required" in lowered
        or "cannot access gated repo" in lowered
        or "gated repo" in lowered
        or ("access to model" in lowered and "restricted" in lowered)
    ):
        return (
            f"Hugging Face refused access to {repo_id}. "
            "If the repo is gated or private, make sure your account has access "
            "and add a read-enabled HF_TOKEN in Settings."
        )
    if (
        "connecterror" in lowered
        or "name or service not known" in lowered
        or "nodename nor servname provided" in lowered
        or "temporary failure in name resolution" in lowered
        or "timed out" in lowered
    ):
        return (
            f"ChaosEngineAI could not reach Hugging Face while checking {repo_id}. "
            "Check the backend network connection and retry."
        )
    condensed = _condense_hf_error(error)
    return condensed or f"Download failed for {repo_id}."


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


def _hf_repo_preflight_size_gb(repo_id: str) -> float | None:
    payload = _hub_repo_files(repo_id)
    total_gb = payload.get("totalSizeGb")
    if isinstance(total_gb, (int, float)) and total_gb > 0:
        return float(total_gb)
    return None
