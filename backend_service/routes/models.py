from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend_service.i18n import localized_detail
from backend_service.models import (
    LoadModelRequest,
    ConvertModelRequest,
    RevealPathRequest,
    DeleteModelRequest,
    DownloadModelRequest,
)
from backend_service.helpers.discovery import (
    _list_weight_files,
)
from backend_service.helpers.huggingface import (
    _search_huggingface_hub,
    _hub_repo_files,
    _find_quantized_variants,
    _hf_token_value,
)
from backend_service.helpers.hf_resolve import resolve_hf_model

router = APIRouter()

_DISCOVER_SEARCH_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_DISCOVER_SEARCH_ALPHA_NUM_RE = re.compile(r"([a-z])(\d)|(\d)([a-z])")


def _normalize_discover_search_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return ""
    normalized = _DISCOVER_SEARCH_ALPHA_NUM_RE.sub(
        lambda match: f"{match.group(1) or match.group(3)} {match.group(2) or match.group(4)}",
        lowered,
    )
    normalized = _DISCOVER_SEARCH_PUNCT_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _discover_search_tokens(query: str) -> list[str]:
    normalized = _normalize_discover_search_text(query)
    return normalized.split() if normalized else []


def _family_discover_search_haystack(family: dict[str, Any]) -> str:
    fragments: list[str] = [
        str(family.get("name") or ""),
        str(family.get("provider") or ""),
        str(family.get("headline") or ""),
        str(family.get("summary") or ""),
        str(family.get("description") or ""),
        *(str(capability or "") for capability in family.get("capabilities") or []),
        *(str(line or "") for line in family.get("readme") or []),
    ]
    for variant in family.get("variants") or []:
        fragments.extend(
            [
                str(variant.get("name") or ""),
                str(variant.get("repo") or ""),
                str(variant.get("format") or ""),
                str(variant.get("quantization") or ""),
                str(variant.get("note") or ""),
                str(variant.get("contextWindow") or ""),
                *(str(capability or "") for capability in variant.get("capabilities") or []),
            ]
        )
    return _normalize_discover_search_text(" ".join(fragment for fragment in fragments if fragment))


def _family_matches_discover_query(family: dict[str, Any], query: str) -> bool:
    tokens = _discover_search_tokens(query)
    if not tokens:
        return True
    haystack_tokens = set(_discover_search_tokens(_family_discover_search_haystack(family)))
    return all(token in haystack_tokens for token in tokens)


@router.get("/api/models/search")
def search_models(request: Request, query: str = Query("", alias="q", min_length=0, max_length=120)) -> dict[str, Any]:
    state = request.app.state.chaosengine
    from backend_service.helpers.discovery import _model_family_payloads

    system_stats = state._system_snapshot_provider()
    library = state._library()
    catalog = _model_family_payloads(system_stats, library)
    search_query = query.strip()
    if not search_query:
        results = catalog
    else:
        results = [
            family
            for family in catalog
            if _family_matches_discover_query(family, search_query)
        ]

    # Also search HuggingFace Hub when there's a query
    hub_results: list[dict[str, Any]] = []
    if search_query and len(search_query) >= 2:
        hub_results = _search_huggingface_hub(search_query.lower(), library)

    return {"query": query, "results": results, "hubResults": hub_results}


@router.post("/api/models/load")
def load_model(request: Request, body: LoadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    try:
        runtime = state.load_model(body)
        return {"runtime": runtime}
    except HTTPException:
        raise
    except Exception as exc:
        detail = str(exc) or "Unknown error during model loading."
        state.add_log("runtime", "error", f"Load failed for {body.modelRef}: {detail}")
        raise HTTPException(status_code=500, detail=localized_detail(request, detail)) from exc


@router.post("/api/models/unload")
async def unload_model(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    ref: str | None = None
    try:
        body = await request.body()
        if body:
            payload = json.loads(body)
            if isinstance(payload, dict):
                ref = payload.get("ref")
    except Exception:
        ref = None
    runtime = state.unload_model(ref=ref)
    return {"runtime": runtime}


@router.post("/api/models/convert")
def convert_model(request: Request, body: ConvertModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    try:
        return state.convert_model(body)
    except RuntimeError as exc:
        detail = str(exc)
        state.add_log("conversion", "error", f"Conversion failed: {detail}")
        raise HTTPException(status_code=400, detail=localized_detail(request, detail)) from exc


@router.post("/api/models/reveal")
def reveal_model_path(request: Request, body: RevealPathRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.reveal_model_path(body.path)


@router.post("/api/models/delete")
def delete_model_path(request: Request, body: DeleteModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return state.delete_model_path(body.path)


@router.get("/api/models/list-weights")
def list_weights(path: str) -> dict[str, Any]:
    return _list_weight_files(path)


@router.get("/api/models/quantized-variants")
def quantized_variants(
    repo: str = Query(..., min_length=3, max_length=256),
) -> dict[str, Any]:
    """List community-quantized mirrors (GGUF, NF4) of a base HF repo.

    Used by the image + video Discover panes to surface quantized
    alternatives for a selected base model on demand, without
    pre-baking every city96/QuantStack mirror into the catalog.
    """
    return {"repo": repo, "variants": _find_quantized_variants(repo)}


@router.post("/api/models/download")
def download_model(request: Request, body: DownloadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"download": state.start_download(body.repo)}


@router.get("/api/models/download/status")
def download_status(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"downloads": state.download_status()}


@router.post("/api/models/download/cancel")
def cancel_download(request: Request, body: DownloadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"download": state.cancel_download(body.repo)}


@router.post("/api/models/download/delete")
def delete_download(request: Request, body: DownloadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"result": state.delete_download(body.repo)}


@router.get("/api/models/hub-search")
def hub_search(request: Request, query: str = Query("", alias="q", min_length=2, max_length=120)) -> dict[str, Any]:
    state = request.app.state.chaosengine
    library = state._library()
    results = _search_huggingface_hub(query.strip().lower(), library)
    return {"query": query, "results": results}


@router.get("/api/models/hub-files")
def hub_files(request: Request, repo: str = Query(min_length=3, max_length=200)) -> dict[str, Any]:
    if "/" not in repo:
        raise HTTPException(
            status_code=400,
            detail=localized_detail(request, "Repo must be in `owner/name` format."),
        )
    try:
        return _hub_repo_files(repo)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=localized_detail(request, str(exc)),
        ) from exc


class ResolveHfRequest(BaseModel):
    repo: str
    file: str | None = None


def _fetch_hf_config(repo: str) -> dict[str, Any] | None:
    """Best-effort read of a repo's ``config.json`` (tiny). None on any failure."""
    import urllib.error
    import urllib.parse
    import urllib.request

    encoded = urllib.parse.quote(repo, safe="/")
    url = f"https://huggingface.co/{encoded}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "ChaosEngineAI/0.2.0"})
    token = _hf_token_value()
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None


@router.post("/api/models/resolve-hf")
def resolve_hf(request: Request, body: ResolveHfRequest) -> dict[str, Any]:
    """Resolve an arbitrary HF repo into a loadable descriptor (#5).

    Reads the repo's file list + ``config.json`` to classify backend,
    pick a GGUF file, and infer context + capabilities — so off-catalog
    models run without fuzzy-matching to the wrong catalog row. The
    caller loads with ``canonicalRepo=<repo>`` to keep that contract.
    """
    repo = (body.repo or "").strip()
    # Accept a pasted URL as well as a bare ``owner/name``.
    if repo.startswith("http://") or repo.startswith("https://"):
        parts = [p for p in repo.split("huggingface.co/", 1)[-1].split("/") if p]
        repo = "/".join(parts[:2]) if len(parts) >= 2 else repo
    if "/" not in repo:
        raise HTTPException(
            status_code=400,
            detail=localized_detail(request, "Repo must be in `owner/name` format."),
        )
    try:
        files_payload = _hub_repo_files(repo)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=localized_detail(request, str(exc))) from exc

    files = files_payload.get("files") or files_payload.get("allFiles") or []
    config = _fetch_hf_config(repo)
    descriptor = resolve_hf_model(repo, files=files, config=config, requested_file=body.file)
    descriptor["totalSizeGb"] = round(descriptor["sizeBytes"] / 1e9, 2)
    return {"resolved": descriptor}


# ---------------------------------------------------------------------------
# Import existing Ollama / LM Studio models by reference (#4)
# ---------------------------------------------------------------------------


class ImportModelRequest(BaseModel):
    source: str  # "ollama" | "lmstudio"
    path: str
    name: str
    repo: str | None = None


@router.get("/api/models/import/scan")
def import_scan() -> dict[str, Any]:
    """Discover importable models in the Ollama blob store + LM Studio cache."""
    from backend_service.helpers.model_import import scan_importable

    return scan_importable()


@router.post("/api/models/import")
def import_model(request: Request, body: ImportModelRequest) -> dict[str, Any]:
    """Register an existing model by reference (symlink, no copy)."""
    from pathlib import Path

    from backend_service.app import DOCUMENTS_DIR
    from backend_service.helpers.model_import import import_by_reference, imported_dir
    from backend_service.helpers.settings import _save_settings

    if body.source not in {"ollama", "lmstudio"}:
        raise HTTPException(status_code=400, detail=localized_detail(request, "Unknown import source."))

    data_dir = DOCUMENTS_DIR.parent
    try:
        result = import_by_reference(source=body.source, path=body.path, name=body.name, data_dir=data_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=localized_detail(request, str(exc))) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=localized_detail(
                request,
                f"Could not link the model (symlinks may require elevated privileges on this OS): {exc}",
            ),
        ) from exc

    # Register the managed imported dir once so the library scan surfaces
    # every imported model. The fingerprint-based library cache refreshes
    # automatically when modelDirectories changes.
    state = request.app.state.chaosengine
    imported_root = str(imported_dir(data_dir))
    with state._lock:
        dirs = state.settings.setdefault("modelDirectories", [])
        already = any(
            str(Path(str(d.get("path") or "")).expanduser()) == imported_root for d in dirs
        )
        if not already:
            dirs.append(
                {
                    "path": imported_root,
                    "label": "Imported models",
                    "enabled": True,
                    "id": "imported-models",
                }
            )
            _save_settings(state.settings, state._settings_path)
            state.add_log("runtime", "info", f"Registered imported-models directory: {imported_root}")

    return {
        "imported": result,
        "repo": body.repo or body.name.split(":")[0],
        "name": body.name,
        "source": body.source,
    }
