from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

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
)

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
        raise HTTPException(status_code=500, detail=detail) from exc


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
        raise HTTPException(status_code=400, detail=detail) from exc


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
def hub_files(repo: str = Query(min_length=3, max_length=200)) -> dict[str, Any]:
    if "/" not in repo:
        raise HTTPException(status_code=400, detail="Repo must be in `owner/name` format.")
    try:
        return _hub_repo_files(repo)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
