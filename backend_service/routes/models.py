from __future__ import annotations

import json
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
)

router = APIRouter()


@router.get("/api/models/search")
def search_models(request: Request, query: str = Query("", alias="q", min_length=0, max_length=120)) -> dict[str, Any]:
    state = request.app.state.chaosengine
    from backend_service.helpers.discovery import _model_family_payloads

    system_stats = state._system_snapshot_provider()
    library = state._library()
    catalog = _model_family_payloads(system_stats, library)
    haystack = query.strip().lower()
    if not haystack:
        results = catalog
    else:
        results = [
            family
            for family in catalog
            if haystack in family["name"].lower()
            or haystack in family["provider"].lower()
            or any(haystack in capability for capability in family["capabilities"])
            or any(
                haystack in variant["name"].lower()
                or haystack in variant["format"].lower()
                or haystack in variant["quantization"].lower()
                or haystack in variant["repo"].lower()
                for variant in family["variants"]
            )
        ]

    # Also search HuggingFace Hub when there's a query
    hub_results: list[dict[str, Any]] = []
    if haystack and len(haystack) >= 2:
        hub_results = _search_huggingface_hub(haystack, library)

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
