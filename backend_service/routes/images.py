from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.models import (
    DownloadModelRequest,
    ImageGenerationRequest,
    ImageRuntimePreloadRequest,
    ImageRuntimeUnloadRequest,
)
from backend_service.helpers.images import (
    _image_model_payloads,
    _find_image_variant,
    _find_image_variant_by_repo,
    _image_variant_available_locally,
    _image_download_validation_error,
    _image_download_repo_ids,
    _latest_image_model_payloads,
)
from backend_service.app import (
    _generate_image_artifacts,
    _load_image_outputs,
    _find_image_output,
    _delete_image_output,
)

router = APIRouter()


@router.get("/api/images/catalog")
def image_catalog(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    library = state._library()
    return {
        "families": _image_model_payloads(library),
        "latest": _latest_image_model_payloads(library),
    }


@router.get("/api/images/runtime")
def image_runtime_status(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"runtime": state.image_runtime.capabilities()}


@router.post("/api/images/preload")
def preload_image_model(request: Request, body: ImageRuntimePreloadRequest) -> dict[str, Any]:
    import traceback as _tb
    state = request.app.state.chaosengine
    state.add_log("images", "info", f"Preload requested: modelId='{body.modelId}'")
    variant = _find_image_variant(body.modelId)
    if variant is None:
        state.add_log("images", "error", f"Preload failed: model '{body.modelId}' not found")
        raise HTTPException(status_code=404, detail=f"Unknown image model '{body.modelId}'.")
    library = state._library()
    if not _image_variant_available_locally(variant, library):
        validation_error = _image_download_validation_error(variant["repo"])
        detail = validation_error or f"{variant['name']} is not installed locally yet."
        raise HTTPException(status_code=409, detail=detail)
    try:
        runtime = state.image_runtime.preload(variant["repo"])
    except RuntimeError as exc:
        state.add_log("images", "error", f"Failed to preload {variant['name']}: {exc}")
        raise HTTPException(status_code=400, detail=f"Failed to load {variant['name']}: {exc}") from exc
    except Exception as exc:
        state.add_log("images", "error", f"Unexpected error preloading {variant['name']}: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to load {variant['name']}: {type(exc).__name__}: {exc}") from exc
    state.add_log("images", "info", f"Preloaded image model {variant['name']}.")
    state.add_activity("Image model loaded", variant["name"])
    return {"runtime": runtime}


@router.post("/api/images/unload")
def unload_image_model(request: Request, body: ImageRuntimeUnloadRequest | None = None) -> dict[str, Any]:
    state = request.app.state.chaosengine
    requested_repo: str | None = None
    requested_name: str | None = None
    if body and body.modelId:
        variant = _find_image_variant(body.modelId)
        if variant is None:
            raise HTTPException(status_code=404, detail=f"Unknown image model '{body.modelId}'.")
        requested_repo = variant["repo"]
        requested_name = variant["name"]
    current_runtime = state.image_runtime.capabilities()
    current_repo = str(current_runtime.get("loadedModelRepo") or "") or None
    try:
        runtime = state.image_runtime.unload(requested_repo)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    unloaded_repo = requested_repo or current_repo
    if unloaded_repo and (requested_repo is None or requested_repo == current_repo):
        unloaded_variant = _find_image_variant_by_repo(unloaded_repo)
        unloaded_name = unloaded_variant["name"] if unloaded_variant else requested_name or unloaded_repo
        state.add_log("images", "info", f"Unloaded image model {unloaded_name}.")
        state.add_activity("Image model unloaded", unloaded_name)
    return {"runtime": runtime}


@router.get("/api/images/library")
def image_library(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    library = state._library()
    installed_models: list[dict[str, Any]] = []
    for family in _image_model_payloads(library):
        for variant in family["variants"]:
            if variant.get("availableLocally"):
                installed_models.append(
                    {
                        **variant,
                        "familyName": family["name"],
                    }
                )
    return {"models": installed_models}


@router.post("/api/images/download")
def download_image_model(request: Request, body: DownloadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"download": state.start_download(body.repo)}


@router.get("/api/images/download/status")
def image_download_status(request: Request) -> dict[str, Any]:
    state = request.app.state.chaosengine
    image_repos = _image_download_repo_ids()
    downloads = [
        item
        for item in state.download_status()
        if str(item.get("repo") or "") in image_repos
    ]
    return {"downloads": downloads}


@router.post("/api/images/download/cancel")
def cancel_image_download(request: Request, body: DownloadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"download": state.cancel_download(body.repo)}


@router.post("/api/images/download/delete")
def delete_image_download(request: Request, body: DownloadModelRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    return {"result": state.delete_download(body.repo)}


@router.post("/api/images/generate")
def generate_image(request: Request, body: ImageGenerationRequest) -> dict[str, Any]:
    import traceback as _tb
    state = request.app.state.chaosengine
    state.add_log("images", "info", f"Image generation requested: modelId='{body.modelId}', {body.width}x{body.height}, {body.steps} steps")
    variant = _find_image_variant(body.modelId)
    if variant is None:
        state.add_log("images", "error", f"Image model not found in catalog or tracked seeds: '{body.modelId}'")
        raise HTTPException(status_code=404, detail=f"Unknown image model '{body.modelId}'. The model isn't in the curated catalog or tracked seeds.")
    state.add_log("images", "info", f"Resolved variant: {variant.get('name')} (repo={variant.get('repo')})")
    try:
        artifacts, runtime = _generate_image_artifacts(body, variant, state.image_runtime)
    except Exception as exc:
        tb_str = _tb.format_exc()
        state.add_log("images", "error", f"Image generation FAILED for {variant.get('name')}: {type(exc).__name__}: {exc}")
        state.add_log("images", "error", f"Traceback:\n{tb_str[-500:]}")
        raise HTTPException(status_code=500, detail=f"Image generation failed for {variant.get('name')}: {type(exc).__name__}: {exc}") from exc
    state.add_log(
        "images",
        "info",
        f"Generated {len(artifacts)} image(s) with {variant['name']} via {runtime.get('activeEngine', 'unknown')}.",
    )
    state.add_activity(
        "Image generated",
        f"{variant['name']} \u00b7 {body.width}x{body.height}",
    )
    return {"artifacts": artifacts, "outputs": _load_image_outputs(), "runtime": runtime}


@router.get("/api/images/outputs")
def image_outputs() -> dict[str, Any]:
    return {"outputs": _load_image_outputs()}


@router.get("/api/images/outputs/{artifact_id}")
def image_output_detail(artifact_id: str) -> dict[str, Any]:
    output = _find_image_output(artifact_id)
    if output is None:
        raise HTTPException(status_code=404, detail=f"Image output '{artifact_id}' not found.")
    return {"artifact": output}


@router.delete("/api/images/outputs/{artifact_id}")
def delete_image_output_endpoint(request: Request, artifact_id: str) -> dict[str, Any]:
    state = request.app.state.chaosengine
    deleted = _delete_image_output(artifact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Image output '{artifact_id}' not found.")
    state.add_log("images", "info", f"Deleted image output {artifact_id}.")
    return {"deleted": artifact_id, "outputs": _load_image_outputs()}
