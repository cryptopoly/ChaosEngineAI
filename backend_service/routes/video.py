"""Video generation API routes.

Backed by ``backend_service.video_runtime.VideoRuntimeManager``. The runtime
supports probe / preload / unload today — generation and downloads still
return 501 and land in the next phase.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.helpers.video import (
    _find_video_variant,
    _find_video_variant_by_repo,
    _video_download_validation_error,
    _video_model_payloads,
    _video_variant_available_locally,
)
from backend_service.models import (
    VideoRuntimePreloadRequest,
    VideoRuntimeUnloadRequest,
)


router = APIRouter()


@router.get("/api/video/catalog")
def video_catalog(request: Request) -> dict[str, Any]:
    """Return the curated catalog of video generation models."""
    library = request.app.state.chaosengine._library()
    return {
        "families": _video_model_payloads(library),
        "latest": [],
    }


@router.get("/api/video/runtime")
def video_runtime_status(request: Request) -> dict[str, Any]:
    """Report the live video runtime capability from diffusers + torch."""
    state = request.app.state.chaosengine
    return {"runtime": state.video_runtime.capabilities()}


@router.post("/api/video/preload")
def preload_video_model(request: Request, body: VideoRuntimePreloadRequest) -> dict[str, Any]:
    state = request.app.state.chaosengine
    state.add_log("video", "info", f"Preload requested: modelId='{body.modelId}'")
    variant = _find_video_variant(body.modelId)
    if variant is None:
        state.add_log("video", "error", f"Preload failed: model '{body.modelId}' not found")
        raise HTTPException(status_code=404, detail=f"Unknown video model '{body.modelId}'.")

    if not _video_variant_available_locally(variant):
        validation_error = _video_download_validation_error(variant["repo"])
        detail = validation_error or f"{variant['name']} is not installed locally yet."
        raise HTTPException(status_code=409, detail=detail)

    try:
        runtime = state.video_runtime.preload(variant["repo"])
    except RuntimeError as exc:
        state.add_log("video", "error", f"Failed to preload {variant['name']}: {exc}")
        raise HTTPException(status_code=400, detail=f"Failed to load {variant['name']}: {exc}") from exc
    except Exception as exc:
        state.add_log(
            "video",
            "error",
            f"Unexpected error preloading {variant['name']}: {type(exc).__name__}: {exc}",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load {variant['name']}: {type(exc).__name__}: {exc}",
        ) from exc

    state.add_log("video", "info", f"Preloaded video model {variant['name']}.")
    state.add_activity("Video model loaded", variant["name"])
    return {"runtime": runtime}


@router.post("/api/video/unload")
def unload_video_model(request: Request, body: VideoRuntimeUnloadRequest | None = None) -> dict[str, Any]:
    state = request.app.state.chaosengine
    requested_repo: str | None = None
    requested_name: str | None = None
    if body and body.modelId:
        variant = _find_video_variant(body.modelId)
        if variant is None:
            raise HTTPException(status_code=404, detail=f"Unknown video model '{body.modelId}'.")
        requested_repo = variant["repo"]
        requested_name = variant["name"]

    current_runtime = state.video_runtime.capabilities()
    current_repo = str(current_runtime.get("loadedModelRepo") or "") or None
    try:
        runtime = state.video_runtime.unload(requested_repo)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    unloaded_repo = requested_repo or current_repo
    if unloaded_repo and (requested_repo is None or requested_repo == current_repo):
        unloaded_variant = _find_video_variant_by_repo(unloaded_repo)
        unloaded_name = (
            unloaded_variant["name"]
            if unloaded_variant
            else requested_name or unloaded_repo
        )
        state.add_log("video", "info", f"Unloaded video model {unloaded_name}.")
        state.add_activity("Video model unloaded", unloaded_name)
    return {"runtime": runtime}


@router.get("/api/video/library")
def video_library(request: Request) -> dict[str, Any]:
    """Return the list of locally-installed video models."""
    state = request.app.state.chaosengine
    library = state._library()
    installed_models: list[dict[str, Any]] = []
    for family in _video_model_payloads(library):
        for variant in family["variants"]:
            if variant.get("availableLocally"):
                installed_models.append(variant)
    return {"models": installed_models}


@router.get("/api/video/outputs")
def video_outputs() -> dict[str, Any]:
    """Return saved video outputs. Empty until generation is implemented."""
    return {"outputs": []}


@router.post("/api/video/generate")
def generate_video(request: Request) -> dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Video generation is not implemented yet. The runtime is on the roadmap.",
    )


@router.post("/api/video/download")
def download_video_model(request: Request) -> dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Video model downloads are not implemented yet.",
    )
