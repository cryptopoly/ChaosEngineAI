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
from backend_service.progress import GenerationCancelled, IMAGE_PROGRESS, VIDEO_PROGRESS

router = APIRouter()


def _unload_idle_video_runtime_for_image(request: Request, action: str) -> None:
    """Free resident video diffusion weights before image work starts.

    Image and video pipelines live in separate managers, so loading an image
    model no longer implicitly releases a previously-loaded video model. That
    can leave tens of GB resident across Studio switches. If video generation
    is actively running, fail fast instead of blocking the image request behind
    a long render.
    """
    state = request.app.state.chaosengine
    if VIDEO_PROGRESS.snapshot().get("active"):
        raise HTTPException(
            status_code=409,
            detail=(
                "A video generation is still running. Wait for it to finish or cancel it "
                "before loading an image model."
            ),
        )
    try:
        runtime = state.video_runtime.capabilities()
    except Exception:
        return
    loaded_repo = str(runtime.get("loadedModelRepo") or "")
    if not loaded_repo:
        return
    try:
        state.video_runtime.unload()
    except Exception as exc:
        state.add_log(
            "images",
            "warning",
            f"Could not unload video model before {action}: {type(exc).__name__}: {exc}",
        )
        return
    state.add_log(
        "images",
        "info",
        f"Unloaded video model {loaded_repo} before {action} to free memory.",
    )
    state.add_activity("Video model unloaded", f"Freed memory for {action}")


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


@router.get("/api/images/progress")
def image_generation_progress() -> dict[str, Any]:
    """Live progress snapshot for the in-flight image generation.

    Polled by the generation modal every ~500ms while the bar is visible.
    When ``active`` is false the UI falls back to its own client-side
    estimates rather than freezing the bar at 0%.
    """
    return {"progress": IMAGE_PROGRESS.snapshot()}


@router.post("/api/images/cancel")
def cancel_image_generation(request: Request) -> dict[str, Any]:
    """Signal the running image generation to abort at the next step.

    Returns ``{"cancelled": true}`` when a run was in flight and received
    the signal, ``{"cancelled": false}`` when nothing was running (treated
    as success — the UI's intent ("make it stop") is already satisfied).
    The actual abort is cooperative: the pipeline's step-end callback
    reads ``IMAGE_PROGRESS.is_cancelled()`` and raises, typically within a
    second of this call returning.
    """
    state = request.app.state.chaosengine
    signalled = IMAGE_PROGRESS.request_cancel()
    if signalled:
        state.add_log("images", "info", "Cancel signal sent to running image generation.")
    return {"cancelled": signalled}


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
    _unload_idle_video_runtime_for_image(request, "image preload")
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
    # Phase 2.0.5-H: pre-flight memory gate. Refuse before invoking the
    # diffusion pipeline if the host is already memory-starved — image
    # gen on a swap-thrashing laptop typically takes minutes to recover
    # and can wedge the desktop entirely. Gate failure (psutil error)
    # never blocks legitimate work; logged + skipped.
    try:
        from backend_service.helpers.memory_gate import (
            gate_image_generation,
            snapshot_memory_signals,
        )

        available_gb, pressure_percent = snapshot_memory_signals()
        refusal = gate_image_generation(available_gb, pressure_percent)
        if refusal is not None:
            state.add_log(
                "images", "warning",
                f"Memory gate refused image gen: {refusal['code']} "
                f"(avail={available_gb:.1f} GB, pressure={pressure_percent:.0f}%).",
            )
            raise HTTPException(status_code=503, detail=refusal["message"])
    except HTTPException:
        raise
    except Exception as gate_exc:
        state.add_log("images", "warning", f"Memory gate skipped: {gate_exc}")
    _unload_idle_video_runtime_for_image(request, "image generation")
    try:
        artifacts, runtime = _generate_image_artifacts(body, variant, state.image_runtime)
    except GenerationCancelled:
        # User hit Cancel on the modal. 409 (Conflict) carries a clearer
        # semantics than 500 ("something broke") for the frontend to tell
        # "we stopped because you asked us to" apart from an actual crash,
        # and 499 is a non-standard nginx extension the Python stdlib
        # HTTPException doesn't recognise.
        state.add_log("images", "info", f"Image generation cancelled for {variant.get('name')} by user.")
        raise HTTPException(status_code=409, detail="cancelled") from None
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
