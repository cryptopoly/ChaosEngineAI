"""Video generation API routes.

The runtime is not wired yet — these endpoints expose the curated catalog and
a stub runtime status so the frontend can light up Discover/My Models cleanly
ahead of the engine landing. Generation, preload, and download routes return
501 until the video runtime ships.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend_service.catalog import VIDEO_MODEL_FAMILIES


router = APIRouter()


def _variant_payload(variant: dict[str, Any], family_name: str) -> dict[str, Any]:
    """Return a variant dict with frontend-friendly fields populated."""
    payload = dict(variant)
    payload.setdefault("availableLocally", False)
    payload.setdefault("hasLocalData", False)
    payload["familyName"] = family_name
    return payload


def _video_model_payloads() -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for family in VIDEO_MODEL_FAMILIES:
        variants = [_variant_payload(v, family["name"]) for v in family["variants"]]
        payload = dict(family)
        payload["variants"] = variants
        families.append(payload)
    return families


@router.get("/api/video/catalog")
def video_catalog(request: Request) -> dict[str, Any]:
    """Return the curated catalog of planned video generation models."""
    return {
        "families": _video_model_payloads(),
        "latest": [],
    }


@router.get("/api/video/runtime")
def video_runtime_status(request: Request) -> dict[str, Any]:
    """Report the video runtime status.

    Today this is always "not available" — the runtime hasn't shipped yet.
    The shape mirrors ``ImageRuntimeStatus`` so the frontend can reuse
    the same rendering logic.
    """
    return {
        "runtime": {
            "activeEngine": "placeholder",
            "realGenerationAvailable": False,
            "message": "Video runtime not yet available. Tab is scaffolded — engine work is on the roadmap.",
            "device": None,
            "pythonExecutable": None,
            "missingDependencies": ["diffusers", "torch"],
            "loadedModelRepo": None,
        }
    }


@router.get("/api/video/library")
def video_library(request: Request) -> dict[str, Any]:
    """Return the list of locally-installed video models.

    Empty until the video runtime ships and users can download weights.
    """
    return {"models": []}


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


@router.post("/api/video/preload")
def preload_video_model(request: Request) -> dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Video model preload is not implemented yet.",
    )


@router.post("/api/video/download")
def download_video_model(request: Request) -> dict[str, Any]:
    raise HTTPException(
        status_code=501,
        detail="Video model downloads are not implemented yet.",
    )
