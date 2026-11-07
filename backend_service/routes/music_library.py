"""Music API routes: the Music Library, the music pack install job, settings and weight downloads.

Library ids are checked with ``music_library.is_valid_id`` before anything touches the disk, and the
audio route serves only the library's own files (Starlette's ``FileResponse`` answers ``Range`` requests,
which the player needs for seeking). The install routes mirror the voice pack's; weights go through the
app's shared Hugging Face downloader, restricted here to music repos.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter()

_MEDIA_TYPES = {".flac": "audio/flac", ".wav": "audio/wav"}


class LibraryUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    tags: list[str] | None = Field(default=None, max_length=40)


class LibraryDeleteRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)


class MusicSettingsRequest(BaseModel):
    idleUnloadMinutes: int | None = Field(default=None, ge=0, le=240)


class MusicDownloadRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=200)


def _library_id(item_id: str) -> str:
    from backend_service.music_library import is_valid_id

    if not is_valid_id(item_id):
        raise HTTPException(status_code=400, detail="Invalid item id.")
    return item_id


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

@router.get("/api/music/library")
def list_library(request: Request, q: str = "", tag: str = "") -> dict[str, Any]:
    from backend_service import music_library

    items = music_library.list_items(q[:200], tag[:40])
    return {"items": items, "count": len(items), "diskBytes": music_library.disk_bytes()}


@router.post("/api/music/library/delete")
def delete_library_items(request: Request, body: LibraryDeleteRequest) -> dict[str, Any]:
    from backend_service import music_library

    ids = [_library_id(item_id) for item_id in body.ids]
    return {"deleted": [item_id for item_id in ids if music_library.delete(item_id)]}


@router.get("/api/music/library/{item_id}")
def get_library_item(request: Request, item_id: str) -> dict[str, Any]:
    from backend_service import music_library

    item = music_library.get(_library_id(item_id))
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found.")
    return item


@router.post("/api/music/library/{item_id}")
def update_library_item(request: Request, item_id: str, body: LibraryUpdateRequest) -> dict[str, Any]:
    from backend_service import music_library

    item = music_library.update(_library_id(item_id), title=body.title, tags=body.tags)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found.")
    return item


@router.delete("/api/music/library/{item_id}")
def delete_library_item(request: Request, item_id: str) -> dict[str, Any]:
    from backend_service import music_library

    if not music_library.delete(_library_id(item_id)):
        raise HTTPException(status_code=404, detail="Item not found.")
    return {"deleted": True}


@router.get("/api/music/library/{item_id}/audio")
def get_library_audio(request: Request, item_id: str) -> FileResponse:
    from backend_service import music_library

    path = music_library.audio_path(_library_id(item_id))
    if path is None:
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(path, media_type=_MEDIA_TYPES.get(path.suffix, "application/octet-stream"),
                        filename=f"{item_id}{path.suffix}", content_disposition_type="inline")


# ---------------------------------------------------------------------------
# Install, settings
# ---------------------------------------------------------------------------

@router.post("/api/music/install")
def start_install(request: Request) -> dict[str, Any]:
    from backend_service import music_env

    return music_env.start_install()


@router.get("/api/music/install/status")
def install_status(request: Request) -> dict[str, Any]:
    from backend_service import music_env

    return music_env.job_status()


@router.post("/api/music/install/cancel")
def cancel_install(request: Request) -> dict[str, Any]:
    from backend_service import music_env

    return music_env.cancel_install()


@router.get("/api/music/settings")
def get_settings(request: Request) -> dict[str, Any]:
    from backend_service.music_runtime import load_settings

    return load_settings()


@router.post("/api/music/settings")
def put_settings(request: Request, body: MusicSettingsRequest) -> dict[str, Any]:
    from backend_service.music_runtime import save_settings

    return save_settings(body.model_dump())


# ---------------------------------------------------------------------------
# Weight downloads (the shared downloader, music repos only)
# ---------------------------------------------------------------------------

def _music_repo(repo: str) -> str:
    from backend_service.helpers.music import is_music_repo

    if not is_music_repo(repo):
        raise HTTPException(status_code=400, detail=f"'{repo}' is not a music model in the catalog.")
    return repo


@router.post("/api/music/download")
def download_music_model(request: Request, body: MusicDownloadRequest) -> dict[str, Any]:
    return {"download": request.app.state.chaosengine.start_download(_music_repo(body.repo))}


@router.get("/api/music/download/status")
def music_download_status(request: Request) -> dict[str, Any]:
    from backend_service.helpers.music import is_music_repo

    downloads = [d for d in request.app.state.chaosengine.download_status() if is_music_repo(str(d.get("repo") or ""))]
    return {"downloads": downloads}


@router.post("/api/music/download/cancel")
def cancel_music_download(request: Request, body: MusicDownloadRequest) -> dict[str, Any]:
    return {"download": request.app.state.chaosengine.cancel_download(_music_repo(body.repo))}


@router.post("/api/music/download/delete")
def delete_music_download(request: Request, body: MusicDownloadRequest) -> dict[str, Any]:
    return {"result": request.app.state.chaosengine.delete_download(_music_repo(body.repo))}
