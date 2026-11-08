"""Skills API (D-62): list, import from a folder or .zip path, switch on or off, delete.

The import path comes from the user (the desktop file dialog or a typed path) and is validated in
``skills.importer``; ids are checked against the store's pattern before any path is built. A new skill is
stored switched off, so the user sees its description and files before the chat model can read it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillImportRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class SkillEnabledRequest(BaseModel):
    enabled: bool


def _skill_id(skill_id: str) -> str:
    from backend_service.skills.store import is_valid_id

    if not is_valid_id(skill_id):
        raise HTTPException(status_code=400, detail="Invalid skill id.")
    return skill_id


@router.get("")
def list_skills(request: Request) -> dict[str, Any]:
    from backend_service.skills import store

    skills = store.list_skills()
    return {"skills": skills, "count": len(skills)}


@router.post("/import")
def import_skill(request: Request, body: SkillImportRequest) -> dict[str, Any]:
    from backend_service.skills import importer, store

    try:
        skill = importer.load(body.path)
        meta = store.save(skill, source=body.path.strip())
    except store.SkillExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except importer.SkillImportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state = getattr(request.app.state, "chaosengine", None)
    if state is not None:
        state.add_log("skills", "info", f"Imported skill '{meta['id']}' ({len(meta['files'])} files, "
                                        f"{len(meta['skipped'])} skipped); it starts switched off.")
    return {"skill": meta}


@router.post("/{skill_id}/enabled")
def set_skill_enabled(request: Request, skill_id: str, body: SkillEnabledRequest) -> dict[str, Any]:
    from backend_service.skills import store

    meta = store.set_enabled(_skill_id(skill_id), body.enabled)
    if meta is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    return {"skill": meta}


@router.delete("/{skill_id}")
def delete_skill(request: Request, skill_id: str) -> dict[str, Any]:
    from backend_service.skills import store

    if not store.delete(_skill_id(skill_id)):
        raise HTTPException(status_code=404, detail="Skill not found.")
    return {"deleted": skill_id}
