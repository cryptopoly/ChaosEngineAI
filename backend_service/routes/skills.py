"""Skills API (D-62): list, import from a folder or .zip path, switch on or off, delete.

The import path comes from the user (the desktop file dialog or a typed path) and is validated in
``skills.importer``; ids are checked against the store's pattern before any path is built. A new skill is
stored switched off, so the user sees its description, files and security scan (D-68) before the chat
model can read it. A skill the scan rated High is switched on only with ``acknowledgeFindings: true``.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillImportRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class SkillEnabledRequest(BaseModel):
    enabled: bool
    acknowledgeFindings: bool = False


class SkillScanFinding(BaseModel):
    file: str
    line: int = Field(ge=0)
    rule: str
    severity: Literal["low", "medium", "high"]
    message: str
    excerpt: str = ""


class SkillScan(BaseModel):
    rating: Literal["low", "medium", "high"]
    findings: list[SkillScanFinding]
    truncated: bool = False
    scannedAt: str
    rulesVersion: int


class SkillOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    enabled: bool
    scan: SkillScan


class SkillResponse(BaseModel):
    skill: SkillOut


class SkillListResponse(BaseModel):
    skills: list[SkillOut]
    count: int


def _skill_id(skill_id: str) -> str:
    from backend_service.skills.store import is_valid_id

    if not is_valid_id(skill_id):
        raise HTTPException(status_code=400, detail="Invalid skill id.")
    return skill_id


@router.get("", response_model=SkillListResponse)
def list_skills(request: Request) -> dict[str, Any]:
    from backend_service.skills import store

    skills = store.list_skills()
    return {"skills": skills, "count": len(skills)}


@router.post("/import", response_model=SkillResponse)
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
                                        f"{len(meta['skipped'])} skipped, scan rated {meta['scan']['rating']}); "
                                        "it starts switched off.")
    return {"skill": meta}


@router.post("/{skill_id}/enabled", response_model=SkillResponse)
def set_skill_enabled(request: Request, skill_id: str, body: SkillEnabledRequest) -> dict[str, Any]:
    from backend_service.skills import store

    try:
        meta = store.set_enabled(_skill_id(skill_id), body.enabled, acknowledged=body.acknowledgeFindings)
    except store.ScanNotAcknowledgedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if meta is None:
        raise HTTPException(status_code=404, detail="Skill not found.")
    return {"skill": meta}


@router.delete("/{skill_id}")
def delete_skill(request: Request, skill_id: str) -> dict[str, Any]:
    from backend_service.skills import store

    if not store.delete(_skill_id(skill_id)):
        raise HTTPException(status_code=404, detail="Skill not found.")
    return {"deleted": skill_id}
