"""Imported skills on disk: ``~/.chaosengine/skills/<id>/skill.json`` plus ``files/<relative path>``.

The id is a slug of the skill's name, checked against a fixed pattern before any path is built, and every
file path is resolved and checked to sit inside its skill's ``files`` folder. A new skill is written to a
temporary folder and renamed into place, so a failed import leaves nothing behind. Skills start off.
``CHAOSENGINE_SKILLS_DIR`` moves the folder (tests).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from backend_service.skills.importer import SKILL_FILE, ImportedSkill, safe_relpath, slugify, SkillImportError

_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_META = "skill.json"
_FILES = "files"
_LOCK = threading.Lock()


class SkillExistsError(ValueError):
    pass


def skills_dir() -> Path:
    path = Path(os.environ.get("CHAOSENGINE_SKILLS_DIR") or Path.home() / ".chaosengine" / "skills")
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_valid_id(skill_id: str) -> bool:
    return bool(_ID_PATTERN.fullmatch(skill_id or ""))


def _skill_path(skill_id: str) -> Path:
    if not is_valid_id(skill_id):
        raise KeyError(skill_id)
    return skills_dir() / skill_id


def _read_meta(folder: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads((folder / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) and meta.get("id") == folder.name else None


def _write_meta(folder: Path, meta: dict[str, Any]) -> None:
    tmp = folder / f"{_META}.tmp"
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(folder / _META)


def save(skill: ImportedSkill, source: str = "") -> dict[str, Any]:
    """Store an imported skill (disabled). SkillExistsError when a skill with the same id is there."""
    skill_id = slugify(skill.name)
    if not is_valid_id(skill_id):
        raise SkillImportError("The skill name does not make a usable id.")
    root = skills_dir()
    with _LOCK:
        final = root / skill_id
        if final.exists():
            raise SkillExistsError(f"A skill called '{skill_id}' is already imported. Delete it first to replace it.")
        staging = root / f".import-{uuid.uuid4().hex[:8]}"
        files_root = (staging / _FILES)
        try:
            files_root.mkdir(parents=True)
            base = files_root.resolve()
            for rel, text in skill.files.items():
                target = (files_root / safe_relpath(rel)).resolve()
                if not target.is_relative_to(base):
                    raise SkillImportError(f"Unsafe path in the skill: {rel!r}.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            meta = {
                "id": skill_id,
                "name": skill.name,
                "description": skill.description,
                "enabled": False,
                "importedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": Path(source).name[:120] if source else "",
                "files": [{"path": rel, "bytes": len(text.encode("utf-8"))} for rel, text in sorted(skill.files.items())],
                "skipped": skill.skipped,
            }
            _write_meta(staging, meta)
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return meta


def list_skills() -> list[dict[str, Any]]:
    out = []
    for folder in sorted(skills_dir().iterdir()):
        if folder.is_dir() and not folder.is_symlink() and is_valid_id(folder.name):
            meta = _read_meta(folder)
            if meta is not None:
                out.append(meta)
    return out


def get(skill_id: str) -> dict[str, Any] | None:
    try:
        folder = _skill_path(skill_id)
    except KeyError:
        return None
    return _read_meta(folder) if folder.is_dir() and not folder.is_symlink() else None


def set_enabled(skill_id: str, enabled: bool) -> dict[str, Any] | None:
    with _LOCK:
        meta = get(skill_id)
        if meta is None:
            return None
        meta["enabled"] = bool(enabled)
        _write_meta(_skill_path(skill_id), meta)
    return meta


def delete(skill_id: str) -> bool:
    with _LOCK:
        if get(skill_id) is None:
            return False
        shutil.rmtree(_skill_path(skill_id))
    return True


def read_file(skill_id: str, rel: str = SKILL_FILE) -> str:
    """One stored text file of a skill. KeyError for an unknown skill or a path not in its file list."""
    meta = get(skill_id)
    if meta is None:
        raise KeyError(skill_id)
    try:
        rel = safe_relpath(rel)
    except SkillImportError as exc:
        raise KeyError(rel) from exc
    if rel not in {f["path"] for f in meta.get("files", [])}:
        raise KeyError(rel)
    base = (_skill_path(skill_id) / _FILES).resolve()
    target = (base / rel).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise KeyError(rel)
    return target.read_text(encoding="utf-8")
