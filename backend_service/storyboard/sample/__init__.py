"""The bundled sample film (S12): a finished four-shot project that a new install can open in the first minute.

The files sit in ``film/`` beside this module (``project.json`` and its ``media/``) and ship with the backend
(``scripts/stage-runtime.mjs`` copies ``backend_service`` whole and checks this folder is there). Installing it
packs that folder as a ``.chaosfilm`` bundle and runs the ordinary guarded import, so the copy is a normal project
in the user's storyboards folder with a new id: it can be edited, redone, exported or deleted like any other, and
the bundled files are never touched. The stills were made with Iris-3B Turbo (Apache 2.0) and the clips with LTX-2.5
(LTX-2 community licence: the licensor claims no rights in outputs); see ``docs/progress/v0100-f.md``.
"""

from __future__ import annotations

import json
import secrets
import time
import zipfile
from pathlib import Path
from typing import Any

from backend_service.storyboard.models import Project
from backend_service.storyboard.store import PROJECT_FILE, ProjectStore, StoreError, parse_project

FILM = Path(__file__).resolve().parent / "film"


def _bundled() -> Project:
    path = FILM / PROJECT_FILE
    if not path.is_file():
        raise StoreError("The sample project is not part of this install.")
    return parse_project(path.read_bytes())


def files() -> list[str]:
    """The media paths the sample uses (project-relative), every one of which must ship."""
    from backend_service.storyboard.bundle import referenced

    return sorted(referenced(_bundled()))


def info() -> dict[str, Any]:
    """What the "Try the sample project" card shows; ``available`` False when the files are not installed."""
    try:
        project = _bundled()
        missing = [rel for rel in files() if not (FILM / rel).is_file()]
    except (StoreError, ValueError, OSError):
        return {"available": False}
    if missing:
        return {"available": False}
    d = project.defaults
    seconds = sum((shot.active().measuredSeconds or shot.seconds) if shot.active() else shot.seconds for shot in project.shots)
    return {"available": True, "name": project.name, "shots": len(project.shots), "seconds": round(seconds, 1),
            "aspect": d.exportPreset, "resolution": d.resolution, "fps": d.fps, "poster": poster_rel(project) is not None}


def poster_rel(project: Project | None = None) -> str | None:
    """The first shot's still, for the card."""
    project = project or _bundled()
    for shot in project.shots:
        take = shot.active()
        if take is not None and take.still and (FILM / take.still).is_file():
            return take.still
    return None


def poster_path() -> Path:
    rel = poster_rel()
    if rel is None:
        raise StoreError("The sample project has no picture.")
    return FILM / rel


def install(store: ProjectStore, name: str | None = None) -> Project:
    """Copy the sample into ``store`` as a new project (new id, created now, optional new name) and return it."""
    from backend_service.storyboard import bundle

    project = _bundled()
    data = json.loads((FILM / PROJECT_FILE).read_text(encoding="utf-8"))
    data["createdAt"] = time.time()
    if name:
        data["name"] = name
    store.root.mkdir(parents=True, exist_ok=True)
    tmp = store.root / f".sample-{secrets.token_hex(6)}{bundle.BUNDLE_SUFFIX}"
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            zf.writestr(PROJECT_FILE, json.dumps(data, ensure_ascii=False))
            for rel in bundle.referenced(project):
                source = FILM / rel
                if not source.is_file():
                    raise StoreError("The sample project is incomplete in this install (a media file is missing).")
                zf.write(source, rel, zipfile.ZIP_STORED)
        return bundle.import_bundle(store, tmp)
    finally:
        tmp.unlink(missing_ok=True)
