"""``GET /api/storyboard/projects/{id}/films``: the finished films in a project's ``exports/`` folder.

Read-only. Export storage is unchanged: films stay in ``exports/`` and are not added to the Video gallery.
Each row is a file name relative to the project (``exports/<name>``), which the existing project files
route serves; no absolute path is returned.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()

FILM_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm")


@router.get("/api/storyboard/projects/{project_id}/films")
def list_films(project_id: str) -> dict[str, Any]:
    from backend_service.routes.storyboard import _EXPORT_NAME_RE, _guard, get_store

    folder = _guard(get_store().folder, project_id)
    exports = folder / "exports"
    films: list[dict[str, Any]] = []
    if exports.is_dir():
        for path in exports.iterdir():
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix.lower() not in FILM_SUFFIXES or not _EXPORT_NAME_RE.match(path.name):
                continue
            if ".partial" in path.name or path.name.startswith("."):
                continue  # an unfinished export
            stat = path.stat()
            films.append({"file": f"exports/{path.name}", "name": path.name, "bytes": stat.st_size,
                          "modifiedAt": stat.st_mtime})
    films.sort(key=lambda f: f["modifiedAt"], reverse=True)
    return {"films": films}
