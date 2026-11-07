"""``.chaosfilm`` bundles: a zip of a project folder for sharing and backup, and its guarded import.

Export writes ``project.json`` (batches, undo stacks and statuses of unfinished work dropped; it only ever
holds project-relative paths) and every media file the project references, nothing else, so no absolute
local path or stray file leaves the machine. Import treats the zip as hostile: entry names must be
``project.json`` or a safe ``media/`` path (no absolute names, drive letters, '..', backslashes or
symlinks), the file count and total uncompressed size are capped and counted while extracting (header
sizes are not trusted), newer schema versions are refused and older ones migrated, and files whose sha256
does not match are dropped so the shot shows ``missing``. The project gets a new id.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import time
import zipfile
from pathlib import Path
from typing import Any

from backend_service.storyboard import media
from backend_service.storyboard.models import Project, check_rel_path, dump, new_project_id
from backend_service.storyboard.store import (
    _RECOVERED,
    MAX_PROJECT_BYTES,
    PROJECT_FILE,
    ProjectStore,
    StoreError,
    atomic_write,
    parse_project,
)

MAX_FILES = 5000
MAX_TOTAL_BYTES = 64 * 1024**3
BUNDLE_SUFFIX = ".chaosfilm"


class BundleError(StoreError):
    pass


def referenced(project: Project) -> dict[str, str | None]:
    """Every media path the project uses, with its expected sha256 when known."""
    files: dict[str, str | None] = {}
    for ref in project.references.characters + project.references.places:
        if ref.image:
            files[ref.image] = ref.hash
    for shot in project.shots:
        if shot.endImage:
            files.setdefault(shot.endImage, None)
        for take in shot.takes:
            if take.still:
                files[take.still] = take.hashes.get("still")
            if take.clip:
                files[take.clip] = take.hashes.get("clip")
    for clip in project.audio:
        files[clip.source.path] = clip.source.hash
    return files


def portable(project: Project) -> dict[str, Any]:
    data = dump(project)
    data["batches"], data["undoStack"], data["redoStack"] = [], [], []
    for shot in data["shots"]:
        if shot["status"] in ("queued", "running", "missing"):
            shot["status"] = "done" if shot.get("activeTake") else "idle"
    for clip in data["audio"]:
        clip["status"] = "ok"
    return data


def export_bundle(store: ProjectStore, project_id: str) -> Path:
    project = store.load(project_id)
    folder = store.folder(project_id)
    exports = folder / "exports"
    exports.mkdir(exist_ok=True)
    slug = "".join(c if c.isalnum() else "-" for c in project.name).strip("-")[:40] or "film"
    out = exports / f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}{BUNDLE_SUFFIX}"
    tmp = out.with_name(out.name + ".part")
    try:
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
            zf.writestr(PROJECT_FILE, json.dumps(portable(project), indent=1, ensure_ascii=False), zipfile.ZIP_DEFLATED)
            for rel in sorted(referenced(project)):
                try:
                    real = media.inside(folder, rel)
                except media.MediaError:
                    continue  # missing here, missing there
                zf.write(real, rel, zipfile.ZIP_STORED)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)  # a failed export leaves nothing in the (downloadable) exports folder
    return out


def _check_entry(info: zipfile.ZipInfo) -> None:
    name = info.filename
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise BundleError(f"Refused bundle: '{name[:80]}' is a symlink.")
    if name == PROJECT_FILE:
        return
    try:
        check_rel_path(name.rstrip("/") if info.is_dir() else name)
    except ValueError as exc:
        raise BundleError(f"Refused bundle entry '{name[:80]}': {exc}") from exc


def _check_no_clash(infos: list[zipfile.ZipInfo]) -> None:
    """Refuse a file that is also a folder (``media/a`` and ``media/a/b``), or two names that differ only
    in case (one disk folder on macOS and Windows)."""
    files = [i.filename.casefold() for i in infos if not i.is_dir()]
    taken = set(files)
    if len(taken) != len(files):
        raise BundleError("Refused: unsafe bundle (two entries differ only in upper and lower case).")
    for info in infos:
        name = info.filename.rstrip("/").casefold()
        parents = name.split("/")[:-1]
        clash = (info.is_dir() and name in taken) or any("/".join(parents[:n]) in taken for n in range(1, len(parents) + 1))
        if clash:
            raise BundleError(f"Refused: unsafe bundle ('{info.filename[:80]}' is inside a name that is also a file).")


def import_bundle(store: ProjectStore, src: Path) -> Project:
    try:
        zf = zipfile.ZipFile(src)
    except (zipfile.BadZipFile, OSError) as exc:
        raise BundleError("Not a .chaosfilm bundle (not a zip file).") from exc
    store.root.mkdir(parents=True, exist_ok=True)
    staging = store.root / f".import-{secrets.token_hex(6)}"
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_FILES:
            raise BundleError(f"Refused bundle: more than {MAX_FILES} files.")
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise BundleError("Refused bundle: duplicate entry names.")
        for info in infos:
            _check_entry(info)
        _check_no_clash(infos)
        if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
            raise BundleError(f"Refused bundle: larger than {MAX_TOTAL_BYTES // 1024**3} GB unpacked.")
        if PROJECT_FILE not in names:
            raise BundleError("Refused bundle: no project.json.")
        with zf.open(PROJECT_FILE) as handle:
            raw = handle.read(MAX_PROJECT_BYTES + 1)
        project = parse_project(raw)
        try:
            staging.mkdir()
            total = 0
            for info in infos:
                if info.is_dir() or info.filename == PROJECT_FILE:
                    continue
                dest = media.inside(staging, info.filename, must_exist=False)
                dest.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with zf.open(info) as source, dest.open("wb") as out:
                    for block in iter(lambda: source.read(1 << 20), b""):
                        written += len(block)
                        total += len(block)
                        if written > info.file_size or total > MAX_TOTAL_BYTES:
                            raise BundleError("Refused bundle: an entry unpacks larger than it says.")
                        out.write(block)
            _drop_bad_files(project, staging)
            project = Project.model_validate({**portable(project), "id": new_project_id(), "revision": 1})
            for sub in ("media/audio", "cache", "history", "exports"):
                (staging / sub).mkdir(parents=True, exist_ok=True)
            atomic_write(staging / PROJECT_FILE, json.dumps(dump(project), indent=1, ensure_ascii=False).encode("utf-8"))
            final = store.folder(project.id)
            os.rename(staging, final)
            _RECOVERED.add(str(final.resolve()))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return store.load(project.id)


def _drop_bad_files(project: Project, staging: Path) -> None:
    for rel, expected in referenced(project).items():
        try:
            real = media.inside(staging, rel)
        except media.MediaError:
            continue
        if expected and media.sha256_file(real) != expected:
            real.unlink()
