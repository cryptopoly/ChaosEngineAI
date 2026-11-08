"""Read and validate a skill the user picked: a folder or a ``.zip`` with a ``SKILL.md`` at its top.

The path and everything inside it are hostile. A folder is walked without following links and any
symbolic link rejects the import; a zip is read entry by entry with entries that are absolute, climb with
``..``, are links or are encrypted rejected. File count, per-file size, total size and the zip itself are
capped, and a zip entry's real size is checked while reading (a header can lie). Only text files are
kept; binaries and hidden files are skipped and listed. Nothing is written here: ``store.save`` does that.
"""

from __future__ import annotations

import os
import re
import stat
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

MAX_FILES = 64                     # text files kept, SKILL.md included
MAX_ENTRIES = 512                  # everything looked at, skipped files included
MAX_FILE_BYTES = 256 * 1024        # one text file
MAX_TOTAL_BYTES = 2 * 1024 * 1024  # all text files together
MAX_ZIP_BYTES = 16 * 1024 * 1024   # the archive on disk
MAX_RATIO = 100                    # uncompressed / compressed, per entry (zip bombs)
MAX_FRONT_MATTER = 8 * 1024
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 1024
MAX_PATH_CHARS = 200

SKILL_FILE = "SKILL.md"
TEXT_SUFFIXES = frozenset({
    ".md", ".markdown", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".tsv",
    ".xml", ".html", ".htm", ".css", ".svg", ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".sh", ".bash",
    ".zsh", ".ps1", ".bat", ".rb", ".lua", ".gd", ".cs", ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".java",
    ".kt", ".swift", ".sql", ".glsl", ".hlsl", ".shader", ".tex", ".j2", ".jinja", ".tmpl", ".env.example",
})
TEXT_NAMES = frozenset({"LICENSE", "LICENCE", "NOTICE", "README", "CHANGELOG", "Makefile", "Dockerfile"})
_SKIP_DIRS = frozenset({"__MACOSX", "__pycache__", "node_modules", ".git"})


class SkillImportError(ValueError):
    """The skill cannot be imported; the message is shown to the user as is."""


@dataclass
class ImportedSkill:
    name: str
    description: str
    files: dict[str, str]                              # relative POSIX path -> text, SKILL.md included
    skipped: list[dict[str, str]] = field(default_factory=list)  # {"path", "reason"}


def slugify(name: str) -> str:
    """A safe id from a skill name: lower-case letters, digits and single dashes, at most 64 characters."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:MAX_NAME_CHARS].strip("-")


def safe_relpath(raw: str) -> str:
    """A relative POSIX path with no empty, ``.`` or ``..`` parts, no drive, no backslash; else SkillImportError."""
    if not raw or "\x00" in raw or "\\" in raw or len(raw) > MAX_PATH_CHARS:
        raise SkillImportError(f"Unsafe path in the skill: {raw[:80]!r}.")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise SkillImportError(f"Absolute path in the skill: {raw[:80]!r}.")
    parts = PurePosixPath(raw).parts
    if not parts or any(p in ("", ".", "..") for p in parts) or ".." in raw.split("/"):
        raise SkillImportError(f"Path climbs out of the skill: {raw[:80]!r}.")
    return "/".join(parts)


def parse_skill_md(text: str) -> tuple[str, str]:
    """``(name, description)`` from SKILL.md's front matter; SkillImportError when missing or malformed."""
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        raise SkillImportError("SKILL.md must start with front matter (a '---' line, name and description, '---').")
    end = text.find("\n---", 3)
    if end < 0:
        raise SkillImportError("SKILL.md front matter has no closing '---' line.")
    front = text[3:end]
    if len(front) > MAX_FRONT_MATTER:
        raise SkillImportError("SKILL.md front matter is too long.")
    import yaml  # lazy: only an import needs it

    try:
        meta = yaml.safe_load(front)
    except yaml.YAMLError as exc:
        raise SkillImportError(f"SKILL.md front matter is not valid YAML: {str(exc).splitlines()[0]}") from exc
    if not isinstance(meta, dict):
        raise SkillImportError("SKILL.md front matter must be key: value lines.")
    name, description = meta.get("name"), meta.get("description")
    if not isinstance(name, str) or not name.strip():
        raise SkillImportError("SKILL.md front matter needs a 'name'.")
    if not isinstance(description, str) or not description.strip():
        raise SkillImportError("SKILL.md front matter needs a 'description'.")
    name = " ".join(name.split())
    if len(name) > MAX_NAME_CHARS or not slugify(name):
        raise SkillImportError(f"The skill name must be 1 to {MAX_NAME_CHARS} characters with letters or digits.")
    description = " ".join(description.split())
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise SkillImportError(f"The skill description is longer than {MAX_DESCRIPTION_CHARS} characters.")
    return name, description


def skill_body(text: str) -> str:
    """SKILL.md without its front matter."""
    text = text.lstrip("﻿")
    end = text.find("\n---", 3) if text.startswith("---") else -1
    return text if end < 0 else text[end + 4:].lstrip("\r\n")


def _is_text_name(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name in TEXT_NAMES or any(name.lower().endswith(s) for s in TEXT_SUFFIXES)


def _hidden(rel: str) -> bool:
    return any(p.startswith(".") or p in _SKIP_DIRS for p in rel.split("/"))


def _decode(rel: str, data: bytes) -> str | None:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


class _Collector:
    """Applies the count and size caps as files arrive, then builds the ImportedSkill."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.skipped: list[dict[str, str]] = []
        self.total = 0
        self.seen = 0

    def see(self) -> None:
        self.seen += 1
        if self.seen > MAX_ENTRIES:
            raise SkillImportError(f"The skill has more than {MAX_ENTRIES} entries.")

    def skip(self, rel: str, reason: str) -> None:
        if len(self.skipped) < 100:
            self.skipped.append({"path": rel, "reason": reason})

    def check_size(self, rel: str, size: int) -> None:
        if size > MAX_FILE_BYTES:
            raise SkillImportError(f"{rel} is larger than {MAX_FILE_BYTES // 1024} KB.")
        if self.total + size > MAX_TOTAL_BYTES:
            raise SkillImportError(f"The skill's text files are larger than {MAX_TOTAL_BYTES // (1024 * 1024)} MB in all.")

    def add(self, rel: str, data: bytes) -> None:
        self.check_size(rel, len(data))
        text = _decode(rel, data)
        if text is None:
            self.skip(rel, "not a text file")
            return
        if len(self.files) >= MAX_FILES:
            raise SkillImportError(f"The skill has more than {MAX_FILES} text files.")
        self.files[rel] = text
        self.total += len(data)

    def build(self) -> ImportedSkill:
        if SKILL_FILE not in self.files:
            raise SkillImportError("No SKILL.md at the top of the skill.")
        name, description = parse_skill_md(self.files[SKILL_FILE])
        return ImportedSkill(name=name, description=description, files=self.files, skipped=self.skipped)


def from_folder(path: str | os.PathLike[str]) -> ImportedSkill:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise SkillImportError("The skill folder is a symbolic link; choose the real folder.")
    try:
        root = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SkillImportError("The skill folder does not exist.") from exc
    if not root.is_dir():
        raise SkillImportError("Choose a skill folder or a .zip file.")
    out = _Collector()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        for name in sorted(dirnames + filenames):
            entry = here / name
            out.see()
            try:
                info = entry.lstat()
            except OSError as exc:
                raise SkillImportError(f"Cannot read {name}.") from exc
            rel = safe_relpath(entry.relative_to(root).as_posix())
            if stat.S_ISLNK(info.st_mode):
                raise SkillImportError(f"The skill contains a symbolic link ({rel}); links are not allowed.")
            if not entry.resolve().is_relative_to(root):  # belt and braces after the link check
                raise SkillImportError(f"{rel} points outside the skill folder.")
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                out.skip(rel, "not a regular file")
            elif _hidden(rel):
                out.skip(rel, "hidden file")
            elif not _is_text_name(rel):
                out.skip(rel, "not a text file")
            else:
                out.check_size(rel, info.st_size)
                with open(entry, "rb") as handle:
                    out.add(rel, handle.read(MAX_FILE_BYTES + 1))
        dirnames[:] = sorted(d for d in dirnames if not (here / d).is_symlink() and d not in _SKIP_DIRS)
    return out.build()


def _zip_root(names: list[str]) -> str:
    """'' when SKILL.md is at the top, else 'folder/' when the archive holds one folder with SKILL.md in it."""
    if SKILL_FILE in names:
        return ""
    tops = {n.split("/", 1)[0] for n in names if not _hidden(n)}
    if len(tops) == 1:
        top = next(iter(tops))
        if f"{top}/{SKILL_FILE}" in names:
            return f"{top}/"
    raise SkillImportError("No SKILL.md at the top of the archive (or inside its one top folder).")


def from_zip(path: str | os.PathLike[str]) -> ImportedSkill:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise SkillImportError("The archive is a symbolic link; choose the real file.")
    try:
        archive = raw.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SkillImportError("The archive does not exist.") from exc
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise SkillImportError("Choose a skill folder or a .zip file.")
    if archive.stat().st_size > MAX_ZIP_BYTES:
        raise SkillImportError(f"The archive is larger than {MAX_ZIP_BYTES // (1024 * 1024)} MB.")
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise SkillImportError("The file is not a valid .zip archive.") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ENTRIES:
            raise SkillImportError(f"The archive has more than {MAX_ENTRIES} entries.")
        entries: list[tuple[str, zipfile.ZipInfo]] = []
        for info in infos:
            rel = safe_relpath(info.filename.rstrip("/")) if info.filename.rstrip("/") else ""
            if not rel:
                raise SkillImportError("The archive has an entry with an empty name.")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise SkillImportError(f"The archive contains a symbolic link ({rel}); links are not allowed.")
            if info.flag_bits & 0x1:
                raise SkillImportError("The archive is encrypted.")
            if not info.is_dir():
                entries.append((rel, info))
        prefix = _zip_root([rel for rel, _ in entries])
        out = _Collector()
        for rel, info in entries:
            out.see()
            if prefix and not rel.startswith(prefix):
                out.skip(rel, "outside the skill folder")
                continue
            rel = rel[len(prefix):]
            if _hidden(rel):
                out.skip(rel, "hidden file")
                continue
            if not _is_text_name(rel):
                out.skip(rel, "not a text file")
                continue
            out.check_size(rel, info.file_size)
            if info.file_size > 64 * 1024 and info.file_size > MAX_RATIO * max(info.compress_size, 1):
                raise SkillImportError(f"{rel} is compressed too tightly to be safe (possible zip bomb).")
            try:
                with zf.open(info) as handle:
                    data = handle.read(MAX_FILE_BYTES + 1)
            except (zipfile.BadZipFile, OSError, EOFError, zlib.error, NotImplementedError) as exc:
                raise SkillImportError(f"{rel} is damaged in the archive.") from exc
            out.add(rel, data)  # the size check runs again on the real bytes
        return out.build()


def load(path: str) -> ImportedSkill:
    """A skill from a folder or a .zip path."""
    text = (path or "").strip()
    if not text or "\x00" in text:
        raise SkillImportError("Choose a skill folder or a .zip file.")
    candidate = Path(text).expanduser()
    if candidate.suffix.lower() == ".zip" and not candidate.is_dir():
        return from_zip(candidate)
    return from_folder(candidate)
