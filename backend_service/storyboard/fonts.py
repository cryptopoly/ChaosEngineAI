"""The font families installed on this machine, for the caption look of the brand kit (S15).

Reads the family name of every TrueType / OpenType file in the usual font folders of macOS, Windows and Linux
with Pillow (no network, nothing installed). The list is read once per process; a font installed while the
app runs shows after a restart. Families whose names start with "." are system-private on macOS and skipped.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

_SUFFIXES = {".ttf", ".otf", ".ttc"}
_MAX_FILES = 4000
_CACHE: dict[str, str] | None = None
_LOCK = threading.Lock()


def font_dirs() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        return [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
                home / "Library" / "Fonts"]
    if sys.platform.startswith("win"):
        windir = Path(os.environ.get("WINDIR", "C:\\Windows"))
        local = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
        return [windir / "Fonts", local / "Microsoft" / "Windows" / "Fonts"]
    return [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), home / ".local" / "share" / "fonts", home / ".fonts"]


def _scan(dirs: list[Path]) -> dict[str, str]:
    from PIL import ImageFont

    families: dict[str, str] = {}
    seen = 0
    for folder in dirs:
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in _SUFFIXES:
                continue
            seen += 1
            if seen > _MAX_FILES:
                return families
            try:
                family = ImageFont.truetype(str(path), 12).getname()[0]
            except (OSError, ValueError):
                continue
            if family and not family.startswith(".") and family not in families:
                families[family] = str(path)
    return families


def installed_fonts(dirs: list[Path] | None = None) -> list[str]:
    """Family names, sorted without regard to case."""
    global _CACHE
    if dirs is not None:
        return sorted(_scan(dirs), key=str.casefold)
    with _LOCK:
        if _CACHE is None:
            _CACHE = _scan(font_dirs())
        return sorted(_CACHE, key=str.casefold)


def font_file(family: str) -> str | None:
    """The file of an installed family, for a caption burn-in that needs a path (None when unknown)."""
    installed_fonts()
    return (_CACHE or {}).get(family)
