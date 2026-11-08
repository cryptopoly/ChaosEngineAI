"""Small copies of project pictures for grids (the Cast cards, the picture picker, the Overview sheet).

A 4K still is tens of MB as a PNG; a grid of 24 of them would hold hundreds of MB in the window. A thumbnail
is the picture scaled so its short side is at most ``size`` pixels (never scaled up, so a small picture keeps
its real size and the "small picture" note stays true), saved as JPEG under the project's ``cache/thumbs/``
and named from the source path, its size and its modification time, so a changed file gets a new one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

THUMB_SIZES = (128, 256, 512)


def thumbnail(folder: Path, real: Path, size: int) -> Path:
    """The thumbnail of ``real`` (a picture inside ``folder``), made on first use. Raises ``ValueError`` if the
    file is not a readable picture."""
    from PIL import Image, ImageOps  # lazy: only when a grid asks

    stat = real.stat()
    key = hashlib.sha256(f"{real.relative_to(folder.resolve())}|{stat.st_size}|{stat.st_mtime_ns}|{size}".encode()).hexdigest()[:32]
    out = folder / "cache" / "thumbs" / f"{key}.jpg"
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(real) as img:
            img = ImageOps.exif_transpose(img)
            short = min(img.size)
            if short > size:
                scale = size / short
                img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
            tmp = out.with_suffix(".part")
            img.convert("RGB").save(tmp, "JPEG", quality=82)
            tmp.replace(out)
    except OSError as exc:
        raise ValueError(f"Not a readable picture: {real.name}") from exc
    return out
