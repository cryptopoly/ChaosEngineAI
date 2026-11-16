"""The main colours of a few pictures, read locally (S15): a small k-means on a downscaled copy, no network.

``palette_of`` returns up to ``count`` colours as ``#rrggbb``, the most used (and most colourful) first, with near-duplicates
merged so a mostly white screenshot does not give five whites. ``colour_name`` turns a colour into plain
words ("deep green", "warm cream") for prompts written to a still model that cannot take reference pictures.
Pillow and numpy are imported inside the calls; both ship with the backend.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

SAMPLE_SIDE = 64  # pixels on the long side of the copy the colours are read from
_MERGE_DISTANCE = 28.0  # RGB distance under which two cluster centres count as one colour
_ITERATIONS = 12


def _pixels(paths: list[Path]):  # noqa: ANN202 - numpy array
    import numpy as np
    from PIL import Image

    chunks = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.draft("RGB", (SAMPLE_SIDE * 4, SAMPLE_SIDE * 4))  # JPEG: decode small, much faster
                rgba = image.convert("RGBA")
                rgba.thumbnail((SAMPLE_SIDE, SAMPLE_SIDE))
                data = np.asarray(rgba, dtype=np.float32).reshape(-1, 4)
        except (OSError, ValueError):
            continue  # an unreadable picture adds no colour; the op still keeps the picture
        chunks.append(data[data[:, 3] >= 128][:, :3])  # transparent pixels are not part of the look
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks)


def _kmeans(pixels, k: int):  # noqa: ANN001, ANN202 - numpy arrays
    import numpy as np

    rng = np.random.default_rng(0)  # the same pictures always give the same palette
    centres = [pixels[rng.integers(len(pixels))]]
    for _ in range(1, k):  # k-means++ start
        dist = np.min(((pixels[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        total = float(dist.sum())
        if total <= 0:
            break
        centres.append(pixels[rng.choice(len(pixels), p=dist / total)])
    c = np.array(centres, dtype=np.float32)
    for _ in range(_ITERATIONS):
        labels = np.argmin(((pixels[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        moved = np.array([pixels[labels == i].mean(0) if np.any(labels == i) else c[i] for i in range(len(c))])
        if np.allclose(moved, c, atol=0.5):
            c = moved
            break
        c = moved
    labels = np.argmin(((pixels[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
    counts = np.bincount(labels, minlength=len(c))
    return c, counts


def palette_of(paths: list[Path], count: int = 5) -> list[str]:
    """Up to ``count`` main colours of the pictures, the most used first, as lowercase ``#rrggbb``."""
    import numpy as np

    pixels = _pixels(paths)
    if len(pixels) == 0:
        return []
    centres, counts = _kmeans(pixels, min(count + 7, len(pixels)))
    order = np.argsort(-counts)
    kept: list[tuple[np.ndarray, int]] = []
    for i in order:
        if counts[i] == 0:
            continue
        for j, (centre, weight) in enumerate(kept):
            if float(np.linalg.norm(centre - centres[i])) < _MERGE_DISTANCE:
                kept[j] = ((centre * weight + centres[i] * counts[i]) / (weight + counts[i]), weight + int(counts[i]))
                break
        else:
            kept.append((centres[i], int(counts[i])))
    # Rank by how much of the picture a colour covers, lifted by how colourful it is, so the amber of the tea
    # is not pushed out by four shades of grey wall.
    total = sum(weight for _, weight in kept)
    kept = [item for item in kept if item[1] >= total * 0.005] or kept  # specks are not part of the look
    kept.sort(key=lambda item: -(item[1] ** 0.5) * (0.15 + _chroma(item[0])))
    return [_hex(centre) for centre, _ in kept[:count]]


def _chroma(rgb) -> float:  # noqa: ANN001 - numpy row
    return (float(max(rgb)) - float(min(rgb))) / 255.0


def _hex(rgb) -> str:  # noqa: ANN001 - numpy row
    r, g, b = (max(0, min(255, int(round(float(v))))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


_HUES = (
    (15, "red"), (40, "orange"), (65, "amber"), (75, "yellow"), (160, "green"), (195, "teal"),
    (250, "blue"), (290, "purple"), (335, "pink"), (360, "red"),
)


def colour_name(hex_colour: str) -> str:
    """Plain words for a colour: "black", "warm cream", "deep green", "pale blue"..."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    if light < 0.12:
        return "black"
    if light > 0.93 and sat < 0.5:
        return "white"
    if sat < 0.12:
        return "light grey" if light > 0.65 else "dark grey" if light < 0.35 else "grey"
    degrees = hue * 360
    name = next(label for limit, label in _HUES if degrees < limit)
    if name in ("orange", "amber") and light < 0.35:
        name = "brown"
    if name in ("orange", "amber", "yellow") and light > 0.8 and sat < 0.7:
        return "warm cream"
    if light < 0.3:
        return f"deep {name}"
    if light > 0.75:
        return f"pale {name}"
    if sat < 0.35:
        return f"muted {name}"
    return name


def palette_words(colours: list[str]) -> str:
    """"deep green, warm cream and amber" (each name once, in palette order)."""
    names = list(dict.fromkeys(colour_name(c) for c in colours))
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"
