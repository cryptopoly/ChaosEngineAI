"""Draft, Standard and Final for a single image or clip in Chat (FU-241, D-91, D-92).

One definition of what a tier sets: size, fps and steps. The model keeps the content (prompt, length, seed). A call
that asks for more than the tier runs at the tier, and the result says so. The Storyboard keeps its own draft and
final passes (``draft: true``, ``what: final``); these tiers are the same names for a single render.

The Final size is a proposal until the clip timings of claim C10 are measured: 1.5 times the recommended size,
capped by the model's largest side.
"""

from __future__ import annotations

import re
from typing import Any

TIERS = ("draft", "standard", "final")
DEFAULT_TIER = "standard"
SINGLE_RENDERS = ("generate_image", "generate_video")
_FLOOR = 256
_FPS = 24

_DRAFT_WORDS = re.compile(r"\b(draft|drafts|quick|quickly|rough|preview)\b", re.IGNORECASE)
_FINAL_WORDS = re.compile(r"\b(final|full quality|best quality|highest quality|high quality)\b", re.IGNORECASE)
_STANDARD_WORDS = re.compile(r"\bstandard quality\b", re.IGNORECASE)


def tier_from_text(text: str | None) -> str | None:
    """The tier a message names in plain words, or None. Final wins over draft ("a quick draft, then the final")."""
    text = text or ""
    if _FINAL_WORDS.search(text):
        return "final"
    if _DRAFT_WORDS.search(text):
        return "draft"
    return "standard" if _STANDARD_WORDS.search(text) else None


def _multiple_of_64(value: float) -> int:
    return max(64, int(round(value / 64.0)) * 64)


def tier_size(tier: str, variant: dict[str, Any]) -> tuple[int, int] | None:
    """The tier's width and height for a catalog variant, or None when the row has no recommended size."""
    from backend_service.helpers.generation_limits import parse_resolution

    recommended = parse_resolution(variant.get("recommendedResolution"))
    if recommended is None or tier == "standard":
        return recommended
    width, height = recommended
    if tier == "draft":
        return (min(width, max(_FLOOR, _multiple_of_64(width / 2))), min(height, max(_FLOOR, _multiple_of_64(height / 2))))
    scale = 1.5
    max_side = variant.get("maxSide")
    if isinstance(max_side, (int, float)) and max_side > 0:
        scale = min(scale, float(max_side) / max(width, height))
    scale = max(scale, 1.0)
    return (_multiple_of_64(width * scale), _multiple_of_64(height * scale))


def tier_settings(tier: str, kind: str, variant: dict[str, Any]) -> dict[str, Any]:
    """What the tier sets for a variant: ``width``, ``height``, ``fps`` (clips) and ``steps`` (None when unknown)."""
    from backend_service.mcp_bridge.tools import _recommended

    size = tier_size(tier, variant)
    settings: dict[str, Any] = {"width": size[0] if size else None, "height": size[1] if size else None,
                                "steps": _recommended(variant).get("steps")}
    if kind == "video":
        settings["fps"] = _FPS
    return settings


def apply_tier(tier: str, name: str, args: dict[str, Any], variant: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """``args`` as the tier lets them run, and a note for each value brought down or filled in.

    Draft and Final name their size even when the call gave none; Standard leaves a call without a size to the
    handler, which uses the recommended size. A size with more pixels than the tier's, an fps above the tier's and
    steps above the recommended count are brought to the tier.
    """
    if name not in SINGLE_RENDERS or variant is None or tier not in TIERS:
        return args, []
    kind = "image" if name == "generate_image" else "video"
    settings = tier_settings(tier, kind, variant)
    out, notes = dict(args), []
    label = tier.capitalize()
    width, height = settings["width"], settings["height"]
    if width and height:
        asked = (out.get("width"), out.get("height"))
        if all(asked):
            if int(asked[0]) * int(asked[1]) > width * height:
                out["width"], out["height"] = width, height
                notes.append(f"{label} quality: size {asked[0]}x{asked[1]} brought to {width}x{height}.")
        elif tier != "standard":
            out["width"], out["height"] = width, height
            notes.append(f"{label} quality: size {width}x{height}.")
    steps = settings.get("steps")
    if steps is not None and isinstance(out.get("steps"), (int, float)) and out["steps"] > steps:
        notes.append(f"{label} quality: steps {out['steps']} brought to {steps}.")
        out["steps"] = steps
    if kind == "video" and isinstance(out.get("fps"), (int, float)) and out["fps"] > settings["fps"]:
        notes.append(f"{label} quality: {out['fps']} fps brought to {settings['fps']}.")
        out["fps"] = settings["fps"]
    return out, notes


def variant_for(name: str, model: Any, state: Any) -> dict[str, Any] | None:
    """The catalog variant a single render would use (the named model, else the default pick), or None."""
    from backend_service.mcp_bridge import models as bridge_models

    kind = "image" if name == "generate_image" else "video"
    try:
        if model:
            rows = bridge_models.installed_models(state, kind)
            row = next((r for r in rows if str(r.get("id")) == str(model)), None)
        else:
            row = bridge_models.default_model(state, kind)
        if not row:
            return None
        if kind == "image":
            from backend_service.helpers.images import _find_image_variant as find
        else:
            from backend_service.helpers.video import _find_video_variant as find
        return find(str(row["id"]))
    except Exception:  # noqa: BLE001 - a tier never blocks a render; without a variant the call runs as asked
        return None


def tiers_table(state: Any) -> list[dict[str, Any]]:
    """Every installed image and video model with what each tier would run and a time for exactly that."""
    from backend_service.helpers.generation_limits import align_video_frames
    from backend_service.helpers.time_estimate import variant_seconds
    from backend_service.mcp_bridge import models as bridge_models

    table: list[dict[str, Any]] = []
    for kind in ("image", "video"):
        for row in bridge_models.installed_models(state, kind):
            variant = variant_for("generate_image" if kind == "image" else "generate_video", row.get("id"), state)
            if variant is None:
                continue
            frames = 1
            if kind == "video":
                frames, _note = align_video_frames(
                    str(variant.get("repo") or ""), round(float(variant.get("defaultDurationSeconds") or 4) * _FPS))
            tiers: dict[str, Any] = {}
            for tier in TIERS:
                settings = tier_settings(tier, kind, variant)
                seconds = None
                if settings["width"] and settings["height"]:
                    seconds = variant_seconds(variant, kind, settings["width"], settings["height"], frames, settings["steps"])
                tiers[tier] = {**settings, "frames": frames if kind == "video" else None,
                               "estimatedSeconds": round(seconds) if seconds else None}
            table.append({"id": row.get("id"), "name": row.get("name"), "kind": kind, "tiers": tiers})
    return table
