"""Installed and recommended models for the bridge tools. Read only; no downloads."""

from __future__ import annotations

from typing import Any

KINDS = ("image", "video", "voice", "stt")


def _families(state: Any, kind: str) -> list[dict[str, Any]]:
    from backend_service.helpers.fit import annotate_fit
    from backend_service.helpers.images import _image_model_payloads
    from backend_service.helpers.video import _video_model_payloads

    library = state._library()
    families = _image_model_payloads(library) if kind == "image" else _video_model_payloads(library)
    return annotate_fit(families, kind)


def _text_to_media(variant: dict[str, Any], kind: str) -> bool:
    """v1 tools start from a prompt only; skip models that need an input image."""
    tasks = [str(t) for t in variant.get("taskSupport") or []]
    return not tasks or ("txt2img" if kind == "image" else "txt2video") in tasks


def _licence_hints(family: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    """The catalog has no licence field; read the family badges and the variant note."""
    text = f"{variant.get('note') or ''} {family.get('description') or ''}".lower()
    badges = [str(b).lower() for b in family.get("badges") or []]
    return {
        "gated": "gated" in badges or "gated" in text,
        "nonCommercial": "non-commercial" in text or "noncommercial" in text,
    }


def _row(family: dict[str, Any], variant: dict[str, Any], kind: str) -> dict[str, Any]:
    fit = variant.get("fit") or {}
    row: dict[str, Any] = {
        "id": variant.get("id"),
        "name": variant.get("name"),
        "family": family.get("name"),
        "sizeGb": variant.get("sizeGb"),
        "installed": bool(variant.get("availableLocally")),
        "fit": fit.get("status"),
        "fitReason": fit.get("reason"),
        "recommendedResolution": variant.get("recommendedResolution"),
        "estimatedSeconds": variant.get("estimatedGenerationSeconds"),
        "maxSide": variant.get("maxSide"),
        "note": variant.get("note"),
        **_licence_hints(family, variant),
    }
    if kind == "video":
        from backend_service.helpers.generation_limits import align_video_frames

        repo = str(variant.get("repo") or "")
        probe, _ = align_video_frames(repo, 97)
        row["frameRule"] = (
            "Frame counts are rounded down to a valid value (for example 97 becomes "
            f"{probe}). Pass seconds and the tool converts."
            if probe != 97
            else "Any frame count works."
        )
        row["defaultSeconds"] = variant.get("defaultDurationSeconds")
    return row


def installed_models(state: Any, kind: str) -> list[dict[str, Any]]:
    if kind in ("image", "video"):
        return [
            _row(family, variant, kind)
            for family in _families(state, kind)
            for variant in family.get("variants") or []
            if variant.get("availableLocally")
            and _text_to_media(variant, kind)
            and (variant.get("fit") or {}).get("status") != "wont-fit"
        ]
    from backend_service import voice_runtime

    if kind == "stt":
        return [m for m in voice_runtime.list_stt_models() if m.get("installed")]
    return [
        {"id": "kokoro", "name": "Kokoro", "voices": len([v for v in voice_runtime.list_tts_voices() if v["available"]])}
    ]


def list_voices() -> list[dict[str, Any]]:
    from backend_service import voice_runtime

    return [
        {"id": v["id"], "name": v["name"], "gender": v["gender"], "language": v["languageLabel"]}
        for v in voice_runtime.list_tts_voices()
        if v.get("available")
    ]


def recommend(state: Any, kind: str, limit: int = 3) -> dict[str, Any]:
    """Best not-installed models that fit this machine, newest first (the catalog's own best-fit order)."""
    if kind == "voice":
        return {
            "kind": kind,
            "models": [],
            "note": "Speech uses the built-in voice engine. Install it from ChaosEngineAI: Voice > Setup.",
        }
    if kind == "stt":
        from backend_service import voice_runtime

        rows = [m for m in voice_runtime.list_stt_models() if not m.get("installed")]
        return {"kind": kind, "models": rows[:limit]}
    from backend_service.helpers.fit import best_fit_variant_id

    picks: list[dict[str, Any]] = []
    for family in _families(state, kind):
        usable = [v for v in family.get("variants") or [] if _text_to_media(v, kind)]
        best_id = best_fit_variant_id(usable, kind)
        for variant in usable:
            if variant.get("id") == best_id and not variant.get("availableLocally"):
                picks.append({**_row(family, variant, kind), "releaseDate": variant.get("releaseDate")})
    # Open licence and ungated first (an agent can send the user straight there), then newest.
    picks.sort(key=lambda r: str(r.get("releaseDate") or ""), reverse=True)
    picks.sort(key=lambda r: (bool(r.get("nonCommercial")), bool(r.get("gated"))))
    for row in picks:
        row.pop("releaseDate", None)
    return {
        "kind": kind,
        "models": picks[:limit],
        "howToGet": "Ask the user to open ChaosEngineAI > Models and download one. Gated models need "
        "the licence accepted on Hugging Face first.",
    }


def default_model(state: Any, kind: str) -> dict[str, Any] | None:
    """The installed model to use when the agent names none: the catalog's best fit, else the newest that fits."""
    rows = installed_models(state, kind)
    fitting = [r for r in rows if r.get("fit") == "fits"] or rows
    return fitting[0] if fitting else None
