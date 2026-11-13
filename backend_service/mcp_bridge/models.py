"""Installed and recommended models for the bridge tools. Read only; no downloads."""

from __future__ import annotations

from typing import Any

KINDS = ("image", "video", "voice", "stt", "music")
_TEXT_TASK = {"image": "txt2img", "video": "txt2video", "music": "txt2music"}


def _families(state: Any, kind: str) -> list[dict[str, Any]]:
    """Catalog families of a generating kind with ``fit``; any other kind is an error, never video."""
    from backend_service.helpers.fit import annotate_fit

    if kind == "image":
        from backend_service.helpers.images import _image_model_payloads

        return annotate_fit(_image_model_payloads(state._library()), kind)
    if kind == "video":
        from backend_service.helpers.video import _video_model_payloads

        return annotate_fit(_video_model_payloads(state._library()), kind)
    if kind == "music":
        from backend_service.helpers.music import music_model_payloads

        return music_model_payloads()
    raise ValueError(f"Unknown model kind '{kind}'.")


def _text_to_media(variant: dict[str, Any], kind: str) -> bool:
    """v1 tools start from a prompt only; skip models that need an input image or audio."""
    tasks = [str(t) for t in variant.get("taskSupport") or []]
    return not tasks or _TEXT_TASK[kind] in tasks


def _music_row(variant: dict[str, Any]) -> dict[str, Any]:
    fit = variant.get("fit") or {}
    licence = variant.get("licence") or {}
    estimates = variant.get("estimates") or {}
    return {
        "id": variant.get("id"),
        "name": variant.get("name"),
        "sizeGb": variant.get("sizeGb"),
        "installed": bool(variant.get("availableLocally") and not variant.get("missingRequires")),
        "packInstalled": bool(variant.get("packInstalled")),
        "fit": fit.get("status"),
        "fitReason": fit.get("reason"),
        "makes": variant.get("makes"),
        "maxSeconds": variant.get("maxSeconds"),
        "secondsPer10sAudio": estimates.get("secondsPer10sAudio"),
        "peakGb": estimates.get("peakGb"),
        "licence": licence.get("name"),
        "commercial": licence.get("commercial"),
        "gated": bool(licence.get("gated")),
        "nonCommercial": licence.get("commercial") == "no",
        "note": variant.get("note"),
    }


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
    if kind == "image":
        from backend_service.image_runtime.mflux_engine import reference_image_limit

        limit = reference_image_limit(str(variant.get("repo") or "")) if variant.get("engine") == "mflux" else 0
        row["acceptsImageInput"] = limit > 0
        row["maxReferenceImages"] = limit
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
        from backend_service.video_runtime.repos import image_input_support

        row["acceptsImageInput"], row["acceptsEndImage"] = image_input_support(repo)
    return row


def installed_models(state: Any, kind: str) -> list[dict[str, Any]]:
    if kind not in KINDS:
        raise ValueError(f"Unknown model kind '{kind}'.")
    if kind == "music":
        return [
            _music_row(variant)
            for family in _families(state, kind)
            for variant in family.get("variants") or []
            if variant.get("availableLocally") and not variant.get("missingRequires")
            and _text_to_media(variant, kind)
            and (variant.get("fit") or {}).get("status") != "wont-fit"
        ]
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
    # kind == "voice": Kokoro when its quick voices are installed (an add-on since D-75)
    from backend_service import voice_env

    if not voice_env.kokoro_installed():
        return []
    return [
        {"id": "kokoro", "name": "Kokoro", "voices": len([v for v in voice_runtime.list_tts_voices() if v["available"]])}
    ]


def list_voices() -> list[dict[str, Any]]:
    from backend_service import voice_speak

    # Cloned voices have their own tools (list_cloned_voices, speak_cloned).
    return [
        {"id": v["id"], "name": v["name"], "gender": v["gender"], "language": v["languageLabel"], "kind": v["kind"]}
        for v in voice_speak.list_voices(include_clones=False)
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
    if kind == "music":
        rows = [
            _music_row(v) for family in _families(state, kind) for v in family.get("variants") or []
            if not v.get("availableLocally") and (v.get("fit") or {}).get("status") in ("fits", "tight")
        ]
        rows.sort(key=lambda r: (r["fit"] != "fits", r["sizeGb"] or 0))
        return {
            "kind": kind,
            "models": rows[:limit],
            "howToGet": "Ask the user to open ChaosEngineAI > Music > Models, install the music pack and "
            "download a model.",
        }
    if kind not in ("image", "video"):
        raise ValueError(f"Unknown model kind '{kind}'.")
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
