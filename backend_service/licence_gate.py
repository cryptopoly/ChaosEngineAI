"""Territory-limited model licences: the user's own statement, recorded, and checked everywhere a row is reached.

Some weights may only be used outside certain territories (MiniMax H3: not in the EU, the UK, the Republic of
Korea or the USA, D-76). The app never guesses where the user is (no network lookup): the user ticks a sentence in
the app's window and that tick is the record. Until then a gated row is hidden from every list and refused by every
route that downloads, loads or runs it, for the app's UI and for agents alike. Only the UI route can confirm (an
agent token, ``/mcp`` or another machine is refused), and no agent tool confirms.

``H3_RELEASED`` is the release switch: while it is False the rows stay hidden and nothing can be confirmed, whatever
is on disk. It stays False until the owner has had the licence read (plan v0.10.0 task B).

The record (key, time, app version, the exact sentence shown) lives in ``licence-gate.json`` beside
``settings.json``; every confirm and withdraw is also appended to ``licence-gate-audit.jsonl`` there.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

# The release switch (owner's call after a legal read). Flipping it is the only change needed to ship the rows.
H3_RELEASED = False

_LOCK = threading.Lock()

H3_KEY = "minimax-h3"
H3_LICENCE_URL = (
    "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/LICENSE"
)

TERRITORY_MODELS: dict[str, dict[str, Any]] = {
    H3_KEY: {
        "key": H3_KEY,
        "name": "MiniMax H3",
        "kinds": ["video"],
        "excludedTerritories": ["European Union", "United Kingdom", "Republic of Korea", "United States"],
        "licenceName": "MiniMax H3 Community License",
        "licenceUrl": H3_LICENCE_URL,
        "summaryLines": [
            "Licensed only outside the European Union, the United Kingdom, the Republic of Korea and the United "
            "States. Using the model or what it makes in those places is not allowed.",
            "Free to use, including commercially, until your products earn more than US$20 million a year; above "
            "that you need written permission from MiniMax.",
            "A commercial product that uses it must show \"MiniMax H3\" in its interface.",
            "You may not use the model or its videos to improve another AI model.",
            "Use must follow MiniMax's acceptable use policy (for example: no impersonation without consent, label "
            "machine-made content you publish, no military use).",
        ],
        "confirmSentence": (
            "I confirm that I am not in the European Union, the United Kingdom, the Republic of Korea or the United "
            "States, and that I will not use this model or what it makes there."
        ),
        "note": (
            "The weights are never bundled with ChaosEngineAI: you download them from Hugging Face yourself. It "
            "runs on the stable-diffusion.cpp runtime (sd), which release builds may not include. Not tested by "
            "the maintainers."
        ),
    },
}

_RELEASED: dict[str, Any] = {H3_KEY: lambda: H3_RELEASED}


class LicenceGateError(Exception):
    """A gated model was reached without a confirmation (or before release). Routes map it to HTTP 403."""

    status_code = 403

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key
        self.message = message


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    from backend_service.app import SETTINGS_DIR  # noqa: PLC0415 - the app module is loaded when this runs

    return Path(SETTINGS_DIR)


def _path() -> Path:
    return _data_dir() / "licence-gate.json"


def _audit_path() -> Path:
    return _data_dir() / "licence-gate-audit.jsonl"


def _read() -> dict[str, Any]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: dict[str, Any]) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _audit(event: str, **fields: Any) -> None:
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    try:
        path = _audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def is_released(key: str) -> bool:
    check = _RELEASED.get(key)
    return bool(check and check())


def record(key: str) -> dict[str, Any] | None:
    entry = _read().get(key)
    return entry if isinstance(entry, dict) else None


def is_confirmed(key: str) -> bool:
    """True only when the key is released and the user confirmed the sentence that is shown now."""
    if key not in TERRITORY_MODELS or not is_released(key):
        return False
    entry = record(key)
    return bool(entry and entry.get("sentence") == TERRITORY_MODELS[key]["confirmSentence"])


def refusal_message(key: str) -> str:
    model = TERRITORY_MODELS.get(key) or {"name": key}
    if not is_released(key):
        return f"{model['name']} is not available in this version of ChaosEngineAI."
    return (
        f"{model['name']} is limited by territory. Confirm the territory statement in ChaosEngineAI "
        "(Video > Models > Models with territory limits > Review) before downloading or running it."
    )


def require(key: str) -> None:
    """Raise ``LicenceGateError`` (HTTP 403) unless ``key`` is released and confirmed."""
    if not is_confirmed(key):
        raise LicenceGateError(key, refusal_message(key))


def confirm(key: str, *, sentence: str, app_version: str | None = None) -> dict[str, Any]:
    """Store the user's statement. ``sentence`` must be the exact sentence the screen showed."""
    model = TERRITORY_MODELS.get(key)
    if model is None:
        raise KeyError(key)
    if not is_released(key):
        raise LicenceGateError(key, refusal_message(key))
    if sentence != model["confirmSentence"]:
        raise ValueError("The statement does not match the one this version shows. Reopen the screen and tick again.")
    entry = {
        "key": key,
        "confirmedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "appVersion": app_version or "unknown",
        "sentence": sentence,
        "licenceUrl": model["licenceUrl"],
    }
    with _LOCK:
        data = _read()
        data[key] = entry
        _write(data)
    _audit("confirm", key=key, appVersion=entry["appVersion"], sentence=sentence)
    return entry


def withdraw(key: str) -> None:
    if key not in TERRITORY_MODELS:
        raise KeyError(key)
    with _LOCK:
        data = _read()
        had = data.pop(key, None) is not None
        _write(data)
    if had:
        _audit("withdraw", key=key)


def status() -> dict[str, Any]:
    """What the Models pages read: one entry per released territory model (nothing while unreleased)."""
    models = []
    for key, model in TERRITORY_MODELS.items():
        if not is_released(key):
            continue
        entry = record(key) if is_confirmed(key) else None
        models.append({
            **model,
            "confirmed": entry is not None,
            "confirmedAt": entry.get("confirmedAt") if entry else None,
        })
    return {"models": models}


# ---------------------------------------------------------------------------
# Catalog rows
# ---------------------------------------------------------------------------

def row_key(variant: dict[str, Any]) -> str | None:
    key = variant.get("territoryGate")
    return str(key) if key else None


def variant_visible(variant: dict[str, Any]) -> bool:
    key = row_key(variant)
    return key is None or is_confirmed(key)


def require_variant(variant: dict[str, Any] | None) -> None:
    key = row_key(variant or {})
    if key:
        require(key)


def filter_families(families: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Drop hidden gated rows (and families left empty). Returns the families and the keys of released models
    whose rows were hidden, so the page can show one neutral notice."""
    kept: list[dict[str, Any]] = []
    hidden: list[str] = []
    for family in families:
        variants = list(family.get("variants") or [])
        visible = [v for v in variants if variant_visible(v)]
        for variant in variants:
            key = row_key(variant)
            if key and not variant_visible(variant) and is_released(key) and key not in hidden:
                hidden.append(key)
        if len(visible) == len(variants):
            kept.append(family)
        elif visible:
            kept.append({**family, "variants": visible})
    return kept, hidden


def _catalog_rows() -> Iterable[dict[str, Any]]:
    from backend_service.catalog import IMAGE_MODEL_FAMILIES, VIDEO_MODEL_FAMILIES  # noqa: PLC0415

    for family in (*IMAGE_MODEL_FAMILIES, *VIDEO_MODEL_FAMILIES):
        yield from family.get("variants") or []


def key_for_repo(repo: str | None) -> str | None:
    """The gate key of any catalog row that downloads ``repo`` (its own repo, GGUF repo or a companion)."""
    if not repo:
        return None
    for variant in _catalog_rows():
        key = row_key(variant)
        if not key:
            continue
        repos = {str(variant.get(name) or "") for name in ("repo", "ggufRepo", "textEncoderRepo")}
        repos.update(str(c.get("repo") or "") for c in variant.get("downloadCompanions") or [])
        if repo in repos:
            return key
    return None


def require_repo(repo: str | None) -> None:
    key = key_for_repo(repo)
    if key:
        require(key)
