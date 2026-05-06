"""Data persistence: benchmark runs and chat sessions."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MAX_CHAT_SESSIONS = 200
MAX_BENCHMARK_RUNS = 48
LEGACY_SEEDED_CHAT_IDS = {"ui-direction", "model-shortlist"}
LEGACY_SEEDED_BENCHMARK_IDS = {"baseline", "native-34", "native-36", "native-44"}

LIBRARY_CACHE_VERSION = 3


def _default_chat_variant() -> dict[str, Any]:
    from backend_service.catalog import CATALOG

    direct_variants = sorted(
        [entry for entry in CATALOG if entry.get("launchMode") == "direct"],
        key=lambda entry: (float(entry.get("paramsB") or 0), float(entry.get("sizeGb") or 0)),
    )
    return direct_variants[0] if direct_variants else CATALOG[0]


def _seed_chat_sessions() -> list[dict[str, Any]]:
    return []


def _seed_benchmark_runs() -> list[dict[str, Any]]:
    return []


def _load_benchmark_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    valid_runs = [
        item
        for item in payload
        if isinstance(item, dict)
        and item.get("id")
        and item.get("label")
        and item.get("id") not in LEGACY_SEEDED_BENCHMARK_IDS
    ]
    return valid_runs[:MAX_BENCHMARK_RUNS]


def _save_benchmark_runs(runs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runs[:MAX_BENCHMARK_RUNS], indent=2), encoding="utf-8")


def _load_chat_sessions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []

    valid = [
        s
        for s in payload
        if isinstance(s, dict)
        and s.get("id")
        and s.get("title")
        and s.get("id") not in LEGACY_SEEDED_CHAT_IDS
    ]
    return valid[:MAX_CHAT_SESSIONS]


def _save_chat_sessions(sessions: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sessions[:MAX_CHAT_SESSIONS], indent=2, default=str), encoding="utf-8")
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(str(tmp), str(path))


def _library_cache_fingerprint(model_directories: list[dict[str, Any]]) -> dict[str, float]:
    fingerprint: dict[str, float] = {}
    for directory in model_directories:
        if not directory.get("enabled", True):
            continue
        raw_path = str(directory.get("path") or "").strip()
        if not raw_path:
            continue
        root = Path(os.path.expanduser(raw_path))
        fingerprint[raw_path] = 0.0
        if not root.exists():
            continue
        max_mtime = 0.0
        try:
            max_mtime = root.stat().st_mtime
        except OSError:
            pass
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        mtime = entry.stat(follow_symlinks=False).st_mtime
                    except OSError:
                        continue
                    if mtime > max_mtime:
                        max_mtime = mtime
        except OSError:
            pass
        fingerprint[raw_path] = max_mtime
    return fingerprint


def _load_library_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != LIBRARY_CACHE_VERSION:
        return None
    fingerprint = payload.get("fingerprint")
    items = payload.get("items")
    if not isinstance(fingerprint, dict) or not isinstance(items, list):
        return None
    return payload


def _save_library_cache(
    items: list[dict[str, Any]],
    fingerprint: dict[str, float],
    path: Path,
) -> None:
    import time as _time

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": LIBRARY_CACHE_VERSION,
        "scannedAt": _time.time(),
        "fingerprint": fingerprint,
        "items": items,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _build_benchmarks() -> list[dict[str, Any]]:
    return [
        {
            "id": "baseline",
            "label": "FP16 baseline",
            "bits": 16,
            "fp16Layers": 32,
            "cacheGb": 14.0,
            "compression": 1.0,
            "tokS": 52.1,
            "quality": 100,
        },
        {
            "id": "native-34",
            "label": "Native 3-bit 4+4",
            "bits": 3,
            "fp16Layers": 4,
            "cacheGb": 5.9,
            "compression": 2.4,
            "tokS": 30.7,
            "quality": 98,
        },
        {
            "id": "native-36",
            "label": "Native 3-bit 6+6",
            "bits": 3,
            "fp16Layers": 6,
            "cacheGb": 7.5,
            "compression": 1.9,
            "tokS": 33.0,
            "quality": 98,
        },
        {
            "id": "native-44",
            "label": "Native 4-bit 4+4",
            "bits": 4,
            "fp16Layers": 4,
            "cacheGb": 7.1,
            "compression": 2.0,
            "tokS": 35.8,
            "quality": 99,
        },
    ]
