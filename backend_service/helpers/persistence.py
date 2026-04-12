"""Data persistence: benchmark runs, chat sessions, and seeding."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from backend_service.catalog import CATALOG


MAX_CHAT_SESSIONS = 200
MAX_BENCHMARK_RUNS = 48


def _default_chat_variant() -> dict[str, Any]:
    direct_variants = sorted(
        [entry for entry in CATALOG if entry.get("launchMode") == "direct"],
        key=lambda entry: (float(entry.get("paramsB") or 0), float(entry.get("sizeGb") or 0)),
    )
    return direct_variants[0] if direct_variants else CATALOG[0]


def _seed_chat_sessions() -> list[dict[str, Any]]:
    default_variant = _default_chat_variant()
    return [
        {
            "id": "ui-direction",
            "title": "Compact desktop layout",
            "updatedAt": "Today 17:18",
            "model": "Devstral Small 2507 GGUF",
            "modelRef": "mistralai/Devstral-Small-2507_gguf",
            "modelSource": "catalog",
            "modelPath": None,
            "modelBackend": "llama.cpp",
            "pinned": True,
            "cacheLabel": "Native 3-bit 4+4",
            "messages": [
                {
                    "role": "user",
                    "text": "Make the desktop UI feel tighter and more like a serious desktop tool.",
                },
                {
                    "role": "assistant",
                    "text": "I would reduce padding, tighten nav density, simplify the dashboard, and make threads and models feel more task-oriented than card-oriented.",
                    "metrics": None,
                },
            ],
        },
        {
            "id": "model-shortlist",
            "title": "Try newer local models",
            "updatedAt": "Today 15:42",
            "model": default_variant["name"],
            "modelRef": default_variant["id"],
            "modelSource": "catalog",
            "modelPath": None,
            "modelBackend": default_variant.get("backend", "auto"),
            "pinned": False,
            "cacheLabel": "Native 3-bit 4+4",
            "messages": [
                {
                    "role": "user",
                    "text": "Which local-first models feel fresher than the old Qwen2.5 shortlist?",
                },
                {
                    "role": "assistant",
                    "text": "Gemma 4, Qwen 3.5, Nemotron 3 Nano, and Devstral are stronger directions to browse first depending on whether you care most about vision, reasoning, or coding.",
                    "metrics": None,
                },
            ],
        },
    ]


def _seed_benchmark_runs() -> list[dict[str, Any]]:
    return [
        {
            "id": "baseline",
            "label": "Nemotron 3 Nano 4B GGUF / Native f16 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:12:08",
            "bits": 16,
            "fp16Layers": 0,
            "cacheStrategy": "native",
            "cacheLabel": "Native f16 cache",
            "cacheGb": 14.0,
            "baselineCacheGb": 14.0,
            "compression": 1.0,
            "tokS": 52.1,
            "quality": 100,
            "responseSeconds": 4.2,
            "loadSeconds": 5.8,
            "totalSeconds": 10.0,
            "promptTokens": 78,
            "completionTokens": 219,
            "totalTokens": 297,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed baseline run for comparison.",
        },
        {
            "id": "native-34",
            "label": "Nemotron 3 Nano 4B GGUF / Native 3-bit 4+4 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:18:44",
            "bits": 3,
            "fp16Layers": 4,
            "cacheStrategy": "native",
            "cacheLabel": "Native 3-bit 4+4",
            "cacheGb": 5.9,
            "baselineCacheGb": 14.0,
            "compression": 2.4,
            "tokS": 30.7,
            "quality": 98,
            "responseSeconds": 7.1,
            "loadSeconds": 6.0,
            "totalSeconds": 13.1,
            "promptTokens": 78,
            "completionTokens": 218,
            "totalTokens": 296,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed adaptive cache strategy run.",
        },
        {
            "id": "native-36",
            "label": "Nemotron 3 Nano 4B GGUF / Native 3-bit 6+6 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:25:19",
            "bits": 3,
            "fp16Layers": 6,
            "cacheStrategy": "native",
            "cacheLabel": "Native 3-bit 6+6",
            "cacheGb": 7.5,
            "baselineCacheGb": 14.0,
            "compression": 1.9,
            "tokS": 33.0,
            "quality": 98,
            "responseSeconds": 6.7,
            "loadSeconds": 6.1,
            "totalSeconds": 12.8,
            "promptTokens": 78,
            "completionTokens": 220,
            "totalTokens": 298,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed higher-FP16 edge run.",
        },
        {
            "id": "native-44",
            "label": "Nemotron 3 Nano 4B GGUF / Native 4-bit 4+4 / 8K ctx",
            "model": "Nemotron 3 Nano 4B GGUF",
            "modelRef": "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            "backend": "llama.cpp",
            "engineLabel": "llama.cpp + GGUF",
            "source": "catalog",
            "measuredAt": "2026-04-05 12:32:57",
            "bits": 4,
            "fp16Layers": 4,
            "cacheStrategy": "native",
            "cacheLabel": "Native 4-bit 4+4",
            "cacheGb": 7.1,
            "baselineCacheGb": 14.0,
            "compression": 2.0,
            "tokS": 35.8,
            "quality": 99,
            "responseSeconds": 6.0,
            "loadSeconds": 6.0,
            "totalSeconds": 12.0,
            "promptTokens": 78,
            "completionTokens": 215,
            "totalTokens": 293,
            "contextTokens": 8192,
            "maxTokens": 256,
            "notes": "Seed higher-quality cache strategy run.",
        },
    ]


def _load_benchmark_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return _seed_benchmark_runs()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _seed_benchmark_runs()

    if not isinstance(payload, list):
        return _seed_benchmark_runs()

    valid_runs = [item for item in payload if isinstance(item, dict) and item.get("id") and item.get("label")]
    return valid_runs[:MAX_BENCHMARK_RUNS] or _seed_benchmark_runs()


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

    valid = [s for s in payload if isinstance(s, dict) and s.get("id") and s.get("title")]
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
