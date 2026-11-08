"""Qwen3-TTS speech models through mlx-audio: Base (clone), CustomVoice (preset speakers plus a style
instruction) and VoiceDesign (a voice from a description) (D-28, D-45, D-58).

Only engines wired end to end are listed. Each row pins a full Hugging Face revision (Base rows checked on
2026-11-07, CustomVoice and VoiceDesign rows on 2026-11-08, through the Hub API, Apache-2.0, not gated), so
a repo update upstream cannot change what users run until this table changes. Sizes are the repo totals on
that day. Apple Silicon only for now: the CUDA path (``qwen-tts`` / transformers) is a follow-up.

``capabilities`` says what a row can do: ``speak`` (plain text to speech with a preset or designed voice),
``style`` (a written style instruction; the 0.6B CustomVoice model ignores it), ``clone`` (a voice from a
reference clip) and ``design`` (a voice from a description). No 0.6B VoiceDesign model exists upstream.
"""

from __future__ import annotations

from typing import Any

CLONE_LICENCE: dict[str, Any] = {
    "id": "apache-2.0",
    "name": "Apache 2.0",
    "commercial": "yes",
    "url": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

VOICE_MODELS: list[dict[str, Any]] = [
    {
        "key": "qwen3-tts-0.6b",
        "name": "Qwen3-TTS 0.6B Base",
        "summary": "Smaller in memory (about 4 GB). Good likeness from 10 to 20 s of clean speech.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "revision": "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
        "upstream": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "sizeGb": 2.52,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": True,
        "kind": "base",
        "capabilities": {"speak": False, "style": False, "clone": True, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 3.0,
            "secondsPer10sAudio": 3.2,
            "wallSecondsShortLine": 2.8,
            "peakGb": 4.3,
            "idleGb": 3.7,
        },
    },
    {
        "key": "qwen3-tts-1.7b",
        "name": "Qwen3-TTS 1.7B Base",
        "summary": "Closer likeness; needs about 6.5 GB of memory. About as quick as the 0.6B model.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "revision": "a6eb4f68e4b056f1215157bb696209bc82a6db48",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sizeGb": 4.54,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": False,
        "kind": "base",
        "capabilities": {"speak": False, "style": False, "clone": True, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 3.6,
            "secondsPer10sAudio": 3.6,
            "wallSecondsShortLine": 2.2,
            "peakGb": 6.5,
            "idleGb": 5.6,
        },
    },
    {
        "key": "qwen3-tts-custom-0.6b",
        "name": "Qwen3-TTS 0.6B CustomVoice",
        "summary": "Smallest in memory (about 3 GB). Nine preset voices; ignores the style instruction. No quicker than the 1.7B model.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "revision": "6415d95f88be018ff9e46813119dc3bc12261328",
        "upstream": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "sizeGb": 2.50,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": False,
        "kind": "custom",
        "capabilities": {"speak": True, "style": False, "clone": False, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 2.4,
            "secondsPer10sAudio": 2.65,
            "wallSecondsShortLine": 1.9,
            "peakGb": 2.93,
            "idleGb": 2.91,
        },
    },
    {
        "key": "qwen3-tts-custom-1.7b",
        "name": "Qwen3-TTS 1.7B CustomVoice",
        "summary": "Nine preset voices and a written style instruction (8-bit).",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "revision": "41d3337e8b7f2843a75841595fc14e4b9a7a4b96",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "sizeGb": 3.08,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": False,
        "kind": "custom",
        "capabilities": {"speak": True, "style": True, "clone": False, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 2.4,
            "secondsPer10sAudio": 2.45,
            "wallSecondsShortLine": 1.8,
            "peakGb": 3.5,
            "idleGb": 3.48,
        },
    },
    {
        "key": "qwen3-tts-design-1.7b",
        "name": "Qwen3-TTS 1.7B VoiceDesign",
        "summary": "Describe a voice in words and it speaks in that voice (8-bit).",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "revision": "f90d617701d9f7f4ca499291e0b57f2b3c2fd2ee",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "sizeGb": 3.08,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": False,
        "kind": "design",
        "capabilities": {"speak": True, "style": True, "clone": False, "design": True},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 2.3,
            "secondsPer10sAudio": 2.46,
            "wallSecondsShortLine": 1.5,
            "peakGb": 3.51,
            "idleGb": 3.49,
        },
    },
]

# The cloning screen and the clone tools only know the Base rows.
CLONE_MODELS: list[dict[str, Any]] = [row for row in VOICE_MODELS if row["capabilities"]["clone"]]

# CustomVoice preset speakers: id, language code, language name for the model, gender, one-line feel.
CUSTOM_SPEAKERS: list[dict[str, str]] = [
    {"id": "Ryan", "language": "en", "languageLabel": "English", "gender": "male", "about": "Dynamic, strong rhythm"},
    {"id": "Aiden", "language": "en", "languageLabel": "English", "gender": "male", "about": "Sunny American, clear midrange"},
    {"id": "Vivian", "language": "zh", "languageLabel": "Chinese", "gender": "female", "about": "Bright, slightly edgy"},
    {"id": "Serena", "language": "zh", "languageLabel": "Chinese", "gender": "female", "about": "Warm and gentle"},
    {"id": "Uncle_Fu", "language": "zh", "languageLabel": "Chinese", "gender": "male", "about": "Seasoned, low and mellow"},
    {"id": "Dylan", "language": "zh", "languageLabel": "Chinese (Beijing)", "gender": "male", "about": "Youthful Beijing"},
    {"id": "Eric", "language": "zh", "languageLabel": "Chinese (Sichuan)", "gender": "male", "about": "Lively, slightly husky"},
    {"id": "Ono_Anna", "language": "ja", "languageLabel": "Japanese", "gender": "female", "about": "Playful"},
    {"id": "Sohee", "language": "ko", "languageLabel": "Korean", "gender": "female", "about": "Warm, rich emotion"},
]


def clone_model(key: str | None) -> dict[str, Any]:
    """The cloning row for ``key``; empty means the default. Raises ValueError for anything else."""
    wanted = (key or "").strip()
    for row in CLONE_MODELS:
        if (not wanted and row["default"]) or wanted in (row["key"], row["repo"]):
            return row
    raise ValueError(f"Unknown voice cloning model '{wanted}'.")


def voice_model(key: str) -> dict[str, Any]:
    """Any row by key or repo id. Raises ValueError for anything else."""
    wanted = (key or "").strip()
    for row in VOICE_MODELS:
        if wanted in (row["key"], row["repo"]):
            return row
    raise ValueError(f"Unknown voice model '{wanted}'.")
