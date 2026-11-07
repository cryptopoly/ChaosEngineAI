"""Voice cloning models: Qwen3-TTS Base through mlx-audio (D-28, D-45).

Only engines wired end to end are listed. Each row pins a full Hugging Face revision (checked on
2026-11-07 through the Hub API, Apache-2.0, not gated), so a repo update upstream cannot change what
users run until this table changes. Sizes are the repo totals on that day. Apple Silicon only for
now: the CUDA path (``qwen-tts`` / transformers) is a follow-up.
"""

from __future__ import annotations

from typing import Any

CLONE_LICENCE: dict[str, Any] = {
    "id": "apache-2.0",
    "name": "Apache 2.0",
    "commercial": "yes",
    "url": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

CLONE_MODELS: list[dict[str, Any]] = [
    {
        "key": "qwen3-tts-0.6b",
        "name": "Qwen3-TTS 0.6B Base",
        "summary": "Faster. Good likeness from 10 to 20 s of clean speech.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "revision": "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
        "upstream": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "sizeGb": 2.52,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": True,
    },
    {
        "key": "qwen3-tts-1.7b",
        "name": "Qwen3-TTS 1.7B Base",
        "summary": "Slower, closer likeness.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "revision": "a6eb4f68e4b056f1215157bb696209bc82a6db48",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sizeGb": 4.54,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "default": False,
    },
]


def clone_model(key: str | None) -> dict[str, Any]:
    """The row for ``key``; empty means the default. Raises ValueError for anything else."""
    wanted = (key or "").strip()
    for row in CLONE_MODELS:
        if (not wanted and row["default"]) or wanted in (row["key"], row["repo"]):
            return row
    raise ValueError(f"Unknown voice cloning model '{wanted}'.")
