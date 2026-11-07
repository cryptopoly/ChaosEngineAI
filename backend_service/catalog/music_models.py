"""Curated catalog of music models: only engines that are wired end to end.

Same family -> variants shape as the image and video catalogs, plus the fields the Music pages need
(``licence``, ``makes``, ``platforms``, ``estimates``, ``defaultSettings``). Every number in
``estimates`` comes from the ACE-Step spike on an M4 Max with 64 GB (``docs/plans/music-spike.md``,
sections 3.2, 4.2 and 5); ``estimate_run`` in ``music_runtime`` reads this table and nothing else,
so a new measurement is a change here. Weights are pinned to full Hugging Face revisions.
"""

from __future__ import annotations

from typing import Any

# Wording the plan asks for (home-studio.md section 6) plus the authors' own disclaimer, short.
_ACESTEP_LICENCE: dict[str, Any] = {
    "id": "mit",
    "name": "MIT",
    "commercial": "yes",
    "gated": False,
    "summaryLines": [
        "Weights are MIT-licensed. The authors state the model was trained on legally compliant data and "
        "that output may be used commercially. You are responsible for checking that your output does not "
        "copy existing works.",
        "The authors encourage you to check that a track is original before you publish it and to say it "
        "was made with AI.",
        "You are responsible for rights in lyrics you type.",
    ],
    "url": "https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/LICENSE",
}

# The worker's generation defaults (spike section 4). Not user controls in the first version except
# the ones the request model exposes.
_TURBO_DEFAULTS: dict[str, Any] = {
    "seconds": 60,
    "variations": 1,
    "steps": 8,
    "shift": 3.0,
    "sampler": "ode",
    "planner": False,
    "vocalLanguage": "en",
    "normalizePeakDb": -1.0,
    "format": "flac24",
    "sampleRate": 48000,
    "channels": 2,
}

MUSIC_MODEL_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "ace-step-1.5",
        "name": "ACE-Step 1.5",
        "provider": "ACE Studio and StepFun",
        "headline": "Songs with vocals and lyrics, or instrumentals, 10 s to 10 minutes.",
        "summary": "Text to music with BPM, key and seed control. Runs on MLX on Apple Silicon.",
        "badges": ["MIT", "Songs", "Instrumental"],
        "defaultVariantId": "ACE-Step/Ace-Step1.5",
        "variants": [
            {
                "id": "ACE-Step/Ace-Step1.5",
                "familyId": "ace-step-1.5",
                "name": "ACE-Step 1.5 turbo",
                "provider": "ACE-Step",
                "repo": "ACE-Step/Ace-Step1.5",
                "revision": "19671f406d603126926c1b7e2adc169acbcade22",
                "link": "https://huggingface.co/ACE-Step/Ace-Step1.5",
                "engine": "acestep",
                "engineModel": "acestep-v15-turbo",
                "runtime": "ACE-Step 1.5 on MLX (isolated music venv)",
                "makes": ["songs", "instrumental"],
                "platforms": ["apple-silicon", "cuda"],
                "taskSupport": ["txt2music"],
                "sizeGb": 10.09,
                "sizeNote": "Includes the 1.7B planner LM (3.7 GB) the library requires on disk; it is not loaded by default.",
                "requires": [],
                # Where the snapshot goes under <music data>/checkpoints ("" = the root) and the folders
                # that must then hold weights (the library's own MAIN_MODEL_COMPONENTS check).
                "checkpointSubdir": "",
                "weightFolders": ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B"],
                "large": False,
                "minMemoryGb": None,
                "unloadsOthers": False,
                "maxSeconds": 600,
                "minSeconds": 10,
                "estimatedGenerationSeconds": 5.9,
                "licence": _ACESTEP_LICENCE,
                "estimates": {
                    "basis": "measured",
                    "measuredOn": "Apple M4 Max, 64 GB, 2026-11-06",
                    # Warm worker, recommended settings, one variation: (audio seconds, wall seconds).
                    "wallSecondsAt": [[30, 3.6], [60, 5.9], [180, 18.1], [300, 33.5], [600, 82.4]],
                    "secondsPer10sAudio": 1.0,
                    "coldStartSeconds": 25.0,
                    "vaeSecondsPerAudioSecond": 0.06,
                    "peakGb": 22.9,
                    "idleGb": 15.2,
                    "diskGb": 10.09,
                    # FLAC 24-bit 48 kHz stereo written by the worker (refined by the real check).
                    "outputMbPerMinute": 15.0,
                    # No CUDA lane yet and no peak figure on the card for the 2B: fit says "unknown" there.
                    "cuda": {"peakGb": None, "basis": "estimated", "source": "not measured; no peak figure on the model card"},
                },
                "defaultSettings": dict(_TURBO_DEFAULTS),
                "note": "Default music model. About 1 s per 10 s of audio warm; about 22 GB while generating, 15 GB idle.",
            },
            {
                "id": "ACE-Step/acestep-v15-xl-turbo",
                "familyId": "ace-step-1.5",
                "name": "ACE-Step 1.5 XL turbo (large)",
                "provider": "ACE-Step",
                "repo": "ACE-Step/acestep-v15-xl-turbo",
                "revision": "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
                "link": "https://huggingface.co/ACE-Step/acestep-v15-xl-turbo",
                "engine": "acestep",
                "engineModel": "acestep-v15-xl-turbo",
                "runtime": "ACE-Step 1.5 on MLX (isolated music venv)",
                "makes": ["songs", "instrumental"],
                "platforms": ["apple-silicon", "cuda"],
                "taskSupport": ["txt2music"],
                "sizeGb": 19.95,
                "sizeNote": "Needs the ACE-Step 1.5 turbo pack too (VAE and text encoder).",
                "requires": ["ACE-Step/Ace-Step1.5"],
                "checkpointSubdir": "acestep-v15-xl-turbo",
                "weightFolders": ["acestep-v15-xl-turbo"],
                "large": True,
                # Apple Silicon only: it idles at 40 GB and peaks at 47 GB, so smaller Macs are refused.
                "minMemoryGb": 64,
                "unloadsOthers": True,
                "maxSeconds": 600,
                "minSeconds": 10,
                "estimatedGenerationSeconds": 12.4,
                "licence": _ACESTEP_LICENCE,
                "estimates": {
                    "basis": "measured",
                    "measuredOn": "Apple M4 Max, 64 GB, 2026-11-06 (30 s and 60 s; longer tracks scaled 2x from the 2B)",
                    "wallSecondsAt": [[30, 5.4], [60, 12.4], [180, 36.2], [300, 67.0], [600, 165.0]],
                    "secondsPer10sAudio": 2.0,
                    "coldStartSeconds": 30.0,
                    "vaeSecondsPerAudioSecond": 0.08,
                    "peakGb": 47.0,
                    "idleGb": 40.2,
                    "diskGb": 19.95,
                    "outputMbPerMinute": 15.0,
                    # Supported by the model card on CUDA ("20 GB or more without offload"); not measured.
                    "cuda": {"peakGb": 20.0, "basis": "estimated", "source": "ACE-Step XL model card"},
                },
                "defaultSettings": dict(_TURBO_DEFAULTS),
                "note": "Opt-in large model for 64 GB Macs. About 2x slower than turbo; 47 GB peak, so everything else is unloaded first.",
            },
        ],
    },
]


def music_variants() -> list[dict[str, Any]]:
    """Every music variant, flat (catalog order)."""
    return [variant for family in MUSIC_MODEL_FAMILIES for variant in family["variants"]]
