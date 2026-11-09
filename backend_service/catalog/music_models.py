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

# Stability AI Community Licence (last updated 2024-07-05), read on 2026-11-07 from the LICENSE file of
# stabilityai/stable-audio-open-small at the pinned revision. Plain-language lines, not legal advice.
# Stable Audio Open 1.0 uses the same agreement (its model card says "Stability AI Community License");
# the licence file of that gated repo could not be read without accepting it, so its lines are the Small
# row's, checked against the public text at stability.ai/community-license-agreement (same date).
def _stability_licence(repo: str, revision: str, filename: str) -> dict[str, Any]:
    return {
        "id": "stability-community",
        "name": "Stability AI Community Licence",
        "commercial": "limited",
        "gated": True,
        "attribution": "Powered by Stability AI",
        "revenueCapUsd": 1_000_000,
        "summaryLines": [
            "Free for research, personal use and commercial use while you and your affiliates earn under US$1 million a "
            "year. Over that, the licence ends and you need one from Stability AI.",
            "Commercial use means registering with Stability AI (stability.ai/community-license).",
            "If you share the model or a product built on it, you must show \"Powered by Stability AI\" and keep its notice file.",
            "You own the sounds you make, but may not use the model or its output to train another foundation model.",
            "Use must follow Stability's acceptable use policy. The model is gated: you accept the terms on Hugging Face first.",
        ],
        "note": "The weights are never bundled with ChaosEngineAI. You download them from Hugging Face under your own account.",
        "url": f"https://huggingface.co/{repo}/blob/{revision}/{filename}",
    }


_STABILITY_LICENCE = _stability_licence(
    "stabilityai/stable-audio-open-small", "ea2772170fc06f80dfd02bea6e57a44d9d4bc9d3", "LICENSE")
_STABILITY_LICENCE_1_0 = _stability_licence(
    "stabilityai/stable-audio-open-1.0", "f21265c1e2710b3bd2386596943f0007f55f802e", "LICENSE.md")

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
                    # FLAC 24-bit 48 kHz stereo as the worker writes it: 8.9 MB/min (lo-fi) to 11.5
                    # (orchestral) measured in the P1 real check, 2026-11-07; the higher figure is used.
                    "outputMbPerMinute": 11.5,
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
                    "outputMbPerMinute": 11.5,
                    # Supported by the model card on CUDA ("20 GB or more without offload"); not measured.
                    "cuda": {"peakGb": 20.0, "basis": "estimated", "source": "ACE-Step XL model card"},
                },
                "defaultSettings": dict(_TURBO_DEFAULTS),
                "note": "Opt-in large model for 64 GB Macs. About 2x slower than turbo; 47 GB peak, so everything else is unloaded first.",
            },
        ],
    },
    {
        "id": "stable-audio-open",
        "name": "Stable Audio Open",
        "kind": "sound",
        "provider": "Stability AI",
        "headline": "Sound effects and loops: Small up to 11 s, the full 1.0 model up to 47 s. Not songs, not vocals.",
        "summary": "Text to sound: footsteps, doors, ambience, drum loops. Runs on PyTorch (Apple GPU, else CPU) in its own pack.",
        "badges": ["Gated", "Sounds"],
        "defaultVariantId": "stabilityai/stable-audio-open-small",
        "variants": [
            {
                "id": "stabilityai/stable-audio-open-small",
                "familyId": "stable-audio-open",
                "kind": "sound",
                "pack": "sound",
                "name": "Stable Audio Open Small",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-audio-open-small",
                "revision": "ea2772170fc06f80dfd02bea6e57a44d9d4bc9d3",
                "link": "https://huggingface.co/stabilityai/stable-audio-open-small",
                "engine": "stableaudio",
                "engineModel": "stable-audio-open-small",
                "runtime": "stable-audio-tools on PyTorch (isolated sounds environment)",
                "makes": ["sfx", "loops"],
                "platforms": ["apple-silicon"],
                "taskSupport": ["txt2sound"],
                "sizeGb": 2.57,
                "sizeNote": "The 1.7 GB model plus the 0.9 GB T5-base text encoder (Apache 2.0) it reads prompts with.",
                "requires": [],
                "checkpointSubdir": "stable-audio-open-small",
                "weightFolders": ["stable-audio-open-small"],
                # Only these files are fetched (the repo also holds a second 1.7 GB checkpoint copy,
                # the base model and 1.4 GB of mobile builds we do not use).
                "downloadPatterns": ["model.safetensors", "model_config.json", "LICENSE", "README.md"],
                "downloadCompanions": [
                    {"repo": "google-t5/t5-base",
                     "patterns": ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"]},
                ],
                "textEncoder": {
                    "repo": "google-t5/t5-base",
                    "revision": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
                    "checkpointSubdir": "t5-base",
                    "files": ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"],
                },
                "large": False,
                "minMemoryGb": None,
                "unloadsOthers": False,
                "maxSeconds": 11,
                "minSeconds": 1,
                "estimatedGenerationSeconds": 6.0,
                "licence": _STABILITY_LICENCE,
                "estimates": {
                    # Measured 2026-11-07 (docs/research/sound-effects-real-test.md): warm worker, 8 steps, MPS,
                    # one variation. peakGb is the footprint the worker reported while generating.
                    "basis": "measured",
                    "measuredOn": "Apple M4 Max, 64 GB, 2026-11-07",
                    "wallSecondsAt": [[3, 0.7], [6, 0.7], [10, 0.7]],
                    # The model works on its whole window whatever length is asked, so the time does not scale down.
                    "fixedWindow": True,
                    "secondsPer10sAudio": 0.7,
                    "coldStartSeconds": 8.0,
                    "vaeSecondsPerAudioSecond": 0.1,
                    "peakGb": 5.0,
                    "idleGb": 3.8,
                    "diskGb": 2.57,
                    # 44.1 kHz stereo FLAC 24-bit, noisy material.
                    "outputMbPerMinute": 14.0,
                    "cuda": {"peakGb": None, "basis": "estimated", "source": "not measured; the CUDA sound lane is not built yet"},
                },
                "defaultSettings": {
                    "seconds": 5,
                    "variations": 1,
                    "steps": 8,
                    "cfgScale": 1.0,
                    "sampler": "pingpong",
                    "loop": False,
                    "normalizePeakDb": -1.0,
                    "format": "flac24",
                    "sampleRate": 44100,
                    "channels": 2,
                },
                "note": "Opt-in sound effects model. English prompts, 1 to 11 s, stereo 44.1 kHz. Gated: accept the licence on Hugging Face.",
            },
            {
                "id": "stabilityai/stable-audio-open-1.0",
                "familyId": "stable-audio-open",
                "kind": "sound",
                "pack": "sound",
                "name": "Stable Audio Open 1.0",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-audio-open-1.0",
                # Head of the repo on 2026-11-07 (last modified 2025-06-19).
                "revision": "f21265c1e2710b3bd2386596943f0007f55f802e",
                "link": "https://huggingface.co/stabilityai/stable-audio-open-1.0",
                "engine": "stableaudio",
                "engineModel": "stable-audio-open-1.0",
                "runtime": "stable-audio-tools on PyTorch (isolated sounds environment)",
                "makes": ["sfx", "loops"],
                "platforms": ["apple-silicon"],
                "taskSupport": ["txt2sound"],
                "sizeGb": 5.72,
                "sizeNote": "The 4.85 GB model plus the 0.9 GB T5-base text encoder (Apache 2.0) it reads prompts with; "
                            "the encoder is shared with Stable Audio Open Small, so it is fetched once.",
                "requires": [],
                "checkpointSubdir": "stable-audio-open-1.0",
                "weightFolders": ["stable-audio-open-1.0"],
                # model.safetensors is the stable-audio-tools checkpoint (diffusion model + autoencoder). The repo also
                # holds the same weights as model.ckpt (4.85 GB), a diffusers copy (5.5 GB), a separate VAE
                # checkpoint and 52 MB of dataset attribution CSVs; none of those are fetched.
                "downloadPatterns": ["model.safetensors", "model_config.json", "LICENSE.md", "README.md"],
                "downloadCompanions": [
                    {"repo": "google-t5/t5-base",
                     "patterns": ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"]},
                ],
                "textEncoder": {
                    "repo": "google-t5/t5-base",
                    "revision": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
                    "checkpointSubdir": "t5-base",
                    "files": ["config.json", "model.safetensors", "spiece.model", "tokenizer.json"],
                },
                "large": False,
                "minMemoryGb": None,
                "unloadsOthers": False,
                # Model card: "variable-length (up to 47s) stereo audio at 44.1kHz". The config's sample_size
                # is fixed, so every clip costs the same to make whatever length is asked for.
                "maxSeconds": 47,
                "minSeconds": 1,
                "estimatedGenerationSeconds": 208.0,
                "licence": _STABILITY_LICENCE_1_0,
                "estimates": {
                    # Measured 2026-11-08 with scripts/real-test-sound-1.0.py: 100 steps, cfg 7, dpmpp-3m-sde. The model
                    # always works on its full 47 s window, so a 20 s and a 45 s clip took the same time (about 208 s).
                    "basis": "measured",
                    "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
                    "wallSecondsAt": [[20, 207.7], [45, 208.0]],
                    "fixedWindow": True,
                    "secondsPer10sAudio": None,
                    "coldStartSeconds": 7.0,
                    "vaeSecondsPerAudioSecond": 0.1,
                    "peakGb": 22.8,
                    "idleGb": 7.1,
                    "diskGb": 5.72,
                    "outputMbPerMinute": 14.0,
                    "cuda": {"peakGb": None, "basis": "estimated", "source": "not measured; the CUDA sound lane is not built yet"},
                },
                # Model card example for stable-audio-tools: 100 steps, cfg 7, dpmpp-3m-sde, sigma 0.3 to 500.
                "defaultSettings": {
                    "seconds": 10,
                    "variations": 1,
                    "steps": 100,
                    "cfgScale": 7.0,
                    "sampler": "dpmpp-3m-sde",
                    "loop": False,
                    "normalizePeakDb": -1.0,
                    "format": "flac24",
                    "sampleRate": 44100,
                    "channels": 2,
                },
                "note": "Opt-in, larger and slower sound effects model: clips up to 47 s with better detail, about 5 GB on disk "
                        "and about 3.5 minutes per clip, using about 23 GB of memory while it works. English prompts, stereo 44.1 kHz. Gated: accept the licence on Hugging Face.",
            },
        ],
    },
]


def music_variants() -> list[dict[str, Any]]:
    """Every music variant, flat (catalog order)."""
    return [variant for family in MUSIC_MODEL_FAMILIES for variant in family["variants"]]


def sound_max_seconds() -> int:
    """The longest clip any sound row makes (the widest request limit; each row's own ``maxSeconds`` is the real one)."""
    return max(int(v["maxSeconds"]) for v in music_variants() if v.get("kind") == "sound")
