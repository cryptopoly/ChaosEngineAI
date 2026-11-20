"""Curated catalog of music models: only engines that are wired end to end.

Same family -> variants shape as the image and video catalogs, plus the fields the Music pages need
(``licence``, ``makes``, ``platforms``, ``estimates``, ``defaultSettings``). Every number in
``estimates`` comes from the ACE-Step spike on an M4 Max with 64 GB (``docs/plans/music-spike.md``,
sections 3.2, 4.2 and 5); ``estimate_run`` in ``music_runtime`` reads this table and nothing else,
so a new measurement is a change here. Weights are pinned to full Hugging Face revisions.

``platforms`` lists only where a row actually runs. The ACE-Step rows carry ``"cuda"`` since the CUDA lane was
built and measured on an RTX 4090 (FU-139); ``estimate_run`` reads ``estimates.cuda`` on that lane once it holds a
``wallSecondsAt`` table. The sound rows carry ``"cuda"`` since the CUDA sound lane was built and measured on the
same card (FU-274); the Stable Audio 3 rows list only ``"cuda"`` (not run on a Mac yet).
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


# Stable Audio 3 (read 2026-11-20 at the pinned revisions): LICENSE.md is the same Stability AI Community License
# (last updated 2024-07-05); the T5Gemma text encoder inside each repo is redistributed under the Gemma Terms of Use
# (LICENSE_GEMMA.md, NOTICE), whose section 3.2 use restrictions (Gemma Prohibited Use Policy) the gate also accepts.
def _sa3_licence(repo: str, revision: str) -> dict[str, Any]:
    licence = _stability_licence(repo, revision, "LICENSE.md")
    licence["id"] = "stability-community-gemma"
    licence["name"] = "Stability AI Community Licence + Gemma Terms of Use"
    licence["summaryLines"] = licence["summaryLines"] + [
        "Its T5Gemma text encoder is under Google's Gemma Terms of Use: you also accept the Gemma Prohibited Use "
        "Policy (section 3.2), and a copy you share must carry the Gemma notice file.",
    ]
    licence["gemmaUrl"] = f"https://huggingface.co/{repo}/blob/{revision}/LICENSE_GEMMA.md"
    return licence


# Files a Stable Audio 3 row fetches: the model, its config, licences and notice, and the T5Gemma encoder folder.
_SA3_PATTERNS = ["model.safetensors", "model_config.json", "LICENSE.md", "LICENSE_GEMMA.md", "NOTICE", "README.md",
                 "t5gemma-b-b-ul2/*"]
_SA3_DEFAULTS: dict[str, Any] = {
    # Model card: 8 steps, cfg 1.0, the pingpong sampler (the models are adversarially post-trained for few steps).
    "variations": 1,
    "steps": 8,
    "cfgScale": 1.0,
    "sampler": "pingpong",
    "loop": False,
    "normalizePeakDb": -1.0,
    "format": "flac24",
    "sampleRate": 44100,
    "channels": 2,
}
_SA3_STEPS = {"min": 4, "max": 32, "presets": [4, 8, 16]}


def _sa3_estimates(cuda: dict[str, Any], disk_gb: float, mb_per_minute: float) -> dict[str, Any]:
    """Stable Audio 3 has only been measured on CUDA (FU-274), so the row's own figures are the CUDA ones.

    Measured through the app's routes on Windows 11 with an RTX 4090, warm worker, float16, 8 steps, one variation.
    ``wallSecondsAt`` is the client's wall time per request; ``peakGb`` is torch's reserved VRAM peak (the CUDA context
    adds about 0.4 GB in nvidia-smi); ``hostPeakGb`` is the worker's working set, which the memory gate checks on CUDA.
    The work grows with the clip asked for (variable length), so there is no ``fixedWindow``.
    """
    cuda = {"basis": "measured", "measuredOn": "NVIDIA RTX 4090 24 GB, Windows 11, 2026-11-20", **cuda}
    return {**cuda, "diskGb": disk_gb, "outputMbPerMinute": mb_per_minute, "cuda": dict(cuda)}

MUSIC_MODEL_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "ace-step-1.5",
        "name": "ACE-Step 1.5",
        "provider": "ACE Studio and StepFun",
        "headline": "Songs with vocals and lyrics, or instrumentals, 10 s to 10 minutes.",
        "summary": "Text to music with BPM, key and seed control. Runs on MLX on Apple Silicon and on PyTorch on NVIDIA GPUs.",
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
                "runtime": "ACE-Step 1.5 on MLX or CUDA (isolated music venv)",
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
                    # FU-139: measured through the app's routes on Windows 11, warm worker, bf16, no offload, one
                    # variation, 8 steps. peakGb is torch's reserved VRAM peak (allocated 7.6 to 7.9 GB); hostPeakGb
                    # is the worker's working set, which the memory gate checks on CUDA instead of the Mac figure.
                    "cuda": {
                        "basis": "measured",
                        "measuredOn": "NVIDIA RTX 4090 24 GB, Windows 11, 2026-11-19",
                        "wallSecondsAt": [[30, 0.97], [60, 1.32], [180, 3.1], [600, 10.6]],
                        "secondsPer10sAudio": 0.2,
                        "coldStartSeconds": 15.0,
                        "vaeSecondsPerAudioSecond": 0.007,
                        "peakGb": 8.6,
                        "idleGb": 7.9,
                        "hostPeakGb": 2.7,
                        "hostIdleGb": 1.0,
                    },
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
                "runtime": "ACE-Step 1.5 on MLX or CUDA (isolated music venv)",
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
                    # FU-139: measured on the 4090 like the turbo row (bf16, no offload). The card says "20 GB or
                    # more without offload"; the run held 12.1 GB allocated, 12.4 GB reserved (14.3 GB at 600 s).
                    # With another process holding 12 GB of the card, ACE-Step's own pre-flight refused every run.
                    "cuda": {
                        "basis": "measured",
                        "measuredOn": "NVIDIA RTX 4090 24 GB, Windows 11, 2026-11-19",
                        "wallSecondsAt": [[30, 1.12], [60, 1.64], [180, 4.1], [600, 15.8]],
                        "secondsPer10sAudio": 0.27,
                        "coldStartSeconds": 20.0,
                        "vaeSecondsPerAudioSecond": 0.008,
                        "peakGb": 14.3,
                        "idleGb": 12.1,
                        "hostPeakGb": 2.7,
                        "hostIdleGb": 1.0,
                    },
                },
                "defaultSettings": dict(_TURBO_DEFAULTS),
                "note": "Opt-in large model for 64 GB Macs (47 GB peak, everything else is unloaded first) or an NVIDIA GPU with about "
                        "15 GB of free VRAM (measured on a 24 GB RTX 4090: 60 s in 1.6 s).",
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
                "platforms": ["apple-silicon", "cuda"],
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
                    # FU-274: measured through the app's routes on Windows 11, warm worker, float32, 8 steps, one variation.
                    # peakGb is torch's reserved VRAM peak; hostPeakGb the worker's working set (the CUDA memory gate).
                    "cuda": {
                        "basis": "measured",
                        "measuredOn": "NVIDIA RTX 4090 24 GB, Windows 11, 2026-11-20",
                        "wallSecondsAt": [[5, 0.27], [10, 0.28]],
                        "secondsPer10sAudio": 0.28,
                        "coldStartSeconds": 10.0,
                        "peakGb": 3.6,
                        "idleGb": 2.8,
                        "hostPeakGb": 2.5,
                        "hostIdleGb": 1.7,
                    },
                },
                # Steps the person may choose (D-74): presets plus a free number inside min..max. The default is
                # defaultSettings.steps, which is also the step count the estimate was measured at.
                "stepsOptions": {"min": 1, "max": 32, "presets": [4, 8, 16]},
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
                "platforms": ["apple-silicon", "cuda"],
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
                    # FU-274: measured through the app's routes on Windows 11, warm worker, float32, 100 steps, cfg 7,
                    # dpmpp-3m-sde; still the full 47 s window whatever the length. peakGb is torch's reserved VRAM peak
                    # (allocated 10.4 GB), hostPeakGb the worker's working set.
                    "cuda": {
                        "basis": "measured",
                        "measuredOn": "NVIDIA RTX 4090 24 GB, Windows 11, 2026-11-20",
                        "wallSecondsAt": [[5, 15.56], [10, 15.6]],
                        "secondsPer10sAudio": None,
                        "coldStartSeconds": 12.0,
                        "peakGb": 13.8,
                        "hostPeakGb": 2.6,
                        "hostIdleGb": 1.7,
                    },
                },
                # Model card example for stable-audio-tools: 100 steps, cfg 7, dpmpp-3m-sde, sigma 0.3 to 500.
                "stepsOptions": {"min": 10, "max": 150, "presets": [25, 50, 100]},
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
    {
        "id": "stable-audio-3",
        "name": "Stable Audio 3",
        "kind": "sound",
        "provider": "Stability AI",
        "headline": "Sound effects, loops and instrumentals of variable length; no vocals.",
        "summary": "Text to audio from Stability AI: Small SFX for effects, Small Music for instrumentals, Medium for both. "
                   "Runs on PyTorch on an NVIDIA GPU in the sounds pack.",
        "badges": ["Gated", "Sounds", "Instrumental"],
        "defaultVariantId": "stabilityai/stable-audio-3-small-sfx",
        "variants": [
            {
                "id": "stabilityai/stable-audio-3-small-sfx",
                "familyId": "stable-audio-3",
                "kind": "sound",
                "pack": "sound",
                "name": "Stable Audio 3 Small SFX",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-audio-3-small-sfx",
                # Head of the repo on 2026-11-20 (last modified 2026-10-06).
                "revision": "05ca3857881c94a2b84026b1111c537d78f5f9f0",
                "link": "https://huggingface.co/stabilityai/stable-audio-3-small-sfx",
                "engine": "stableaudio",
                "engineLibrary": "stable-audio-3",
                "engineModel": "stable-audio-3-small-sfx",
                "runtime": "stable-audio-3 on PyTorch (isolated sounds environment)",
                "makes": ["sfx", "loops"],
                "platforms": ["cuda"],
                "taskSupport": ["txt2sound"],
                "sizeGb": 3.49,
                "sizeNote": "The 2.27 GB model (0.6B diffusion transformer and its autoencoder) plus its 1.22 GB T5Gemma "
                            "text encoder, both from the same repo.",
                "requires": [],
                "checkpointSubdir": "stable-audio-3-small-sfx",
                "weightFolders": ["stable-audio-3-small-sfx", "stable-audio-3-small-sfx/t5gemma-b-b-ul2"],
                "downloadPatterns": list(_SA3_PATTERNS),
                "large": False,
                "minMemoryGb": None,
                "unloadsOthers": False,
                # The config's window is 5,292,032 samples (120 s at 44.1 kHz); the work scales with the clip asked for.
                "maxSeconds": 120,
                "minSeconds": 1,
                "estimatedGenerationSeconds": 0.41,
                "licence": _sa3_licence("stabilityai/stable-audio-3-small-sfx", "05ca3857881c94a2b84026b1111c537d78f5f9f0"),
                "estimates": _sa3_estimates({
                    "wallSecondsAt": [[5, 0.41], [10, 0.43], [120, 0.87]], "secondsPer10sAudio": 0.07, "coldStartSeconds": 11.0,
                    "vaeSecondsPerAudioSecond": 0.002, "peakGb": 2.0, "idleGb": 1.8, "hostPeakGb": 2.2, "hostIdleGb": 1.6,
                }, disk_gb=3.49, mb_per_minute=8.0),
                "stepsOptions": dict(_SA3_STEPS),
                "defaultSettings": {**_SA3_DEFAULTS, "seconds": 5},
                "note": "Opt-in sound effects model, 1 to 120 s, stereo 44.1 kHz, English prompts. Gated: accept the licence "
                        "(and the Gemma terms of its text encoder) on Hugging Face.",
            },
            {
                "id": "stabilityai/stable-audio-3-small-music",
                "familyId": "stable-audio-3",
                "kind": "sound",
                "pack": "sound",
                "name": "Stable Audio 3 Small Music",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-audio-3-small-music",
                "revision": "8bdfb53c6b0fbc62113cd571fd37373efe15fa14",
                "link": "https://huggingface.co/stabilityai/stable-audio-3-small-music",
                "engine": "stableaudio",
                "engineLibrary": "stable-audio-3",
                "engineModel": "stable-audio-3-small-music",
                "runtime": "stable-audio-3 on PyTorch (isolated sounds environment)",
                "makes": ["instrumental", "loops"],
                "platforms": ["cuda"],
                "taskSupport": ["txt2sound"],
                "sizeGb": 3.49,
                "sizeNote": "The 2.27 GB model (0.6B diffusion transformer and its autoencoder) plus its 1.22 GB T5Gemma "
                            "text encoder, both from the same repo.",
                "requires": [],
                "checkpointSubdir": "stable-audio-3-small-music",
                "weightFolders": ["stable-audio-3-small-music", "stable-audio-3-small-music/t5gemma-b-b-ul2"],
                "downloadPatterns": list(_SA3_PATTERNS),
                "large": False,
                "minMemoryGb": None,
                "unloadsOthers": False,
                "maxSeconds": 120,
                "minSeconds": 1,
                "estimatedGenerationSeconds": 0.51,
                "licence": _sa3_licence("stabilityai/stable-audio-3-small-music", "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"),
                "estimates": _sa3_estimates({
                    "wallSecondsAt": [[30, 0.51], [60, 0.63], [120, 0.87]], "secondsPer10sAudio": 0.07, "coldStartSeconds": 10.0,
                    "vaeSecondsPerAudioSecond": 0.002, "peakGb": 2.0, "idleGb": 1.8, "hostPeakGb": 2.2, "hostIdleGb": 1.6,
                }, disk_gb=3.49, mb_per_minute=12.5),
                "stepsOptions": dict(_SA3_STEPS),
                "defaultSettings": {**_SA3_DEFAULTS, "seconds": 30},
                "note": "Opt-in instrumental music model (no vocals), 1 to 120 s, stereo 44.1 kHz; put the tempo in the prompt "
                        "(for example 120 BPM). Gated: accept the licence and the Gemma terms on Hugging Face.",
            },
            {
                "id": "stabilityai/stable-audio-3-medium",
                "familyId": "stable-audio-3",
                "kind": "sound",
                "pack": "sound",
                "name": "Stable Audio 3 Medium",
                "provider": "Stability AI",
                "repo": "stabilityai/stable-audio-3-medium",
                "revision": "d4d3579395a8ac0df4ab13b5b1d7138ce7290f81",
                "link": "https://huggingface.co/stabilityai/stable-audio-3-medium",
                "engine": "stableaudio",
                "engineLibrary": "stable-audio-3",
                "engineModel": "stable-audio-3-medium",
                "runtime": "stable-audio-3 on PyTorch (isolated sounds environment)",
                "makes": ["sfx", "instrumental", "loops"],
                "platforms": ["cuda"],
                "taskSupport": ["txt2sound"],
                "sizeGb": 10.44,
                "sizeNote": "The 9.22 GB model (2B diffusion transformer and its larger autoencoder) plus its 1.22 GB T5Gemma "
                            "text encoder, both from the same repo.",
                "requires": [],
                "checkpointSubdir": "stable-audio-3-medium",
                "weightFolders": ["stable-audio-3-medium", "stable-audio-3-medium/t5gemma-b-b-ul2"],
                "downloadPatterns": list(_SA3_PATTERNS),
                "large": False,
                "minMemoryGb": None,
                "unloadsOthers": False,
                # The config's window is 16,777,216 samples (380 s at 44.1 kHz).
                "maxSeconds": 380,
                "minSeconds": 1,
                "estimatedGenerationSeconds": 1.75,
                "licence": _sa3_licence("stabilityai/stable-audio-3-medium", "d4d3579395a8ac0df4ab13b5b1d7138ce7290f81"),
                # The first run after a load takes about 9 s more (kernel warm-up); coldStartSeconds counts it.
                "estimates": _sa3_estimates({
                    "wallSecondsAt": [[5, 0.76], [10, 1.11], [30, 1.75], [60, 2.97], [180, 7.4], [380, 15.35]],
                    "secondsPer10sAudio": 0.4, "coldStartSeconds": 26.0, "vaeSecondsPerAudioSecond": 0.005,
                    "peakGb": 8.4, "idleGb": 6.7, "hostPeakGb": 2.5, "hostIdleGb": 1.6,
                }, disk_gb=10.44, mb_per_minute=11.5),
                "stepsOptions": dict(_SA3_STEPS),
                "defaultSettings": {**_SA3_DEFAULTS, "seconds": 30},
                "note": "Opt-in larger model for effects and instrumentals (no vocals), 1 to 380 s, stereo 44.1 kHz. "
                        "Gated: accept the licence and the Gemma terms on Hugging Face.",
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
