"""MiniMax H3 (video with stereo sound) through stable-diffusion.cpp: the catalog family (D-76, FU-166).

Territory-limited: the row carries ``territoryGate`` and stays hidden from every list, and refused by every route,
until the user confirms the territory statement (``backend_service/licence_gate.py``), and while
``licence_gate.H3_RELEASED`` is False it is hidden whatever the user does.

Files and sizes were read from the Hub file lists on 2026-11-14 (leejet/MiniMax-H3-GGUF at eee1dd2f,
Comfy-Org/MiniMax-H3 at 54472221), and the flags from ``docs/minimax_h3.md`` at sd.cpp 3f8527a
(re-read unchanged in ``docs/models/minimax_h3.md`` at 5dcfa79 and at the pin 532d2c1).
Nothing here was run: the maintainers are in an excluded territory. Memory is the sum of the weights (estimated),
and there is no time estimate.

H3-Image is not offered: no row is shipped (the pinned sd.cpp build already carries its support, upstream 32d516c).
"""

from __future__ import annotations

from typing import Any

from backend_service.licence_gate import H3_KEY, H3_LICENCE_URL, H3_NOTICE, TERRITORY_MODELS

H3_GGUF_REPO = "leejet/MiniMax-H3-GGUF"
H3_VAE_REPO = "Comfy-Org/MiniMax-H3"
H3_DIT_FILE = "minimax_h3_fl2va_pruned-Q4_K_M.gguf"  # 11,420,663,904 bytes
H3_TEXT_ENCODER_FILE = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"  # 18,218,065,024 bytes
H3_VIDEO_VAE_FILE = "vae/minimax_h3_video_vae_fp16.safetensors"  # 5,207,808,496 bytes
H3_AUDIO_VAE_FILE = "vae/minimax_h3_audio_vae_fp32.safetensors"  # 605,254,808 bytes
H3_ID = f"{H3_GGUF_REPO}~fl2va-pruned-q4km"

_SIZE_GB = round((11_420_663_904 + 18_218_065_024 + 5_207_808_496 + 605_254_808) / 1e9, 1)  # 35.5

_MODEL = TERRITORY_MODELS[H3_KEY]

H3_LICENCE: dict[str, Any] = {
    "id": "minimax-h3-community",
    "name": _MODEL["licenceName"],
    "commercial": "limited",
    "gated": False,
    "territoryRestricted": True,
    "revenueCapUsd": 20_000_000,
    "attribution": "MiniMax H3",
    "summary": "Only outside the EU, the UK, the Republic of Korea and the USA; commercial use under US$20 million a year.",
    "summaryLines": list(_MODEL["summaryLines"]),
    "note": _MODEL["note"],
    "url": H3_LICENCE_URL,
    "liveUrl": _MODEL["liveLicenceUrl"],
}

MINIMAX_H3_FAMILY: dict[str, Any] = {
    "id": "minimax-h3",
    "name": "MiniMax H3",
    "provider": "MiniMax",
    "headline": "Video with stereo sound from one model. Territory-limited licence; not tested by the maintainers.",
    "summary": (
        "MiniMax H3 makes a video and its stereo soundtrack together. This row runs the author-converted GGUF "
        "weights through stable-diffusion.cpp. The licence covers use only outside the European Union, the United "
        "Kingdom, the Republic of Korea and the United States, so the row appears only after you confirm that "
        "statement. Not tested by the maintainers: no speed or memory was measured."
    ),
    "updatedLabel": "sd.cpp · territory-limited",
    "badges": ["Sound", "Territory-limited", "Untested"],
    "defaultVariantId": H3_ID,
    "variants": [
        {
            "id": H3_ID,
            "familyId": "minimax-h3",
            "name": "MiniMax H3 · FL2VA pruned GGUF Q4_K_M (sd.cpp)",
            "provider": "MiniMax · leejet",
            "repo": H3_GGUF_REPO,
            "engine": "sdcpp",
            "link": f"https://huggingface.co/{H3_GGUF_REPO}",
            "runtime": "stable-diffusion.cpp (MiniMax H3)",
            "styleTags": ["general", "audio", "gguf", "untested"],
            "taskSupport": ["txt2video"],
            "territoryGate": H3_KEY,
            "licence": H3_LICENCE,
            # Only the files the run reads (35.5 GB), not the whole repo.
            "downloadFiles": [H3_DIT_FILE, H3_TEXT_ENCODER_FILE, "README.md"],
            "downloadCompanions": [{"repo": H3_VAE_REPO, "patterns": [H3_VIDEO_VAE_FILE, H3_AUDIO_VAE_FILE]}],
            # Flag -> (repo, file) for sd.cpp; the DiT itself goes to --diffusion-model.
            "sdcppFiles": {
                "--diffusion-model": {"repo": H3_GGUF_REPO, "file": H3_DIT_FILE},
                "--llm": {"repo": H3_GGUF_REPO, "file": H3_TEXT_ENCODER_FILE},
                "--vae": {"repo": H3_VAE_REPO, "file": H3_VIDEO_VAE_FILE},
                "--audio-vae": {"repo": H3_VAE_REPO, "file": H3_AUDIO_VAE_FILE},
            },
            "sizeGb": _SIZE_GB,
            # Weights only (estimated): the DiT, the Qwen3-VL-32B text encoder and both VAEs.
            "minMemoryGb": _SIZE_GB,
            "runtimeFootprintGb": _SIZE_GB,
            "footprintBasis": "estimated",
            # The author's example: 864x480, 56 frames (17k+5 grid), 24 fps, guidance 1 (CFG-distilled weights).
            "recommendedResolution": "864x480",
            "defaultDurationSeconds": 2.0,
            "defaultSteps": 20,  # sd.cpp's default; the author's example passes no step count
            "cfgOverride": 1.0,
            # No estimatedGenerationSeconds and no estimateRef: nothing was measured, so the strip shows no time.
            "note": (
                "Territory-limited: licensed only outside the EU, the UK, the Republic of Korea and the USA. "
                "Commercial use needs written permission from MiniMax above US$20 million a year, and a commercial "
                "product must show \"MiniMax H3\". Not tested by the maintainers: memory is the size of the "
                "weights and there is no time estimate. Needs an sd binary with MiniMax H3 support (sd.cpp 3f8527a "
                "or newer); release builds may not include one. " + H3_NOTICE
            ),
            "availableLocally": False,
            "releaseDate": "2026-08",
        },
    ],
}
