"""LTX-2.5 on NVIDIA GPUs (diffusers, ``Lightricks/LTX-2.5-Diffusers``): the catalog family.

Kept apart from ``video_models.py`` (over the size cap). Two rows share the one gated repo: the 4-bit-on-load row
(``loadQuantization: "nf4"``, the runtime quantises the transformer and the Gemma 4 text encoder while it reads the
shards) and the bf16 row. Every figure here is an estimate: nothing has been downloaded or run (the repo is gated and
72 GB), so the rows say so. The Apple Silicon MLX LTX-2.5 row stays in ``video_models.py``.
"""

from __future__ import annotations

from typing import Any

LTX25_REPO = "Lightricks/LTX-2.5-Diffusers"
LTX25_NF4_ID = f"{LTX25_REPO}~nf4"
LTX25_BF16_ID = LTX25_REPO

# The distilled pipeline only: no transformer_full (dev), no prompt_enhancer (passed as None), no distilled LoRA.
# Component index files are listed so the shard maps can be read; shard globs are pinned to the shard counts
# that were read from the repo's own indexes (transformer 4, connectors 2; the other shard sets in the listing are stale).
LTX25_DOWNLOAD_FILES: list[str] = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "processor/*",
    "duration_head/*",
    "transformer/config.json",
    "transformer/*.index.json",
    "transformer/*-of-00004.safetensors",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/*.index.json",
    "text_encoder/*.safetensors",
    "connectors/config.json",
    "connectors/*.index.json",
    "connectors/*-of-00002.safetensors",
    "vae/*",
    "audio_vae/*",
    "vocoder/*",
    "latent_upsampler/*",
    "LICENSE*",
]

LICENCE_NOTE = (
    "Weights: LTX-2.x Community License. Free commercial use while the entity (counting affiliates) has under "
    "US$10M annual revenue; above that a paid Commercial Use Agreement with Lightricks; transferring fine-tunes may "
    "need a paid licence. The repo is gated: accept the terms on its Hugging Face page, then add a token with read "
    "access in Settings."
)

_COMMON: dict[str, Any] = {
    "familyId": "ltx-2-5-diffusers",
    "provider": "Lightricks",
    "repo": LTX25_REPO,
    "link": f"https://huggingface.co/{LTX25_REPO}",
    "runtime": "diffusers LTX2Pipeline",
    "taskSupport": ["txt2video", "img2video"],
    "cudaOnly": True,
    "gated": True,
    "gatedLicenceUrl": f"https://huggingface.co/{LTX25_REPO}",
    "sizeGb": 71.1,
    "footprintBasis": "estimated",
    "recommendedResolution": "768x512",
    "defaultDurationSeconds": 4.0,
    "estimatedGenerationSecondsBasis": "estimated",
    "downloadFiles": LTX25_DOWNLOAD_FILES,
    "availableLocally": False,
    "releaseDate": "2026-08",
}

LTX25_NVIDIA_FAMILY: dict[str, Any] = {
    "id": "ltx-2-5-diffusers",
    "name": "LTX-2.5 (NVIDIA)",
    "provider": "Lightricks",
    "headline": "LTX-2.5 with sound on NVIDIA GPUs: 4-bit on load for 24 GB cards, bf16 for 80 GB cards.",
    "summary": (
        "LTX-2.5 distilled: video and stereo sound made together, text to video or from a start and an end picture, "
        "8 steps and no guidance. The official weights are bf16 only (a 38 GB transformer and a 24 GB Gemma 4 text "
        "encoder, 71 GB to download), so the 4-bit row converts both to 4-bit while it loads and runs on 24 GB "
        "cards with offloading; the bf16 row is for 80 GB cards and unified-memory machines. Gated: accept the "
        "licence on Hugging Face and add a token in Settings first. Memory and speed figures are estimates, not "
        "yet measured. " + LICENCE_NOTE
    ),
    "updatedLabel": "Diffusers · NVIDIA · gated",
    "badges": ["Sound", "NVIDIA", "Gated", "Quantized"],
    "defaultVariantId": LTX25_NF4_ID,
    "variants": [
        {
            **_COMMON,
            "id": LTX25_NF4_ID,
            "name": "LTX-2.5 · distilled 4-bit on load (diffusers)",
            "styleTags": ["general", "fast", "motion", "audio", "quantized"],
            "loadQuantization": "nf4",
            # Estimated from the LTX-2.3 4-bit pack (14.4 GB transformer + 8.4 GB Gemma 3 encoder, peak 22.3 GB
            # measured on a 24 GB card); Gemma 4 12B is a different encoder, so this is not measured.
            "runtimeFootprintGb": 22.5,
            # Shards are read bf16 and quantised as they load, one at a time (10 GB at most) without the memory map.
            "loadMemoryGb": 26.0,
            "estimatedGenerationSeconds": 70.0,
            "note": (
                "Distilled LTX-2.5 converted to 4-bit (NF4) while it loads: the transformer and the language-model "
                "layers of the Gemma 4 text encoder (embeddings, the output head and the vision and audio parts stay "
                "as they are). For NVIDIA cards from 24 GB to about 48 GB; on a card with 48 GB free or more the "
                "text encoder stays bf16. Stereo sound, 8 steps, no guidance (steps, guidance and negative prompt do "
                "not apply). Takes a start picture and an end picture; sizes are multiples of 32, clips 8k+1 frames. "
                "Needs about 71 GB of disk for the download and the GPU runtime (bitsandbytes). Memory and speed are "
                "estimates from the LTX-2.3 4-bit pack: not measured on LTX-2.5. " + LICENCE_NOTE
            ),
        },
        {
            **_COMMON,
            "id": LTX25_BF16_ID,
            "name": "LTX-2.5 · distilled bf16 (diffusers)",
            "styleTags": ["general", "fast", "motion", "audio", "heavy"],
            "runtimeFootprintGb": 76.0,
            "loadMemoryGb": 80.0,
            "estimatedGenerationSeconds": 60.0,
            "note": (
                "Distilled LTX-2.5 at full precision: stereo sound, 8 steps, no guidance. About 76 GB in memory "
                "(estimate): for 80 GB NVIDIA cards and unified-memory machines such as the DGX Spark. Smaller "
                "cards should use the 4-bit row. Takes a start picture and an end picture. Memory and speed are "
                "not measured. " + LICENCE_NOTE
            ),
        },
    ],
}
