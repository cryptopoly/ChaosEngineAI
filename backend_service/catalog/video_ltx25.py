"""LTX-2.5 on NVIDIA GPUs (diffusers, ``Lightricks/LTX-2.5-Diffusers``): the catalog family.

Kept apart from ``video_models.py`` (over the size cap). Two rows share the one gated repo: the 4-bit-on-load row
(``loadQuantization: "nf4"``, the runtime quantises the transformer and the Gemma 4 text encoder while it reads the
shards) and the bf16 row. The 4-bit row is measured on a 24 GB NVIDIA card with 64 GB of system memory (2026-11-13);
the bf16 row is still an estimate (it needs an 80 GB card). The Apple Silicon MLX LTX-2.5 row stays in
``video_models.py``.
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
    "recommendedResolution": "768x512",
    "defaultDurationSeconds": 4.0,
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
        "licence on Hugging Face and add a token in Settings first. The 4-bit row's memory and speed are measured; "
        "the bf16 row's are estimates. " + LICENCE_NOTE
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
            # Measured 2026-11-13 on a 24 GB NVIDIA card, 64 GB system memory, model CPU offload: 768x512, 97 frames,
            # 8 steps: GPU peak 20.1 GB with the desktop's 0.7 GB (512x320, 25 frames: 18.8 GB).
            "runtimeFootprintGb": 19.5,
            "footprintBasis": "measured",
            "footprintMeasuredOn": "cuda",
            # System memory held while loaded (the offloaded components wait there): 37.5 GB private. Assembling the
            # pipeline peaks higher for a few seconds (51 GB private, commit 69 GB against a 68 GB limit: the
            # page file grew), so a 64 GB machine with ~49 GB free loads it and a 32 GB machine does not.
            "loadMemoryGb": 38.0,
            # Warm, 768x512, 97 frames: 62.4 / 61.3 s denoise (62.2 s wall); the first load adds 107 to 124 s.
            "estimatedGenerationSeconds": 62.0,
            "estimatedGenerationSecondsBasis": "measured",
            "note": (
                "Distilled LTX-2.5 converted to 4-bit (NF4) while it loads: the transformer and the language-model "
                "layers of the Gemma 4 text encoder (embeddings, the output head and the vision and audio parts stay "
                "as they are). For NVIDIA cards from 24 GB to about 48 GB; on a card with 48 GB free or more the "
                "text encoder stays bf16. Stereo sound, 8 steps, no guidance (steps, guidance and negative prompt do "
                "not apply). Takes a start picture and an end picture; sizes are multiples of 32, clips 8k+1 frames. "
                "Needs about 71 GB of disk for the download, the GPU runtime (bitsandbytes) and about 38 GB of free "
                "system memory (51 GB for a few seconds while it loads). Measured on a 24 GB card: about 20 GB on "
                "the GPU and 62 s for a 4 s 768x512 clip once loaded; the first load takes about 2 minutes. "
                + LICENCE_NOTE
            ),
        },
        {
            **_COMMON,
            "id": LTX25_BF16_ID,
            "name": "LTX-2.5 · distilled bf16 (diffusers)",
            "styleTags": ["general", "fast", "motion", "audio", "heavy"],
            "runtimeFootprintGb": 76.0,
            "loadMemoryGb": 80.0,
            "footprintBasis": "estimated",
            "estimatedGenerationSeconds": 60.0,
            "estimatedGenerationSecondsBasis": "estimated",
            "note": (
                "Distilled LTX-2.5 at full precision: stereo sound, 8 steps, no guidance. About 76 GB in memory "
                "(estimate): for 80 GB NVIDIA cards and unified-memory machines such as the DGX Spark. Smaller "
                "cards should use the 4-bit row. Takes a start picture and an end picture. Memory and speed are "
                "not measured. " + LICENCE_NOTE
            ),
        },
    ],
}
