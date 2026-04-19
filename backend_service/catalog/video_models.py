"""Curated catalog of video generation models we plan to support.

This module mirrors the shape of ``image_models.py`` so the frontend can reuse
the same UI patterns (families -> variants, downloads, discover tab).

Only the first-wave candidate engines live here today. The runtime is not
wired yet — see ``backend_service/routes/video.py`` for the API surface and
``VideoPlaceholderTab`` on the frontend for the current UX.
"""

from __future__ import annotations

from typing import Any


VIDEO_MODEL_FAMILIES: list[dict[str, Any]] = [
    {
        "id": "ltx-video",
        "name": "LTX-Video",
        "provider": "Lightricks",
        "headline": "Fast text-to-video model tuned for consumer hardware.",
        "summary": "First target for local video generation. Short clips (2-5s) at 768x512 with solid motion quality.",
        "updatedLabel": "Planned — first wave",
        "badges": ["Fast", "Small", "Apache 2.0"],
        "defaultVariantId": "Lightricks/LTX-Video",
        "variants": [
            {
                "id": "Lightricks/LTX-Video",
                "familyId": "ltx-video",
                "name": "LTX-Video",
                "provider": "Lightricks",
                "repo": "Lightricks/LTX-Video",
                "link": "https://huggingface.co/Lightricks/LTX-Video",
                "runtime": "diffusers LTXPipeline (planned)",
                "styleTags": ["general", "fast", "motion"],
                "taskSupport": ["txt2video"],
                "sizeGb": 2.0,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "Small, fast, Apache 2.0 — best starter pick for a local video runtime.",
                "estimatedGenerationSeconds": 45.0,
                "availableLocally": False,
            }
        ],
    },
    {
        "id": "wan-2-1",
        "name": "Wan 2.1",
        "provider": "Alibaba",
        "headline": "Smaller Wan variants — the 1.3B is the fastest starter pick for local video.",
        "summary": "Wan 2.1 ships in a 1.3B size that fits on modest hardware and a 14B size for higher quality. Both use the same WanPipeline in diffusers.",
        "updatedLabel": "Planned — first wave",
        "badges": ["Small", "Fast", "Apache 2.0"],
        "defaultVariantId": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "variants": [
            {
                "id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 1.3B",
                "provider": "Alibaba",
                # The -Diffusers mirror ships the standard diffusers layout
                # (model_index.json, scheduler/, text_encoder/, transformer/,
                # vae/, tokenizer/) — the base Wan-AI repo uses a native Wan
                # format that WanPipeline.from_pretrained can't load.
                "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "link": "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "runtime": "diffusers WanPipeline",
                "styleTags": ["general", "fast", "small"],
                "taskSupport": ["txt2video"],
                # ~16GB on disk — 1.3B is just the transformer. The repo also
                # ships a UMT5-XXL text encoder (~11GB) and VAE/CLIP weights.
                "sizeGb": 16.4,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 4.0,
                "note": "1.3B transformer + UMT5 text encoder. ~16GB on disk. Best starter pick for trying local video end-to-end on modest hardware.",
                "estimatedGenerationSeconds": 60.0,
                "availableLocally": False,
            },
            {
                "id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 14B",
                "provider": "Alibaba",
                "repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "link": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "runtime": "diffusers WanPipeline",
                "styleTags": ["general", "quality", "motion"],
                "taskSupport": ["txt2video"],
                # 14B transformer in bf16 (~28GB) + UMT5-XXL text encoder (~11GB)
                # + VAE/CLIP weights.
                "sizeGb": 45.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Wan 2.1 quality tier. ~45GB. Same WanPipeline class as the 1.3B and Wan 2.2.",
                "estimatedGenerationSeconds": 180.0,
                "availableLocally": False,
            },
        ],
    },
    {
        "id": "wan-2-2",
        "name": "Wan 2.2",
        "provider": "Alibaba",
        "headline": "Strong text-to-video quality with competitive motion consistency.",
        "summary": "Mid-sized Wan model that runs on 24GB+ VRAM or Apple Silicon with unified memory.",
        "updatedLabel": "Planned — first wave",
        "badges": ["Balanced", "Quality", "Apache 2.0"],
        "defaultVariantId": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "variants": [
            {
                "id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "familyId": "wan-2-2",
                "name": "Wan 2.2 T2V A14B",
                "provider": "Alibaba",
                # -Diffusers mirror ships the standard diffusers layout; the
                # base Wan-AI/Wan2.2-T2V-A14B repo uses the native Wan format.
                "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "link": "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "runtime": "diffusers WanPipeline",
                "styleTags": ["general", "quality", "motion"],
                "taskSupport": ["txt2video"],
                "sizeGb": 14.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Balanced quality vs size. Works on 24GB VRAM or 64GB unified memory.",
                "estimatedGenerationSeconds": 180.0,
                "availableLocally": False,
            }
        ],
    },
    {
        "id": "hunyuan-video",
        "name": "HunyuanVideo",
        "provider": "Tencent",
        "headline": "High-fidelity text-to-video with longer clips and stronger scene cohesion.",
        "summary": "Heavy-duty model that needs 40GB+ class hardware. Ships longer clips and nicer compositions.",
        "updatedLabel": "Planned — stretch target",
        "badges": ["Quality", "Heavy", "Apache 2.0"],
        "defaultVariantId": "hunyuanvideo-community/HunyuanVideo",
        "variants": [
            {
                "id": "hunyuanvideo-community/HunyuanVideo",
                "familyId": "hunyuan-video",
                "name": "HunyuanVideo",
                "provider": "Tencent",
                # Community-maintained diffusers port of tencent/HunyuanVideo.
                # The base tencent repo doesn't ship model_index.json — the
                # -community mirror is the one HunyuanVideoPipeline loads.
                "repo": "hunyuanvideo-community/HunyuanVideo",
                "link": "https://huggingface.co/hunyuanvideo-community/HunyuanVideo",
                "runtime": "diffusers HunyuanVideoPipeline",
                "styleTags": ["general", "quality", "cinematic"],
                "taskSupport": ["txt2video"],
                "sizeGb": 25.0,
                "recommendedResolution": "1280x720",
                "defaultDurationSeconds": 5.0,
                "note": "High quality. Needs 40GB+ VRAM or Apple Silicon Max/Ultra class memory.",
                "estimatedGenerationSeconds": 420.0,
                "availableLocally": False,
            }
        ],
    },
    {
        "id": "mochi-1",
        "name": "Mochi 1",
        "provider": "Genmo",
        "headline": "Open-weight video model with competitive motion quality.",
        "summary": "Apache 2.0 licence, solid motion handling, mid-sized footprint.",
        "updatedLabel": "Planned — first wave",
        "badges": ["Open", "Balanced", "Apache 2.0"],
        "defaultVariantId": "genmo/mochi-1-preview",
        "variants": [
            {
                "id": "genmo/mochi-1-preview",
                "familyId": "mochi-1",
                "name": "Mochi 1 Preview",
                "provider": "Genmo",
                "repo": "genmo/mochi-1-preview",
                "link": "https://huggingface.co/genmo/mochi-1-preview",
                "runtime": "diffusers MochiPipeline (planned)",
                "styleTags": ["general", "motion", "balanced"],
                "taskSupport": ["txt2video"],
                "sizeGb": 10.0,
                "recommendedResolution": "848x480",
                "defaultDurationSeconds": 5.4,
                "note": "Apache 2.0, balanced footprint, strong motion quality.",
                "estimatedGenerationSeconds": 150.0,
                "availableLocally": False,
            }
        ],
    },
    {
        "id": "cogvideox",
        "name": "CogVideoX",
        "provider": "THUDM",
        "headline": "Tsinghua's open-weight video model — 2B fits 8 GB VRAM, 5B is the quality tier.",
        "summary": (
            "CogVideoX ships in a 2B size that runs on 8 GB consumer GPUs and a 5B size that "
            "delivers higher fidelity on 24 GB+ cards or unified-memory Macs. Both use the same "
            "CogVideoXPipeline in diffusers."
        ),
        "updatedLabel": "Planned — first wave",
        "badges": ["Small", "Open", "Apache 2.0"],
        "defaultVariantId": "THUDM/CogVideoX-2b",
        "variants": [
            {
                "id": "THUDM/CogVideoX-2b",
                "familyId": "cogvideox",
                "name": "CogVideoX 2B",
                "provider": "THUDM",
                "repo": "THUDM/CogVideoX-2b",
                "link": "https://huggingface.co/THUDM/CogVideoX-2b",
                "runtime": "diffusers CogVideoXPipeline",
                "styleTags": ["general", "fast", "small"],
                "taskSupport": ["txt2video"],
                # 2B transformer in fp16 (~4 GB) + T5 text encoder (~5 GB) +
                # VAE. Fits comfortably on a 12 GB card; 8 GB works with
                # CPU-offload tricks. Smaller than Wan 2.1 1.3B because there's
                # no UMT5-XXL — just the standard T5.
                "sizeGb": 9.0,
                "recommendedResolution": "720x480",
                "defaultDurationSeconds": 6.0,
                "note": "Smallest CogVideoX. Apache 2.0 weights, ~9 GB on disk, runs on consumer GPUs.",
                "estimatedGenerationSeconds": 90.0,
                "availableLocally": False,
            },
            {
                "id": "THUDM/CogVideoX-5b",
                "familyId": "cogvideox",
                "name": "CogVideoX 5B",
                "provider": "THUDM",
                "repo": "THUDM/CogVideoX-5b",
                "link": "https://huggingface.co/THUDM/CogVideoX-5b",
                "runtime": "diffusers CogVideoXPipeline",
                "styleTags": ["general", "quality", "balanced"],
                "taskSupport": ["txt2video"],
                # 5B transformer (~10 GB) + T5 (~5 GB) + VAE. Lands in the
                # same envelope as Wan 2.2 — needs 24 GB VRAM or 32 GB+
                # unified memory.
                "sizeGb": 18.0,
                "recommendedResolution": "720x480",
                "defaultDurationSeconds": 6.0,
                "note": "Quality tier. ~18 GB on disk. Same CogVideoXPipeline class as the 2B.",
                "estimatedGenerationSeconds": 200.0,
                "availableLocally": False,
            },
        ],
    },
]
