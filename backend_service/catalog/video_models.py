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
                "releaseDate": "2024-11",
            },
            {
                "id": "Lightricks/LTX-Video-gguf-q4km",
                "familyId": "ltx-video",
                "name": "LTX-Video · GGUF Q4_K_M",
                "provider": "Lightricks · city96",
                "repo": "Lightricks/LTX-Video",
                "ggufRepo": "city96/LTX-Video-gguf",
                "ggufFile": "ltx-video-2b-v0.9-Q4_K_M.gguf",
                "link": "https://huggingface.co/city96/LTX-Video-gguf",
                "runtime": "diffusers LTXPipeline + GGUF transformer",
                "styleTags": ["general", "fast", "motion", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 1.4,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "GGUF Q4_K_M — runs on 6-8 GB VRAM / Apple Silicon at near-native quality.",
                "estimatedGenerationSeconds": 50.0,
                "availableLocally": False,
                "releaseDate": "2024-12",
            },
            {
                "id": "Lightricks/LTX-Video-gguf-q6k",
                "familyId": "ltx-video",
                "name": "LTX-Video · GGUF Q6_K",
                "provider": "Lightricks · city96",
                "repo": "Lightricks/LTX-Video",
                "ggufRepo": "city96/LTX-Video-gguf",
                "ggufFile": "ltx-video-2b-v0.9-Q6_K.gguf",
                "link": "https://huggingface.co/city96/LTX-Video-gguf",
                "runtime": "diffusers LTXPipeline + GGUF transformer",
                "styleTags": ["general", "motion", "quality", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 1.7,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "GGUF Q6_K — mid-point between Q4 footprint and Q8 fidelity.",
                "estimatedGenerationSeconds": 48.0,
                "availableLocally": False,
                "releaseDate": "2024-12",
            },
            {
                "id": "Lightricks/LTX-Video-gguf-q8",
                "familyId": "ltx-video",
                "name": "LTX-Video · GGUF Q8_0",
                "provider": "Lightricks · city96",
                "repo": "Lightricks/LTX-Video",
                "ggufRepo": "city96/LTX-Video-gguf",
                "ggufFile": "ltx-video-2b-v0.9-Q8_0.gguf",
                "link": "https://huggingface.co/city96/LTX-Video-gguf",
                "runtime": "diffusers LTXPipeline + GGUF transformer",
                "styleTags": ["general", "motion", "quality", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 2.2,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "GGUF Q8_0 — near-bf16 quality at roughly half the memory.",
                "estimatedGenerationSeconds": 46.0,
                "availableLocally": False,
                "releaseDate": "2024-12",
            },
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
                "releaseDate": "2025-02",
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
                "releaseDate": "2025-02",
            },
        ],
    },
    {
        "id": "wan-2-2",
        "name": "Wan 2.2",
        "provider": "Alibaba",
        "headline": "Wan 2.2 ships a dense 5B that fits consumer GPUs and an MoE A14B quality tier.",
        "summary": (
            "Two very different models under one family name. The TI2V-5B is a dense 5B dual-"
            "task model (text-to-video + image-to-video) that runs on a 24 GB GPU or a 32 GB+ "
            "Mac. The T2V-A14B is a mixture-of-experts model with separate high-noise and low-"
            "noise transformers — roughly 27B total weights on disk and only viable on data-"
            "center-class hardware."
        ),
        "updatedLabel": "Planned — first wave",
        "badges": ["Balanced", "Quality", "Apache 2.0"],
        "defaultVariantId": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "variants": [
            {
                "id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "familyId": "wan-2-2",
                "name": "Wan 2.2 TI2V 5B",
                "provider": "Alibaba",
                # Dense 5B text+image-to-video model. Unlike A14B there's no
                # expert split — standard WanPipeline loads it directly.
                "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "link": "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "runtime": "diffusers WanPipeline",
                "styleTags": ["general", "balanced", "small"],
                "taskSupport": ["txt2video"],
                # 5B transformer bf16 (~10 GB) + UMT5-XXL text encoder (~11 GB)
                # + VAE. Sits right around 24 GB on disk which is a similar
                # footprint to Wan 2.1 1.3B but with a much larger transformer.
                "sizeGb": 24.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Best Wan 2.2 pick for consumer hardware. 24 GB on disk, runs on a 24 GB GPU or a 32 GB+ Mac.",
                "estimatedGenerationSeconds": 150.0,
                "availableLocally": False,
                "releaseDate": "2025-07",
            },
            {
                "id": "QuantStack/Wan2.2-TI2V-5B-GGUF-q4km",
                "familyId": "wan-2-2",
                "name": "Wan 2.2 TI2V 5B · GGUF Q4_K_M",
                "provider": "Alibaba · QuantStack",
                "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "ggufRepo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
                "ggufFile": "Wan2.2-TI2V-5B-Q4_K_M.gguf",
                "link": "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "balanced", "small", "gguf"],
                "taskSupport": ["txt2video"],
                # ~3.5 GB quantized transformer + shared UMT5-XXL / VAE from
                # the base repo. Users installing this variant still pay the
                # ~14 GB text-encoder+VAE download once.
                "sizeGb": 17.5,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Q4_K_M — smallest Wan 2.2 that still generates usable quality. Best fit for 16 GB unified memory.",
                "estimatedGenerationSeconds": 160.0,
                "availableLocally": False,
                "releaseDate": "2025-08",
            },
            {
                "id": "QuantStack/Wan2.2-TI2V-5B-GGUF-q6k",
                "familyId": "wan-2-2",
                "name": "Wan 2.2 TI2V 5B · GGUF Q6_K",
                "provider": "Alibaba · QuantStack",
                "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "ggufRepo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
                "ggufFile": "Wan2.2-TI2V-5B-Q6_K.gguf",
                "link": "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "balanced", "quality", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 18.2,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Q6_K — mid-point between Q4 footprint and Q8 fidelity.",
                "estimatedGenerationSeconds": 155.0,
                "availableLocally": False,
                "releaseDate": "2025-08",
            },
            {
                "id": "QuantStack/Wan2.2-TI2V-5B-GGUF-q8",
                "familyId": "wan-2-2",
                "name": "Wan 2.2 TI2V 5B · GGUF Q8_0",
                "provider": "Alibaba · QuantStack",
                "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "ggufRepo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
                "ggufFile": "Wan2.2-TI2V-5B-Q8_0.gguf",
                "link": "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "quality", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 19.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Q8_0 — near-bf16 quality at roughly half the transformer footprint.",
                "estimatedGenerationSeconds": 150.0,
                "availableLocally": False,
                "releaseDate": "2025-08",
            },
            {
                "id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "familyId": "wan-2-2",
                "name": "Wan 2.2 T2V A14B",
                "provider": "Alibaba",
                # -Diffusers mirror ships the standard diffusers layout; the
                # base Wan-AI/Wan2.2-T2V-A14B repo uses the native Wan format.
                # This is an MoE model with two transformer folders
                # (``transformer/`` + ``transformer_2/``, high-noise and
                # low-noise experts), so the on-disk footprint is roughly
                # double what a dense 14B would be — ~126 GB total including
                # the UMT5-XXL text encoder and VAE. Active params per step
                # are ~14B; total params are ~27B.
                "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "link": "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                "runtime": "diffusers WanPipeline",
                "styleTags": ["general", "quality", "motion", "heavy"],
                "taskSupport": ["txt2video"],
                "sizeGb": 126.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": (
                    "MoE architecture with dual high-/low-noise experts — ~126 GB on disk. "
                    "Needs a data-center GPU (80 GB+) or 192 GB+ unified memory. "
                    "Consumer hardware should use TI2V-5B above."
                ),
                "estimatedGenerationSeconds": 300.0,
                "availableLocally": False,
                "releaseDate": "2025-07",
            },
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
                "releaseDate": "2024-12",
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
                "releaseDate": "2024-10",
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
                "releaseDate": "2024-08",
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
                "releaseDate": "2024-08",
            },
        ],
    },
    {
        "id": "longlive",
        "name": "LongLive 1.3B",
        "provider": "NVlabs",
        "headline": "Real-time, causal long-form video — up to 240s on a Wan 2.1 1.3B base.",
        "summary": (
            "LongLive (ICLR 2026) extends Wan 2.1 T2V 1.3B with a causal "
            "streaming pipeline: 20.7 FPS on a single H100, up to 240s. "
            "CUDA only — installed into an isolated venv via "
            "scripts/install-longlive.sh."
        ),
        "updatedLabel": "Experimental — long-form",
        "badges": ["Long-form", "Real-time", "Apache 2.0", "CUDA"],
        "defaultVariantId": "NVlabs/LongLive-1.3B",
        "variants": [
            {
                "id": "NVlabs/LongLive-1.3B",
                "familyId": "longlive",
                "name": "LongLive 1.3B",
                "provider": "NVlabs",
                "repo": "NVlabs/LongLive-1.3B",
                "link": "https://huggingface.co/Efficient-Large-Model/LongLive-1.3B",
                "runtime": "LongLive subprocess (torchrun)",
                "styleTags": ["general", "long-form", "motion", "causal"],
                "taskSupport": ["txt2video"],
                # Wan 2.1 1.3B base (~3 GB) + LongLive generator checkpoint
                # + LoRA (~1.5 GB). Isolated venv + CUDA-only deps weigh in
                # at ~10 GB total after install.
                "sizeGb": 10.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 30.0,
                "note": (
                    "CUDA only. Needs scripts/install-longlive.sh (isolated "
                    "venv + LongLive + Wan 2.1 base weights). Output at 16 FPS."
                ),
                "estimatedGenerationSeconds": 60.0,
                "availableLocally": False,
                "releaseDate": "2025-09",
            },
        ],
    },
]
