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
        "id": "ltx-2",
        "name": "LTX-2 (MLX)",
        "provider": "Lightricks · prince-canuma",
        "headline": "LTX-2 19B with pre-converted MLX weights — native Apple Silicon path.",
        "summary": (
            "Pre-converted LTX-2 weights for mlx-video on Apple Silicon. Distilled variants "
            "are fast iteration paths with fixed low-step sampling; 'dev' variants run CFG steps for "
            "best fidelity. Routes through Blaizzy/mlx-video — no torch/MPS round trip."
        ),
        "updatedLabel": "Native MLX",
        "badges": ["MLX Native", "Apple Silicon", "Apache 2.0"],
        "defaultVariantId": "prince-canuma/LTX-2-distilled",
        "variants": [
            {
                "id": "prince-canuma/LTX-2-distilled",
                "familyId": "ltx-2",
                "name": "LTX-2 · distilled (MLX)",
                "provider": "Lightricks · prince-canuma",
                "repo": "prince-canuma/LTX-2-distilled",
                "link": "https://huggingface.co/prince-canuma/LTX-2-distilled",
                "runtime": "mlx-video (MLX native)",
                "styleTags": ["general", "fast", "motion", "mlx"],
                "taskSupport": ["txt2video"],
                "sizeGb": 19.0,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "Distilled LTX-2 — fastest MLX path for previews. Use the dev variant for final fidelity.",
                "estimatedGenerationSeconds": 60.0,
                "availableLocally": False,
                "releaseDate": "2026-01",
            },
            {
                "id": "prince-canuma/LTX-2-dev",
                "familyId": "ltx-2",
                "name": "LTX-2 · dev (MLX)",
                "provider": "Lightricks · prince-canuma",
                "repo": "prince-canuma/LTX-2-dev",
                "link": "https://huggingface.co/prince-canuma/LTX-2-dev",
                "runtime": "mlx-video (MLX native)",
                "styleTags": ["general", "quality", "motion", "mlx"],
                "taskSupport": ["txt2video"],
                "sizeGb": 19.0,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "Full LTX-2 dev weights — higher fidelity, longer sampling than distilled.",
                "estimatedGenerationSeconds": 180.0,
                "availableLocally": False,
                "releaseDate": "2026-01",
            },
            {
                "id": "prince-canuma/LTX-2.3-distilled",
                "familyId": "ltx-2",
                "name": "LTX-2.3 · distilled (MLX)",
                "provider": "Lightricks · prince-canuma",
                "repo": "prince-canuma/LTX-2.3-distilled",
                "textEncoderRepo": "prince-canuma/LTX-2-distilled",
                "link": "https://huggingface.co/prince-canuma/LTX-2.3-distilled",
                "runtime": "mlx-video (MLX native)",
                "styleTags": ["general", "fast", "motion", "mlx"],
                "taskSupport": ["txt2video"],
                "sizeGb": 19.0,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "LTX-2.3 distilled — refreshed fast preview path with sharper texture detail vs LTX-2. Use the dev variant for final fidelity.",
                "estimatedGenerationSeconds": 60.0,
                "availableLocally": False,
                "releaseDate": "2026-03",
            },
            {
                "id": "prince-canuma/LTX-2.3-dev",
                "familyId": "ltx-2",
                "name": "LTX-2.3 · dev (MLX)",
                "provider": "Lightricks · prince-canuma",
                "repo": "prince-canuma/LTX-2.3-dev",
                "textEncoderRepo": "prince-canuma/LTX-2-distilled",
                "link": "https://huggingface.co/prince-canuma/LTX-2.3-dev",
                "runtime": "mlx-video (MLX native)",
                "styleTags": ["general", "quality", "motion", "mlx"],
                "taskSupport": ["txt2video"],
                "sizeGb": 19.0,
                "recommendedResolution": "768x512",
                "defaultDurationSeconds": 4.0,
                "note": "LTX-2.3 dev — quality tier; full sampler steps for best output. Apple Silicon native via MLX. Install mlx-video from Setup → GPU runtime bundle to enable.",
                "estimatedGenerationSeconds": 180.0,
                "availableLocally": False,
                "releaseDate": "2026-03",
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
                # Resident peak ~14 GB during text encoding (UMT5-XXL bf16);
                # drops to ~4 GB during diffusion when encoder is freed.
                "runtimeFootprintGb": 14.0,
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
                # 14B trans bf16 (~28 GB) + UMT5-XXL (~11 GB) peak.
                # 24 GB CUDA needs NF4 (~7 GB trans → ~18 GB peak).
                "runtimeFootprintGb": 39.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Wan 2.1 quality tier. ~45GB. Same WanPipeline class as the 1.3B and Wan 2.2.",
                "estimatedGenerationSeconds": 180.0,
                "availableLocally": False,
                "releaseDate": "2025-02",
            },
            {
                "id": "city96/Wan2.1-T2V-1.3B-gguf-q4km",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 1.3B · GGUF Q4_K_M",
                "provider": "Alibaba · city96",
                "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "ggufRepo": "city96/Wan2.1-T2V-1.3B-gguf",
                "ggufFile": "wan2.1-t2v-1.3B-Q4_K_M.gguf",
                "link": "https://huggingface.co/city96/Wan2.1-T2V-1.3B-gguf",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "fast", "small", "gguf"],
                "taskSupport": ["txt2video"],
                # ~0.9 GB GGUF transformer + ~14 GB shared UMT5-XXL/VAE base.
                "sizeGb": 14.9,
                "runtimeFootprintGb": 12.5,  # Q4_K_M trans (~0.9 GB) + UMT5 (~11 GB)
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 4.0,
                "note": "Q4_K_M — smallest quantized 1.3B; runs in <8 GB unified memory once base is cached.",
                "estimatedGenerationSeconds": 70.0,
                "availableLocally": False,
                "releaseDate": "2025-03",
            },
            {
                "id": "city96/Wan2.1-T2V-1.3B-gguf-q6k",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 1.3B · GGUF Q6_K",
                "provider": "Alibaba · city96",
                "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "ggufRepo": "city96/Wan2.1-T2V-1.3B-gguf",
                "ggufFile": "wan2.1-t2v-1.3B-Q6_K.gguf",
                "link": "https://huggingface.co/city96/Wan2.1-T2V-1.3B-gguf",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "fast", "small", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 15.2,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 4.0,
                "note": "Q6_K — mid-point between Q4 footprint and Q8 fidelity.",
                "estimatedGenerationSeconds": 68.0,
                "availableLocally": False,
                "releaseDate": "2025-03",
            },
            {
                "id": "city96/Wan2.1-T2V-1.3B-gguf-q8",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 1.3B · GGUF Q8_0",
                "provider": "Alibaba · city96",
                "repo": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "ggufRepo": "city96/Wan2.1-T2V-1.3B-gguf",
                "ggufFile": "wan2.1-t2v-1.3B-Q8_0.gguf",
                "link": "https://huggingface.co/city96/Wan2.1-T2V-1.3B-gguf",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "quality", "small", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 15.5,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 4.0,
                "note": "Q8_0 — near-bf16 quality at roughly half the transformer footprint.",
                "estimatedGenerationSeconds": 65.0,
                "availableLocally": False,
                "releaseDate": "2025-03",
            },
            {
                "id": "city96/Wan2.1-T2V-14B-gguf-q4km",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 14B · GGUF Q4_K_M",
                "provider": "Alibaba · city96",
                "repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "ggufRepo": "city96/Wan2.1-T2V-14B-gguf",
                "ggufFile": "wan2.1-t2v-14B-Q4_K_M.gguf",
                "link": "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "quality", "motion", "gguf"],
                "taskSupport": ["txt2video"],
                # ~7 GB GGUF transformer + ~14 GB shared UMT5-XXL/VAE — fits
                # comfortably on a 24 GB RTX 4090 with VAE headroom.
                "sizeGb": 21.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Q4_K_M — unlocks Wan 2.1 14B on 24 GB VRAM (RTX 4090) without bnb.",
                "estimatedGenerationSeconds": 220.0,
                "availableLocally": False,
                "releaseDate": "2025-03",
            },
            {
                "id": "city96/Wan2.1-T2V-14B-gguf-q6k",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 14B · GGUF Q6_K",
                "provider": "Alibaba · city96",
                "repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "ggufRepo": "city96/Wan2.1-T2V-14B-gguf",
                "ggufFile": "wan2.1-t2v-14B-Q6_K.gguf",
                "link": "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "quality", "motion", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 24.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Q6_K — mid-point between Q4 footprint and Q8 fidelity.",
                "estimatedGenerationSeconds": 210.0,
                "availableLocally": False,
                "releaseDate": "2025-03",
            },
            {
                "id": "city96/Wan2.1-T2V-14B-gguf-q8",
                "familyId": "wan-2-1",
                "name": "Wan 2.1 T2V 14B · GGUF Q8_0",
                "provider": "Alibaba · city96",
                "repo": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "ggufRepo": "city96/Wan2.1-T2V-14B-gguf",
                "ggufFile": "wan2.1-t2v-14B-Q8_0.gguf",
                "link": "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf",
                "runtime": "diffusers WanPipeline + GGUF transformer",
                "styleTags": ["general", "quality", "motion", "gguf"],
                "taskSupport": ["txt2video"],
                "sizeGb": 28.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": "Q8_0 — near-bf16 quality at roughly half the transformer footprint.",
                "estimatedGenerationSeconds": 200.0,
                "availableLocally": False,
                "releaseDate": "2025-03",
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
                # Runtime resident peak (text encode phase): ~22 GB. Drops to
                # ~12 GB during diffusion once UMT5 is freed. Disk size
                # over-estimates resident because the repo carries duplicate
                # sharded safetensors + tokenizer caches.
                "runtimeFootprintGb": 22.0,
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
                # GGUF Q4_K_M trans (~3.5 GB) + UMT5-XXL during encode (~11 GB).
                "runtimeFootprintGb": 14.5,
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
                "runtimeFootprintGb": 16.5,  # Q6_K trans ~5 GB + UMT5 ~11 GB
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
                "runtimeFootprintGb": 18.0,  # Q8 trans ~7 GB + UMT5 ~11 GB
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
                # MoE: two 14B experts (high-/low-noise) at ~28 GB each on disk,
                # plus UMT5-XXL ~11 GB. Naive diffusers load keeps BOTH experts
                # resident → ~67 GB peak (crashes 64 GB Mac). With diffusers'
                # ``enable_sequential_cpu_offload()`` only one expert is in
                # memory at a time → ~30 GB peak (active expert ~28 GB +
                # UMT5 during encode, dropping to ~14 GB during diffusion).
                # We declare the offloaded peak so 64 GB+ Macs don't see a
                # bogus "needs 176 GB" warning, but the note flags that the
                # offload mode is required.
                "runtimeFootprintGb": 30.0,
                "recommendedResolution": "832x480",
                "defaultDurationSeconds": 5.0,
                "note": (
                    "MoE — dual high-/low-noise experts, ~27B total / 14B active per step. "
                    "126 GB on disk. Runs on 64 GB+ Apple Silicon or 40 GB+ CUDA with "
                    "enable_sequential_cpu_offload() (peak ~30 GB resident). For 24 GB CUDA "
                    "use the TI2V-5B variant above."
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
                # 13B trans bf16 (~26 GB) + T5-XXL + LLaMA encoders (~12 GB)
                # peak during text encode; drops to ~28 GB during diffusion.
                # Apple Silicon 64 GB+ Max/Ultra runs this comfortably with
                # diffusers' enable_model_cpu_offload(); 24 GB CUDA needs NF4.
                "runtimeFootprintGb": 34.0,
                "recommendedResolution": "1280x720",
                "defaultDurationSeconds": 5.0,
                "note": "High quality. Runs on 64 GB+ Apple Silicon Max/Ultra or 40GB+ CUDA (24 GB CUDA with NF4). Sequential text-encoder loading keeps the diffusion phase under ~28 GB resident.",
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
            "CUDA only — installed into an isolated venv by the Studio "
            "or Video Discover install action."
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
                    "CUDA only. Click Install LongLive to set up the isolated "
                    "venv + LongLive + Wan 2.1 base weights. Output at 16 FPS."
                ),
                "estimatedGenerationSeconds": 60.0,
                "availableLocally": False,
                "releaseDate": "2025-09",
            },
        ],
    },
]


# FU-003: LongLive HF repo paths are stale (`NVlabs/LongLive-1.3B` 404s; the
# linked `Efficient-Large-Model/LongLive-1.3B` mirror is also unreachable).
# Hide the family from the catalog until the working HF repo is verified
# and the install script is Mac-aware. Preserving the entry above so the
# data isn't lost when the flag flips back on.
_LONGLIVE_ENABLED = False

if not _LONGLIVE_ENABLED:
    VIDEO_MODEL_FAMILIES = [
        family for family in VIDEO_MODEL_FAMILIES if family.get("id") != "longlive"
    ]
