"""Speech models on Windows and Linux with an NVIDIA GPU (D-70 Phase 6, FU-149). NOT VERIFIED: written on a Mac.

Same keys of information as ``voice_models.VOICE_MODELS``, with ``lanes: ["cuda"]`` and the upstream PyTorch or
GGUF weights instead of the MLX conversions:

* Chatterbox-Turbo and Chatterbox Multilingual V3 run through ``chatterbox-tts`` (installed ``--no-deps`` by the
  speech GPU add-on, ``voice_env.speech_gpu_commands``). That package adds Resemble's PerTh watermark to every
  output (``builtinWatermark: "perth"``); the AI tag on cloned speech records it.
* VoxCPM2 runs through ``voxcpm`` (also ``--no-deps``), denoiser off.
* Orpheus 3B runs as a GGUF through the ``llama-server`` we already ship, and its audio codes are decoded by the SNAC
  decoder in ONNX on onnxruntime (no PyTorch, no vLLM).

Each row pins a full revision (checked through the Hub API with no token on 2026-11-09; none is gated) and lists
only the ``files`` it needs, so the 13.9 GB Chatterbox repo downloads its V3 files only. No row has measured
numbers yet, so none is offered (``listed``) until the RTX 4090 session measures it (docs/plans/rtx4090-test.md
section 2); ``CHAOSENGINE_VOICE_UNMEASURED=1`` lists them for that session.
"""

from __future__ import annotations

from typing import Any

from backend_service.catalog import voice_models as _mlx

CUDA_VOICE_MODELS: list[dict[str, Any]] = [
    {
        "key": "chatterbox-turbo-cuda",
        "name": "Chatterbox-Turbo (GPU)",
        "summary": "Quick, natural English with sound tags; clones from 5 s of speech. Adds Resemble's PerTh watermark.",
        "repo": "ResembleAI/chatterbox-turbo",
        "revision": "749d1c1a46eb10492095d68fbcf55691ccf137cd",
        "files": [
            # Both decoders: which one from_local reads is confirmed on the 4090 (then the other can go).
            "added_tokens.json", "conds.pt", "merges.txt", "s3gen.safetensors", "s3gen_meanflow.safetensors",
            "special_tokens_map.json", "t3_turbo_v1.safetensors", "t3_turbo_v1.yaml", "tokenizer_config.json",
            "ve.safetensors", "vocab.json",
        ],
        "upstream": "ResembleAI/chatterbox-turbo",
        "sizeGb": 4.04,
        "engine": "torch",
        "lanes": ["cuda"],
        "languages": "English",
        "kind": "turbo",
        "family": "chatterbox-turbo",
        "capabilities": {"speak": True, "clone": True, "tags": True, "presets": True},
        "controls": {"tags": _mlx.TURBO_TAGS, "styleTags": _mlx.TURBO_STYLE_TAGS},
        "licence": {"id": "mit", "name": "MIT", "commercial": "yes", "url": "https://huggingface.co/ResembleAI/chatterbox-turbo"},
        "companions": [],
        "sampleRate": 24_000,
        "minReferenceS": 5.0,
        "builtinWatermark": "perth",
        "estimates": None,
    },
    {
        "key": "chatterbox-mtl-cuda",
        "name": "Chatterbox Multilingual V3 (GPU)",
        "summary": "23 languages with an Expression dial; clones from a voice sample. Adds Resemble's PerTh watermark.",
        "repo": "ResembleAI/chatterbox",
        "revision": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
        # V3 text model with the shared voice encoder, decoder, tokenizer and default voice (conds.pt).
        "files": ["conds.pt", "grapheme_mtl_merged_expanded_v1.json", "s3gen.pt", "t3_mtl23ls_v3.safetensors", "ve.pt", "Cangjie5_TC.json"],
        "upstream": "ResembleAI/chatterbox",
        "sizeGb": 3.21,
        "engine": "torch",
        "lanes": ["cuda"],
        "languages": "23 languages",
        "kind": "chatterbox",
        "family": "chatterbox",
        "capabilities": {"speak": True, "clone": True, "expression": True, "presets": True, "languageSelect": True},
        "controls": {"expression": _mlx._EXPRESSION, "guidance": _mlx._GUIDANCE, "languages": _mlx.CHATTERBOX_LANGUAGES},
        "licence": {"id": "mit", "name": "MIT", "commercial": "yes", "url": "https://huggingface.co/ResembleAI/chatterbox"},
        "companions": [],
        "sampleRate": 24_000,
        "minReferenceS": 3.0,
        "builtinWatermark": "perth",
        "estimates": None,
    },
    {
        "key": "voxcpm2-cuda",
        "name": "VoxCPM2 (GPU)",
        "summary": "Describe a voice, or clone one, in 30 languages. 48 kHz output.",
        "repo": "openbmb/VoxCPM2",
        "revision": "7936080b41163874f38610f46f438e2c0757f9fd",
        "files": ["audiovae.pth", "config.json", "model.safetensors", "special_tokens_map.json", "tokenization_voxcpm2.py",
                  "tokenizer.json", "tokenizer_config.json"],
        "upstream": "openbmb/VoxCPM2",
        "sizeGb": 4.96,
        "engine": "torch",
        "lanes": ["cuda"],
        "languages": "30 languages",
        "kind": "voxcpm",
        "family": "voxcpm2",
        "capabilities": {"speak": True, "style": True, "clone": True, "design": True},
        "controls": {},
        "licence": {"id": "apache-2.0", "name": "Apache 2.0", "commercial": "yes", "url": "https://huggingface.co/openbmb/VoxCPM2"},
        "companions": [],
        "sampleRate": 48_000,
        "minReferenceS": 3.0,
        "estimates": None,
    },
    {
        "key": "orpheus-3b-gguf",
        "name": "Orpheus 3B (GGUF)",
        "summary": "Eight expressive English voices with sound tags, through llama-server. No cloning.",
        "repo": "unsloth/orpheus-3b-0.1-ft-GGUF",
        "revision": "e2b00302c46af8205f521f60016600aa25a068e6",
        "files": ["orpheus-3b-0.1-ft-Q8_0.gguf"],
        "upstream": "canopylabs/orpheus-3b-0.1-ft",
        "sizeGb": 3.52,
        "engine": "llama-gguf",
        "lanes": ["cuda"],
        "languages": "English",
        "kind": "orpheus",
        "family": "orpheus",
        "capabilities": {"speak": True, "tags": True, "presets": True},
        "controls": {"tags": _mlx.ORPHEUS_TAGS},
        "licence": next(r for r in _mlx.VOICE_MODELS if r["key"] == "orpheus-3b")["licence"],
        "companions": [
            {
                # SNAC 24 kHz decoder in ONNX (MIT): turns Orpheus's audio codes into sound on onnxruntime.
                "repo": "onnx-community/snac_24khz-ONNX", "revision": "f317cdff7efb7b2a63148c9e68bbb0320fe629a1",
                "files": ["onnx/decoder_model.onnx"], "sizeGb": 0.05, "role": "snacDecoder", "licence": "mit",
            },
        ],
        "sampleRate": 24_000,
        "chunkChars": 250,
        "maxTokens": 2400,
        "minReferenceS": None,
        "estimates": None,
    },
]
