"""Speech models on Windows and Linux with an NVIDIA GPU (D-70 Phase 6, FU-149). NOT VERIFIED: written on a Mac.

Same keys of information as ``voice_models.VOICE_MODELS``, with ``lanes: ["cuda"]`` and the upstream PyTorch or
GGUF weights instead of the MLX conversions:

* Chatterbox-Turbo and Chatterbox Multilingual V3 run through ``chatterbox-tts`` (installed ``--no-deps`` by the
  speech GPU add-on, ``voice_env.speech_gpu_commands``). That package adds Resemble's PerTh watermark to every
  output (``builtinWatermark: "perth"``); the AI tag on cloned speech records it.
* VoxCPM2 runs through ``voxcpm`` (also ``--no-deps``), denoiser off.
* Qwen3-TTS (Base, CustomVoice, VoiceDesign, 0.6B and 1.7B) runs through ``qwen-tts``. It needs transformers 4.57, so
  it has its own worker and its own transformers folder (``voice_env.QWEN_TTS_SITE_PACKAGES``). These rows have
  numbers measured on an RTX 4090 and carry ``minVramGb`` and ``minComputeCap``, which ``voice_gpu`` checks against
  the GPU in this computer.
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

# Qwen3-TTS through ``qwen-tts`` (Apache-2.0), the upstream bf16 weights. It needs transformers 4.57, so it runs in its
# own worker with its own transformers folder (``voice_env.QWEN_TTS_SITE_PACKAGES``, ``voice_client.get_qwen_worker``).
# Revisions checked through the Hub API with no token on 2026-11-11; none is gated. Only the files the loader reads.
_QWEN_FILES = [
    "config.json", "generation_config.json", "merges.txt", "model.safetensors", "preprocessor_config.json",
    "speech_tokenizer/config.json", "speech_tokenizer/configuration.json", "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json", "tokenizer_config.json", "vocab.json",
]
_QWEN_MEASURED_ON = "NVIDIA RTX 4090 24 GB, Windows 11, driver 591.86, 2026-11-11"
_QWEN_GPU_NEEDS = {"minComputeCap": 7.0}  # PyTorch's CUDA 12.8 wheels start at Volta; bf16 from Ampere, fp16 below


def _qwen(key: str, name: str, summary: str, variant: str, revision: str, size_gb: float, kind: str,
          capabilities: dict[str, bool], *, cold: float, per10: float, short: float, peak: float) -> dict[str, Any]:
    """One Qwen3-TTS GPU row. Numbers from the live run through the app (docs/progress/rtx4090-progress.md): ``cold``
    is the first short line minus the same line warm (worker start excluded, model load, and for a cloning row the
    watermark worker), ``per10`` the wall seconds per 10 s of speech for a 600-character text in 6 chunks, ``short``
    one 12-word line warm, ``peak`` the GPU memory the worker added over idle (nvidia-smi, CUDA context included).
    The model stays resident after a request, so idle equals peak. ``minVramGb`` is the peak plus 0.7 GB, rounded."""
    return {
        "key": key, "name": name, "summary": summary,
        "repo": f"Qwen/Qwen3-TTS-12Hz-{variant}", "revision": revision, "files": list(_QWEN_FILES),
        "upstream": f"Qwen/Qwen3-TTS-12Hz-{variant}", "sizeGb": size_gb, "engine": "torch", "lanes": ["cuda"],
        "languages": "10 languages", "kind": kind, "family": "qwen3-tts", "capabilities": capabilities,
        "minVramGb": round(peak + 0.7, 1), **_QWEN_GPU_NEEDS,
        "estimates": {
            "basis": "measured", "measuredOn": _QWEN_MEASURED_ON, "coldStartSeconds": cold,
            "secondsPer10sAudio": per10, "wallSecondsShortLine": short, "peakGb": peak, "idleGb": peak,
        },
    }


CUDA_VOICE_MODELS: list[dict[str, Any]] = [
    _qwen("qwen3-tts-0.6b-cuda", "Qwen3-TTS 0.6B Base (GPU)",
          "Clones a voice from 10 to 20 s of clean speech; needs about 3.5 GB of GPU memory.", "0.6B-Base",
          "5d83992436eae1d760afd27aff78a71d676296fc", 2.52, "base", {"clone": True},
          cold=19.9, per10=19.09, short=9.69, peak=3.42),
    _qwen("qwen3-tts-1.7b-cuda", "Qwen3-TTS 1.7B Base (GPU)",
          "Closer likeness from a voice sample; needs about 5.3 GB of GPU memory. No quicker than the 0.6B model.",
          "1.7B-Base", "fd4b254389122332181a7c3db7f27e918eec64e3", 4.54, "base", {"clone": True},
          cold=15.7, per10=24.07, short=8.9, peak=5.31),
    _qwen("qwen3-tts-custom-0.6b-cuda", "Qwen3-TTS 0.6B CustomVoice (GPU)",
          "Nine preset voices; ignores the style instruction. Smallest in GPU memory (about 2.8 GB), no quicker.",
          "0.6B-CustomVoice", "85e237c12c027371202489a0ec509ded67b5e4b5", 2.50, "custom", {"speak": True},
          cold=8.5, per10=17.06, short=8.74, peak=2.83),
    _qwen("qwen3-tts-custom-1.7b-cuda", "Qwen3-TTS 1.7B CustomVoice (GPU)",
          "Nine preset voices and a written style instruction; about 4.7 GB of GPU memory.", "1.7B-CustomVoice",
          "0c0e3051f131929182e2c023b9537f8b1c68adfe", 4.52, "custom", {"speak": True, "style": True},
          cold=9.3, per10=17.19, short=8.25, peak=4.68),
    _qwen("qwen3-tts-design-1.7b-cuda", "Qwen3-TTS 1.7B VoiceDesign (GPU)",
          "Describe a voice in words and it speaks in that voice; about 4.6 GB of GPU memory.", "1.7B-VoiceDesign",
          "5ecdb67327fd37bb2e042aab12ff7391903235d3", 4.52, "design", {"speak": True, "style": True, "design": True},
          cold=9.3, per10=17.16, short=7.6, peak=4.58),
    {
        "key": "chatterbox-turbo-cuda",
        "name": "Chatterbox-Turbo (GPU)",
        "summary": "Quick, natural English with sound tags; clones from 5 s of speech. Adds Resemble's PerTh watermark.",
        "repo": "ResembleAI/chatterbox-turbo",
        "revision": "749d1c1a46eb10492095d68fbcf55691ccf137cd",
        "files": [
            # chatterbox-tts 0.1.7 ChatterboxTurboTTS.from_local builds S3Gen(meanflow=True) and loads only
            # s3gen_meanflow.safetensors (checked on the RTX 4090 run); s3gen.safetensors (about 1 GB) is never read.
            "added_tokens.json", "conds.pt", "merges.txt", "s3gen_meanflow.safetensors",
            "special_tokens_map.json", "t3_turbo_v1.safetensors", "t3_turbo_v1.yaml", "tokenizer_config.json",
            "ve.safetensors", "vocab.json",
        ],
        "upstream": "ResembleAI/chatterbox-turbo",
        "sizeGb": 2.99,
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
