"""Speech-to-text models on an NVIDIA GPU (Windows and Linux): the ``transformers-asr`` backend.

They run through Transformers 5.23 in the ASR worker (``voice_worker_asr``) once the speech-to-text GPU add-on
(``stt-gpu``, ``voice_env_asr``) is installed. Each row names the files it needs, so a download skips the NeMo
archive, the GGUF and the pictures that share the repo; ``minVramGb`` and ``minComputeCap`` are checked against this
computer's GPU by ``voice_gpu`` before a request runs. Repos, licences and file sizes checked through the Hub API on
2026-11-19 (none gated). Numbers marked measured come from the RTX 4090 run (docs/progress/rtx4090-progress.md).

``voice_runtime`` merges these into its catalogue: ``GPU_STT_MODELS`` are new rows, ``GPU_STT_SOURCES`` adds a GPU
source to a row that already runs on the Mac.
"""

from __future__ import annotations

from typing import Any

BACKEND = "transformers-asr"
# PyTorch's CUDA 12.8 wheels start at Volta; the audio LLMs use bf16 from Ampere and fp16 below it.
_CAP = 7.0
_LLM_FILES = [
    "chat_template.jinja", "config.json", "generation_config.json", "model.safetensors", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json",
]
_TRANSDUCER_FILES = [
    "config.json", "generation_config.json", "model.safetensors", "processor_config.json", "tokenizer.json",
    "tokenizer_config.json",
]


def _gpu(family: str, licence: str, min_vram_gb: float, files: list[str], note: str | None = None) -> dict[str, Any]:
    return {"family": family, "licence": licence, "minVramGb": min_vram_gb, "minComputeCap": _CAP, "files": files,
            "note": note}


# New rows, in the order the Dictation models panel lists them (the first with ``default`` that is downloaded becomes
# the default on this lane; ``voice_runtime.list_stt_models``).
GPU_STT_MODELS: list[dict[str, Any]] = [
    {
        "key": "qwen3-asr-1.7b",
        "name": "Qwen3-ASR 1.7B",
        "languages": "30 languages and 22 Chinese dialects, automatic detection",
        "default": True,
        "translate": False,
        # The Transformers conversion of Qwen/Qwen3-ASR-1.7B (Apache-2.0), bf16.
        "sources": {BACKEND: ("Qwen/Qwen3-ASR-1.7B-hf", 4.09)},
        "gpu": _gpu("qwen3-asr", "Apache-2.0", 6.0, _LLM_FILES),
    },
    {
        "key": "qwen3-asr-0.6b",
        "name": "Qwen3-ASR 0.6B",
        "languages": "30 languages and 22 Chinese dialects, automatic detection",
        "default": False,
        "translate": False,
        "sources": {BACKEND: ("Qwen/Qwen3-ASR-0.6B-hf", 1.58)},
        "gpu": _gpu("qwen3-asr", "Apache-2.0", 3.0, _LLM_FILES),
    },
    {
        "key": "granite-speech-5.0-2b",
        "name": "Granite Speech 5.0 2B",
        "languages": "English, French, German, Spanish, Portuguese, Japanese",
        "default": False,
        "translate": False,
        "sources": {BACKEND: ("ibm-granite/granite-speech-5.0-2b", 4.64)},
        "gpu": _gpu("granite-speech", "Apache-2.0", 6.5, [
            "added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
            "model.safetensors", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
        ]),
    },
    {
        "key": "nemotron-3.5-transcribe-30b",
        "name": "Nemotron 3.5 Transcribe 30B-A4B",
        "languages": "40 languages",
        "default": False,
        "translate": False,
        # 67.5 GB of bf16 weights: for 80 GB cards. Untested here (an RTX 4090 has 24 GB).
        "sources": {BACKEND: ("nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", 67.5)},
        "gpu": _gpu("nemotron-transcribe", "OpenMDW-1.1", 72.0, [
            *_LLM_FILES, "alignment_tokenizer/tokenizer.json", "alignment_tokenizer/tokenizer_config.json",
            "llm_backbone/config.json",
        ], note="Needs a GPU with about 72 GB of memory. Untested here."),
    },
]

# GPU sources for rows that already exist on the Mac (keyed by the row's ``key``).
GPU_STT_SOURCES: dict[str, dict[str, Any]] = {
    # NVIDIA's own Transformers weights (CC-BY-4.0), float32; the repo also holds the .nemo archive and a GGUF.
    "parakeet-tdt-v3": {"source": ("nvidia/parakeet-tdt-0.6b-v3", 2.51),
                        "gpu": _gpu("parakeet-tdt", "CC-BY-4.0", 4.0, _TRANSDUCER_FILES)},
    "nemotron-3.5-asr": {"source": ("nvidia/nemotron-3.5-asr-streaming-0.6b", 2.55),
                         "gpu": _gpu("nemotron-asr", "OpenMDW-1.1", 4.0, _TRANSDUCER_FILES)},
}
