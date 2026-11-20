"""Speech-to-text models on an NVIDIA GPU (Windows and Linux): the ``transformers-asr`` backend.

They run through Transformers 5.23 in the ASR worker (``voice_worker_asr``) once the speech-to-text GPU add-on
(``stt-gpu``, ``voice_env_asr``) is installed. Each row names the files it needs, so a download skips the NeMo
archive, the GGUF and the pictures that share the repo; ``minVramGb`` and ``minComputeCap`` are checked against this
computer's GPU by ``voice_gpu`` before a request runs. Repos, licences and file sizes checked through the Hub API on
2026-11-19 (none gated). Numbers marked measured come from the RTX 4090 run (docs/progress/rtx4090-progress.md).

Every row pins a full Hugging Face ``revision`` (the repo's commit on 2026-11-20, FU-271) and the download fetches
exactly it (``voice_runtime.stt_repo_revision``), so a repo change upstream cannot change what users run until this
table changes. Bump a revision with the model's notes, after a run on the new files. ``ALIGNER`` is the forced aligner
that gives the audio-LLM rows their word timings (FU-270); it is not a dictation model, so the Dictation models panel
shows it as its own small "Word timings" download (``voice_runtime.aligner_row``, D-101).

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


# ``minVramGb``: the worker's own PyTorch peak (reserved) on the RTX 4090 run, 2026-11-19, 9.4 s clip, plus about 1 GB
# for the CUDA context and margin: Qwen3-ASR 1.7B 3.97 GiB, 0.6B 1.63, Parakeet 2.49, Nemotron ASR 2.55, Granite 4.38.
# Nemotron 3.5 Transcribe is the 64.7 GB of bf16 weights (32.4B parameters, counted on the meta device) plus room.
# ``vramMeasured`` False: the figure is worked out from the weights, not a run (the card shows it as an estimate).
def _gpu(family: str, licence: str, min_vram_gb: float, files: list[str], revision: str,
         note: str | None = None, vram_measured: bool = True) -> dict[str, Any]:
    return {"family": family, "licence": licence, "minVramGb": min_vram_gb, "minComputeCap": _CAP, "files": files,
            "revision": revision, "note": note, "vramMeasured": vram_measured}


# Qwen3-ForcedAligner-0.6B (Transformers conversion, Apache-2.0, 1.84 GB bf16): word timings for a transcript, in
# Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian and Spanish, up to five
# minutes of audio (the worker aligns window by window, 30 s each). Japanese and Korean are not used: splitting their
# words needs the nagisa and soynlp packages, which the add-on does not install.
ALIGNER: dict[str, Any] = {
    "key": "qwen3-forced-aligner-0.6b",
    "name": "Qwen3-ForcedAligner 0.6B",
    "repo": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
    "revision": "c07281df297b9905d24a508279258cccf987a064",
    "sizeGb": 1.84,
    "licence": "Apache-2.0",
    "files": [*_LLM_FILES[:2], "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json"],
    # The families that use it. The first three have no timings of their own and need it; Nemotron ASR has its own
    # (late by its streaming latency) and only prefers the aligner's when it is downloaded.
    "families": ("qwen3-asr", "granite-speech", "nemotron-transcribe", "nemotron-asr"),
    "optionalFor": ("nemotron-asr",),
    # GPU memory it adds while loaded (FU-310), beside the transcription model.
    "vramGb": 1.9,
}


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
        "gpu": _gpu("qwen3-asr", "Apache-2.0", 5.0, _LLM_FILES, "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"),
    },
    {
        "key": "qwen3-asr-0.6b",
        "name": "Qwen3-ASR 0.6B",
        "languages": "30 languages and 22 Chinese dialects, automatic detection",
        "default": False,
        "translate": False,
        "sources": {BACKEND: ("Qwen/Qwen3-ASR-0.6B-hf", 1.58)},
        "gpu": _gpu("qwen3-asr", "Apache-2.0", 2.5, _LLM_FILES, "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"),
    },
    {
        "key": "granite-speech-5.0-2b",
        "name": "Granite Speech 5.0 2B",
        "languages": "English, French, German, Spanish, Portuguese, Japanese",
        "default": False,
        "translate": False,
        "sources": {BACKEND: ("ibm-granite/granite-speech-5.0-2b", 4.64)},
        "gpu": _gpu("granite-speech", "Apache-2.0", 5.5, [
            "added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
            "model.safetensors", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
        ], "ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7"),
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
        ], "c17667641642c4265ff2b7037dd101b7d4567634", note="Needs a GPU with about 72 GB of memory. Untested here.",
            vram_measured=False),
    },
]

# GPU sources for rows that already exist on the Mac (keyed by the row's ``key``).
GPU_STT_SOURCES: dict[str, dict[str, Any]] = {
    # NVIDIA's own Transformers weights (CC-BY-4.0), float32; the repo also holds the .nemo archive and a GGUF.
    "parakeet-tdt-v3": {"source": ("nvidia/parakeet-tdt-0.6b-v3", 2.51),
                        "gpu": _gpu("parakeet-tdt", "CC-BY-4.0", 3.5, _TRANSDUCER_FILES,
                                    "541d1f99c6b0c3cd0b11a95167540bb8edefd82b")},
    "nemotron-3.5-asr": {"source": ("nvidia/nemotron-3.5-asr-streaming-0.6b", 2.55),
                         "gpu": _gpu("nemotron-asr", "OpenMDW-1.1", 3.5, _TRANSDUCER_FILES,
                                     "ea30d66debe3740a08b573244286791d423d6b3e")},
}
