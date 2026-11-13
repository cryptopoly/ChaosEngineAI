"""Speech models that run through mlx-audio in the voice worker (D-28, D-45, D-58, D-70).

Qwen3-TTS: Base (clone), CustomVoice (preset speakers plus a style instruction) and VoiceDesign (a voice from a
description). Wave 2 (D-70, FU-149): Chatterbox-Turbo (built-in voice, sound tags, clone), VoxCPM2 (design and
two cloning modes, 30 languages, 48 kHz), Chatterbox Multilingual V3 (23 languages, Expression and Guidance,
clone) and Orpheus 3B (eight preset voices, sound tags, no cloning).

Only engines wired end to end are listed. Each row pins a full Hugging Face revision (Qwen Base rows checked on
2026-11-07, the other Qwen rows on 2026-11-08, wave-2 rows on 2026-11-08 and again on 2026-11-09, through the
Hub API with no token; none is gated), so a repo update upstream cannot change what users run until this table
changes. ``companions`` are extra repos a loader needs, pinned the same way and downloaded with the model:
``role: "loader"`` repos are handed to the worker, which points the loader at them instead of letting it fetch
them unpinned (``voice_worker_models.install_redirect``); ``role: "defaultVoice"`` is a voice file the worker
loads itself. Sizes are repo totals (only the listed files for a companion). Apple Silicon only (``lanes``):
the CUDA rows are in ``voice_models_cuda.py`` and are not measured yet.

Per row:

* ``capabilities``: ``speak`` (plain text with a preset, built-in or designed voice), ``style`` (a written style
  instruction), ``clone`` (a voice from a reference clip), ``design`` (a voice from a description),
  ``expression`` (the Expression slider), ``tags`` (sound tags such as ``[laugh]`` in the text), ``presets`` (named
  voices of its own), ``languageSelect`` (the request's ``language`` picks the language).
* ``controls``: what the screen draws from data: ``tags`` (documented sound tags), ``styleTags`` (Turbo's style
  tokens, which its authors do not document), ``expression`` and ``guidance`` ranges, ``languages`` (codes).
* ``licence``: the upstream model's licence (an mlx-community conversion sometimes lists another; the upstream
  author's value is shown, with its link).
* ``family`` groups rows and picks the worker call; ``sampleRate`` is the output rate; ``chunkChars`` the longest
  piece of text sent in one worker request; ``minReferenceS`` the shortest voice sample the model clones from;
  ``rank`` the per-family English quality rank (lower is better; None until the owner's listening check);
  ``builtinWatermark`` names a watermark the model adds itself (None on the MLX lane: mlx-audio adds none).
* ``estimates``: measured speed and memory (``basis: "measured"``), filled by ``scripts/measure-voice-models.py``.
  A wave-2 row that has not been measured on this hardware stays out of the picker (``listed`` False).
"""

from __future__ import annotations

import os
from typing import Any

# The model a cloned voice uses when the request names none. The owner switches this to "voxcpm2" after the
# listening check (D-70 answer 2); everything else follows from it (the rows' ``default`` flags below).
DEFAULT_CLONE_MODEL = "qwen3-tts-0.6b"

_QWEN_LICENCE: dict[str, Any] = {
    "id": "apache-2.0",
    "name": "Apache 2.0",
    "commercial": "yes",
    "url": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
# Kept for older imports: the licence of the Qwen3-TTS rows. Each row now carries its own ``licence``.
CLONE_LICENCE = _QWEN_LICENCE

_S3_TOKENIZER = {
    "repo": "mlx-community/S3TokenizerV2",
    "revision": "e0c9886f0e1c35ae85b1f27277416fb19fc72bec",
    "files": ["config.json", "model.safetensors"],
    "sizeGb": 0.49,
    "role": "loader",
    # The repo has no licence field; the code it comes from (xingchensong/S3Tokenizer) is Apache-2.0.
    "licence": "apache-2.0",
}

# Chatterbox Multilingual V3 (23 languages, ``lang_code``).
CHATTERBOX_LANGUAGES = [
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt",
    "ru", "sv", "sw", "tr", "zh",
]
# Chatterbox-Turbo: the nine sound tags its mlx-community card documents, and ten style tokens that are in its
# tokenizer but not documented by its authors (shown as chips with that warning, D-70 answer 3).
TURBO_TAGS = ["laugh", "chuckle", "sigh", "gasp", "cough", "clear throat", "sniff", "groan", "shush"]
TURBO_STYLE_TAGS = [
    "angry", "fear", "surprised", "whispering", "advertisement", "dramatic", "narration", "crying", "happy",
    "sarcastic",
]
# Orpheus 3B's tags (written ``[laugh]`` in the text box, sent as ``<laugh>``); ``[sniff]`` is read as sniffle.
ORPHEUS_TAGS = ["laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"]
# Qwen3-TTS takes a language name; these are the ten it lists.
QWEN_LANGUAGES = {
    "zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french",
    "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian",
}

_EXPRESSION = {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5}
_GUIDANCE = {"min": 0.0, "max": 1.0, "step": 0.05, "default": 0.5}

# Chat read-aloud and voice calls (D-67 W6): ``scripts/measure-voice-models.py --chat``, two runs on 2026-11-09, the
# slower of the two kept. First audio of one chat sentence on a fresh worker (worker start and model load included,
# "cold") and warm (median of three); peak is the worker's resident memory over those runs.
CHAT_MEASURED_ON = "Apple M4 Max, 64 GB, 2026-11-09"
# A voice whose warm first audio is above this is labelled slow for calls (still allowed).
SLOW_FOR_CALLS_S = 2.0


def _chat(*, warm: float, cold: float, peak: float) -> dict[str, Any]:
    return {"chatMeasuredOn": CHAT_MEASURED_ON, "chatFirstAudioS": warm, "chatFirstAudioColdS": cold, "chatPeakGb": peak}


# Kokoro has no catalog row (it is part of the voice pack); its chat numbers live here.
KOKORO_CHAT_ESTIMATES: dict[str, Any] = {"basis": "measured", **_chat(warm=0.13, cold=5.7, peak=0.87)}

VOICE_MODELS: list[dict[str, Any]] = [
    {
        "key": "qwen3-tts-0.6b",
        "name": "Qwen3-TTS 0.6B Base",
        "summary": "Smaller in memory (about 4 GB). Good likeness from 10 to 20 s of clean speech.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        "revision": "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
        "upstream": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "sizeGb": 2.52,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "kind": "base",
        "capabilities": {"speak": False, "style": False, "clone": True, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 3.0,
            "secondsPer10sAudio": 3.2,
            "wallSecondsShortLine": 2.8,
            "peakGb": 4.3,
            "idleGb": 3.7,
        },
    },
    {
        "key": "qwen3-tts-1.7b",
        "name": "Qwen3-TTS 1.7B Base",
        "summary": "Closer likeness; needs about 6.5 GB of memory. About as quick as the 0.6B model.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
        "revision": "a6eb4f68e4b056f1215157bb696209bc82a6db48",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sizeGb": 4.54,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "kind": "base",
        "capabilities": {"speak": False, "style": False, "clone": True, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 3.6,
            "secondsPer10sAudio": 3.6,
            "wallSecondsShortLine": 2.2,
            "peakGb": 6.5,
            "idleGb": 5.6,
        },
    },
    {
        "key": "qwen3-tts-custom-0.6b",
        "name": "Qwen3-TTS 0.6B CustomVoice",
        "summary": "Smallest in memory (about 3 GB). Nine preset voices; ignores the style instruction. No quicker than the 1.7B model.",
        "repo": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
        "revision": "6415d95f88be018ff9e46813119dc3bc12261328",
        "upstream": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "sizeGb": 2.50,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "kind": "custom",
        "capabilities": {"speak": True, "style": False, "clone": False, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 2.4,
            "secondsPer10sAudio": 2.65,
            "wallSecondsShortLine": 1.9,
            "peakGb": 2.93,
            "idleGb": 2.91,
        },
    },
    {
        "key": "qwen3-tts-custom-1.7b",
        "name": "Qwen3-TTS 1.7B CustomVoice",
        "summary": "Nine preset voices and a written style instruction (8-bit).",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "revision": "41d3337e8b7f2843a75841595fc14e4b9a7a4b96",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "sizeGb": 3.08,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "kind": "custom",
        "capabilities": {"speak": True, "style": True, "clone": False, "design": False},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 2.4,
            "secondsPer10sAudio": 2.45,
            "wallSecondsShortLine": 1.8,
            "peakGb": 3.5,
            "idleGb": 3.48,
            **_chat(warm=1.3, cold=4.1, peak=3.51),
        },
    },
    {
        "key": "qwen3-tts-design-1.7b",
        "name": "Qwen3-TTS 1.7B VoiceDesign",
        "summary": "Describe a voice in words and it speaks in that voice (8-bit).",
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "revision": "f90d617701d9f7f4ca499291e0b57f2b3c2fd2ee",
        "upstream": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "sizeGb": 3.08,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "10 languages",
        "kind": "design",
        "capabilities": {"speak": True, "style": True, "clone": False, "design": True},
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-08",
            # Fresh worker, weights already on disk; peak and idle are the worker's resident memory (ps / psutil RSS)
            # while speaking and after it finished. Wall seconds per 10 s of speech come from a 600-character text.
            "coldStartSeconds": 2.3,
            "secondsPer10sAudio": 2.46,
            "wallSecondsShortLine": 1.5,
            "peakGb": 3.51,
            "idleGb": 3.49,
        },
    },
    # ------------------------------------------------------------------
    # Wave 2 (D-70, FU-149)
    # ------------------------------------------------------------------
    {
        "key": "chatterbox-turbo",
        "name": "Chatterbox-Turbo",
        "summary": "Quick, natural English with sound tags such as [laugh] and [sigh]. Also clones from 5 s of speech.",
        "repo": "mlx-community/chatterbox-turbo-8bit",
        "revision": "2f2e21a03863f86a1274d1060dcc188e7cde77e1",
        "upstream": "ResembleAI/chatterbox-turbo",
        "sizeGb": 0.71,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "English",
        "kind": "turbo",
        "family": "chatterbox-turbo",
        "capabilities": {"speak": True, "style": False, "clone": True, "design": False, "tags": True, "presets": True},
        "controls": {"tags": TURBO_TAGS, "styleTags": TURBO_STYLE_TAGS},
        "licence": {"id": "mit", "name": "MIT", "commercial": "yes", "url": "https://huggingface.co/ResembleAI/chatterbox-turbo"},
        "companions": [_S3_TOKENIZER],
        "sampleRate": 24_000,
        "minReferenceS": 5.0,
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-09",
            # scripts/measure-voice-models.py, fresh worker, weights on disk: cold start = first short line minus the same
            # line warm; seconds per 10 s from the fixed 600-character text in this model's chunks; peak and idle are
            # the worker's resident memory (psutil RSS) during the long text and 5 s after.
            "coldStartSeconds": 2.4,
            "secondsPer10sAudio": 2.1,
            "wallSecondsShortLine": 0.8,
            "peakGb": 1.72,
            "idleGb": 1.72,
            **_chat(warm=0.6, cold=3.8, peak=1.65),
        },
    },
    {
        "key": "voxcpm2",
        "name": "VoxCPM2",
        "summary": "Describe a voice, or clone one, in 30 languages (found from the text). 48 kHz output.",
        "repo": "mlx-community/VoxCPM2-8bit",
        "revision": "d52725898a0675703f7f9ddc5a4d1a3cdbb99032",
        "upstream": "openbmb/VoxCPM2",
        "sizeGb": 3.23,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "30 languages",
        "kind": "voxcpm",
        "family": "voxcpm2",
        "capabilities": {"speak": True, "style": True, "clone": True, "design": True},
        "controls": {},
        "licence": {"id": "apache-2.0", "name": "Apache 2.0", "commercial": "yes", "url": "https://huggingface.co/openbmb/VoxCPM2"},
        "companions": [],
        "sampleRate": 48_000,
        "minReferenceS": 3.0,
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-09",
            # scripts/measure-voice-models.py, fresh worker, weights on disk: cold start = first short line minus the same
            # line warm; seconds per 10 s from the fixed 600-character text in this model's chunks; peak and idle are
            # the worker's resident memory (psutil RSS) during the long text and 5 s after.
            # Design. 48 kHz output.
            "coldStartSeconds": 2.0,
            "secondsPer10sAudio": 5.22,
            "wallSecondsShortLine": 1.9,
            "peakGb": 3.8,
            "idleGb": 3.8,
            **_chat(warm=1.7, cold=3.8, peak=3.63),
        },
    },
    {
        "key": "chatterbox-mtl",
        "name": "Chatterbox Multilingual V3",
        "summary": "23 languages with an Expression dial from calm to dramatic. Clones from a voice sample.",
        "repo": "mlx-community/chatterbox-multilingual-v3",
        "revision": "03565773edd72e949572557597af8063bb49a18a",
        "upstream": "ResembleAI/chatterbox",
        "sizeGb": 2.71,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "23 languages",
        "kind": "chatterbox",
        "family": "chatterbox",
        "capabilities": {
            "speak": True, "style": False, "clone": True, "design": False, "expression": True, "presets": True,
            "languageSelect": True,
        },
        "controls": {"expression": _EXPRESSION, "guidance": _GUIDANCE, "languages": CHATTERBOX_LANGUAGES},
        "licence": {"id": "mit", "name": "MIT", "commercial": "yes", "url": "https://huggingface.co/ResembleAI/chatterbox"},
        "companions": [
            _S3_TOKENIZER,
            {
                # The V3 conversion has no default voice; the V2 8-bit build's conds.safetensors (same voice encoder
                # and tokenizer) works with V3 weights (P0 spike, 2026-11-09). Only that file is fetched.
                "repo": "mlx-community/chatterbox-8bit", "revision": "9617d61b596a03d1bed766a28c341680e993a1b9",
                "files": ["conds.safetensors"], "sizeGb": 0.0, "role": "defaultVoice", "licence": "mit",
            },
        ],
        "sampleRate": 24_000,
        "minReferenceS": 3.0,
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-09",
            # scripts/measure-voice-models.py, fresh worker, weights on disk: cold start = first short line minus the same
            # line warm; seconds per 10 s from the fixed 600-character text in this model's chunks; peak and idle are
            # the worker's resident memory (psutil RSS) during the long text and 5 s after.
            # Default voice, English.
            "coldStartSeconds": 2.9,
            "secondsPer10sAudio": 3.63,
            "wallSecondsShortLine": 1.4,
            "peakGb": 3.67,
            "idleGb": 3.66,
        },
    },
    {
        "key": "orpheus-3b",
        "name": "Orpheus 3B",
        "summary": "Eight expressive English voices with sound tags such as [laugh] and [yawn]. No cloning.",
        "repo": "mlx-community/orpheus-3b-0.1-ft-8bit",
        "revision": "c8521bf4ea42f8dd96df099219d7871fc9873142",
        "upstream": "canopylabs/orpheus-3b-0.1-ft",
        "sizeGb": 3.54,
        "engine": "mlx-audio",
        "lanes": ["mlx"],
        "languages": "English",
        "kind": "orpheus",
        "family": "orpheus",
        "capabilities": {"speak": True, "style": False, "clone": False, "design": False, "tags": True, "presets": True},
        "controls": {"tags": ORPHEUS_TAGS},
        # The Hub licence field is apache-2.0; the card lists meta-llama/Llama-3.2-3B-Instruct as its base model
        # (D-70 answer 5: show the Hub field and note the base, with both links).
        "licence": {
            "id": "apache-2.0", "name": "Apache 2.0", "commercial": "yes",
            "url": "https://huggingface.co/canopylabs/orpheus-3b-0.1-ft",
            "note": "Fine-tuned from Llama 3.2 3B Instruct (Llama 3.2 Community Licence).",
            "baseUrl": "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct",
        },
        "companions": [
            {
                "repo": "mlx-community/snac_24khz", "revision": "556af1cd3b1c5f2d294f6aa9bb886245d7b716ac",
                "files": ["config.json", "model.safetensors"], "sizeGb": 0.08, "role": "loader", "licence": "mit",
            },
            {
                # The loader reads its tokenizer from the bf16 repo by name; only the tokenizer files are fetched.
                "repo": "mlx-community/orpheus-3b-0.1-ft-bf16", "revision": "3786aac51797181edddeb0d89a8253deaee99aab",
                "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"], "sizeGb": 0.02,
                "role": "loader", "licence": "apache-2.0",
            },
        ],
        "sampleRate": 24_000,
        # One worker request per chunk; mlx-audio stops a segment at max_tokens (about 84 codes per second of
        # speech), so the chunk is kept short enough for the budget (measured, see estimates).
        "chunkChars": 250,
        "maxTokens": 2400,
        "minReferenceS": None,
        "estimates": {
            "basis": "measured",
            "measuredOn": "Apple M4 Max, 64 GB, 2026-11-09",
            # scripts/measure-voice-models.py, fresh worker, weights on disk: cold start = first short line minus the same
            # line warm; seconds per 10 s from the fixed 600-character text in this model's chunks; peak and idle are
            # the worker's resident memory (psutil RSS) during the long text and 5 s after.
            # Voice tara, 250-character chunks, max_tokens 2400.
            "coldStartSeconds": 3.6,
            "secondsPer10sAudio": 7.22,
            "wallSecondsShortLine": 3.3,
            "peakGb": 4.19,
            "idleGb": 4.19,
        },
    },
]

_QWEN_DEFAULTS: dict[str, Any] = {
    "family": "qwen3-tts",
    "controls": {},
    "licence": _QWEN_LICENCE,
    "companions": [],
    "sampleRate": 24_000,
    "minReferenceS": 3.0,
}
_CAPABILITIES = ("speak", "style", "clone", "design", "expression", "tags", "presets", "languageSelect")

# The CUDA lane's rows (Windows and Linux with an NVIDIA GPU, Phase 6): same shape, other weights and engines.
from backend_service.catalog.voice_models_cuda import CUDA_VOICE_MODELS  # noqa: E402

VOICE_MODELS.extend(CUDA_VOICE_MODELS)
# The RTX 4090 session lists the unmeasured CUDA rows to measure them; nothing else does.
_LIST_UNMEASURED_CUDA = os.environ.get("CHAOSENGINE_VOICE_UNMEASURED", "").strip() == "1"

for _row in VOICE_MODELS:
    for _name, _value in _QWEN_DEFAULTS.items():
        _row.setdefault(_name, _value)
    _row["capabilities"] = {name: bool(_row["capabilities"].get(name)) for name in _CAPABILITIES}
    _row.setdefault("chunkChars", 400)
    _row.setdefault("rank", None)
    _row.setdefault("builtinWatermark", None)
    # The default and its GPU twin ("-cuda"); the cloning picker takes the one that runs here.
    _row["default"] = _row["key"] in (DEFAULT_CLONE_MODEL, f"{DEFAULT_CLONE_MODEL}-cuda")
    _row["downloadGb"] = round(_row["sizeGb"] + sum(c.get("sizeGb", 0.0) for c in _row["companions"]), 2)
    # A wave-2 row is offered only once it has measured numbers on this hardware (D-70: no unmeasured rows).
    _row["listed"] = (
        _row["family"] == "qwen3-tts"
        or bool((_row.get("estimates") or {}).get("basis") == "measured")
        or (_LIST_UNMEASURED_CUDA and "cuda" in _row["lanes"])
    )

# Every model that can clone. The default (``DEFAULT_CLONE_MODEL``) comes first.
CLONE_MODELS: list[dict[str, Any]] = sorted(
    (row for row in VOICE_MODELS if row["capabilities"]["clone"] and row["listed"]),
    key=lambda row: not row["default"],
)

# CustomVoice preset speakers: id, language code, language name for the model, gender, one-line feel.
CUSTOM_SPEAKERS: list[dict[str, str]] = [
    {"id": "Ryan", "language": "en", "languageLabel": "English", "gender": "male", "about": "Dynamic, strong rhythm"},
    {"id": "Aiden", "language": "en", "languageLabel": "English", "gender": "male", "about": "Sunny American, clear midrange"},
    {"id": "Vivian", "language": "zh", "languageLabel": "Chinese", "gender": "female", "about": "Bright, slightly edgy"},
    {"id": "Serena", "language": "zh", "languageLabel": "Chinese", "gender": "female", "about": "Warm and gentle"},
    {"id": "Uncle_Fu", "language": "zh", "languageLabel": "Chinese", "gender": "male", "about": "Seasoned, low and mellow"},
    {"id": "Dylan", "language": "zh", "languageLabel": "Chinese (Beijing)", "gender": "male", "about": "Youthful Beijing"},
    {"id": "Eric", "language": "zh", "languageLabel": "Chinese (Sichuan)", "gender": "male", "about": "Lively, slightly husky"},
    {"id": "Ono_Anna", "language": "ja", "languageLabel": "Japanese", "gender": "female", "about": "Playful"},
    {"id": "Sohee", "language": "ko", "languageLabel": "Korean", "gender": "female", "about": "Warm, rich emotion"},
]

# Orpheus preset voices, in the authors' order of conversational realism (README); the prompt is "name: text".
ORPHEUS_VOICES: list[dict[str, str]] = [
    {"id": "tara", "gender": "female", "about": "Most natural of the eight (authors' rank 1)"},
    {"id": "leah", "gender": "female", "about": "Authors' rank 2"},
    {"id": "jess", "gender": "female", "about": "Authors' rank 3"},
    {"id": "leo", "gender": "male", "about": "Authors' rank 4"},
    {"id": "dan", "gender": "male", "about": "Authors' rank 5"},
    {"id": "mia", "gender": "female", "about": "Authors' rank 6"},
    {"id": "zac", "gender": "male", "about": "Authors' rank 7"},
    {"id": "zoe", "gender": "female", "about": "Authors' rank 8"},
]


def clone_model(key: str | None) -> dict[str, Any]:
    """The cloning row for ``key``; empty means the default. Raises ValueError for anything else."""
    wanted = (key or "").strip()
    for row in CLONE_MODELS:
        if (not wanted and row["default"]) or wanted in (row["key"], row["repo"]):
            return row
    raise ValueError(f"Unknown voice cloning model '{wanted}'.")


def voice_model(key: str) -> dict[str, Any]:
    """Any row by key or repo id. Raises ValueError for anything else."""
    wanted = (key or "").strip()
    for row in VOICE_MODELS:
        if wanted in (row["key"], row["repo"]):
            return row
    raise ValueError(f"Unknown voice model '{wanted}'.")


def listed_models() -> list[dict[str, Any]]:
    """The rows offered on this build: every Qwen3-TTS row, and wave-2 rows that have measured numbers. CUDA rows
    appear only on the portable lane (Windows, Linux), never on a Mac."""
    from backend_service import voice_env  # noqa: PLC0415

    portable = voice_env.lane() == "portable"
    return [row for row in VOICE_MODELS if row["listed"] and (portable or "cuda" not in row["lanes"])]
