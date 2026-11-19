"""The speech-to-text GPU add-on (``stt-gpu``): what it installs into the voice venv and how its worker starts.

Qwen3-ASR, Parakeet TDT v3, Nemotron 3.5 ASR, Granite Speech 5.0 and Nemotron 3.5 Transcribe run natively in
Transformers 5.23 (``voice_worker_asr``). The speech GPU add-on pins Transformers 5.2 and safetensors 0.5.3 for
Chatterbox, and 5.23 needs newer tokenizers and safetensors, so, as for Qwen3-TTS, this add-on's Transformers goes
into its own folder inside the voice venv (``asr-site``, ``--no-deps``) and only the ASR worker puts it first on
``PYTHONPATH``. PyTorch (the same CUDA 12.8 build as the speech GPU add-on, shared when both are ticked) and its
torchaudio (Granite's feature extractor) go into the venv itself. huggingface-hub, numpy, tqdm, pyyaml, packaging and
filelock come from the base voice pack. All Apache-2.0 except regex (Apache-2.0 AND CNRI-Python) and torch (BSD-3).
"""

from __future__ import annotations

from pathlib import Path

from backend_service import voice_env

# Checked on the RTX 4090 run (2026-11-19). transformers 5.23.0 requires tokenizers >=0.23.1,<0.24, safetensors >=0.8
# and regex >=2025.10.22; huggingface-hub >=1.31,<3 is the venv's own (2.x on a base pack, 1.33 once the speech GPU
# add-on's Transformers 5.2 has pulled it below 2.0).
ASR_SITE_PACKAGES = ["transformers==5.23.0", "tokenizers==0.23.3", "safetensors==0.8.0", "regex==2026.11.19"]
_ASR_SITE_CHECK = (
    "import sys; sys.path.insert(0, {site!r}); import transformers, torch, torchaudio; "
    "assert transformers.__version__ == '5.23.0', transformers.__version__; "
    "from transformers import AutoModelForMultimodalLM, AutoModelForTDT, AutoModelForRNNT; "
    "assert torch.cuda.is_available(), 'PyTorch cannot see the NVIDIA GPU'; print('asr ok')"
)


def asr_site_dir() -> Path:
    """Where the add-on's Transformers lives: inside the voice venv, so removing the venv or the add-on removes it."""
    return voice_env.VOICE_VENV_DIR / "asr-site"


def asr_site_installed() -> bool:
    site = asr_site_dir()
    return (site / "transformers").is_dir() and any(site.glob("transformers-5.23.0*"))


def stt_gpu_installed() -> bool:
    return voice_env.is_installed() and "stt-gpu" in voice_env.installed_extras() and asr_site_installed()


def stt_gpu_ready() -> bool:
    """The add-on is installed and this machine can run it (an NVIDIA GPU on Windows or Linux x86-64)."""
    return stt_gpu_installed() and voice_env.speech_gpu_supported()[0]


def asr_worker_env() -> dict[str, str]:
    """Environment of the ASR worker: the add-on's Transformers first on the import path, never the backend's own
    ``PYTHONPATH`` (another torch and Transformers). Offline: models load from the snapshots the app downloaded."""
    return {"PYTHONPATH": str(asr_site_dir()), "HF_HUB_OFFLINE": "1"}


def stt_gpu_commands(python: str) -> list[list[str]]:
    """The add-on's install commands, in order, then an import check that also asks PyTorch for the GPU."""
    pip = [python, "-m", "pip", "install"]
    return [
        [*pip, *voice_env.SPEECH_GPU_TORCH, "--index-url", voice_env.SPEECH_GPU_TORCH_INDEX],
        [*pip, "--target", str(asr_site_dir()), "--no-deps", "--upgrade", *ASR_SITE_PACKAGES],
        [python, "-I", "-c", _ASR_SITE_CHECK.format(site=str(asr_site_dir()))],
    ]
