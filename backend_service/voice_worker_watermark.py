"""Hidden watermark for cloned-voice audio, inside the voice worker (D-47): Meta's AudioSeal.

AudioSeal works on 16 kHz mono. ``embed`` resamples the speech to 16 kHz, asks the generator for the
watermark signal carrying a fixed 16-bit ChaosEngineAI message, resamples that signal back to the
file's own rate and adds it, so the speech itself is never resampled. ``detect`` resamples any file to
16 kHz and reports the detector's probability and the decoded message.

The weights are the pinned files the backend downloaded (``voice_env.WATERMARK_*``); they are loaded
from those local paths through AudioSeal's own model cards, never fetched here. Imports only numpy,
torch, librosa and audioseal from the voice venv; nothing from ``backend_service`` except this sibling.
"""

from __future__ import annotations

import os
from typing import Any

SAMPLE_RATE = 16_000
NBITS = 16
# "CE" "5A": a fixed message, so a detector can tell ChaosEngineAI output from other AudioSeal users.
MESSAGE = 0xCE5A
_CARDS = {"generator": "audioseal_wm_16bits", "detector": "audioseal_detector_16bits"}
_models: dict[str, Any] = {}


def message_bits(value: int = MESSAGE) -> list[int]:
    """Most significant bit first."""
    return [(value >> (NBITS - 1 - i)) & 1 for i in range(NBITS)]


def bits_value(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (int(bit) & 1)
    return value


def _device() -> str:
    import torch  # type: ignore[import-untyped]

    if os.environ.get("CHAOSENGINE_WATERMARK_DEVICE"):
        return os.environ["CHAOSENGINE_WATERMARK_DEVICE"]
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(role: str, path: str) -> Any:
    """AudioSeal model from its packaged card, with the card's checkpoint URL swapped for ``path``."""
    key = (role, path)
    if _models.get(role, (None, None))[0] == key:
        return _models[role][1]
    import audioseal.loader as loader  # type: ignore[import-untyped]

    card = loader.load_local_model_config(_CARDS[role])
    if card is None:
        raise RuntimeError(f"AudioSeal card {_CARDS[role]} is missing; reinstall the watermark component.")
    card.checkpoint = str(path)
    original = loader.load_local_model_config
    loader.load_local_model_config = lambda name: card if name == "chaosengine-pinned" else original(name)
    try:
        builder = loader.AudioSeal.load_generator if role == "generator" else loader.AudioSeal.load_detector
        model = builder("chaosengine-pinned", nbits=NBITS)
    finally:
        loader.load_local_model_config = original
    model = model.to(_device()).eval()
    _models[role] = (key, model)
    return model


def _resample(samples, source_rate: int, target_rate: int):
    import librosa  # type: ignore[import-untyped]
    import numpy as np

    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    return librosa.resample(np.asarray(samples, dtype=np.float32), orig_sr=source_rate, target_sr=target_rate).astype(np.float32)


def watermark_samples(generator: Any, samples, sample_rate: int):
    """``samples`` (mono float32 at ``sample_rate``) with the watermark added, same length and rate."""
    import numpy as np
    import torch  # type: ignore[import-untyped]

    low = _resample(samples, sample_rate, SAMPLE_RATE)
    device = next(generator.parameters()).device
    with torch.no_grad():
        audio = torch.from_numpy(low).to(device)[None, None, :]
        message = torch.tensor([message_bits()], dtype=torch.int32, device=device)
        mark = generator.get_watermark(audio, message=message)[0, 0].float().cpu().numpy()
    mark = _resample(mark, SAMPLE_RATE, sample_rate)
    out = np.asarray(samples, dtype=np.float32).copy()
    n = min(len(out), len(mark))
    out[:n] += mark[:n]
    return np.clip(out, -1.0, 1.0)


def detect_samples(detector: Any, samples, sample_rate: int) -> dict[str, Any]:
    import torch  # type: ignore[import-untyped]

    low = _resample(samples, sample_rate, SAMPLE_RATE)
    device = next(detector.parameters()).device
    with torch.no_grad():
        prob, message = detector.detect_watermark(torch.from_numpy(low).to(device)[None, None, :])
    bits = [int(b) for b in message[0].cpu().tolist()]
    value = bits_value(bits)
    return {"probability": round(float(prob[0]), 4), "message": f"{value:04X}", "matches": value == MESSAGE}


def embed(request: dict[str, Any], decode_audio, wav_bytes) -> dict[str, Any]:
    """Watermark the WAV at ``path`` in place (``out`` defaults to ``path``)."""
    import wave

    path = str(request["path"])
    with wave.open(path, "rb") as handle:
        rate = handle.getframerate()
    samples = decode_audio(path, rate)
    marked = watermark_samples(_load("generator", str(request["generator"])), samples, rate)
    out = str(request.get("out") or path)
    with open(out, "wb") as handle:
        handle.write(wav_bytes(marked, rate))
    return {"out": out, "sampleRate": rate}


def detect(request: dict[str, Any], decode_audio) -> dict[str, Any]:
    samples = decode_audio(str(request["path"]), SAMPLE_RATE)
    return detect_samples(_load("detector", str(request["detector"])), samples, SAMPLE_RATE)


def unload() -> None:
    _models.clear()
