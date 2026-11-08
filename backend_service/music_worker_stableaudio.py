"""The Stable Audio Open Small engine for the music worker (sound effects and loops), PyTorch.

Loaded by path from ``music_worker.py`` inside the sounds venv (``music_sound_env``); imports nothing from
``backend_service`` and nothing heavy until ``load``. The model and its T5 text encoder are read from the
checkpoints layout inside the music data folder (the worker may only touch that folder), with the Hub
offline: ``model_config.json`` is rewritten in memory so the T5 conditioner reads its local folder. The
device is the Apple GPU (MPS) when present, else the CPU (``CHAOSENGINE_SOUND_DEVICE`` forces one); an MPS
failure retries once on the CPU. Progress comes from the sampler's own per-step callback, which is also where
a cancel is raised. Clips are trimmed to the requested length, peak-normalised, and optionally made into a
seamless loop (the tail is cross-faded into the head, so the last sample runs straight into the first).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

# What each model needs that the catalog row does not carry: its folder under ``checkpoints``, its longest clip
# (the catalog's ``maxSeconds`` is the user-facing limit; this is the engine's own guard), whether its
# conditioning also takes ``seconds_start`` (1.0 does, Small does not), and extra sampler arguments (1.0's
# k-diffusion sampler takes the sigma range from its model card; Small's pingpong sampler takes none).
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "stable-audio-open-small": {"dir": "stable-audio-open-small", "max_seconds": 11, "seconds_start": False, "sampler_kwargs": {}},
    "stable-audio-open-1.0": {
        "dir": "stable-audio-open-1.0", "max_seconds": 47, "seconds_start": True,
        "sampler_kwargs": {"sigma_min": 0.3, "sigma_max": 500},
    },
}

# What ``music_worker.check_generate`` validates a request against for this engine: the widest model's range;
# ``generate`` then holds the clip to the loaded model's own maximum.
SECONDS_RANGE = (1, max(int(p["max_seconds"]) for p in MODEL_PROFILES.values()))
TASKS = ("text2sound",)

ENCODER_DIR = "t5-base"
ENGINE_MODEL = "stable-audio-open-small"  # the default when a request names no model
LOOP_FADE_SECONDS = 0.08
EDGE_FADE_SECONDS = 0.008
DECODE_SECONDS_PER_AUDIO_SECOND = 0.1


def make_loop(samples: Any, sample_rate: int, fade_seconds: float = LOOP_FADE_SECONDS) -> Any:
    """A seamless loop from ``samples`` (frames x channels): the last ``fade`` frames are cross-faded into the
    first ``fade`` frames (equal power) and dropped from the end, so the last frame continues into the first."""
    import numpy as np

    frames = samples.shape[0]
    fade = max(1, min(int(fade_seconds * sample_rate), frames // 4))
    if frames < 8:
        return samples
    out = np.array(samples[: frames - fade], dtype=np.float32, copy=True)
    ramp = np.linspace(0.0, np.pi / 2, fade, dtype=np.float32)[:, None]
    head = samples[:fade].astype(np.float32)
    tail = samples[frames - fade:].astype(np.float32)
    out[:fade] = head * np.sin(ramp) + tail * np.cos(ramp)
    return out


def edge_fades(samples: Any, sample_rate: int, seconds: float = EDGE_FADE_SECONDS) -> Any:
    """A few milliseconds of fade at both ends so a trimmed clip never starts or stops with a click."""
    import numpy as np

    fade = min(int(seconds * sample_rate), samples.shape[0] // 2)
    if fade < 2:
        return samples
    out = np.array(samples, dtype=np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


def normalise_peak(samples: Any, peak_db: float | None) -> Any:
    """Scale so the loudest sample sits at ``peak_db`` dBFS (silence is left alone)."""
    import numpy as np

    if peak_db is None:
        return samples
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-6:
        return samples
    return (samples * (10 ** (peak_db / 20.0) / peak)).astype(np.float32)


class StableAudioEngine:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.model: Any = None
        self.name: str | None = None
        self.device = "cpu"
        self._fell_back = False
        self.sample_rate = 44_100
        self.sample_size = 524_288

    # -- availability and loading ---------------------------------------

    def unavailable_reason(self) -> str | None:
        import importlib.util

        for name in ("torch", "soundfile", "stable_audio_tools", "transformers"):
            if importlib.util.find_spec(name) is None:
                return f"The package '{name}' is missing from the sounds environment. Reinstall the sounds pack."
        return None

    def loaded(self) -> dict[str, Any]:
        return {"model": self.name, "lm": None}

    def _pick_device(self) -> str:
        import torch

        forced = os.environ.get("CHAOSENGINE_SOUND_DEVICE", "").strip().lower()
        if forced in ("cpu", "mps", "cuda"):
            return forced
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _model_files(self, model: str) -> tuple[Path, Path, Path]:
        root = self.data_dir / "checkpoints"
        folder = root / MODEL_PROFILES[model]["dir"]
        return folder / "model_config.json", folder / "model.safetensors", root / ENCODER_DIR

    def _build(self, config: dict[str, Any], weights: Path) -> Any:
        """The model with its weights, on the CPU in float32 (moved to the device by the caller)."""
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        model = create_model_from_config(config)
        model.load_state_dict(load_ckpt_state_dict(str(weights)))
        return model

    def load(self, model: str, lm: str | None = None) -> dict[str, Any]:
        if model not in MODEL_PROFILES:
            raise ValueError(f"Unknown sound model {model!r}.")
        if self.model is not None and self.name == model:
            return {"model": model, "lm": None, "already_loaded": True, "device": self.device}
        self.unload()
        config_path, weights, encoder = self._model_files(model)
        for needed in (config_path, weights, encoder / "model.safetensors"):
            if not needed.exists():
                raise FileNotFoundError(f"{needed.name} is missing from {needed.parent}. Download the model from Music > Models.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for entry in (config.get("model", {}).get("conditioning", {}) or {}).get("configs", []):
            if entry.get("type") == "t5":
                entry.setdefault("config", {})["model_path"] = str(encoder)
        self.sample_rate = int(config.get("sample_rate") or self.sample_rate)
        self.sample_size = int(config.get("sample_size") or self.sample_size)
        loaded = self._build(config, weights)
        self.device = self._pick_device()
        self.model = loaded.to(self.device).eval().requires_grad_(False)
        self.name = model
        return {"model": model, "lm": None, "device": self.device, "sample_rate": self.sample_rate}

    def unload(self) -> None:
        self.model, self.name = None, None
        gc.collect()
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - nothing to release
            pass

    # -- generation -----------------------------------------------------

    def _run_one(self, request: dict[str, Any], seed: int, seconds: float, hooks: Any) -> Any:
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        profile = MODEL_PROFILES[self.name or ENGINE_MODEL]
        steps = int(request.get("steps") or 8)
        conditioning: dict[str, Any] = {"prompt": str(request.get("caption") or ""), "seconds_total": seconds}
        if profile["seconds_start"]:
            conditioning["seconds_start"] = 0

        def callback(info: Any) -> None:
            done = int(info["i"]) + 1 if isinstance(info, dict) else 0
            hooks.step(done, steps)
            hooks.check_cancel()
            if done >= steps:
                hooks.stage("decode", DECODE_SECONDS_PER_AUDIO_SECOND * seconds)

        hooks.check_cancel()
        if self.device == "cpu":
            hooks.device("cpu", fallback=self._fell_back)
        with torch.inference_mode():
            audio = generate_diffusion_cond(
                self.model,
                steps=steps,
                cfg_scale=float(request.get("cfg_scale") or 1.0),
                conditioning=[conditioning],
                sample_size=self.sample_size,
                sample_rate=self.sample_rate,
                sampler_type=str(request.get("sampler") or "pingpong"),
                device=self.device,
                seed=seed,
                batch_size=1,
                callback=callback,
                **profile["sampler_kwargs"],
            )
        return audio

    def _generate_audio(self, request: dict[str, Any], seed: int, seconds: float, hooks: Any) -> Any:
        try:
            return self._run_one(request, seed, seconds, hooks)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "Cancelled" or self.device == "cpu":
                raise
            # One retry on the CPU when the Apple GPU could not run an operation.
            self.model = self.model.to("cpu")
            self.device = "cpu"
            self._fell_back = True
            hooks.check_cancel()
            return self._run_one(request, seed, seconds, hooks)

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

        model = str(request.get("model") or ENGINE_MODEL)
        if self.model is None or self.name != model:
            hooks.stage("load")
            self.load(model)
        hooks.check_cancel()
        hooks.stage("encode")
        seconds = float(request["seconds"])
        limit = float(MODEL_PROFILES[model]["max_seconds"])
        if seconds > limit:
            raise ValueError(f"{model} makes clips of up to {limit:g} s; {seconds:g} s was asked for.")
        variations = int(request.get("variations") or 1)
        base_seed = int(request["seed"])
        loop = bool(request.get("loop"))
        peak_db = request.get("normalize_peak_db")
        out_dir = Path(str(request["out_dir"]))
        tracks: list[dict[str, Any]] = []
        for index in range(variations):
            seed = base_seed + index
            audio = self._generate_audio(request, seed, seconds, hooks)
            hooks.check_cancel()
            hooks.stage("write")
            samples = audio[0].detach().float().cpu().numpy().T  # frames x channels
            samples = np.clip(samples[: int(seconds * self.sample_rate)], -1.0, 1.0)
            samples = make_loop(samples, self.sample_rate) if loop else edge_fades(samples, self.sample_rate)
            samples = np.clip(normalise_peak(samples, float(peak_db) if peak_db is not None else None), -1.0, 1.0)
            path = out_dir / f"{request['job']}_{index}.flac"
            sf.write(str(path), samples, self.sample_rate, subtype="PCM_24", format="FLAC")
            tracks.append({
                "path": str(path),
                "seed": seed,
                "sample_rate": self.sample_rate,
                "channels": int(samples.shape[1]) if samples.ndim == 2 else 1,
                "duration_s": round(samples.shape[0] / self.sample_rate, 3),
                "loop": loop,
            })
        return {"tracks": tracks, "timings": {"device": self.device}}


def create_engine(data_dir: Path) -> StableAudioEngine:
    return StableAudioEngine(data_dir)
