"""The Stable Audio engine for the music worker (sound effects, loops, short instrumentals), PyTorch.

Loaded by path from ``music_worker.py`` inside the sounds venv (``music_sound_env``); imports nothing from
``backend_service`` and nothing heavy until ``load``. Two libraries: Stable Audio Open (Small, 1.0) runs on
``stable-audio-tools``, Stable Audio 3 (small-sfx, small-music, medium) on the ``stable-audio-3`` library. The
model and its text encoder are read from the checkpoints layout inside the music data folder (the worker may
only touch that folder), with the Hub offline: ``model_config.json`` is rewritten in memory so the text
conditioner (T5 or T5Gemma) reads its local folder. The device is the NVIDIA GPU (CUDA) or the Apple GPU (MPS)
when present, else the CPU (``CHAOSENGINE_SOUND_DEVICE`` forces one); a GPU failure retries once on the CPU.
Stable Audio 3 runs in float16 on CUDA, as its authors' loader does. Progress comes from the sampler's own
per-step callback, which is also where a cancel is raised. Clips are trimmed to the requested length,
peak-normalised, and optionally made into a seamless loop (the tail is cross-faded into the head, so the last
sample runs straight into the first).
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
    # Stable Audio 3: variable length (the work scales with the clip), the window from each config's sample_size
    # (5,292,032 samples = 120 s for the small models, 16,777,216 = 380 s for medium). ``vram_gb`` is the measured
    # CUDA peak at the longest clip the catalog offers plus headroom; a run is refused with a plain sentence when
    # less VRAM than that is free (another program on the GPU).
    "stable-audio-3-small-sfx": {"dir": "stable-audio-3-small-sfx", "max_seconds": 120, "library": "sa3", "vram_gb": 3.0},
    "stable-audio-3-small-music": {"dir": "stable-audio-3-small-music", "max_seconds": 120, "library": "sa3", "vram_gb": 3.0},
    "stable-audio-3-medium": {"dir": "stable-audio-3-medium", "max_seconds": 380, "library": "sa3", "vram_gb": 9.0},
}
SA3_ENCODER_DIR = "t5gemma-b-b-ul2"  # inside each Stable Audio 3 repo

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


def vram_shortfall_message(model: str, free_gb: float, held_gb: float) -> str | None:
    """A plain sentence when the GPU has too little free memory for ``model`` (counting what this worker already
    holds), else ``None``. Only rows with a ``vram_gb`` figure are checked."""
    need = MODEL_PROFILES.get(model, {}).get("vram_gb")
    if not need or free_gb + held_gb >= float(need):
        return None
    return (f"The GPU has {free_gb:.1f} GB free and this model needs about {float(need):.0f} GB while it works. "
            "Another program may be using the GPU: close it, or pick a smaller sound model.")


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

        import sys

        # Stable Audio 3 is installed on the CUDA lane only (FU-493); a Mac pack has no ``stable_audio_3``.
        names = ["torch", "soundfile", "stable_audio_tools", "transformers"]
        if sys.platform != "darwin":
            names.append("stable_audio_3")
        for name in names:
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
        encoder = folder / SA3_ENCODER_DIR if MODEL_PROFILES[model].get("library") == "sa3" else root / ENCODER_DIR
        return folder / "model_config.json", folder / "model.safetensors", encoder

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
            if entry.get("type") in ("t5", "t5gemma"):
                settings = entry.setdefault("config", {})
                settings["model_path"] = str(encoder)
                # Stable Audio 3 names its own Hub repo and subfolder; the local folder replaces both (offline).
                settings.pop("repo_id", None)
                settings.pop("subfolder", None)
        self.sample_rate = int(config.get("sample_rate") or self.sample_rate)
        self.sample_size = int(config.get("sample_size") or self.sample_size)
        self.device = self._pick_device()
        if MODEL_PROFILES[model].get("library") == "sa3":
            self.model = self._build_sa3(config, weights)
        else:
            loaded = self._build(config, weights)
            self.model = loaded.to(self.device).eval().requires_grad_(False)
        self.name = model
        return {"model": model, "lm": None, "device": self.device, "sample_rate": self.sample_rate}

    def _build_sa3(self, config: dict[str, Any], weights: Path) -> Any:
        """A ``stable_audio_3.StableAudioModel`` on the device; float16 on CUDA (the authors' loader default)."""
        import torch
        from stable_audio_3 import StableAudioModel
        from stable_audio_3.loading_utils import load_diffusion_cond

        half = self.device == "cuda"
        model = load_diffusion_cond(config, str(weights), device=self.device, model_half=half)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return StableAudioModel(model, config, self.device, half)

    def unload(self) -> None:
        self.model, self.name = None, None
        gc.collect()
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - nothing to release
            pass

    # -- generation -----------------------------------------------------

    def _run_one(self, request: dict[str, Any], seed: int, seconds: float, hooks: Any) -> Any:
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        profile = MODEL_PROFILES[self.name or ENGINE_MODEL]
        steps = int(request.get("steps") or 8)
        conditioning: dict[str, Any] = {"prompt": str(request.get("caption") or ""), "seconds_total": seconds}
        if profile.get("seconds_start"):
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
        if profile.get("library") == "sa3":
            # Variable length: the library pads the clip by a few seconds and trims back to ``duration``.
            return self.model.generate(
                prompt=str(request.get("caption") or ""),
                duration=seconds,
                steps=steps,
                cfg_scale=float(request.get("cfg_scale") or 1.0),
                sample_size=self.sample_size,
                seed=seed,
                sampler_type=str(request.get("sampler") or "pingpong"),
                callback=callback,
                disable_tqdm=True,
            )
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
            if type(exc).__name__ == "Cancelled" or self.device in ("cpu", "cuda"):
                raise  # a CUDA failure (out of memory, a driver error) is reported, not retried for minutes on the CPU
            if MODEL_PROFILES[self.name or ENGINE_MODEL].get("library") == "sa3":
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
        if self.device == "cuda":
            import torch

            free_bytes, _total = torch.cuda.mem_get_info()
            problem = vram_shortfall_message(model, free_bytes / 1e9, torch.cuda.memory_reserved() / 1e9)
            if problem:
                raise RuntimeError(problem)
            torch.cuda.reset_peak_memory_stats()
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
        timings: dict[str, Any] = {"device": self.device}
        if self.device == "cuda":
            import torch

            timings["vram_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            timings["vram_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return {"tracks": tracks, "timings": timings}


def create_engine(data_dir: Path) -> StableAudioEngine:
    return StableAudioEngine(data_dir)
