"""The ACE-Step 1.5 engine for the music worker (MLX on Apple Silicon, PyTorch on CUDA), at the pinned commit.

Loaded by path from ``music_worker.py`` inside the music venv; imports nothing from ``backend_service``
and nothing heavy until ``load``. On MLX it applies the three memory settings from the spike (VAE chunk 256
via the environment, a 2 GB MLX cache limit, dropping the torch copy of the DiT decoder once MLX runs it)
and reports real diffusion steps by wrapping the pinned private name ``acestep.models.mlx.dit_generate.tqdm``.
On CUDA (FU-139) the DiT runs in PyTorch bf16 with no ``tqdm`` in its loop, so steps come from forward hooks
on ``handler.model.decoder`` (one decoder call per ODE step); everything stays on the GPU when enough VRAM is free
for the model (``CUDA_FULL_GPU_NEED_GB``) and ACE-Step's own CPU offload is used otherwise. Either hook cancels by raising; an unexpected layout
leaves stage-only progress, never a crash. FLAC 24-bit 48 kHz is written here from the returned tensor.
"""

from __future__ import annotations

import gc
import importlib
import os
import platform
from pathlib import Path
from typing import Any, Callable

STEP_HOOK_MODULE = "acestep.models.mlx.dit_generate"
MLX_CACHE_LIMIT_BYTES = 2 * 1024 ** 3
SAMPLE_RATE = 48_000
# Free VRAM a model needs to keep the DiT, text encoder and VAE on the card together (measured on the 4090:
# turbo peaks at 8.6 GB reserved; XL holds 11.5 GB once loaded, plus activations). Below it, ACE-Step's own CPU
# offload moves the text encoder and VAE off the card between uses; the DiT stays on it.
CUDA_FULL_GPU_NEED_GB = {"acestep-v15-turbo": 10.0, "acestep-v15-xl-turbo": 16.0}
CUDA_DEFAULT_NEED_GB = 20.0


def cuda_offload(model: str, free_gb: float, forced: str = "") -> bool:
    """True when the model should load with CPU offload: forced by the setting, or too little free VRAM now."""
    forced = forced.strip().lower()
    if forced:
        return forced in ("1", "true", "yes")
    return free_gb < CUDA_FULL_GPU_NEED_GB.get(model, CUDA_DEFAULT_NEED_GB)
# Total VRAM a run holds on the card at its peak (torch reserved, measured on the 4090: turbo 8.1 to 8.6 GB, XL turbo
# 12.4 GB, 14.3 GB for a 10 minute track). A run that cannot get it fails inside ACE-Step with a message about batch
# size and duration, so the worker checks first and says what is wrong.
CUDA_RUN_PEAK_GB = {"acestep-v15-turbo": 8.7, "acestep-v15-xl-turbo": 12.5}
CUDA_RUN_LONG_EXTRA_GB = 2.0
CUDA_RUN_LONG_SECONDS = 300.0


def vram_shortfall_message(model: str, seconds: float, free_gb: float, held_gb: float) -> str | None:
    """A plain sentence when the card lacks room for this run, else None.

    ``held_gb`` is what this process already has reserved on the card (the loaded model), ``free_gb`` what the
    driver reports free besides that. Unknown models are left to ACE-Step's own check.
    """
    peak = CUDA_RUN_PEAK_GB.get(model)
    if peak is None:
        return None
    if seconds > CUDA_RUN_LONG_SECONDS:
        peak += CUDA_RUN_LONG_EXTRA_GB
    extra = max(0.0, peak - held_gb)
    if free_gb >= extra:
        return None
    name = "ACE-Step XL" if "xl" in model else "ACE-Step"
    advice = "Close it, or use the turbo model (it needs about 9 GB)." if "xl" in model else "Close it and try again."
    return (
        f"Not enough free GPU memory to run {name}: it needs about {peak:.1f} GB on the card, this app already "
        f"holds {held_gb:.1f} GB and only {free_gb:.1f} GB more is free. Another program may be using the GPU. {advice}"
    )
# The worker runs in its own process and imports nothing from the backend, so this sentence is a copy of
# ``music_env.UNSUPPORTED_PLATFORM_REASON``; tests/test_music_worker.py keeps the two equal.
UNSUPPORTED_PLATFORM_REASON = (
    "Music generation needs an Apple Silicon Mac or an NVIDIA GPU on Windows or Linux; "
    "nothing is installed or downloaded on this computer."
)
NO_CUDA_REASON = (
    "The music environment sees no NVIDIA GPU (CUDA). Update the NVIDIA driver, or reinstall the music pack."
)


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")

# Who receives step events right now: ``(before_step(index, total), after_step(done, total))``.
_ACTIVE: dict[str, Callable[[int, int], None] | None] = {"before": None, "after": None}


def install_step_hook(module_name: str = STEP_HOOK_MODULE) -> bool:
    """Wrap the module's ``tqdm`` so each diffusion step calls the active hooks. False when it cannot.

    The wrapper materialises the iterable, calls ``before`` (which may raise to cancel) and ``after``
    around each item, and falls through to the original ``tqdm`` when nothing is listening.
    """
    try:
        module = importlib.import_module(module_name)
        original = getattr(module, "tqdm")
    except Exception:  # noqa: BLE001 - any layout change means stage-only progress
        return False
    if not callable(original) or getattr(original, "_chaosengine_hook", False):
        return bool(getattr(original, "_chaosengine_hook", False))

    def hooked(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
        before, after = _ACTIVE["before"], _ACTIVE["after"]
        if iterable is None or (before is None and after is None):
            return original(iterable, *args, **kwargs)
        items = list(iterable)
        total = len(items)

        def steps() -> Any:
            for index, item in enumerate(items):
                if before is not None:
                    before(index, total)
                yield item
                if after is not None:
                    after(index + 1, total)

        return steps()

    hooked._chaosengine_hook = True  # type: ignore[attr-defined]
    try:
        setattr(module, "tqdm", hooked)
    except Exception:  # noqa: BLE001
        return False
    return True


def attach_decoder_hooks(decoder: Any, total: int) -> list[Any]:
    """Forward hooks on the torch DiT decoder: one call is one ODE step. Returns the handles to remove.

    ``before`` runs ahead of each call (it may raise to cancel), ``after`` once it returns. Calls beyond
    ``total`` (a sampler that evaluates twice) are reported as the last step, never past it.
    """
    count = {"n": 0}

    def pre(_module: Any, _args: Any) -> None:
        before = _ACTIVE["before"]
        if before is not None:
            before(min(count["n"], total - 1), total)

    def post(_module: Any, _args: Any, _output: Any) -> None:
        count["n"] += 1
        after = _ACTIVE["after"]
        if after is not None and count["n"] <= total:
            after(count["n"], total)

    try:
        return [decoder.register_forward_pre_hook(pre), decoder.register_forward_hook(post)]
    except Exception:  # noqa: BLE001 - no hooks means stage-only progress
        return []


def _stage_for(desc: Any) -> str | None:
    text = str(desc or "").lower()
    if "preparing inputs" in text:
        return "encode"
    if "generating" in text:
        return "diffusion"
    if "decoding" in text:
        return "decode"
    return None


class AceStepEngine:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.handler: Any = None
        self.llm: Any = None
        self.model: str | None = None
        self.lm: str | None = None
        self.step_hook = False
        self.decoder_dropped = False
        self.cuda = not is_apple_silicon()
        self.offload = False

    # -- availability and loading ---------------------------------------

    def unavailable_reason(self) -> str | None:
        import importlib.util

        if self.cuda and platform.system() not in ("Windows", "Linux"):
            return UNSUPPORTED_PLATFORM_REASON
        for name in ("torch", "acestep", "soundfile") if self.cuda else ("mlx", "torch", "acestep", "soundfile"):
            if importlib.util.find_spec(name) is None:
                return f"The package '{name}' is missing from the music environment. Reinstall the music pack."
        if self.cuda:
            import torch

            if not torch.cuda.is_available():
                return NO_CUDA_REASON
        return None

    def loaded(self) -> dict[str, Any]:
        return {"model": self.model, "lm": self.lm}

    def load(self, model: str, lm: str | None = None) -> dict[str, Any]:
        if not model:
            raise ValueError("model is required.")
        if self.handler is not None and self.model == model and self.lm == lm:
            return {"model": model, "lm": lm, "already_loaded": True, "step_hook": self.step_hook}
        self.unload()
        if self.cuda:
            return self._load_cuda(model, lm)
        import mlx.core as mx
        import torch

        mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        self.step_hook = install_step_hook()
        handler = AceStepHandler()
        status, ok = handler.initialize_service(project_root=str(self.data_dir), config_path=model, device="auto")
        if not ok:
            raise RuntimeError(f"ACE-Step could not load {model}: {str(status)[:400]}")
        self.decoder_dropped = False
        if getattr(handler, "mlx_decoder", None) is not None:
            # Only lyric scoring and the silent PyTorch fallback use it; saves 4 to 5 GB and turns a
            # failed MLX run (or a cancel) into an error instead of a slow float32 rerun.
            handler.model.decoder.to("meta")
            gc.collect()
            torch.mps.empty_cache()
            self.decoder_dropped = True
        llm = LLMHandler()
        if lm:
            lm_status, lm_ok = llm.initialize(
                checkpoint_dir=str(self.data_dir / "checkpoints"), lm_model_path=lm, backend="mlx", device="auto"
            )
            if not lm_ok:
                raise RuntimeError(f"The planner {lm} could not load: {str(lm_status)[:300]}")
        self.handler, self.llm, self.model, self.lm = handler, llm, model, lm
        return {
            "model": model,
            "lm": lm,
            "mlx_dit": getattr(handler, "mlx_decoder", None) is not None,
            "decoder_dropped": self.decoder_dropped,
            "step_hook": self.step_hook,
            "device": "mps",
        }

    def _load_cuda(self, model: str, lm: str | None) -> dict[str, Any]:
        """The PyTorch path: bf16 on the GPU, ACE-Step's CPU offload only on cards under 20 GB."""
        import torch
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        # Free, not total: other apps (or another model) may already hold part of the card.
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        self.offload = cuda_offload(model, free_gb, os.environ.get("CHAOSENGINE_MUSIC_OFFLOAD", ""))
        handler = AceStepHandler()
        status, ok = handler.initialize_service(
            project_root=str(self.data_dir), config_path=model, device="cuda", use_mlx_dit=False,
            offload_to_cpu=self.offload, offload_dit_to_cpu=False,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step could not load {model}: {str(status)[:400]}")
        # Loading leaves several GB cached in torch's allocator (XL: 10 GB allocated, 21 GB held), and ACE-Step's
        # pre-flight reads the driver's free VRAM, so it refused every XL run until the cache was handed back.
        gc.collect()
        torch.cuda.empty_cache()
        self.decoder_dropped = False
        self.step_hook = getattr(getattr(handler, "model", None), "decoder", None) is not None
        llm = LLMHandler()
        if lm:
            lm_status, lm_ok = llm.initialize(
                checkpoint_dir=str(self.data_dir / "checkpoints"), lm_model_path=lm, backend="pt", device="cuda"
            )
            if not lm_ok:
                raise RuntimeError(f"The planner {lm} could not load: {str(lm_status)[:300]}")
        self.handler, self.llm, self.model, self.lm = handler, llm, model, lm
        return {
            "model": model,
            "lm": lm,
            "mlx_dit": False,
            "decoder_dropped": False,
            "step_hook": self.step_hook,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(getattr(handler, "dtype", "")).replace("torch.", ""),
            "offload": self.offload,
            "free_vram_before_gb": round(free_gb, 2),
            "vram_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
        }

    def unload(self) -> None:
        handler, self.handler, self.llm, self.model, self.lm = self.handler, None, None, None, None
        if handler is not None:
            for name in ("model", "mlx_decoder", "mlx_dit", "vae", "text_encoder", "mlx_vae"):
                try:
                    setattr(handler, name, None)
                except Exception:  # noqa: BLE001
                    pass
            del handler
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch

            if self.cuda:
                torch.cuda.empty_cache()
            else:
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- generation -----------------------------------------------------

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        model = str(request.get("model") or "acestep-v15-turbo")
        lm = request.get("lm") or None
        if self.handler is None or self.model != model or self.lm != lm:
            hooks.stage("load")
            self.load(model, lm)
        hooks.check_cancel()
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        seconds = float(request["seconds"])
        variations = int(request.get("variations") or 1)
        seed = int(request["seed"])
        seeds = [seed + index for index in range(variations)]
        instrumental = bool(request.get("instrumental"))
        peak_db = request.get("normalize_peak_db")
        params = GenerationParams(
            task_type="text2music",
            caption=str(request.get("caption") or ""),
            lyrics="[Instrumental]" if instrumental else str(request.get("lyrics") or ""),
            instrumental=instrumental,
            vocal_language=str(request.get("vocal_language") or "unknown"),
            bpm=int(request["bpm"]) if request.get("bpm") else None,
            keyscale=str(request.get("key") or ""),
            timesignature=str(request.get("time_signature") or ""),
            duration=seconds,
            inference_steps=int(request.get("steps") or 8),
            shift=float(request.get("shift") or 3.0),
            infer_method="ode",
            seed=seed,
            thinking=bool(request.get("planner")),
            enable_normalization=peak_db is not None,
            normalization_db=float(peak_db) if peak_db is not None else -1.0,
        )
        config = GenerationConfig(batch_size=variations, use_random_seed=False, seeds=seeds, audio_format="flac")
        decode_estimate = 0.06 * seconds * variations

        def before(index: int, _total: int) -> None:
            hooks.check_cancel()

        def after(done: int, total: int) -> None:
            hooks.step(done, total)
            if done == total:
                hooks.stage("decode", decode_estimate)

        def progress(_value: Any, desc: Any = None, **_kwargs: Any) -> None:
            stage = _stage_for(desc)
            if stage == "decode":
                hooks.stage("decode", decode_estimate)
            elif stage is not None:
                hooks.stage(stage)

        handles: list[Any] = []
        torch = None
        if self.cuda:
            import torch

            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            problem = vram_shortfall_message(model, seconds, free_gb, torch.cuda.memory_reserved() / 1e9)
            if problem:
                raise RuntimeError(problem)
            torch.cuda.reset_peak_memory_stats()
            decoder = getattr(getattr(self.handler, "model", None), "decoder", None)
            if decoder is not None:
                handles = attach_decoder_hooks(decoder, int(params.inference_steps))
        _ACTIVE["before"], _ACTIVE["after"] = before, after
        try:
            result = generate_music(self.handler, self.llm, params, config, save_dir=None, progress=progress)
        finally:
            _ACTIVE["before"], _ACTIVE["after"] = None, None
            for handle in handles:
                handle.remove()
        hooks.check_cancel()
        if not getattr(result, "success", False):
            raise RuntimeError(str(getattr(result, "error", None) or "ACE-Step generation failed."))
        hooks.stage("write")
        timings = self._timings(result)
        if torch is not None:
            timings["vram_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
            timings["vram_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        return {"tracks": self._write(request, result, seeds), "timings": timings}

    def _write(self, request: dict[str, Any], result: Any, seeds: list[int]) -> list[dict[str, Any]]:
        import numpy as np
        import soundfile as sf

        out_dir = Path(str(request["out_dir"]))
        tracks = []
        for index, audio in enumerate(result.audios):
            tensor = audio.get("tensor")
            if tensor is None:
                continue
            samples = np.clip(tensor.detach().float().cpu().numpy().T, -1.0, 1.0)
            rate = int(audio.get("sample_rate") or SAMPLE_RATE)
            path = out_dir / f"{request['job']}_{index}.flac"
            sf.write(str(path), samples, rate, subtype="PCM_24", format="FLAC")
            tracks.append({
                "path": str(path),
                "seed": seeds[index] if index < len(seeds) else None,
                "sample_rate": rate,
                "channels": int(samples.shape[1]) if samples.ndim == 2 else 1,
                "duration_s": round(samples.shape[0] / rate, 3),
                "bpm": request.get("bpm"),
                "key": request.get("key"),
            })
        return tracks

    @staticmethod
    def _timings(result: Any) -> dict[str, Any]:
        costs = (getattr(result, "extra_outputs", None) or {}).get("time_costs") or {}
        return {
            "diffusion_s": round(float(costs.get("dit_diffusion_time_cost") or 0.0), 2),
            "vae_decode_s": round(float(costs.get("dit_vae_decode_time_cost") or 0.0), 2),
        }


def create_engine(data_dir: Path) -> AceStepEngine:
    return AceStepEngine(data_dir)
