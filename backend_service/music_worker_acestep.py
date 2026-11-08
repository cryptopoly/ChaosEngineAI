"""The ACE-Step 1.5 engine for the music worker (MLX on Apple Silicon), at the pinned commit.

Loaded by path from ``music_worker.py`` inside the music venv; imports nothing from ``backend_service``
and nothing heavy until ``load``. It applies the three memory settings from the spike (VAE chunk 256 via
the environment, a 2 GB MLX cache limit, dropping the torch copy of the DiT decoder once MLX runs it),
reports real diffusion steps by wrapping the pinned private name ``acestep.models.mlx.dit_generate.tqdm``
(an unexpected layout leaves stage-only progress, never a crash), cancels by raising from that hook, and
writes FLAC 24-bit 48 kHz itself from the returned tensor.
"""

from __future__ import annotations

import gc
import importlib
import platform
from pathlib import Path
from typing import Any, Callable

STEP_HOOK_MODULE = "acestep.models.mlx.dit_generate"
MLX_CACHE_LIMIT_BYTES = 2 * 1024 ** 3
SAMPLE_RATE = 48_000

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

    # -- availability and loading ---------------------------------------

    def unavailable_reason(self) -> str | None:
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            return ("ACE-Step runs on Apple Silicon (MLX) in this version; the Windows and Linux (CUDA) "
                    "lane is later work.")
        import importlib.util

        for name in ("mlx", "torch", "acestep", "soundfile"):
            if importlib.util.find_spec(name) is None:
                return f"The package '{name}' is missing from the music environment. Reinstall the music pack."
        return None

    def loaded(self) -> dict[str, Any]:
        return {"model": self.model, "lm": self.lm}

    def load(self, model: str, lm: str | None = None) -> dict[str, Any]:
        if not model:
            raise ValueError("model is required.")
        if self.handler is not None and self.model == model and self.lm == lm:
            return {"model": model, "lm": lm, "already_loaded": True, "step_hook": self.step_hook}
        self.unload()
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

        _ACTIVE["before"], _ACTIVE["after"] = before, after
        try:
            result = generate_music(self.handler, self.llm, params, config, save_dir=None, progress=progress)
        finally:
            _ACTIVE["before"], _ACTIVE["after"] = None, None
        hooks.check_cancel()
        if not getattr(result, "success", False):
            raise RuntimeError(str(getattr(result, "error", None) or "ACE-Step generation failed."))
        hooks.stage("write")
        return {"tracks": self._write(request, result, seeds), "timings": self._timings(result)}

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
