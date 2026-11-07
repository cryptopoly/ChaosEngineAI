"""Warm mflux worker: keeps one image model loaded between pictures (I1).

Runs from the mflux venv (``~/.chaosengine/mflux-venv``), never in the FastAPI backend, and imports
nothing from ``backend_service`` (the protocol loop is ``warm_worker_base.py``, loaded by path).

A ``generate`` request carries the same command name and argument list the CLI path would run
(``MfluxImageEngine._build_command``), so every option the Studio sends (reference images, image
strength, guidance, negative prompt, quantize, seed, size, steps) means exactly what it means on the
command line: the arguments go through that command's own ``build_parser()`` and the model call below
mirrors that command's ``main()`` in mflux 0.24.0. Re-check these handlers when
``scripts/install-mflux.sh`` moves ``MFLUX_VERSION``.

The model stays loaded until the next request needs a different one (other command, model, quantize or
LoRA), an ``unload``, or the backend stops the process (idle stop, another heavy job, memory pressure).
The text encoders stay loaded too, so a new prompt needs no reload; that is the memory this trades.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _base() -> Any:
    spec = importlib.util.spec_from_file_location("chaosengine_warm_worker_base", _HERE / "warm_worker_base.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _base()
BASE.strip_own_dir(__file__)

# command -> (CLI module, model class attribute path inside the CLI module's namespace)
CLI_MODULES = {
    "mflux-generate": "mflux.models.flux.cli.flux_generate",
    "mflux-generate-flux2": "mflux.models.flux2.cli.flux2_generate",
    "mflux-generate-qwen-2.1": "mflux.models.qwen21.cli.qwen21_generate",
    "mflux-generate-flux3": "mflux.models.flux3.cli.flux3_generate",
    "mflux-generate-flux3-edit": "mflux.models.flux3.cli.flux3_edit_generate",
    "mflux-generate-iris": "mflux.models.iris.cli.iris_generate",
}
# Arguments that decide which weights are loaded; anything else is per picture.
_MODEL_ARGS = ("model", "quantize", "model_path", "base_model", "lora_paths", "lora_scales", "compute_precision",
               "small_text_encoder", "small_decoder")
_PROMPT_CACHE_MAX = 16


def _freeze(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


class _Timer:
    """mflux callbacks that time the phases, send step events and stop the loop on cancel."""

    def __init__(self) -> None:
        self.hooks: Any = None
        self.marks: dict[str, float] = {}

    def call_before_loop(self, seed, prompt, latents, config, canny_image=None, depth_image=None, control_images=None):
        self.marks["loop_start"] = time.perf_counter()
        if self.hooks is not None:
            self.hooks.check_cancel()
            self.hooks.stage("denoise")

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if self.hooks is None:
            return
        self.hooks.check_cancel()
        done = int(getattr(time_steps, "n", 0)) + 1
        total = int(getattr(time_steps, "total", 0) or getattr(config, "num_inference_steps", 0) or done)
        self.hooks.step(min(done, total), total)

    def call_after_loop(self, seed, prompt, latents, config):
        self.marks["loop_end"] = time.perf_counter()
        if self.hooks is not None:
            self.hooks.stage("decode")


class MfluxEngine:
    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._model: Any = None
        self._timer = _Timer()

    def loaded(self) -> dict[str, Any]:
        if self._key is None:
            return {"model": None}
        return {"command": self._key[0], "model": dict(self._key[1]).get("model"), "quantize": dict(self._key[1]).get("quantize")}

    def unload(self) -> None:
        self._model, self._key = None, None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

    # -- model construction, one per CLI (mirrors each command's main()) --------------------------

    def _build(self, command: str, module: Any, parser: Any, args: Any) -> Any:
        from mflux.cli.parser.parsers import lora_init_kwargs_from_args
        from mflux.models.common.resolution.config_resolution import ConfigResolution

        if command == "mflux-generate":
            from mflux.models.common.config import ModelConfig

            config = ModelConfig.from_name(model_name=args.model, base_model=args.base_model)
            return module.Flux1(model_config=config, quantize=args.quantize, model_path=args.model_path,
                                **lora_init_kwargs_from_args(args))
        from mflux.models.common.compute_precision import ComputePrecision

        extra = getattr(module, "FAMILY_MODELS", ())
        config = ConfigResolution.resolve_restricted(args.model, module.DEFAULT_MODEL, model_path=args.model_path,
                                                     extra_keys=extra, base_model=args.base_model)
        if command == "mflux-generate-iris":
            return module.Iris(model_config=config, quantize=args.quantize, model_path=args.model_path)
        precision = ComputePrecision.dtype_for(args.compute_precision)
        if command == "mflux-generate-flux2":
            return module.Flux2Klein(model_config=config, quantize=args.quantize, model_path=args.model_path,
                                     compute_precision=precision, small_text_encoder=args.small_text_encoder,
                                     small_decoder=args.small_decoder, **lora_init_kwargs_from_args(args))
        if command == "mflux-generate-qwen-2.1":
            return module.QwenImage21(quantize=args.quantize, model_path=args.model_path, model_config=config,
                                      compute_precision=precision, small_text_encoder=args.small_text_encoder,
                                      **lora_init_kwargs_from_args(args))
        cls = module.Flux3KleinEdit if command == "mflux-generate-flux3-edit" else module.Flux3Klein
        return cls(model_config=config, quantize=args.quantize, model_path=args.model_path,
                   **lora_init_kwargs_from_args(args), compute_precision=precision)

    def _render(self, command: str, module: Any, parser: Any, args: Any, seed: int) -> Any:
        from mflux.utils.dimension_resolver import DimensionResolver
        from mflux.utils.prompt_util import PromptUtil

        model = self._model
        prompt = PromptUtil.read_prompt(args)
        if command == "mflux-generate":
            from mflux.cli.defaults import defaults as ui_defaults

            guidance = ui_defaults.GUIDANCE_SCALE if args.guidance is None else args.guidance
            width, height = DimensionResolver.resolve(height=args.height, width=args.width, reference_image_path=args.image_path)
            return model.generate_image(seed=seed, prompt=prompt, width=width, height=height, guidance=guidance,
                                        scheduler=args.scheduler, image_path=args.image_path,
                                        num_inference_steps=args.steps, image_strength=args.image_strength,
                                        negative_prompt=PromptUtil.read_negative_prompt(args),
                                        pid_decode=args.pid_decode, pid_degrade_sigma=args.pid_degrade_sigma)
        if command == "mflux-generate-iris":
            width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
            return model.generate_image(seed=seed, prompt=prompt, width=width, height=height,
                                        num_inference_steps=args.steps, guidance=args.guidance,
                                        negative_prompt=args.negative_prompt)
        if command == "mflux-generate-flux2":
            from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions

            VAEDecodeOptions.apply(args, model)
            width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=args.image_path)
            return model.generate_image(seed=seed, prompt=prompt, width=width, height=height,
                                        guidance=1.0 if args.guidance is None else args.guidance,
                                        image_path=args.image_path, num_inference_steps=args.steps,
                                        image_strength=args.image_strength, scheduler="flow_match_euler_discrete",
                                        pid_decode=args.pid_decode, pid_degrade_sigma=args.pid_degrade_sigma)
        if command == "mflux-generate-qwen-2.1":
            from mflux.models.common.resolution.config_resolution import ConfigResolution

            module.ViggleTurboScheduler.check_args(parser, args)
            config = ConfigResolution.resolve_restricted(args.model, module.DEFAULT_MODEL, model_path=args.model_path,
                                                         base_model=args.base_model, extra_keys=module.FAMILY_MODELS)
            module.Qwen21TurboScheduler.check_args(parser, args, config)
            width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=args.image_path)
            return model.generate_image(seed=seed, prompt=prompt, negative_prompt=PromptUtil.read_negative_prompt(args),
                                        width=width, height=height,
                                        guidance=args.guidance if args.guidance is not None else 1.0,
                                        scheduler=args.scheduler, image_path=args.image_path,
                                        num_inference_steps=args.steps, image_strength=args.image_strength,
                                        step_cache_ratio=args.step_cache_ratio)
        from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions

        VAEDecodeOptions.apply(args, model)
        if command == "mflux-generate-flux3-edit":
            image_paths = [Path(p) for p in args.image_paths]
            if not 1 <= len(image_paths) <= module.MAX_REFERENCES:
                raise ValueError(f"--image-paths takes 1 to {module.MAX_REFERENCES} images, got {len(image_paths)}")
            width, height = module._explicit_size(args.width, args.height, image_paths[0])
            return model.generate_image(seed=seed, prompt=prompt, image_paths=image_paths, width=width, height=height,
                                        num_inference_steps=args.steps, guidance=args.guidance)
        width, height = DimensionResolver.resolve(width=args.width, height=args.height, reference_image_path=None)
        return model.generate_image(seed=seed, prompt=prompt, width=width, height=height,
                                    num_inference_steps=args.steps, guidance=args.guidance)

    # -- one picture -------------------------------------------------------------------------------

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        import mlx.core as mx

        command = str(request.get("command") or "")
        if command not in CLI_MODULES:
            raise ValueError(f"Unknown mflux command {command!r}.")
        argv = [str(a) for a in request.get("argv") or []]
        module = importlib.import_module(CLI_MODULES[command])
        parser = module.build_parser()
        # mflux's parser reads sys.argv itself (parse_args() takes no list, and it checks which
        # options were typed), so it sees exactly the command line the CLI path would run.
        sys.argv = [command, *argv]
        try:
            args = parser.parse_args()
        except SystemExit as exc:  # argparse reports a bad argument by exiting
            raise ValueError(f"mflux rejected the arguments (exit {exc.code}); see the worker log.") from None
        if "--negative-prompt" in argv and command in ("mflux-generate-flux2", "mflux-generate-flux3", "mflux-generate-flux3-edit"):
            raise ValueError("--negative-prompt is not supported for this model.")
        seeds = list(args.seed or [])
        if len(seeds) != 1:
            raise ValueError("The warm worker makes one picture per request.")
        output = Path(args.output.format(seed=seeds[0]))
        if not output.is_absolute() or not output.parent.is_dir():
            raise ValueError("--output must be an absolute path in an existing folder.")

        started = time.perf_counter()
        key = (command, tuple((name, _freeze(getattr(args, name, None))) for name in _MODEL_ARGS))
        load_s = 0.0
        warm = key == self._key and self._model is not None
        if not warm:
            self.unload()
            hooks.stage("load")
            self._model = self._build(command, module, parser, args)
            self._key = key
            try:
                from mflux.callbacks.callback_manager import CallbackManager

                CallbackManager._register_battery_saver(args, self._model)  # the CLI always registers it
            except Exception:  # noqa: BLE001 - optional parity, never blocks a picture
                pass
            self._model.callbacks.register(self._timer)
            load_s = time.perf_counter() - started
        cache = getattr(self._model, "prompt_cache", None)
        if isinstance(cache, dict) and len(cache) > _PROMPT_CACHE_MAX:
            cache.clear()

        hooks.check_cancel()
        mx.reset_peak_memory()
        self._timer.hooks, self._timer.marks = hooks, {}
        hooks.stage("encode")
        render_start = time.perf_counter()
        try:
            image = self._render(command, module, parser, args, seeds[0])
        except SystemExit as exc:  # a scheduler check calls parser.error()
            raise ValueError(f"mflux rejected the arguments (exit {exc.code}); see the worker log.") from None
        finally:
            self._timer.hooks = None
        decoded_at = time.perf_counter()
        hooks.stage("write")
        image.save(path=str(output), export_json_metadata=args.metadata)
        written_at = time.perf_counter()
        peak = mx.get_peak_memory() / 1e9
        gc.collect()
        mx.clear_cache()  # C3: give freed buffers back between pictures
        marks = self._timer.marks
        loop_start = marks.get("loop_start", render_start)
        loop_end = marks.get("loop_end", decoded_at)
        return {
            "output": str(output),
            "warm": warm,
            "peak_mlx_gb": round(peak, 2),
            "timings": {
                "loadSeconds": round(load_s, 2),
                "encodeSeconds": round(loop_start - render_start, 2),
                "denoiseSeconds": round(loop_end - loop_start, 2),
                "decodeSeconds": round(decoded_at - loop_end, 2),
                "saveSeconds": round(written_at - decoded_at, 2),
                "totalSeconds": round(written_at - started, 2),
                "warm": warm,
            },
        }


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("mflux")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    BASE.serve(MfluxEngine, engine_version=_version())
