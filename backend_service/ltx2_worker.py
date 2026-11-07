"""Warm ltx-2-mlx worker: keeps the LTX-2.3 / 2.5 distilled pipeline loaded between clips (V1).

Runs from the ltx-2-mlx venv (``~/.chaosengine/ltx2-venv``), never in the FastAPI backend, and imports
nothing from ``backend_service`` (the protocol loop is ``warm_worker_base.py``, loaded by path).

A ``generate`` request carries the same ``generate --distilled ...`` argument list the CLI path runs
(``Ltx2MlxVideoEngine._build_command``). It goes through ltx-2-mlx's own parser, and the call below
mirrors the CLI's distilled branch (``ltx_pipelines_mlx.cli._cmd_generate``, ltx-2-mlx 0.18.0) with one
change: ``low_memory=False``. With the CLI's ``low_memory=True`` the pipeline frees the text encoder, the
transformer and the decoders inside every call, so nothing would stay loaded; with ``False`` they stay
resident and the next clip skips the loads (spike 2026-11-07, docs/hub/progress/v-speed.md: output
bit-identical, resident footprint about 29 GB for LTX-2.5 q4). Re-check this file when
``scripts/install-ltx2-mlx.sh`` moves ``LTX2_MLX_REF``.

The pipeline's phase lines and tqdm bars (stderr) are forwarded as ``line`` events, so the backend reads
progress and the timing split with the same parser it uses for the CLI. Cancel works by raising from the
next output write of the engine thread; the backend kills the worker if that does not end the job.
"""

from __future__ import annotations

import gc
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


class Ltx2Engine:
    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._pipe: Any = None

    def loaded(self) -> dict[str, Any]:
        return {"model": self._key[0] if self._key else None}

    def unload(self) -> None:
        self._pipe, self._key = None, None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except ImportError:
            pass

    def generate(self, request: dict[str, Any], hooks: Any) -> dict[str, Any]:
        import mlx.core as mx
        from ltx_pipelines_mlx import cli

        argv = [str(a) for a in request.get("argv") or []]
        try:
            args = cli._build_parser().parse_args(argv)
        except SystemExit as exc:
            raise ValueError(f"ltx-2-mlx rejected the arguments (exit {exc.code}); see the worker log.") from None
        if args.command != "generate" or not args.distilled:
            raise ValueError("The warm worker runs `generate --distilled` only.")
        if any(getattr(args, name, None) for name in ("two_stage", "two_stages_hq", "one_stage", "dfr", "enable_teacache", "segments")):
            raise ValueError("The warm worker runs the plain distilled pipeline only.")
        output = Path(args.output)
        if not output.is_absolute() or not output.parent.is_dir():
            raise ValueError("--output must be an absolute path in an existing folder.")
        keep_resident = bool(request.get("keep_resident", True))

        cli._require_diffusion_decoder_preconditions(args, args.model)
        nag = cli._build_nag_config(args)
        prompt = cli._maybe_enhance_prompt(args)
        lora_paths = [(path, float(strength)) for path, strength in args.lora] if args.lora else []

        started = time.perf_counter()
        tile = cli._build_tile_count_config(args)
        key = (args.model, args.gemma, bool(getattr(args, "low_ram", False)), repr(tile), tuple(lora_paths), keep_resident)
        warm = key == self._key and self._pipe is not None
        if not warm:
            from ltx_pipelines_mlx.distilled import DistilledPipeline

            self.unload()
            hooks.stage("load")
            self._pipe = DistilledPipeline(
                model_dir=args.model,
                gemma_model_id=args.gemma,
                low_memory=not keep_resident,
                low_ram_streaming=getattr(args, "low_ram", False),
                tile_count=tile,
            )
            if lora_paths:
                self._pipe._pending_loras = lora_paths
            self._key = key
        pipe = self._pipe
        pipe.verbose = True  # phase lines feed the backend's progress and timing parser
        pipe.stepwise = cli._build_stepwise(args)
        pipe.generate_audio = not args.no_audio
        pipe.video_decoder = args.video_decoder
        pipe.diffvae_tile = tuple(args.diffvae_tile) if args.diffvae_tile else None
        kwargs: dict[str, Any] = dict(
            prompt=prompt,
            output_path=str(output),
            height=args.height,
            width=args.width,
            num_frames=cli._resolve_num_frames_arg(args),
            frame_rate=args.frame_rate,
            seed=args.seed,
            images=args.images,
            generated_keyframes=args.num_generated_keyframes,
        )
        if args.stage1_steps is not None:
            kwargs["stage1_steps"] = args.stage1_steps
        if args.stage2_steps is not None:
            kwargs["stage2_steps"] = args.stage2_steps
        if nag is not None:
            kwargs.update(negative_prompt=args.negative_prompt, nag=nag)
        hooks.check_cancel()
        try:
            pipe.generate_and_save(**kwargs)
        except BASE.Cancelled:
            raise
        except BaseException:
            # A failure part-way can leave the pipeline half freed; start clean next time.
            self.unload()
            raise
        mx.clear_cache()  # C3: give freed buffers back between clips
        return {"output": str(output), "warm": warm, "wall_s": round(time.perf_counter() - started, 2)}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("ltx-pipelines-mlx")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    BASE.serve(Ltx2Engine, engine_version=_version(), tee_lines=True)
