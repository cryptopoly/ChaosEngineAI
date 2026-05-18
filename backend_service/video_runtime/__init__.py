"""Video runtime for ChaosEngineAI.

Mirrors the shape of ``image_runtime.py`` so the frontend's runtime-status
contract is identical. This phase ships:

- Dependency probe (reports torch / diffusers availability, detected device,
  and any missing packages — including the mp4 encoders needed later for
  ``generate()``).
- Preload / unload lifecycle for one active pipeline at a time.
- Registry routing for the four first-wave engines (LTX-Video, Mochi 1,
  Wan 2.2, HunyuanVideo) to the right diffusers pipeline class.

Generation is intentionally not implemented yet — the preload-to-generate
phase lands next. This keeps the surface area small and testable while
the UX wiring stabilises.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import os
import platform
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from backend_service.helpers.gpu import nvidia_gpu_present, torch_install_warning
from backend_service.image_runtime import validate_local_diffusers_snapshot
from backend_service.progress import (
    GenerationCancelled,
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_LOADING,
    PHASE_SAVING,
    VIDEO_PROGRESS,
)
from backend_service.video_runtime.defaults import (
    _CORE_DEPS,
    _GGUF_QUANT_MULTIPLIERS,
    _SCHEDULER_CLASSES,
    _VIDEO_MODEL_DEPS,
    _VIDEO_MODEL_FOOTPRINT_BF16_GB,
    _VIDEO_OUTPUT_DEPS,
    _align_wan_num_frames,
    _estimate_model_footprint_gb,
    _find_missing,
    _interpolate_frames,
    _resolve_video_defaults,
    _should_apply_memory_savers,
)
from backend_service.video_runtime.device import (
    MAX_VIDEO_SEED,
    WORKSPACE_ROOT,
    _detect_device_memory_gb,
    _guess_video_expected_device,
    _resolve_video_python,
    _resolve_video_seed,
    _windows_cuda_unavailable_message,
)
from backend_service.video_runtime.repos import (
    PIPELINE_REGISTRY,
    _BNB_NF4_VIDEO_TRANSFORMER_CLASSES,
    _GGUF_VIDEO_TRANSFORMER_CLASSES,
    _LTX_DEFAULT_NEGATIVE_PROMPT,
    _PROMPT_ENHANCE_MIN_WORDS,
    _PROMPT_ENHANCEMENT_SUFFIXES,
    _REQUEST_DEFAULT_GUIDANCE,
    _REQUEST_DEFAULT_STEPS,
    _VIDEO_PIPELINE_DEFAULTS,
    _bnb_nf4_transformer_class_for_repo,
    _enhance_prompt,
    _gguf_video_transformer_class_for_repo,
)
from backend_service.video_runtime.pipeline_helpers import (
    build_pipeline_kwargs,
    encode_frames_to_mp4,
    finalize_config,
    invoke_pipeline,
    make_step_callback,
    pipeline_class_for_repo,
    swap_scheduler,
)
from backend_service.video_runtime.transformer_loaders import (
    detect_device,
    preferred_torch_dtype,
    swap_distill_transformers,
    try_load_bnb_nf4_transformer,
    try_load_gguf_transformer,
)
from backend_service.video_runtime.types import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)
from backend_service.video_runtime.warmup import (
    _torch_warmup_lock,
    _torch_warmup_state,
    _torch_warmup_worker,
    start_torch_warmup,
    torch_warmup_status,
)


_LOG = logging.getLogger(__name__)





class DiffusersVideoEngine:
    """Thin wrapper around diffusers video pipelines.

    Single-pipeline at a time; preload() evicts the previous pipeline before
    loading a new one to avoid OOM on unified-memory machines. Generation
    is not implemented in this phase — see ``generate()`` which raises.
    """

    runtime_label = "Diffusers video engine"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_path: str | None = None
        self._loaded_variant_key: str | None = None
        self._device: str | None = None
        # FU-019 / FU-016: notes accumulated during pipeline load (LoRA
        # fuse, attention backend). Reset on each load; surfaced via
        # GeneratedVideo.runtimeNote.
        self._load_notes: list[str] = []

    # ---------- public API ----------

    def probe(self) -> VideoRuntimeStatus:
        # Deliberately does NOT ``import torch`` or trigger the warmup
        # thread. Importing torch loads torch/lib/*.dll into the backend
        # process handle table, and on Windows those locked DLLs block
        # /api/setup/install-gpu-bundle from overwriting them (pip rmtree
        # fails with WinError 5). find_spec answers "is it installable?"
        # without the side effects. Device detection + broken-import
        # checks are deferred to preload/generate where we're about to
        # actually use torch.
        missing_core = _find_missing(_CORE_DEPS)
        missing_output = _find_missing(_VIDEO_OUTPUT_DEPS)
        missing_model = _find_missing(_VIDEO_MODEL_DEPS)

        # All missing deps are reported so the UI can surface a clear install
        # hint, but only ``_CORE_DEPS`` block ``realGenerationAvailable``.
        # ``_VIDEO_MODEL_DEPS`` are pipeline-specific (tiktoken for LTX,
        # sentencepiece for Wan/T5 etc.) — not all of them are needed for
        # every model, but listing them lets the Studio install proactively.
        missing_optional = missing_output + missing_model
        missing_all = missing_core + missing_optional

        if missing_core:
            # Include the missing package names in the message so consumers
            # that only see the RuntimeError string (e.g. preload()'s 500
            # response) still know WHAT to install — missingDependencies is
            # on the structured status but isn't plumbed through every path.
            return VideoRuntimeStatus(
                activeEngine="placeholder",
                realGenerationAvailable=False,
                missingDependencies=missing_all,
                pythonExecutable=_resolve_video_python(),
                expectedDevice=_guess_video_expected_device(),
                torchInstallWarning=torch_install_warning(),
                message=(
                    f"Video runtime needs these packages: {', '.join(missing_core)}. "
                    "Click the 'Install GPU runtime' button above to install the full bundle."
                ),
                loadedModelRepo=self._loaded_repo,
            )

        if missing_output and missing_model:
            message = (
                "Video runtime is ready to load models, but mp4 encoding and tokenizer packages "
                f"are missing — run `pip install {' '.join(missing_optional)}` before generating videos."
            )
        elif missing_output:
            message = (
                "Video runtime is ready to load models, but mp4 encoding packages are missing — "
                "run `pip install imageio imageio-ffmpeg` before generating videos."
            )
        elif missing_model:
            message = (
                "Video runtime is ready, but some models need tokenizer packages that are not "
                f"installed: {', '.join(missing_model)}. Install them now and the affected "
                "models will load on next preload."
            )
        else:
            message = (
                "Real local video generation is available. Download a video model, then Video Studio "
                "will use the diffusers runtime."
            )

        # ``device`` mirrors the currently-loaded model's runtime context —
        # None until preload, because importing torch speculatively locks
        # DLLs on Windows and breaks /api/setup/install-gpu-bundle.
        #
        # ``deviceMemoryGb`` is resolved independently. It reads sysctl on
        # macOS and nvidia-smi on Linux/Windows — neither needs a loaded
        # model, and both are cheap (cached per-process). Gating it behind
        # ``device is not None`` used to leave the frontend safety heuristic
        # with no data until first load, which made it fall back to its
        # 16 GB MPS default and warn a 64 GB M4 Max user as if they were
        # on a base-model Mac.
        device = self._device
        device_memory_gb = _detect_device_memory_gb(device)

        return VideoRuntimeStatus(
            activeEngine="diffusers",
            realGenerationAvailable=True,
            device=device,
            expectedDevice=_guess_video_expected_device(),
            pythonExecutable=_resolve_video_python(),
            missingDependencies=missing_optional,
            message=message,
            loadedModelRepo=self._loaded_repo,
            deviceMemoryGb=device_memory_gb,
            # The earlier replace_all that wired this missed the
            # success-path return because the indentation differs from
            # the placeholder branch above. Without it, the Studio
            # warning chip + banner only fired on the rare path where
            # core deps were also missing -- if torch was importable but
            # +cpu (the actual user case), realGenerationAvailable=True
            # and the field was never set, so the UI silently dropped
            # the warning while every other badge read green.
            torchInstallWarning=torch_install_warning(),
        )

    def preload(self, repo: str) -> VideoRuntimeStatus:
        self._ensure_pipeline(repo)
        return self.probe()

    def unload(self, repo: str | None = None) -> VideoRuntimeStatus:
        with self._lock:
            if repo and self._loaded_repo != repo:
                return self.probe()
            self._release_pipeline()
            return self.probe()

    def generate(self, config: VideoGenerationConfig) -> GeneratedVideo:
        """Run a single text-to-video generation and return the encoded mp4.

        The hot path:
            1. Ensure the right pipeline is loaded.
            2. Build per-model kwargs.
            3. Run the pipeline with a seeded generator.
            4. Encode frames to mp4 via imageio-ffmpeg.
            5. Return bytes + metadata.

        We split the diffusers invocation and mp4 encoding into narrow seams
        (``_invoke_pipeline``, ``_encode_frames_to_mp4``) so tests can stub
        them without needing real 10+GB video weights on disk.
        """
        config, finalize_notes = self._finalize_config(config)
        # Warm cache: skip the ``loading`` phase entirely. ``encoding`` flashes
        # as the initial state so the UI shows the active model name without
        # implying disk I/O that won't happen.
        warm_cache = self._is_variant_loaded(self._compute_variant_key(config))
        VIDEO_PROGRESS.begin(
            run_label=self._format_run_label(config),
            total_steps=max(1, int(config.steps)),
            phase=PHASE_ENCODING if warm_cache else PHASE_LOADING,
            message=(
                f"Reusing {config.modelName}" if warm_cache
                else f"Preparing {config.modelName}"
            ),
        )
        for note in finalize_notes:
            VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=note)
            _LOG.info("video.finalize: %s", note)
        try:
            # mp4 encoding needs imageio-ffmpeg. Check before we spend 60+ seconds
            # doing a full generation we then can't save anywhere.
            missing_output = _find_missing(_VIDEO_OUTPUT_DEPS)
            if missing_output:
                raise RuntimeError(
                    "Video generation requires the mp4 encoding packages: "
                    f"missing {', '.join(missing_output)}. "
                    "Run `pip install imageio imageio-ffmpeg` and retry."
                )

            pipeline = self._ensure_pipeline(
                config.repo,
                gguf_repo=config.ggufRepo,
                gguf_file=config.ggufFile,
                use_nf4=config.useNf4,
                lora_repo=config.loraRepo,
                lora_file=config.loraFile,
                lora_scale=config.loraScale,
                preview_vae=config.previewVae,
                distill_repo=config.distillTransformerRepo,
                distill_high_file=config.distillTransformerHighNoiseFile,
                distill_low_file=config.distillTransformerLowNoiseFile,
                distill_precision=config.distillTransformerPrecision,
            )
            # Early-cancel check after model load — from_pretrained is a
            # blocking C-extension call we can't interrupt. If the user hit
            # Cancel during load we catch up here and bail before we sink
            # time into T5 encoding + the denoising loop.
            if VIDEO_PROGRESS.is_cancelled():
                raise GenerationCancelled("Video generation cancelled by user")

            scheduler_note = self._swap_scheduler(pipeline, config.scheduler)
            if scheduler_note:
                VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=scheduler_note)
                _LOG.info("video.scheduler: %s", scheduler_note)
            torch = self._torch
            if torch is None:
                raise RuntimeError("PyTorch was not initialised for the video runtime.")

            VIDEO_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding prompt")

            base_seed = _resolve_video_seed(config.seed)
            # MPS generators don't seed the same way as CUDA/CPU — follow the
            # diffusers docs and always build the generator on CPU for MPS.
            generator_device = "cpu" if self._device == "mps" else (self._device or "cpu")
            generator = torch.Generator(device=generator_device).manual_seed(base_seed)

            kwargs = self._build_pipeline_kwargs(config, generator)

            VIDEO_PROGRESS.set_phase(
                PHASE_DIFFUSING,
                message=f"Diffusing {config.numFrames} frames",
            )
            VIDEO_PROGRESS.set_step(0, total=max(1, int(config.steps)))

            # TeaCache / other diffusion caches hook here — pipeline is
            # loaded and num_inference_steps is final. Video DiTs are
            # where TeaCache pays off most (1.6–2.1× on HunyuanVideo,
            # ~1.3–2× on Wan). NotImplementedError is swallowed by the
            # helper when the pipeline class has no vendored patch yet;
            # see FU-007 in CLAUDE.md.
            from cache_compression import apply_diffusion_cache_strategy

            apply_diffusion_cache_strategy(
                pipeline,
                strategy_id=config.cacheStrategy,
                num_inference_steps=int(config.steps),
                rel_l1_thresh=config.cacheRelL1Thresh,
                domain="video",
            )

            started = time.perf_counter()
            if config.enableLtxRefiner and config.repo == "Lightricks/LTX-Video":
                try:
                    frames = self._invoke_pipeline_with_ltx_refiner(
                        pipeline, kwargs, torch
                    )
                    VIDEO_PROGRESS.set_phase(
                        PHASE_DIFFUSING,
                        message="LTX two-stage spatial upscale applied.",
                    )
                except Exception as exc:  # noqa: BLE001 — refiner is best-effort
                    note = (
                        f"LTX refiner skipped ({type(exc).__name__}: {exc}) — "
                        "running base pipeline only."
                    )
                    _LOG.info("video.ltx_refiner: %s", note)
                    VIDEO_PROGRESS.set_phase(PHASE_DIFFUSING, message=note)
                    frames = self._invoke_pipeline(pipeline, kwargs)
            else:
                frames = self._invoke_pipeline(pipeline, kwargs)
            elapsed = max(0.1, time.perf_counter() - started)

            if not frames:
                raise RuntimeError(
                    f"The video pipeline returned zero frames for {config.repo}. "
                    "Try a smaller resolution or a different model."
                )

            interpolation_factor = max(1, int(config.interpolationFactor or 1))
            if interpolation_factor > 1:
                VIDEO_PROGRESS.set_phase(
                    PHASE_DECODING,
                    message=f"Interpolating {interpolation_factor}x frames",
                )
                frames = _interpolate_frames(frames, interpolation_factor)
            effective_fps = config.fps * interpolation_factor
            VIDEO_PROGRESS.set_phase(PHASE_DECODING, message="Encoding mp4")
            mp4_bytes = self._encode_frames_to_mp4(frames, effective_fps)
            if not mp4_bytes:
                raise RuntimeError(
                    "mp4 encoding produced an empty buffer. Check that imageio-ffmpeg is "
                    "installed and healthy — run `python -m imageio_ffmpeg` to verify."
                )

            VIDEO_PROGRESS.set_phase(PHASE_SAVING, message="Saving to gallery")
            # FU-019 / FU-016: surface per-pipeline load notes (LoRA
            # fuse, attention backend) on every generated mp4 so the
            # user sees what was applied. Joined with " · " for a
            # single-line UI presentation.
            runtime_note = (
                " · ".join(self._load_notes) if self._load_notes else None
            )
            return GeneratedVideo(
                seed=base_seed,
                bytes=mp4_bytes,
                extension="mp4",
                mimeType="video/mp4",
                durationSeconds=round(elapsed, 2),
                frameCount=len(frames),
                fps=effective_fps,
                width=config.width,
                height=config.height,
                runtimeLabel=f"{self.runtime_label} ({self._device or 'cpu'})",
                runtimeNote=runtime_note,
                effectiveSteps=int(config.steps),
                effectiveGuidance=float(config.guidance),
            )
        finally:
            VIDEO_PROGRESS.finish()

    def _format_run_label(self, config: VideoGenerationConfig) -> str:
        return f"{config.modelName} · {config.numFrames}f @ {config.width}x{config.height}"

    # ---------- internals ----------

    def _finalize_config(
        self, config: VideoGenerationConfig
    ) -> tuple[VideoGenerationConfig, list[str]]:
        return finalize_config(config)

    def _swap_scheduler(self, pipeline: Any, scheduler_id: str | None) -> str | None:
        return swap_scheduler(pipeline, scheduler_id)

    def _build_pipeline_kwargs(
        self,
        config: VideoGenerationConfig,
        generator: Any,
    ) -> dict[str, Any]:
        return build_pipeline_kwargs(config, generator, self._pipeline)

    def _make_step_callback(
        self,
        total_steps: int,
        initial_guidance: float,
        cfg_decay: bool,
        preview_vae: bool = False,
    ) -> Any:
        return make_step_callback(total_steps, initial_guidance, cfg_decay, preview_vae)

    def _invoke_pipeline(self, pipeline: Any, kwargs: dict[str, Any]) -> list[Any]:
        return invoke_pipeline(pipeline, kwargs)

    def _invoke_pipeline_with_ltx_refiner(
        self, pipeline: Any, kwargs: dict[str, Any], torch: Any
    ) -> list[Any]:
        """Run LTX base + LTXLatentUpsamplePipeline spatial 2× upscale.

        Mirrors the upstream Lightricks LTX-Video two-stage pattern:
        sample latents through ``LTXPipeline`` then refine through
        ``LTXLatentUpsamplePipeline`` loaded from the
        ``Lightricks/LTX-Video-0.9.5-spatial-upscaler`` snapshot. Both
        snapshots must be locally cached — we never auto-download from
        within ``generate``. Failure modes (snapshot missing, diffusers
        too old, decode error) propagate to the caller which falls back
        to the base pipeline.
        """
        from huggingface_hub import snapshot_download  # type: ignore

        diffusers = importlib.import_module("diffusers")
        upscaler_cls = getattr(diffusers, "LTXLatentUpsamplePipeline", None)
        if upscaler_cls is None:
            raise RuntimeError(
                "Installed diffusers does not expose LTXLatentUpsamplePipeline."
            )
        upscaler_repo = "Lightricks/LTX-Video-0.9.5-spatial-upscaler"
        upscaler_path = snapshot_download(
            repo_id=upscaler_repo,
            local_files_only=True,
            resume_download=True,
        )

        base_kwargs = dict(kwargs)
        # Strip private kwargs the diffusers pipeline doesn't accept —
        # ``_invoke_pipeline`` pops these before its own pipeline call,
        # but the refiner path bypasses that and would otherwise leak
        # ``__cfg_decay`` / ``__preview_vae`` into ``LTXPipeline.__call__``.
        base_kwargs.pop("__cfg_decay", None)
        base_kwargs.pop("__preview_vae", None)
        base_kwargs["output_type"] = "latent"
        base_result = pipeline(**base_kwargs)
        latents = getattr(base_result, "frames", None)
        if latents is None:
            raise RuntimeError("LTX base pipeline returned no latents.")

        device = self._device or "cpu"
        dtype = self._preferred_torch_dtype(torch, device)
        upscaler = upscaler_cls.from_pretrained(
            upscaler_path,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if device != "cpu":
            try:
                upscaler = upscaler.to(device)
            except (RuntimeError, MemoryError):
                if hasattr(upscaler, "enable_sequential_cpu_offload"):
                    upscaler.enable_sequential_cpu_offload()
                else:
                    raise

        try:
            refined = upscaler(latents=latents)
        finally:
            del upscaler
            gc.collect()

        frames = getattr(refined, "frames", None)
        if frames is None:
            raise RuntimeError("LTX refiner returned no frames.")
        if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
            return list(frames[0])
        return list(frames)

    def _encode_frames_to_mp4(self, frames: list[Any], fps: int) -> bytes:
        return encode_frames_to_mp4(frames, fps)

    def _pipeline_class(self, repo: str) -> Any:
        return pipeline_class_for_repo(repo)

    @staticmethod
    def _build_variant_key(
        *,
        repo: str,
        gguf_file: str | None,
        use_nf4: bool,
        lora_repo: str | None,
        lora_file: str | None,
        lora_scale: float | None,
        preview_vae: bool,
        distill_repo: str | None,
        distill_high_file: str | None,
        distill_low_file: str | None,
        distill_precision: str | None,
    ) -> str:
        variant_parts = [repo]
        if gguf_file:
            variant_parts.append(f"gguf={gguf_file}")
        elif use_nf4:
            variant_parts.append("nf4")
        if lora_repo and lora_file:
            variant_parts.append(f"lora={lora_repo}/{lora_file}@{lora_scale or 1.0}")
        if preview_vae:
            variant_parts.append("preview_vae")
        if distill_repo and distill_high_file and distill_low_file:
            variant_parts.append(
                f"distill={distill_repo}/{distill_precision or 'bf16'}/"
                f"{distill_high_file}/{distill_low_file}"
            )
        return "::".join(variant_parts)

    def _compute_variant_key(self, config: VideoGenerationConfig) -> str:
        return self._build_variant_key(
            repo=config.repo,
            gguf_file=config.ggufFile,
            use_nf4=bool(getattr(config, "useNf4", False)),
            lora_repo=config.loraRepo,
            lora_file=config.loraFile,
            lora_scale=config.loraScale,
            preview_vae=config.previewVae,
            distill_repo=config.distillTransformerRepo,
            distill_high_file=config.distillTransformerHighNoiseFile,
            distill_low_file=config.distillTransformerLowNoiseFile,
            distill_precision=config.distillTransformerPrecision,
        )

    def _is_variant_loaded(self, variant_key: str) -> bool:
        with self._lock:
            return self._pipeline is not None and self._loaded_variant_key == variant_key

    def _ensure_pipeline(
        self,
        repo: str,
        gguf_repo: str | None = None,
        gguf_file: str | None = None,
        use_nf4: bool = False,
        lora_repo: str | None = None,
        lora_file: str | None = None,
        lora_scale: float | None = None,
        preview_vae: bool = False,
        distill_repo: str | None = None,
        distill_high_file: str | None = None,
        distill_low_file: str | None = None,
        distill_precision: str | None = None,
    ) -> Any:
        with self._lock:
            # Variant key folds in LoRA identity — switching LoRAs on the
            # same base repo must rebuild the pipeline because fuse_lora
            # mutates the transformer weights in place. ``preview_vae``
            # joins the same key set so toggling the FU-018 preview-decode
            # knob triggers a clean rebuild. Distilled transformers replace
            # both expert modules outright, so they also key on the variant.
            variant_key = self._build_variant_key(
                repo=repo,
                gguf_file=gguf_file,
                use_nf4=use_nf4,
                lora_repo=lora_repo,
                lora_file=lora_file,
                lora_scale=lora_scale,
                preview_vae=preview_vae,
                distill_repo=distill_repo,
                distill_high_file=distill_high_file,
                distill_low_file=distill_low_file,
                distill_precision=distill_precision,
            )
            if self._pipeline is not None and self._loaded_variant_key == variant_key:
                return self._pipeline

            # Loading a video pipeline can read 10+ GB from disk on cold cache.
            # Publish the phase so the UI explicitly says "Loading model" while
            # snapshot_download + from_pretrained run.
            VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=f"Loading {repo}")

            if self._pipeline is not None and self._loaded_variant_key != variant_key:
                self._release_pipeline()

            import torch  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore

            pipeline_cls = self._pipeline_class(repo)

            local_path = snapshot_download(
                repo_id=repo,
                local_files_only=True,
                resume_download=True,
            )
            local_root = Path(local_path)
            validation_error = validate_local_diffusers_snapshot(local_root, repo)
            if validation_error is not None:
                raise RuntimeError(validation_error)

            device = self._detect_device(torch)
            dtype = self._preferred_torch_dtype(torch, device)

            pipeline_kwargs: dict[str, Any] = {}
            if gguf_file:
                VIDEO_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message=f"Loading GGUF transformer {gguf_file}",
                )
                quantized_transformer, gguf_note = self._try_load_gguf_transformer(
                    repo=repo,
                    gguf_repo=gguf_repo or repo,
                    gguf_file=gguf_file,
                    torch=torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if gguf_note:
                    VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=gguf_note)
                if quantized_transformer is None:
                    raise RuntimeError(
                        gguf_note
                        or f"Could not load requested GGUF transformer {gguf_file}."
                    )
            elif use_nf4:
                VIDEO_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message="Loading NF4 transformer (bitsandbytes)",
                )
                nf4_transformer, nf4_note = self._try_load_bnb_nf4_transformer(
                    repo=repo,
                    local_path=local_path,
                    torch=torch,
                    device=device,
                )
                if nf4_transformer is not None:
                    pipeline_kwargs["transformer"] = nf4_transformer
                if nf4_note:
                    VIDEO_PROGRESS.set_phase(PHASE_LOADING, message=nf4_note)

            pipeline = pipeline_cls.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
                **pipeline_kwargs,
            )

            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)

            # FU-019: clear stale load notes from the previous pipeline
            # and apply distill LoRAs (lightx2v Wan CausVid /
            # Wan2.2-Distill-Models / FastWan) before placement so
            # ``pipeline.to(device)`` moves the fused transformer weights
            # in one pass. Failure is non-fatal — the user gets a note
            # explaining why the LoRA didn't apply.
            self._load_notes = []

            # FU-016: SageAttention CUDA backend. No-op on MPS / CPU.
            # Must run before LoRA fuse so the LoRA's adapter modules
            # don't trip the backend swap (set_attention_backend
            # mutates the attention class on existing modules).
            try:
                from backend_service.helpers.attention_backend import (
                    maybe_apply_sage_attention,
                )
                sage_note = maybe_apply_sage_attention(pipeline)
                if sage_note:
                    self._load_notes.append(sage_note)
            except Exception:
                pass

            # FU-018: TAESD / TAEHV preview-decode VAE swap. No-op when
            # toggle is off or no preview VAE is mapped for this repo.
            # Runs before LoRA fuse so the swap settles before any
            # transformer-side adapters touch the pipeline.
            try:
                from backend_service.helpers.preview_vae import (
                    maybe_apply_preview_vae,
                )
                preview_note = maybe_apply_preview_vae(
                    pipeline, repo=repo, enabled=preview_vae
                )
                if preview_note:
                    self._load_notes.append(preview_note)
            except Exception:
                pass

            # Phase 3 / Wan2.2-Distill 4-step: replace transformer +
            # transformer_2 with the lightx2v distilled experts. Skips
            # LoRA below — distill weights already encode the 4-step
            # schedule and are not LoRA-shaped. Failure is non-fatal:
            # the stock Wan transformers stay in place and the user
            # gets a runtimeNote explaining why.
            distill_active = bool(
                distill_repo and distill_high_file and distill_low_file
            )
            if distill_active:
                distill_note = self._swap_distill_transformers(
                    pipeline,
                    repo=distill_repo,
                    high_file=distill_high_file,
                    low_file=distill_low_file,
                    precision=distill_precision or "bf16",
                    torch=torch,
                )
                self._load_notes.append(distill_note)

            if lora_repo and lora_file and not distill_active:
                try:
                    pipeline.load_lora_weights(
                        lora_repo,
                        weight_name=lora_file,
                        local_files_only=True,
                    )
                    effective_scale = (
                        float(lora_scale) if lora_scale is not None else 1.0
                    )
                    pipeline.fuse_lora(lora_scale=effective_scale)
                    try:
                        pipeline.unload_lora_weights()
                    except Exception:
                        pass
                    self._load_notes.append(
                        f"LoRA: {lora_repo}/{lora_file} @ scale {effective_scale:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001 — non-fatal
                    self._load_notes.append(
                        f"LoRA load failed ({type(exc).__name__}: {exc}). "
                        "Pipeline continuing without LoRA."
                    )

            # Memory-saving knobs. Slicing + tiling are quality-lossy and
            # Reference workflows don't enable them by default — only flip them on
            # when there's real pressure. See ``_should_apply_memory_savers``
            # for the decision matrix.
            total_memory_gb = _detect_device_memory_gb(device)
            estimated_footprint_gb = _estimate_model_footprint_gb(
                repo, str(dtype), gguf_file=gguf_file
            )
            if _should_apply_memory_savers(device, total_memory_gb, estimated_footprint_gb):
                _LOG.info(
                    "video.memory_savers: enabled (device=%s, total_gb=%s, "
                    "estimated_gb=%s)",
                    device,
                    total_memory_gb,
                    estimated_footprint_gb,
                )
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                vae = getattr(pipeline, "vae", None)
                if vae is not None:
                    if hasattr(vae, "enable_slicing"):
                        vae.enable_slicing()
                    if hasattr(vae, "enable_tiling"):
                        vae.enable_tiling()
            else:
                _LOG.info(
                    "video.memory_savers: skipped (device=%s, total_gb=%s, "
                    "estimated_gb=%s) — full quality path.",
                    device,
                    total_memory_gb,
                    estimated_footprint_gb,
                )

            if device != "cpu":
                # MoE pipelines (Wan 2.2 A14B has both ``transformer`` and
                # ``transformer_2``) cannot fit two 28 GB experts in unified
                # memory on a 64 GB Mac. Skip the full-device placement path
                # and engage sequential CPU offload directly so the active
                # expert lives on-device while the inactive one swaps to
                # CPU. Without this, ``.to("mps")`` would raise mid-copy
                # and the user would see a hard crash.
                is_moe = (
                    hasattr(pipeline, "transformer_2")
                    and getattr(pipeline, "transformer_2", None) is not None
                )
                if is_moe and hasattr(pipeline, "enable_sequential_cpu_offload"):
                    _LOG.info(
                        "video.placement: MoE pipeline detected (transformer + transformer_2) — "
                        "engaging enable_sequential_cpu_offload() proactively to keep peak under "
                        "device memory."
                    )
                    pipeline.enable_sequential_cpu_offload()
                else:
                    # Try full-device placement first; fall back to sequential
                    # CPU offload if the model is too big to fit.
                    try:
                        pipeline = pipeline.to(device)
                    except (RuntimeError, MemoryError):
                        if hasattr(pipeline, "enable_sequential_cpu_offload"):
                            pipeline.enable_sequential_cpu_offload()
                        else:
                            raise

            self._pipeline = pipeline
            self._torch = torch
            self._loaded_repo = repo
            self._loaded_path = local_path
            self._loaded_variant_key = variant_key
            self._device = device
            return pipeline

    def _try_load_gguf_transformer(
        self,
        repo: str,
        gguf_repo: str,
        gguf_file: str,
        torch: Any,
    ) -> tuple[Any, str | None]:
        return try_load_gguf_transformer(repo, gguf_repo, gguf_file, torch)

    def _try_load_bnb_nf4_transformer(
        self,
        repo: str,
        local_path: str,
        torch: Any,
        device: str,
    ) -> tuple[Any, str | None]:
        return try_load_bnb_nf4_transformer(repo, local_path, torch, device)

    def _swap_distill_transformers(
        self,
        pipeline: Any,
        *,
        repo: str,
        high_file: str,
        low_file: str,
        precision: str,
        torch: Any,
    ) -> str:
        return swap_distill_transformers(
            pipeline,
            repo=repo,
            high_file=high_file,
            low_file=low_file,
            precision=precision,
            torch=torch,
        )

    def _release_pipeline(self) -> None:
        pipeline = self._pipeline
        torch = self._torch
        device = self._device
        self._pipeline = None
        self._torch = None
        self._loaded_repo = None
        self._loaded_path = None
        self._loaded_variant_key = None
        self._device = None
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch is not None:
            try:
                if device == "cuda" and getattr(torch.cuda, "is_available", lambda: False)():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                mps_backend = getattr(getattr(torch, "mps", None), "empty_cache", None)
                if device == "mps" and callable(mps_backend):
                    mps_backend()
            except Exception:
                pass

    def _detect_device(self, torch: Any) -> str:
        return detect_device(torch)

    def _preferred_torch_dtype(self, torch: Any, device: str) -> Any:
        return preferred_torch_dtype(torch, device)


def _is_longlive_repo(repo: str | None) -> bool:
    """Route LongLive repos to the subprocess engine, everything else to diffusers.

    LongLive is not a diffusers pipeline — it ships as a torchrun-launched
    script with its own CUDA-specific deps that we keep in an isolated
    venv (see ``backend_service.longlive_engine``). Routing happens by
    repo prefix so the rest of the video stack doesn't need to know
    there's a second engine behind the manager.
    """
    if not repo:
        return False
    return repo.startswith("NVlabs/LongLive")


class VideoRuntimeManager:
    """State-level facade that mirrors ``ImageRuntimeManager``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine = DiffusersVideoEngine()
        # Lazy-constructed so the LongLive import (and its probe, which
        # shells out to nvidia-smi) doesn't run on every sidecar start —
        # only when a LongLive repo is actually selected.
        self._longlive: Any | None = None
        # Same pattern for mlx-video (FU-009). Probe-only in this phase
        # — generate() raises, preload/generate are not routed through
        # the manager yet. See ``mlx_video_runtime`` module docstring.
        self._mlx_video: Any | None = None
        # sd.cpp video engine (FU-008). Scaffold only: probe + preload
        # routed; generate() raises NotImplementedError so the manager
        # falls through to diffusers until the CLI subprocess lands.
        self._sdcpp_video: Any | None = None

    def _get_longlive(self) -> Any:
        if self._longlive is None:
            from backend_service.longlive_engine import LongLiveEngine
            self._longlive = LongLiveEngine()
        return self._longlive

    def _get_mlx_video(self) -> Any:
        if self._mlx_video is None:
            from backend_service.mlx_video_runtime import MlxVideoEngine
            self._mlx_video = MlxVideoEngine()
        return self._mlx_video

    def _is_mlx_video_repo(self, repo: str | None) -> bool:
        """Routing predicate for mlx-video. Avoids importing the engine
        module unless the repo prefix actually matches."""
        if not repo:
            return False
        from backend_service.mlx_video_runtime import _is_mlx_video_repo
        return _is_mlx_video_repo(repo)

    def _get_sdcpp_video(self) -> Any:
        if self._sdcpp_video is None:
            from backend_service.sdcpp_video_runtime import SdCppVideoEngine
            self._sdcpp_video = SdCppVideoEngine()
        return self._sdcpp_video

    def _is_sdcpp_video_repo(self, repo: str | None) -> bool:
        if not repo:
            return False
        from backend_service.sdcpp_video_runtime import _is_sdcpp_video_repo
        return _is_sdcpp_video_repo(repo)

    def capabilities(self) -> dict[str, Any]:
        return self._engine.probe().to_dict()

    def longlive_capabilities(self) -> dict[str, Any]:
        """Probe the LongLive engine separately so the Studio can surface install state."""
        return self._get_longlive().probe().to_dict()

    def mlx_video_capabilities(self) -> dict[str, Any]:
        """Probe the mlx-video engine so Setup can surface install state.

        On Apple Silicon with mlx-video installed, the manager routes
        ``prince-canuma/LTX-2-*`` repos here before falling through to
        diffusers — see ``generate``. Wan paths still use diffusers MPS
        until the mlx-video Wan conversion step is bundled.
        """
        return self._get_mlx_video().probe().to_dict()

    def sdcpp_video_capabilities(self) -> dict[str, Any]:
        """Probe the sd.cpp engine so Setup/Studio can surface staging state.

        Scaffold today: ``realGenerationAvailable`` is always ``False``
        because ``generate()`` is unwired. Probe still reports binary
        presence so the UI can prompt the user to stage `sd` ahead of
        the FU-008 generation cutover.
        """
        return self._get_sdcpp_video().probe().to_dict()

    def preload(self, repo: str) -> dict[str, Any]:
        with self._lock:
            if _is_longlive_repo(repo):
                engine = self._get_longlive()
                status = engine.probe()
                if not status.realGenerationAvailable:
                    raise RuntimeError(status.message)
                return engine.preload(repo).to_dict()
            if self._is_mlx_video_repo(repo):
                mlx = self._get_mlx_video()
                status = mlx.probe()
                if not status.realGenerationAvailable:
                    raise RuntimeError(status.message)
                return mlx.preload(repo).to_dict()
            status = self._engine.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            return self._engine.preload(repo).to_dict()

    def unload(self, repo: str | None = None) -> dict[str, Any]:
        with self._lock:
            if _is_longlive_repo(repo):
                return self._get_longlive().unload(repo).to_dict()
            if self._is_mlx_video_repo(repo):
                return self._get_mlx_video().unload(repo).to_dict()
            return self._engine.unload(repo).to_dict()

    def generate(self, config: VideoGenerationConfig) -> tuple[GeneratedVideo, dict[str, Any]]:
        """Run a single video generation, returning (video, runtime_status).

        Unlike the image manager, there is no placeholder fallback — video is
        heavy enough that a silent fake clip would waste the user's time. If
        the runtime isn't ready, raise a clear error so the route can return
        a proper 4xx.
        """
        if _is_longlive_repo(config.repo):
            engine = self._get_longlive()
            status = engine.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            with self._lock:
                video = engine.generate(config)
                runtime = engine.probe().to_dict()
            return video, runtime

        if self._is_mlx_video_repo(config.repo):
            mlx = self._get_mlx_video()
            status = mlx.probe()
            if status.realGenerationAvailable:
                # Wire mlx-video subprocess stdout into the shared
                # VIDEO_PROGRESS tracker. Without this the bar sits at
                # whatever phase the previous diffusers run left it on,
                # because the mlx-video path never published its own
                # lifecycle events.
                total_steps = max(1, int(config.steps))
                VIDEO_PROGRESS.begin(
                    run_label=self._engine._format_run_label(config),
                    total_steps=total_steps,
                    phase=PHASE_LOADING,
                    message=f"Preparing {config.modelName}",
                )

                # Track the last published phase so non-step chatter
                # doesn't keep calling ``set_phase``, which would reset the
                # step counter to zero on every line (set_phase is designed
                # for transitions, not message refreshes).
                last_phase: list[str] = [PHASE_LOADING]

                def _on_mlx_progress(phase: str, message: str, fraction: float | None) -> None:
                    mapped = PHASE_DIFFUSING if phase == "diffusing" else PHASE_LOADING
                    if mapped != last_phase[0]:
                        VIDEO_PROGRESS.set_phase(mapped, message=message)
                        last_phase[0] = mapped
                    else:
                        VIDEO_PROGRESS.set_message(message)
                    if fraction is not None:
                        step = max(0, min(total_steps, int(fraction * total_steps)))
                        VIDEO_PROGRESS.set_step(step, total=total_steps)

                try:
                    with self._lock:
                        video = mlx.generate(config, on_progress=_on_mlx_progress)
                        runtime = mlx.probe().to_dict()
                    VIDEO_PROGRESS.set_phase(PHASE_SAVING, message="Saving to gallery")
                    return video, runtime
                finally:
                    VIDEO_PROGRESS.finish()
            # mlx-video not available (Intel Mac, missing package, etc.) —
            # fall through to diffusers so the supported repo doesn't dead-
            # end. Diffusers won't actually load LTX-2-* (no compatible
            # pipeline yet), so this branch effectively only covers the
            # "supported repo on a non-Apple-Silicon host" edge case.

        status = self._engine.probe()
        if not status.realGenerationAvailable:
            raise RuntimeError(status.message)
        with self._lock:
            video = self._engine.generate(config)
            runtime = self._engine.probe().to_dict()
        return video, runtime
