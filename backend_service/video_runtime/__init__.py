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
from backend_service.video_runtime.types import (
    GeneratedVideo,
    VideoGenerationConfig,
    VideoRuntimeStatus,
)


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch warmup
# ---------------------------------------------------------------------------
# Importing torch for the first time is expensive (30-60s on a cold Windows
# SSD). Because probe() is a sync FastAPI route that calls ``import torch``,
# the first probe blew past the frontend's 30s fetch timeout and surfaced as
# "Video runtime did not respond" with every downstream endpoint cascading to
# "Failed to fetch". We warm torch on a background thread at sidecar startup
# so probe() can return a fast "initializing" status while the import is in
# flight, and an accurate status the moment it completes. The import lock
# means any in-flight probe still ends up serialized behind the warmup
# anyway — the fast-path here is purely to keep the probe route itself from
# blocking so the rest of the video API stays responsive.

_torch_warmup_lock = threading.Lock()
_torch_warmup_state: dict[str, Any] = {
    "status": "not_started",  # "not_started" | "in_progress" | "ready" | "failed"
    "error": None,  # exception message when status == "failed"
    "started_at": None,
}


def _torch_warmup_worker() -> None:
    try:
        import torch  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover - import failure path
        with _torch_warmup_lock:
            _torch_warmup_state["status"] = "failed"
            _torch_warmup_state["error"] = f"{type(exc).__name__}: {exc}"
        return
    # Pre-warm anything else the first probe() call would otherwise pay for
    # inline. On Windows the nvidia-smi shell-out adds 1-2s per probe when
    # uncached, and importlib.util.find_spec on a cold NTFS volume with
    # antivirus scanning can be slow enough to push a probe past the
    # frontend's fetch timeout. Doing both here keeps probe() a hashmap
    # lookup in the common case.
    try:
        from backend_service.helpers.gpu import get_device_vram_total_gb
        get_device_vram_total_gb()
    except Exception:
        pass
    try:
        for _pkg, module_name in _CORE_DEPS + _VIDEO_OUTPUT_DEPS + _VIDEO_MODEL_DEPS:
            try:
                importlib.util.find_spec(module_name)
            except Exception:
                pass
    except Exception:
        pass
    with _torch_warmup_lock:
        _torch_warmup_state["status"] = "ready"
        _torch_warmup_state["error"] = None


def start_torch_warmup() -> None:
    """Kick off a one-shot background import of torch.

    Called from ``create_app()`` at sidecar startup. Safe to call repeatedly —
    only the first call spawns a thread. If torch is already importable
    cheaply (e.g. the interpreter has seen it before in this process), the
    worker finishes almost immediately.
    """
    with _torch_warmup_lock:
        if _torch_warmup_state["status"] != "not_started":
            return
        _torch_warmup_state["status"] = "in_progress"
        _torch_warmup_state["started_at"] = time.monotonic()
    thread = threading.Thread(
        target=_torch_warmup_worker,
        name="chaosengine-torch-warmup",
        daemon=True,
    )
    thread.start()


def torch_warmup_status() -> dict[str, Any]:
    """Snapshot of the warmup state. Used by ``probe()`` to avoid blocking."""
    with _torch_warmup_lock:
        return dict(_torch_warmup_state)







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
        VIDEO_PROGRESS.begin(
            run_label=self._format_run_label(config),
            total_steps=max(1, int(config.steps)),
            phase=PHASE_LOADING,
            message=f"Preparing {config.modelName}",
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
        """Apply per-model defaults + frame alignment + scheduler resolution.

        Centralised here so VIDEO_PROGRESS, the cache strategy hook, and the
        pipeline invocation all see the same resolved values. Returns a new
        (frozen) config + a list of human-readable notes the caller publishes
        to the run log.
        """
        notes: list[str] = []
        resolved = _resolve_video_defaults(config.repo, config.steps, config.guidance)
        steps = int(resolved["steps"])
        guidance = float(resolved["guidance"])
        if resolved.get("substituted"):
            notes.append(
                f"Substituting model-tuned defaults for {config.modelName}: "
                f"steps {config.steps} → {steps}, CFG {config.guidance} → {guidance}."
            )

        aligned_frames, frame_note = _align_wan_num_frames(config.repo, config.numFrames)
        if frame_note:
            notes.append(frame_note)

        # Scheduler: explicit request > model default > leave alone.
        requested_scheduler = (config.scheduler or "").strip().lower() or None
        if requested_scheduler == "auto":
            requested_scheduler = None
        scheduler = requested_scheduler or resolved.get("scheduler")
        if scheduler and scheduler not in _SCHEDULER_CLASSES:
            notes.append(
                f"Unknown scheduler {scheduler!r} — keeping the pipeline default."
            )
            scheduler = None

        # LTX-Video: surface the auto-tuned decode params + frame_rate
        # conditioning so the user sees why output quality matches the
        # Lightricks reference even though we didn't expose new sliders.
        if config.repo == "Lightricks/LTX-Video":
            notes.append(
                f"LTX-Video auto-tuned to Lightricks reference defaults: "
                f"frame_rate={int(config.fps)} (model conditioning), "
                f"decode_timestep=0.05, decode_noise_scale=0.025, "
                f"guidance_rescale=0.7."
            )

        # Phase E1 — auto-enhance short prompts. Default-on; opt-out via
        # config.enhancePrompt=False. Only fires below the word-count
        # threshold so a long custom prompt is never modified.
        enhanced_prompt = config.prompt
        if config.enhancePrompt:
            enhanced_prompt, enhance_note = _enhance_prompt(config.repo, config.prompt)
            if enhance_note:
                notes.append(enhance_note)

        # Phase E2 — CFG decay note. Only surfaces when decay actually
        # has somewhere to ramp (initial CFG > 1.5 — the floor that
        # keeps classifier-free guidance enabled throughout the loop).
        _CFG_DECAY_FLOOR = 1.5
        if config.cfgDecay and guidance > _CFG_DECAY_FLOOR and steps > 1:
            notes.append(
                f"CFG decay enabled: linearly ramping guidance_scale from "
                f"{guidance:.2f} (step 0) to {_CFG_DECAY_FLOOR} (final step) — "
                f"flow-match video models oversaturate when CFG stays high "
                f"throughout sampling. Floor stays above 1.0 so classifier-"
                f"free guidance keeps running 2-batch end-to-end."
            )

        return (
            replace(
                config,
                prompt=enhanced_prompt,
                steps=steps,
                guidance=guidance,
                numFrames=aligned_frames,
                scheduler=scheduler,
            ),
            notes,
        )

    def _swap_scheduler(self, pipeline: Any, scheduler_id: str | None) -> str | None:
        """Replace the pipeline's scheduler with the requested class.

        Returns a status message (non-None) iff the swap actually happened
        or failed in a user-relevant way. ``None`` means "no swap requested
        or pipeline already on this scheduler" — silent path.
        """
        if not scheduler_id:
            return None
        cls_name = _SCHEDULER_CLASSES.get(scheduler_id)
        if cls_name is None:
            return None
        current_cls = type(getattr(pipeline, "scheduler", None)).__name__
        if current_cls == cls_name:
            return None
        try:
            diffusers = importlib.import_module("diffusers")
        except Exception:
            return f"Scheduler swap skipped: diffusers import failed."
        scheduler_cls = getattr(diffusers, cls_name, None)
        if scheduler_cls is None:
            return (
                f"Scheduler {scheduler_id!r} ({cls_name}) not available in the "
                "installed diffusers — keeping the pipeline default."
            )
        try:
            pipeline.scheduler = scheduler_cls.from_config(pipeline.scheduler.config)
        except Exception as exc:  # noqa: BLE001
            return f"Scheduler swap to {scheduler_id!r} failed: {exc}"
        return f"Scheduler swapped to {scheduler_id} ({cls_name})."

    def _build_pipeline_kwargs(
        self,
        config: VideoGenerationConfig,
        generator: Any,
    ) -> dict[str, Any]:
        """Per-model kwarg shaping.

        Most diffusers video pipelines accept the same shape, but there are
        small variations — e.g. HunyuanVideoPipeline does not accept a
        ``negative_prompt`` argument in its canonical signature.
        """
        kwargs: dict[str, Any] = {
            "prompt": config.prompt,
            "width": config.width,
            "height": config.height,
            "num_frames": config.numFrames,
            "num_inference_steps": config.steps,
            "guidance_scale": config.guidance,
            "generator": generator,
            # Force PIL output so ``_encode_frames_to_mp4`` always receives
            # ``list[PIL.Image]``. WanPipeline defaults to ``"np"``, which
            # returns a 5D numpy array (B, F, H, W, C). Our frame
            # post-processing assumes the diffusers PIL convention; a raw
            # numpy tensor leaks through and ``PIL.Image.fromarray`` then
            # raises "Image must have 1, 2, 3 or 4 channels" because it
            # reads the first non-batch dim as height. LTXPipeline
            # already defaults to "pil"; setting it explicitly here is
            # a no-op for LTX and the fix for Wan / Hunyuan / Mochi /
            # CogVideoX (all default to "np").
            "output_type": "pil",
        }
        lowered_repo = config.repo.lower()
        if "hunyuanvideo" not in lowered_repo and config.negativePrompt.strip():
            kwargs["negative_prompt"] = config.negativePrompt

        # LTX-Video kwargs parity with Lightricks' reference defaults.
        # Without these, diffusers' LTXPipeline produces rainbow / blurry
        # output because (1) the model conditions on default frame_rate=25
        # while our exporter writes config.fps, (2) the VAE decodes from
        # final latent without the small denoise pass that cleans
        # compression artifacts, (3) flow-match models oversaturate
        # without rescale. Reference: Lightricks LTX-Video model card.
        pipeline_cls = type(self._pipeline).__name__ if self._pipeline is not None else ""
        if pipeline_cls == "LTXPipeline":
            kwargs["frame_rate"] = int(config.fps)
            kwargs["decode_timestep"] = 0.05
            kwargs["decode_noise_scale"] = 0.025
            kwargs["guidance_rescale"] = 0.7
            # Inject Lightricks' recommended negative-prompt template when
            # the user hasn't overridden — LTX was trained with strong
            # negative-prompt conditioning, so the schema's softer default
            # ("blurry, low quality") leaves quality on the table.
            if not kwargs.get("negative_prompt"):
                kwargs["negative_prompt"] = _LTX_DEFAULT_NEGATIVE_PROMPT
        # Private kwarg consumed by ``_invoke_pipeline`` — pop'd before
        # passing to the diffusers pipeline, so it never reaches the
        # underlying call. Lets the engine plumb decay through one
        # callback factory rather than threading state through self.
        kwargs["__cfg_decay"] = bool(config.cfgDecay)
        # FU-018 part 2: same private-kwarg plumbing for the live
        # denoise thumbnail emit. When on, the step callback decodes
        # the current latent's middle frame via the TAEHV/TAEW preview
        # VAE that ``_ensure_pipeline`` swapped onto ``pipeline.vae``.
        kwargs["__preview_vae"] = bool(config.previewVae)
        return kwargs

    def _make_step_callback(
        self,
        total_steps: int,
        initial_guidance: float,
        cfg_decay: bool,
        preview_vae: bool = False,
    ) -> Any:
        """Build the per-step callback the pipeline calls during sampling.

        Wires four concerns into one callback:
          1. Progress reporting via ``VIDEO_PROGRESS.set_step``.
          2. Cooperative cancel — raise ``GenerationCancelled`` when the
             user hits Cancel on the modal.
          3. Phase E2 CFG decay — linearly ramp ``pipeline.guidance_scale``
             from ``initial_guidance`` at step 0 toward 1.0 at the last
             step. Flow-match video models (LTX, Wan, HunyuanVideo) tend
             to oversaturate when CFG is held high through the whole
             schedule; decaying lets the early steps lock semantics
             (high CFG) while late steps preserve fine detail (low CFG).
          4. FU-018 part 2 — when ``preview_vae`` is on, every Nth step
             decode the current latent's middle frame via the swapped
             TAEHV/TAEW preview VAE and publish a base64 PNG to
             ``VIDEO_PROGRESS.set_thumbnail`` for the modal to render.
        """
        # Floor MUST stay strictly above 1.0 so the pipeline's
        # ``do_classifier_free_guidance`` property (``_guidance_scale > 1.0``)
        # does not flip to False mid-loop. If it did, the pipeline would
        # switch from 2-batch (cond+uncond) to 1-batch on the last step
        # while the embeddings + scheduler state are still 2-batch shape,
        # crashing with shape mismatches or producing garbled frames
        # ("Image must have 1, 2, 3 or 4 channels" on Wan, batch
        # dimension errors on LTX).
        decay_floor = 1.5
        decay_active = cfg_decay and total_steps > 1 and initial_guidance > decay_floor
        thumb_active = bool(preview_vae)
        # Stride keeps the polled endpoint payload small. Video
        # latent decode is more expensive than image (5D tensor), so
        # we cap thumbnails at ~6 per gen.
        thumb_stride = max(1, total_steps // 6) if thumb_active else 1

        def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
            VIDEO_PROGRESS.set_step(step + 1, total=max(1, total_steps))
            if VIDEO_PROGRESS.is_cancelled():
                try:
                    _pipeline._interrupt = True
                except Exception:
                    pass
                raise GenerationCancelled("Video generation cancelled by user")
            if decay_active:
                # Step `step` just finished (step uses scale set BEFORE it).
                # Set the scale for step `step+1`. Linear ramp from initial
                # at step 0 to decay_floor at step total_steps-1.
                next_step = step + 1
                progress = min(1.0, next_step / max(1, total_steps - 1))
                next_scale = initial_guidance * (1.0 - progress) + decay_floor * progress
                try:
                    _pipeline.guidance_scale = float(next_scale)
                except Exception:
                    pass
            if thumb_active:
                is_final = (step + 1) >= total_steps
                if is_final or (step % thumb_stride == 0):
                    latents = callback_kwargs.get("latents") if callback_kwargs else None
                    try:
                        from backend_service.helpers.preview_thumbnails import (
                            decode_video_latent_to_b64,
                        )
                        b64 = decode_video_latent_to_b64(_pipeline, latents)
                        if b64 is not None:
                            VIDEO_PROGRESS.set_thumbnail(b64)
                    except Exception:
                        # Best-effort — never fail the gen on a preview
                        # decode error.
                        pass
            return callback_kwargs

        return _on_step_end

    def _invoke_pipeline(self, pipeline: Any, kwargs: dict[str, Any]) -> list[Any]:
        """Run the diffusers pipeline and return the first batch's frames.

        Carved out as a seam so tests can stub it without loading real
        weights. Diffusers video pipelines return an output with a
        ``.frames`` attribute shaped like ``list[list[PIL.Image]]`` — one
        inner list per batch item. We only ever render batchSize=1, so
        we return ``result.frames[0]``.

        Wires the diffusers per-step callback into ``VIDEO_PROGRESS`` so the
        UI bar tracks denoising in real time. Falls back to a callback-free
        invocation on older diffusers versions that don't expose the kwarg.
        """
        total_steps = int(kwargs.get("num_inference_steps") or 0)
        initial_guidance = float(kwargs.get("guidance_scale") or 1.0)
        # Phase E2: CFG decay flag is plumbed via a private kwarg the
        # caller pops before passing to the pipeline. Default-on when
        # absent so existing call sites pick up the schedule.
        cfg_decay = bool(kwargs.pop("__cfg_decay", True))
        # FU-018 part 2: previewVae flag plumbs through the same
        # private-kwarg pattern. When on, ``_make_step_callback`` emits
        # a per-step base64 thumbnail decoded via the TAESD/TAEHV swap.
        preview_vae = bool(kwargs.pop("__preview_vae", False))
        callback = self._make_step_callback(
            total_steps, initial_guidance, cfg_decay, preview_vae=preview_vae,
        )
        kwargs.setdefault("callback_on_step_end", callback)

        try:
            result = pipeline(**kwargs)
        except TypeError as exc:
            message = str(exc)
            # Older diffusers / pipelines that don't accept ``callback_on_step_end``.
            if "callback_on_step_end" in message:
                kwargs = {k: v for k, v in kwargs.items() if k != "callback_on_step_end"}
                try:
                    result = pipeline(**kwargs)
                except TypeError as inner:
                    if "negative_prompt" in str(inner) and "negative_prompt" in kwargs:
                        kwargs = {k: v for k, v in kwargs.items() if k != "negative_prompt"}
                        result = pipeline(**kwargs)
                    else:
                        raise
            elif "negative_prompt" in message and "negative_prompt" in kwargs:
                # Some pipelines reject ``negative_prompt`` even when given a
                # non-empty value. Fall back once without it rather than crashing
                # the whole generation.
                kwargs = {key: value for key, value in kwargs.items() if key != "negative_prompt"}
                result = pipeline(**kwargs)
            else:
                raise

        frames = getattr(result, "frames", None)
        if frames is None:
            raise RuntimeError(
                "Video pipeline result is missing a `.frames` attribute. "
                "This usually means the installed diffusers version returns a "
                "different output shape. Upgrade diffusers: pip install -U diffusers"
            )
        if isinstance(frames, (list, tuple)) and frames and isinstance(frames[0], (list, tuple)):
            return list(frames[0])
        return list(frames)

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
        """Encode a list of PIL.Image frames to an mp4 byte buffer.

        Carved out as a seam so tests can stub it. We use ``imageio`` +
        ``imageio-ffmpeg`` via the ``diffusers.utils.export_to_video`` helper
        when available (it handles the numpy conversion), and fall back to a
        direct ``imageio`` writer if diffusers hasn't exposed the helper on
        the installed version.
        """
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            tmp_path = handle.name
        try:
            export_to_video = None
            try:
                from diffusers.utils import export_to_video as _export  # type: ignore
                export_to_video = _export
            except Exception:
                export_to_video = None

            if export_to_video is not None:
                export_to_video(frames, tmp_path, fps=fps)
            else:
                # Minimal fallback — avoids tying us to diffusers' helper
                # layout. Uses the same pyav backend imageio-ffmpeg ships.
                import numpy as np  # type: ignore
                import imageio  # type: ignore

                writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
                try:
                    for frame in frames:
                        array = np.asarray(frame)
                        if array.ndim == 2:
                            array = np.stack([array] * 3, axis=-1)
                        writer.append_data(array.astype("uint8"))
                finally:
                    writer.close()

            return Path(tmp_path).read_bytes()
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _pipeline_class(self, repo: str) -> Any:
        entry = PIPELINE_REGISTRY.get(repo)
        if entry is None:
            raise RuntimeError(
                f"No diffusers pipeline is registered for repo '{repo}'. "
                f"Supported repos: {sorted(PIPELINE_REGISTRY.keys())}"
            )
        class_name = entry["class_name"]
        diffusers = importlib.import_module("diffusers")
        pipeline_cls = getattr(diffusers, class_name, None)
        if pipeline_cls is None:
            raise RuntimeError(
                f"The installed diffusers version does not expose '{class_name}'. "
                "Upgrade diffusers: pip install -U diffusers"
            )
        return pipeline_cls

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
            variant_key = "::".join(variant_parts)
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
        """Load a video DiT from a single ``.gguf`` file via diffusers.

        Mirrors the image-side loader: GGUF weights cover the DiT only;
        VAE and text encoders are loaded from the base ``repo`` snapshot.
        The helper itself only reports ``(None, note)`` on failure so tests
        can exercise each missing-dependency path. ``_ensure_pipeline`` treats
        a requested GGUF variant as strict and raises with that note rather
        than silently loading the full fp16 / bf16 transformer.
        """
        if importlib.util.find_spec("gguf") is None:
            return None, (
                "gguf package missing — install it from the Setup page to "
                f"load {gguf_file}. Falling back to the standard transformer."
            )
        try:
            from diffusers import GGUFQuantizationConfig  # type: ignore
        except Exception as exc:
            return None, (
                f"Installed diffusers cannot load GGUFQuantizationConfig "
                f"({type(exc).__name__}: {exc}). Upgrade diffusers via the "
                "Setup page to use GGUF variants."
            )
        transformer_cls_name = _gguf_video_transformer_class_for_repo(repo)
        if transformer_cls_name is None:
            return None, (
                f"No GGUF transformer class registered for {repo}. "
                "Add it to _GGUF_VIDEO_TRANSFORMER_CLASSES."
            )
        try:
            import diffusers  # type: ignore
        except Exception:
            return None, "diffusers import failed — cannot load GGUF transformer."
        transformer_cls = getattr(diffusers, transformer_cls_name, None)
        if transformer_cls is None:
            return None, (
                f"{transformer_cls_name} not in installed diffusers — "
                "upgrade to use this GGUF variant."
            )

        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            gguf_local_path = hf_hub_download(
                repo_id=gguf_repo,
                filename=gguf_file,
                local_files_only=True,
            )
            # ``from_single_file`` defaults the architecture config to the
            # transformer class's largest known variant. For Wan that is the
            # 14 B / A14B layout (cross-attn dim 5120). The TI2V 5B uses
            # cross-attn dim 3072, so loading its GGUF without an explicit
            # config raises:
            #     blocks.0.attn2.to_k.bias expected torch.Size([5120]),
            #     but got torch.Size([3072])
            # Pointing at the base diffusers repo's transformer subfolder
            # makes diffusers build the model from the matching
            # ``transformer/config.json`` before mapping in GGUF tensors,
            # which fixes Wan 2.2 5B and stays correct for every other
            # variant (the config dim happens to match the GGUF anyway).
            transformer = transformer_cls.from_single_file(
                gguf_local_path,
                quantization_config=GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16,
                ),
                torch_dtype=torch.bfloat16,
                config=repo,
                subfolder="transformer",
            )
            return transformer, f"Transformer loaded from GGUF ({gguf_file})"
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            return None, (
                f"GGUF load failed ({type(exc).__name__}: {exc}) — "
                "falling back to the standard transformer."
            )

    def _try_load_bnb_nf4_transformer(
        self,
        repo: str,
        local_path: str,
        torch: Any,
        device: str,
    ) -> tuple[Any, str | None]:
        """Load a video DiT in NF4 4-bit via bitsandbytes.

        CUDA-only — bitsandbytes has no Metal/MPS backend, and the kernels
        wouldn't help on a 64 GB Mac anyway. Failure modes (non-CUDA host,
        missing bitsandbytes, old diffusers without ``BitsAndBytesConfig``,
        unmapped repo, broken snapshot subfolder) all return ``(None, note)``
        so the caller falls back to the standard fp16 / bf16 transformer.

        The transformer subfolder pattern (``from_pretrained(local_path,
        subfolder="transformer", quantization_config=...)``) matches the
        Wan / HunyuanVideo / LTX-Video diffusers snapshots — VAE and text
        encoders still load via the parent pipeline ``from_pretrained`` on
        the same snapshot root.
        """
        if device != "cuda":
            return None, (
                "NF4 (bitsandbytes) requires CUDA. "
                "Falling back to the standard transformer."
            )
        if importlib.util.find_spec("bitsandbytes") is None:
            return None, (
                "bitsandbytes package missing — install it from the Setup "
                "page to enable NF4. Falling back to the standard transformer."
            )
        try:
            from diffusers import BitsAndBytesConfig  # type: ignore
        except ImportError:
            return None, (
                "Installed diffusers doesn't expose BitsAndBytesConfig. "
                "Upgrade diffusers via the Setup page to use NF4 variants."
            )
        transformer_cls_name = _bnb_nf4_transformer_class_for_repo(repo)
        if transformer_cls_name is None:
            return None, (
                f"No NF4 transformer class registered for {repo}. "
                "Add it to _BNB_NF4_VIDEO_TRANSFORMER_CLASSES."
            )
        try:
            import diffusers  # type: ignore
        except Exception:
            return None, "diffusers import failed — cannot load NF4 transformer."
        transformer_cls = getattr(diffusers, transformer_cls_name, None)
        if transformer_cls is None:
            return None, (
                f"{transformer_cls_name} not in installed diffusers — "
                "upgrade to use NF4 quantization."
            )

        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer = transformer_cls.from_pretrained(
                local_path,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            return transformer, "Transformer loaded with NF4 (bitsandbytes)"
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            return None, (
                f"NF4 load failed ({type(exc).__name__}: {exc}) — "
                "falling back to the standard transformer."
            )

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
        """Swap ``pipeline.transformer`` + ``pipeline.transformer_2`` for
        the lightx2v 4-step distilled experts (Wan 2.2 A14B I2V).

        Wan 2.2 A14B is MoE: ``transformer`` is the high-noise expert and
        ``transformer_2`` is the low-noise expert. Distillation publishes
        both as standalone safetensors files; the swap is the load-bearing
        substitution that takes the pipeline from 30-step base to 4-step
        distilled. Returns a runtimeNote describing what happened. Failure
        is non-fatal — the stock transformers stay in place and the user
        sees the failure in the note.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            return (
                f"Distill swap skipped: huggingface_hub unavailable ({exc}). "
                "Pipeline continuing with stock Wan transformers."
            )

        try:
            from diffusers import WanTransformer3DModel
        except ImportError as exc:
            return (
                f"Distill swap skipped: WanTransformer3DModel unavailable "
                f"({exc}). Pipeline continuing with stock Wan transformers."
            )

        # FP8/INT8 distill weights ship pre-quantized; they need a torch
        # backend that exposes the matching kernels (CUDA SM 8.9+ for FP8,
        # CUDA / Metal for INT8). On platforms without those kernels we
        # load as bf16 and let diffusers do the dequant — quality holds
        # but the memory savings disappear. ``bf16`` (no quantization)
        # always loads at native precision.
        torch_dtype = torch.bfloat16
        if precision == "fp8_e4m3":
            torch_dtype = getattr(torch, "float8_e4m3fn", torch.bfloat16)

        try:
            high_local = hf_hub_download(
                repo_id=repo, filename=high_file, local_files_only=False
            )
            low_local = hf_hub_download(
                repo_id=repo, filename=low_file, local_files_only=False
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            return (
                f"Distill download failed ({type(exc).__name__}: {exc}). "
                "Pipeline continuing with stock Wan transformers."
            )

        try:
            high_transformer = WanTransformer3DModel.from_single_file(
                high_local, torch_dtype=torch_dtype
            )
            low_transformer = WanTransformer3DModel.from_single_file(
                low_local, torch_dtype=torch_dtype
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal
            return (
                f"Distill load failed ({type(exc).__name__}: {exc}). "
                "Pipeline continuing with stock Wan transformers."
            )

        if not hasattr(pipeline, "transformer"):
            return (
                "Distill swap skipped: pipeline has no .transformer attribute. "
                "This Wan distill path requires a WanPipeline-shaped object."
            )

        pipeline.transformer = high_transformer
        if hasattr(pipeline, "transformer_2"):
            pipeline.transformer_2 = low_transformer
        else:
            return (
                f"Distill: high-noise expert applied, but pipeline lacks "
                f"transformer_2 (low-noise expert). Verify base repo {repo} "
                "is the A14B MoE pipeline. Quality may be degraded."
            )

        return (
            f"Distill: swapped transformer + transformer_2 from {repo} "
            f"(precision={precision}, 4-step schedule)."
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
        cuda_module = getattr(torch, "cuda", None)
        if cuda_module is not None:
            try:
                if getattr(cuda_module, "is_available", lambda: False)():
                    return "cuda"
            except Exception:
                pass
        cuda_error = _windows_cuda_unavailable_message(torch)
        if cuda_error:
            raise RuntimeError(cuda_error)
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
            return "mps"
        return "cpu"

    def _preferred_torch_dtype(self, torch: Any, device: str) -> Any:
        if device == "cuda":
            return torch.bfloat16
        if device == "mps":
            # M2 and newer support bf16 on MPS; M1 silently downcasts to
            # fp16 inside operators which costs accuracy on long DiT
            # sequences. Probe the capability with a one-element tensor —
            # if MPS rejects it, fall back to fp16 cleanly. Honour an env
            # opt-out so we have a rollback lever if a future MPS update
            # regresses.
            if os.getenv("CHAOSENGINE_VIDEO_MPS_BF16") == "0":
                return torch.float16
            try:
                probe = torch.zeros(1, dtype=torch.bfloat16, device="mps")
                del probe
                return torch.bfloat16
            except (RuntimeError, NotImplementedError, TypeError):
                return torch.float16
        return torch.float32


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
                with self._lock:
                    video = mlx.generate(config)
                    runtime = mlx.probe().to_dict()
                return video, runtime
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
