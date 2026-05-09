from __future__ import annotations

import json
import importlib.util
import io
import os
import platform
import textwrap
import time
import gc
import secrets

from backend_service.helpers.gpu import (
    nvidia_gpu_present as _nvidia_gpu_present,
    torch_install_warning as _torch_install_warning,
)
from colorsys import hsv_to_rgb
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from backend_service.progress import (
    GenerationCancelled,
    IMAGE_PROGRESS,
    PHASE_DECODING,
    PHASE_DIFFUSING,
    PHASE_ENCODING,
    PHASE_LOADING,
    PHASE_SAVING,
)
from backend_service.image_runtime.device import (
    _guess_expected_device,
    _is_cuda_torch_unavailable_error,
    _resolve_image_python as _resolve_image_python_impl,
    _windows_cuda_unavailable_message,
)
from backend_service.image_runtime.placeholder_engine import (
    MAX_IMAGE_SEED,
    PlaceholderImageEngine,
    _mix_channel,
    _resolve_base_seed,
    _rgb_from_hsv,
    _stable_hash,
)
from backend_service.image_runtime.repos import (
    _AYS_TIMESTEPS,
    _SAMPLER_REGISTRY,
    _SDXL_VAE_FIX_REPO,
    _apply_scheduler,
    _gguf_transformer_class_for_repo,
    _is_flow_matching_repo,
    _is_flux_repo,
    _is_sdxl_repo,
    _locate_sdxl_vae_fix_snapshot,
    _nunchaku_transformer_class_for_repo,
)
from backend_service.image_runtime.snapshot import (
    _snapshot_retry_guidance,
    _snapshot_visible_label,
    validate_local_diffusers_snapshot,
)
from backend_service.image_runtime.types import (
    GeneratedImage,
    ImageGenerationConfig,
    ImageRuntimeStatus,
)


def _resolve_image_python() -> str:
    """Wrapper preserving the no-arg signature of the original helper."""
    return _resolve_image_python_impl(WORKSPACE_ROOT)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]








































class DiffusersTextToImageEngine:
    runtime_label = "Diffusers local engine"

    def __init__(self) -> None:
        self._lock = RLock()
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._loaded_repo: str | None = None
        self._loaded_path: str | None = None
        self._loaded_variant_key: str | None = None
        self._device: str | None = None
        # FU-017 / FU-019 / FU-016: notes accumulated during pipeline load
        # (VAE swap, LoRA fuse, attention backend). Surfaced as part of
        # ``runtimeNote`` on every GeneratedImage produced by the loaded
        # pipeline so the user sees what was applied without polling
        # capabilities mid-batch. Reset on each pipeline load.
        self._load_notes: list[str] = []

    def probe(self) -> ImageRuntimeStatus:
        # Deliberately does NOT ``import torch`` — that would load
        # torch/lib/*.dll into the backend process handle table, and on
        # Windows those locked DLLs break /api/setup/install-gpu-bundle
        # (pip's rmtree can't remove files another process has open).
        # find_spec answers "is it installable?" without triggering the
        # import side effects. Device detection (cuda vs cpu) is deferred
        # to preload/generate where we're about to import torch anyway.
        #
        # ``invalidate_caches`` matters when the GPU bundle install has
        # finished mid-process: pip writes the new packages into the
        # extras dir (already on ``sys.path`` from process start), but
        # ``importlib`` keeps a per-finder cache of negative lookups, so
        # the find_spec calls below would still report None even though
        # the .dist-info folders are sitting on disk. Calling
        # ``invalidate_caches`` first re-walks the path entries so the
        # newly installed packages are picked up without a process
        # restart.
        importlib.invalidate_caches()
        missing = [
            package
            for package, module_name in (
                ("diffusers", "diffusers"),
                ("torch", "torch"),
                ("accelerate", "accelerate"),
                ("huggingface_hub", "huggingface_hub"),
                ("pillow", "PIL"),
            )
            if importlib.util.find_spec(module_name) is None
        ]
        if missing:
            message = (
                "Install the GPU image runtime packages to enable real local generation. "
                "Click the 'Install GPU runtime' button above."
            )
            return ImageRuntimeStatus(
                activeEngine="placeholder",
                realGenerationAvailable=False,
                missingDependencies=missing,
                pythonExecutable=_resolve_image_python(),
                message=message,
                loadedModelRepo=self._loaded_repo,
                torchInstallWarning=_torch_install_warning(),
            )

        message = (
            "Real local generation is available. Download an image model locally, then Image Studio "
            "will use the diffusers runtime instead of the placeholder engine."
        )
        device_memory_gb: float | None = None
        try:
            from backend_service.helpers.gpu import get_device_vram_total_gb
            device_memory_gb = get_device_vram_total_gb()
        except Exception:
            device_memory_gb = None
        return ImageRuntimeStatus(
            activeEngine="diffusers",
            realGenerationAvailable=True,
            # ``device`` is the *currently-loaded* model's device, or None
            # if no model is loaded. We no longer speculatively import
            # torch just to report cuda/mps/cpu availability in the empty
            # case — users find out on first Generate which is cheap.
            device=self._device,
            expectedDevice=_guess_expected_device(),
            pythonExecutable=_resolve_image_python(),
            message=message,
            loadedModelRepo=self._loaded_repo,
            deviceMemoryGb=device_memory_gb,
            torchInstallWarning=_torch_install_warning(),
        )

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        # Begin reporting progress before we touch the pipeline. ``_ensure_pipeline``
        # publishes its own ``loading`` phase if it actually has to materialise
        # the pipeline, but we still want a tracker entry from the moment the
        # request lands so the UI's first poll has something to render.
        IMAGE_PROGRESS.begin(
            run_label=self._format_run_label(config),
            total_steps=max(1, int(config.steps)),
            phase=PHASE_LOADING,
            message=f"Preparing {config.modelName}",
        )
        try:
            pipeline = self._ensure_pipeline(
                config.repo,
                gguf_repo=config.ggufRepo,
                gguf_file=config.ggufFile,
                lora_repo=config.loraRepo,
                lora_file=config.loraFile,
                lora_scale=config.loraScale,
                preview_vae=config.previewVae,
                nunchaku_repo=config.nunchakuRepo,
                nunchaku_file=config.nunchakuFile,
                fp8_layerwise_casting=config.fp8LayerwiseCasting,
            )
            # Early-cancel check: the load phase is blocking (from_pretrained
            # is a C-extension call we can't interrupt), so if the user hit
            # Cancel during it we catch up here and bail before kicking off
            # the T5/VAE passes.
            if IMAGE_PROGRESS.is_cancelled():
                raise GenerationCancelled("Image generation cancelled by user")
            # Apply the user's sampler choice (SD1.5/SDXL only). Flow-matching
            # models (FLUX, SD3, Qwen-Image, Sana, HiDream) ship locked
            # schedulers — silently ignore the sampler there rather than
            # producing noise. The returned note lands on GeneratedImage
            # so users see which sampler was applied.
            sampler_note: str | None = None
            if config.sampler and not _is_flow_matching_repo(config.repo):
                sampler_note = _apply_scheduler(pipeline, config.sampler)
            torch = self._torch
            if torch is None:
                raise RuntimeError("PyTorch was not initialised for the diffusers runtime.")
            IMAGE_PROGRESS.set_phase(PHASE_ENCODING, message="Encoding prompt")
            generator_device = "cpu" if self._device == "mps" else (self._device or "cpu")
            base_seed = _resolve_base_seed(config.seed)
            generators = [
                torch.Generator(device=generator_device).manual_seed(base_seed + index)
                for index in range(config.batchSize)
            ]

            kwargs = self._build_pipeline_kwargs(config, generators if len(generators) > 1 else generators[0])
            lowered_repo = config.repo.lower()
            if "flux" in lowered_repo:
                kwargs.pop("negative_prompt", None)
                kwargs["num_inference_steps"] = min(config.steps, 8)
            if "turbo" in lowered_repo:
                kwargs["num_inference_steps"] = min(config.steps, 8)
                kwargs["guidance_scale"] = min(config.guidance, 2.5)

            # Wire the diffusers per-step callback so the UI sees the bar move
            # in lockstep with denoising, which is the bulk of the wall time on
            # most models. ``callback_on_step_end`` is the non-deprecated name
            # in modern diffusers (>=0.27); some pipelines also accept the
            # legacy ``callback`` arg, but we prefer the new one.
            # AYS path passes ``timesteps=[...]`` instead of
            # ``num_inference_steps`` — derive the step count from the
            # array length so the progress bar / decay schedule still
            # report the right total.
            if isinstance(kwargs.get("timesteps"), list):
                total_steps = len(kwargs["timesteps"])
            else:
                total_steps = int(kwargs.get("num_inference_steps", config.steps) or config.steps)
            IMAGE_PROGRESS.set_phase(
                PHASE_DIFFUSING,
                message=self._diffuse_message(config),
            )
            # Re-publish the totalSteps in case ``num_inference_steps`` was
            # clamped above (Flux/Turbo cap at 8).
            IMAGE_PROGRESS.set_step(0, total=max(1, total_steps))

            # TeaCache / other diffusion cache strategies hook here: the
            # pipeline is loaded, num_inference_steps is final, and we
            # haven't kicked off the forward pass yet. If the selected
            # strategy isn't applicable to images or hasn't landed a patch
            # for this pipeline yet we swallow NotImplementedError and run
            # the stock pipeline — the UI surfaces the "Scaffold" badge so
            # users know why speedup didn't appear.
            from cache_compression import apply_diffusion_cache_strategy

            cache_note = apply_diffusion_cache_strategy(
                pipeline,
                strategy_id=config.cacheStrategy,
                num_inference_steps=total_steps,
                rel_l1_thresh=config.cacheRelL1Thresh,
                domain="image",
            )
            if cache_note:
                # Surface for log only; sampler_note already owns the
                # runtime_note slot on GeneratedImage. Adding cache noise
                # to every image's metadata would flood the gallery UI.
                pass

            # FU-021: CFG decay schedule for flow-match image pipelines.
            # Same shape as the video-runtime ramp — linear from initial
            # guidance to a 1.5 floor that keeps
            # ``do_classifier_free_guidance`` True for the entire schedule
            # (dropping below 1.0 mid-loop swaps the pipeline from
            # 2-batch to 1-batch shape and produces shape-mismatch
            # crashes; 1.5 is the documented floor we use on video).
            # Gated to flow-match so SD1.5 / SDXL stay on constant CFG.
            decay_floor = 1.5
            initial_guidance = float(kwargs.get("guidance_scale", config.guidance) or config.guidance)
            decay_active = (
                config.cfgDecay
                and _is_flow_matching_repo(config.repo)
                and total_steps > 1
                and initial_guidance > decay_floor
            )

            # FU-018 part 2: live denoise thumbnails. Emit a base64 PNG
            # of the current latent every Nth step when previewVae is on
            # (the swap to TAESD makes per-step decode cheap enough to do
            # without dragging total wall time). Stride keeps the polled
            # endpoint payload manageable on long schedules — 50 steps at
            # one decode each would push 1.5 MB of base64 through the
            # poller per gen. Always emit on the final step.
            thumb_active = bool(config.previewVae)
            thumb_stride = max(1, total_steps // 8) if thumb_active else 1

            def _on_step_end(_pipeline: Any, step: int, _timestep: Any, callback_kwargs: dict[str, Any]):
                # Diffusers calls this *after* step ``step`` finishes, so step
                # 0 means "one step done". Convert to the 1-indexed value the
                # UI wants to display.
                IMAGE_PROGRESS.set_step(step + 1, total=max(1, total_steps))
                # Cooperative cancel: the Cancel button on the modal sets
                # IMAGE_PROGRESS.request_cancel(); we honor it at the next
                # step boundary by setting ``_interrupt``, which makes
                # diffusers stop the denoising loop cleanly at the next
                # iteration. We also raise here so the outer handler can
                # see a cancellation came from the user (not a pipeline
                # crash) and return the right response.
                if IMAGE_PROGRESS.is_cancelled():
                    try:
                        _pipeline._interrupt = True
                    except Exception:
                        pass
                    raise GenerationCancelled("Image generation cancelled by user")
                if decay_active:
                    next_step = step + 1
                    progress = min(1.0, next_step / max(1, total_steps - 1))
                    next_scale = (
                        initial_guidance * (1.0 - progress)
                        + decay_floor * progress
                    )
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
                                decode_image_latent_to_b64,
                            )
                            b64 = decode_image_latent_to_b64(_pipeline, latents)
                            if b64 is not None:
                                IMAGE_PROGRESS.set_thumbnail(b64)
                        except Exception:
                            # Thumbnail decode is best-effort — never fail
                            # the actual generation because of a preview
                            # decode error.
                            pass
                return callback_kwargs

            kwargs.setdefault("callback_on_step_end", _on_step_end)

            started = time.perf_counter()
            try:
                result = pipeline(**kwargs)
            except TypeError as exc:
                # Older diffusers versions don't accept ``callback_on_step_end``
                # — drop it and retry once before bubbling the original error.
                if "callback_on_step_end" in str(exc):
                    kwargs.pop("callback_on_step_end", None)
                    try:
                        result = pipeline(**kwargs)
                    except TypeError:
                        kwargs.pop("negative_prompt", None)
                        result = pipeline(**kwargs)
                else:
                    kwargs.pop("negative_prompt", None)
                    result = pipeline(**kwargs)
            elapsed = max(0.1, time.perf_counter() - started)

            IMAGE_PROGRESS.set_phase(PHASE_DECODING, message="Decoding pixels")

            artifacts: list[GeneratedImage] = []
            for index, image in enumerate(getattr(result, "images", []) or []):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                if image.getbbox() is None:
                    raise RuntimeError(
                        "The image runtime returned an all-black frame instead of a real image. "
                        f"Model: {config.repo}. Device: {self._device or 'cpu'}. "
                        "Try restarting the backend and generating again. If this keeps happening on Apple Silicon, "
                        "the model likely needs a safer precision path."
                    )
                buffer = io.BytesIO()
                image.save(buffer, format="PNG", optimize=True)
                # Combine all per-load notes (VAE swap, LoRA fuse,
                # attention backend) with the per-generate sampler note.
                # Joined with " · " so the UI can show a single line.
                note_parts: list[str] = list(self._load_notes)
                if sampler_note:
                    note_parts.append(sampler_note)
                if cache_note:
                    note_parts.append(cache_note)
                runtime_note = " · ".join(note_parts) if note_parts else None
                artifacts.append(
                    GeneratedImage(
                        seed=base_seed + index,
                        bytes=buffer.getvalue(),
                        extension="png",
                        mimeType="image/png",
                        durationSeconds=round(elapsed / max(1, config.batchSize), 1),
                        runtimeLabel=f"{self.runtime_label} ({self._device or 'cpu'})",
                        runtimeNote=runtime_note,
                    )
                )
            if not artifacts:
                raise RuntimeError("Diffusers returned no images.")
            IMAGE_PROGRESS.set_phase(PHASE_SAVING, message="Saving to gallery")
            return artifacts
        finally:
            IMAGE_PROGRESS.finish()

    def _diffuse_message(self, config: ImageGenerationConfig) -> str:
        if config.batchSize > 1:
            return f"Diffusing {config.batchSize} images"
        return "Diffusing image"

    def _format_run_label(self, config: ImageGenerationConfig) -> str:
        return f"{config.modelName} · {config.width}x{config.height}"

    def preload(self, repo: str) -> ImageRuntimeStatus:
        self._ensure_pipeline(repo)
        return self.probe()

    def unload(self, repo: str | None = None) -> ImageRuntimeStatus:
        with self._lock:
            if repo and self._loaded_repo != repo:
                return self.probe()
            self._release_pipeline()
            return self.probe()

    def _ensure_pipeline(
        self,
        repo: str,
        gguf_repo: str | None = None,
        gguf_file: str | None = None,
        lora_repo: str | None = None,
        lora_file: str | None = None,
        lora_scale: float | None = None,
        preview_vae: bool = False,
        nunchaku_repo: str | None = None,
        nunchaku_file: str | None = None,
        fp8_layerwise_casting: bool = False,
    ) -> Any:
        with self._lock:
            # Variant key folds LoRA identity in too — switching LoRAs
            # on the same base repo must rebuild the pipeline because
            # ``fuse_lora`` mutates the transformer weights in place.
            # ``preview_vae`` joins the same key set so toggling the
            # FU-018 preview-decode knob triggers a clean rebuild.
            variant_parts = [repo]
            if gguf_file:
                variant_parts.append(f"gguf={gguf_file}")
            if lora_repo and lora_file:
                variant_parts.append(f"lora={lora_repo}/{lora_file}@{lora_scale or 1.0}")
            if preview_vae:
                variant_parts.append("preview_vae")
            if nunchaku_repo:
                variant_parts.append(
                    f"nunchaku={nunchaku_repo}{'/' + nunchaku_file if nunchaku_file else ''}"
                )
            if fp8_layerwise_casting:
                variant_parts.append("fp8_layerwise")
            variant_key = "::".join(variant_parts)
            if self._pipeline is not None and self._loaded_variant_key == variant_key:
                return self._pipeline

            # Loading a pipeline can take 10-60s on cold disk. Surface that
            # explicitly to the UI so the progress bar stops sitting at 0%
            # while we read 5GB of weights from the SSD.
            IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=f"Loading {repo}")

            if self._pipeline is not None and self._loaded_variant_key != variant_key:
                self._release_pipeline()

            import torch  # type: ignore
            from diffusers import AutoPipelineForText2Image  # type: ignore
            from huggingface_hub import snapshot_download  # type: ignore

            local_path = snapshot_download(
                repo_id=repo,
                local_files_only=True,
                resume_download=True,
            )
            local_root = Path(local_path)
            validation_error = validate_local_diffusers_snapshot(local_root, repo)
            if validation_error is not None:
                raise RuntimeError(validation_error)
            detected_device = self._detect_device(torch)
            device = self._preferred_execution_device(repo, detected_device)
            # FU-017: probe the SDXL fp16-fix VAE before deciding dtype so
            # SDXL on MPS can stay on fp16 when the fix snapshot is cached.
            # Probe only fires for SDXL repos on devices that actually
            # benefit (MPS / CUDA) — CPU stays on fp32 regardless.
            sdxl_vae_fix_path: str | None = None
            if _is_sdxl_repo(repo) and device in ("mps", "cuda"):
                sdxl_vae_fix_path = _locate_sdxl_vae_fix_snapshot()
            dtype = self._preferred_torch_dtype(
                torch, repo, device,
                sdxl_vae_fix_available=sdxl_vae_fix_path is not None,
            )
            use_cpu_offload = self._should_use_model_cpu_offload(repo, device)
            # Clear load notes on each pipeline (re)load so stale entries
            # from a previously-loaded model don't bleed into new outputs.
            self._load_notes = []

            # Three transformer-loading strategies, in preference order:
            #   1. GGUF (cross-platform, any quant level the user picked)
            #   2. NF4 via bitsandbytes (CUDA-only, FLUX-only, ~7 GB)
            #   3. Full-precision transformer bundled into the base pipeline
            # GGUF wins when the variant asked for it because the user's
            # quant choice is explicit; NF4 remains the default for FLUX
            # on CUDA when no GGUF file was specified.
            pipeline_kwargs: dict[str, Any] = {}
            gguf_note: str | None = None
            nunchaku_note: str | None = None
            if gguf_file:
                IMAGE_PROGRESS.set_phase(
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
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=gguf_note)
            # FU-023 Nunchaku / SVDQuant — preferred path on CUDA when the
            # variant pins a Nunchaku snapshot. Wins over NF4 / int8wo by
            # roughly 3× on FLUX.1-dev. CUDA only; the helper falls back to
            # the standard transformer when nunchaku isn't installed or the
            # device is mps/cpu so the rest of the runtime keeps working.
            if (
                "transformer" not in pipeline_kwargs
                and nunchaku_repo
                and device == "cuda"
            ):
                IMAGE_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message=f"Loading Nunchaku SVDQuant transformer {nunchaku_repo}",
                )
                quantized_transformer, nunchaku_note = self._try_load_nunchaku_transformer(
                    repo=repo,
                    nunchaku_repo=nunchaku_repo,
                    nunchaku_file=nunchaku_file,
                    torch=torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if nunchaku_note:
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=nunchaku_note)
            if (
                "transformer" not in pipeline_kwargs
                and device == "mps"
                and _is_flux_repo(repo)
            ):
                # MPS has no bitsandbytes/NF4 path — int8wo is the
                # cross-platform fallback that still halves FLUX's
                # memory footprint on Apple Silicon.
                IMAGE_PROGRESS.set_phase(
                    PHASE_LOADING,
                    message=f"Quantizing {repo} transformer to int8",
                )
                quantized_transformer, note = self._try_load_int8wo_flux_transformer(
                    local_path, torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if note:
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=note)
            if "transformer" not in pipeline_kwargs and use_cpu_offload:
                IMAGE_PROGRESS.set_phase(
                    PHASE_LOADING, message=f"Quantizing {repo} transformer to NF4",
                )
                quantized_transformer, note = self._try_load_nf4_flux_transformer(
                    local_path, torch,
                )
                if quantized_transformer is not None:
                    pipeline_kwargs["transformer"] = quantized_transformer
                if note:
                    IMAGE_PROGRESS.set_phase(PHASE_LOADING, message=note)

            pipeline = AutoPipelineForText2Image.from_pretrained(
                local_path,
                torch_dtype=dtype,
                local_files_only=True,
                **pipeline_kwargs,
            )
            # The safety checker adds extra vision-model dependencies and can
            # fail on tiny or oddly shaped test pipelines. For the local app
            # MVP we prioritize generation reliability over post-filtering.
            if hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None
            if hasattr(pipeline, "feature_extractor"):
                pipeline.feature_extractor = None
            if hasattr(pipeline, "requires_safety_checker"):
                pipeline.requires_safety_checker = False
            if hasattr(pipeline, "set_progress_bar_config"):
                pipeline.set_progress_bar_config(disable=True)

            # FU-017: swap in madebyollin's SDXL VAE fp16-fix when the
            # snapshot is cached. The pipeline already loaded with fp16
            # weights (decided above) so the VAE swap is the load-bearing
            # piece — without it the stock SDXL VAE silently NaN-overflows
            # on the fp16 sigmoid and outputs black images on MPS / consumer
            # CUDA. Failure modes (corrupt snapshot, dtype mismatch) fall
            # back to the original VAE so the user still gets *some* image.
            if sdxl_vae_fix_path and getattr(pipeline, "vae", None) is not None:
                try:
                    from diffusers import AutoencoderKL  # type: ignore
                    fix_vae = AutoencoderKL.from_pretrained(
                        sdxl_vae_fix_path,
                        torch_dtype=torch.float16,
                        local_files_only=True,
                    )
                    pipeline.vae = fix_vae
                    self._load_notes.append("VAE: SDXL fp16-fix")
                except Exception as exc:  # noqa: BLE001 — fall back to stock VAE
                    self._load_notes.append(
                        f"SDXL VAE fp16-fix swap failed ({type(exc).__name__}); using stock VAE."
                    )

            # FU-016: SageAttention CUDA backend. No-op on MPS / CPU and
            # when the pipeline lacks ``transformer.set_attention_backend``.
            # Stacks multiplicatively with FBCache. Must run *before*
            # placement so the kernel selection is locked in before the
            # first forward pass.
            try:
                from backend_service.helpers.attention_backend import (
                    maybe_apply_sage_attention,
                )
                sage_note = maybe_apply_sage_attention(pipeline)
                if sage_note:
                    self._load_notes.append(sage_note)
            except Exception:
                # Helper is wrapped in its own try/except; any leakage
                # here is a bug in the helper, not a runtime concern.
                pass

            # FU-018: TAESD preview-decode VAE swap. No-op when toggle
            # is off or no preview VAE is mapped for this repo. Runs
            # before LoRA fuse so the LoRA's adapter modules don't trip
            # the VAE swap (they target the transformer, not the VAE,
            # but ordering keeps the swap close to other VAE-touching
            # code like the SDXL fp16-fix above).
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

            # FU-024 FP8 layerwise casting (CUDA SM 8.9+ / Ada+ / Hopper+).
            # Halves transformer VRAM by storing weights in fp8 and
            # promoting to bf16 only inside the matmul. Diffusers exposes
            # ``enable_layerwise_casting`` on every flow-match DiT we ship.
            # Family-correct fp8 dtype: E4M3 for FLUX / Wan / Qwen-Image,
            # E5M2 for HunyuanVideo (hunyuan team's recommendation in
            # their model card). No-op outside CUDA.
            if fp8_layerwise_casting and device == "cuda":
                try:
                    fp8_note = self._maybe_enable_fp8_layerwise(
                        pipeline, repo=repo, torch=torch,
                    )
                    if fp8_note:
                        self._load_notes.append(fp8_note)
                except Exception as exc:  # noqa: BLE001 — any failure → bf16
                    self._load_notes.append(
                        f"FP8 layerwise casting failed ({type(exc).__name__}: "
                        f"{exc}) — running bf16."
                    )

            # FU-019: distill LoRAs (Hyper-SD FLUX, alimama FLUX.1-Turbo,
            # lightx2v Wan CausVid). Load + fuse at pipeline build time
            # so subsequent ``pipeline(...)`` calls run with the LoRA
            # baked into the transformer — no per-generate fuse cost.
            # ``unload_lora_weights`` after fuse drops the un-fused
            # state dict from RAM (the fused weights live in the
            # transformer itself).
            if lora_repo and lora_file:
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
                        # Best-effort cleanup — older diffusers don't
                        # always succeed at unloading after fuse, and
                        # the fused transformer is correct either way.
                        pass
                    self._load_notes.append(
                        f"LoRA: {lora_repo}/{lora_file} @ scale {effective_scale:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001 — non-fatal
                    self._load_notes.append(
                        f"LoRA load failed ({type(exc).__name__}: {exc}). "
                        "Pipeline continuing without LoRA."
                    )

            if use_cpu_offload:
                # Diffusers' stock recipe for FLUX on <32 GB VRAM: keep only
                # the active component (T5, then transformer, then VAE) on
                # GPU, transferring at component boundaries. Do NOT combine
                # with attention/VAE slicing or .to(device) — slicing issues
                # many tiny kernel launches that saturate PCIe when the
                # active weights are already being DMA'd in, and .to(device)
                # would pin all 33 GB of FLUX weights in VRAM at once
                # (exceeds even a 4090) causing fallback-to-pagefile thrash.
                # Real-world signature of doing it wrong: GPU at 97% util
                # but step 0/8 never completing.
                pipeline.enable_model_cpu_offload()
            else:
                if hasattr(pipeline, "enable_attention_slicing"):
                    pipeline.enable_attention_slicing()
                vae = getattr(pipeline, "vae", None)
                if vae is not None and hasattr(vae, "enable_slicing"):
                    vae.enable_slicing()
                # VAE tiling is a no-op at low resolution (diffusers only
                # activates it when the latent exceeds the VAE's sample_size),
                # so enabling it unconditionally costs nothing at 1024px but
                # prevents the VAE decode from OOM-ing at 1536/2048px on
                # MPS / 8-12 GB CUDA cards. Same pattern as video_runtime.
                if vae is not None and hasattr(vae, "enable_tiling"):
                    vae.enable_tiling()
                if device != "cpu":
                    pipeline = pipeline.to(device)

            self._pipeline = pipeline
            self._torch = torch
            self._loaded_repo = repo
            self._loaded_path = local_path
            self._loaded_variant_key = variant_key
            self._device = device
            return pipeline

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

    def _preferred_torch_dtype(
        self,
        torch: Any,
        repo: str,
        device: str,
        sdxl_vae_fix_available: bool = False,
    ) -> Any:
        if device == "cuda":
            # FLUX was trained and validated in bfloat16. Loading it as
            # float16 produces slightly off saturations and occasional
            # NaN-propagation on long prompts — not catastrophic, but the
            # official Black Forest recipe is bfloat16 and we should match
            # it so output quality is on-spec.
            if _is_flux_repo(repo):
                return torch.bfloat16
            return torch.float16
        if device == "mps":
            lowered_repo = repo.lower()
            # SDXL / Stable Diffusion on MPS can silently decode to black
            # images in fp16 due to the stock SDXL VAE overflowing the
            # fp16 sigmoid. FU-017: when madebyollin/sdxl-vae-fp16-fix is
            # cached locally we swap that VAE in and stay on fp16 (≈2×
            # faster than fp32). Without the fix snapshot we keep the
            # safe fp32 fallback so users still get correct images.
            if any(token in lowered_repo for token in ("stable-diffusion", "sdxl", "sd_xl")):
                if sdxl_vae_fix_available and _is_sdxl_repo(repo):
                    return torch.float16
                return torch.float32
            return torch.float16
        return torch.float32

    def _preferred_execution_device(self, repo: str, detected_device: str) -> str:
        lowered_repo = repo.lower()
        # Qwen-Image's official quick start uses CUDA+bfloat16, otherwise CPU+float32.
        # On Apple MPS, users report black outputs with the naive fp16 path, so prefer
        # the safer CPU execution path instead of silently returning placeholder frames.
        if detected_device == "mps" and "qwen-image" in lowered_repo:
            return "cpu"
        return detected_device

    def _try_load_nf4_flux_transformer(
        self, local_path: str, torch: Any,
    ) -> tuple[Any, str | None]:
        """Load FLUX's transformer quantized to NF4 via bitsandbytes.

        NF4 (4-bit NormalFloat) drops the 12B FLUX transformer from ~24 GB
        (bf16) to ~7 GB with negligible visual quality loss — the exact
        pattern the FLUX community runs on 24 GB consumer GPUs. T5-XXL and
        the VAE are NOT quantized (they're small enough, and quantizing
        text encoders hurts prompt adherence more than it saves memory).

        Returns ``(transformer, note)``. A ``None`` transformer means the
        caller should fall back to the unquantized pipeline — typically
        because bitsandbytes isn't installed yet or the diffusers version
        predates the ``quantization_config`` plumbing. The note is a user-
        visible progress message explaining which path was taken.
        """
        if importlib.util.find_spec("bitsandbytes") is None:
            return None, (
                "bitsandbytes missing — FLUX will load in bf16. "
                "Install it from the Setup page to enable NF4 quantization "
                "(turns 8 min/step into ~10 s/step on a 24 GB GPU)."
            )
        try:
            from diffusers import BitsAndBytesConfig, FluxTransformer2DModel  # type: ignore
        except ImportError:
            return None, (
                "Installed diffusers doesn't expose BitsAndBytesConfig. "
                "Upgrade via the Setup page to use NF4 FLUX."
            )

        try:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                local_path,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            return transformer, "FLUX transformer loaded in NF4 (~7 GB VRAM)"
        except Exception as exc:  # noqa: BLE001 — any failure → fall back to bf16
            # Any error here (missing subfolder, CUDA kernel mismatch,
            # bitsandbytes CPU-only wheel) falls back to the unquantized
            # path rather than breaking image generation entirely.
            return None, (
                f"NF4 quantization failed ({type(exc).__name__}: {exc}) — "
                "falling back to bf16 transformer (slower on <32 GB GPUs)."
            )

    def _try_load_int8wo_flux_transformer(
        self, local_path: str, torch: Any,
    ) -> tuple[Any, str | None]:
        """Load FLUX's transformer with TorchAO int8 weight-only quant.

        int8wo is the Apple-Silicon counterpart to bitsandbytes NF4:
        bitsandbytes ships CUDA kernels only, so an MPS FLUX run would
        otherwise need 24 GB bf16 weights and pagefile-thrash on any
        Mac under 48 GB. int8wo drops that to ~12 GB — not as tight as
        NF4's ~7 GB but wide enough for 32 GB M-series machines.

        Returns ``(transformer, note)`` with the same contract as the
        NF4 helper: ``None`` transformer means the caller should fall
        back, note is a human-readable progress message.
        """
        if importlib.util.find_spec("torchao") is None:
            return None, (
                "torchao missing — FLUX will load in bf16 on MPS. "
                "Install it from the Setup page to enable int8 "
                "quantization (~24 GB → ~12 GB)."
            )
        try:
            from diffusers import FluxTransformer2DModel, TorchAoConfig  # type: ignore
        except ImportError:
            return None, (
                "Installed diffusers doesn't expose TorchAoConfig. "
                "Upgrade via the Setup page to use int8wo FLUX."
            )
        try:
            transformer = FluxTransformer2DModel.from_pretrained(
                local_path,
                subfolder="transformer",
                quantization_config=TorchAoConfig("int8wo"),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            return transformer, "FLUX transformer loaded in int8wo (~12 GB)"
        except Exception as exc:  # noqa: BLE001 — fall back to bf16
            return None, (
                f"int8wo quantization failed ({type(exc).__name__}: {exc}) — "
                "falling back to bf16."
            )

    def _try_load_gguf_transformer(
        self,
        repo: str,
        gguf_repo: str,
        gguf_file: str,
        torch: Any,
    ) -> tuple[Any, str | None]:
        """Load a transformer from a single ``.gguf`` file via diffusers.

        GGUF wins over NF4 for two reasons: it works on Apple Silicon / CPU
        (bitsandbytes is CUDA-only), and the community ships a spread of
        quant levels (Q2_K … Q8_0) so the user can trade quality for VRAM
        at a finer granularity than NF4's single 4-bit point.

        The VAE and text encoders still come from the base ``repo``
        snapshot — GGUF files only carry the transformer/DiT weights.

        Returns ``(transformer, note)``. A ``None`` transformer means the
        caller should fall back (NF4 or bf16). Any failure here is
        non-fatal: missing ``gguf`` pip package, an old diffusers without
        ``GGUFQuantizationConfig``, or an HF cache miss for the chosen
        quant file will all route to the standard pipeline.
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

        # Pick the transformer class from the base repo. Most flow-matching
        # image models expose a dedicated DiT class; for SD1.5/SDXL the
        # GGUF community uses the UNet path which we don't support here —
        # those pipelines stay on the standard loader.
        transformer_cls_name = _gguf_transformer_class_for_repo(repo)
        if transformer_cls_name is None:
            return None, (
                f"No GGUF transformer class registered for {repo}. "
                "Add a mapping in image_runtime._gguf_transformer_class_for_repo."
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
            # Pin the architecture config to the base repo's
            # ``transformer/config.json`` — without this hint
            # ``from_single_file`` falls back to the transformer class's
            # default layout, which is fine for the largest variant in a
            # family but breaks smaller variants (different cross-attn
            # dim, hidden size, layer count). Mirrors the video-side
            # loader. See ``backend_service/video_runtime.py``'s
            # ``_try_load_gguf_transformer`` for the Wan 2.2 5B repro
            # that motivated the fix.
            transformer = transformer_cls.from_single_file(
                gguf_local_path,
                quantization_config=GGUFQuantizationConfig(
                    compute_dtype=torch.bfloat16,
                ),
                torch_dtype=torch.bfloat16,
                config=repo,
                subfolder="transformer",
            )
            return transformer, (
                f"Transformer loaded from GGUF ({gguf_file})"
            )
        except Exception as exc:  # noqa: BLE001 — any failure → fall back
            return None, (
                f"GGUF load failed ({type(exc).__name__}: {exc}) — "
                "falling back to the standard transformer."
            )

    def _should_use_model_cpu_offload(self, repo: str, device: str) -> bool:
        """True when the pipeline should load via enable_model_cpu_offload().

        Currently limited to FLUX on CUDA. FLUX.1-Dev is ~24 GB transformer
        plus ~9 GB T5-XXL text encoder in bf16; on any single consumer GPU
        (≤32 GB VRAM) a plain ``pipeline.to("cuda")`` either OOMs or, worse
        on Windows, silently falls back to pinned host memory + pagefile
        and runs at PCIe speeds — which is what "GPU at 97% but step 0/8
        never completes" looks like. enable_model_cpu_offload swaps whole
        components (not layers) at module boundaries, which is the
        diffusers-recommended pattern for FLUX on consumer hardware.

        Other pipelines (SD 1.5 / SDXL / Qwen-Image) fit comfortably and
        stay on the legacy .to(device) path for best throughput.
        """
        if device != "cuda":
            return False
        return _is_flux_repo(repo)

    def _build_pipeline_kwargs(self, config: ImageGenerationConfig, generator: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": config.prompt,
            "width": config.width,
            "height": config.height,
            "num_inference_steps": config.steps,
            "guidance_scale": config.guidance,
            "num_images_per_prompt": config.batchSize,
            "generator": generator,
        }
        # FU-020: when the user picked an AYS sampler,
        # ``_apply_scheduler`` stashed the precomputed timestep array on
        # the pipeline. Diffusers accepts ``timesteps=`` as an explicit
        # override; when present it takes precedence over
        # ``num_inference_steps`` so we drop the latter to avoid the
        # "got both" warning.
        pipeline = self._pipeline
        if pipeline is not None:
            ays_timesteps = getattr(pipeline, "_chaosengine_ays_timesteps", None)
            if ays_timesteps:
                kwargs["timesteps"] = list(ays_timesteps)
                kwargs.pop("num_inference_steps", None)
        lowered_repo = config.repo.lower()
        if "qwen-image" in lowered_repo:
            kwargs.pop("guidance_scale", None)
            kwargs["true_cfg_scale"] = config.guidance
            # Qwen-Image expects a negative prompt value, even if it is intentionally blank.
            kwargs["negative_prompt"] = config.negativePrompt if config.negativePrompt else " "
            return kwargs
        if config.negativePrompt.strip():
            kwargs["negative_prompt"] = config.negativePrompt
        return kwargs

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

    def _try_load_nunchaku_transformer(
        self,
        repo: str,
        nunchaku_repo: str,
        nunchaku_file: str | None,
        torch: Any,
    ) -> tuple[Any, str | None]:
        """FU-023: load a Nunchaku SVDQuant transformer for FLUX / Qwen-Image
        / SD3.5 / SANA / PixArt-Σ. CUDA only.

        Nunchaku ships dedicated transformer subclasses
        (``NunchakuFluxTransformer2dModel``, ``NunchakuQwenImageTransformer2DModel``,
        etc.) that load precompiled INT4 SVDQuant weights and expose the
        same forward signature as the stock diffusers transformer, so the
        rest of ``_ensure_pipeline`` keeps working without further
        plumbing. ~3× perf over NF4 on FLUX.1-dev.

        Returns ``(transformer, note)`` matching the NF4 / GGUF helper
        contract — ``None`` transformer means the caller should fall back.
        """
        if importlib.util.find_spec("nunchaku") is None:
            return None, (
                "Nunchaku package not installed — install it from the Setup "
                "page to enable SVDQuant 4-bit on CUDA. Falling back to "
                "the standard transformer."
            )
        cls_name = _nunchaku_transformer_class_for_repo(repo)
        if cls_name is None:
            return None, (
                f"No Nunchaku transformer class registered for {repo}. "
                "Add a mapping in image_runtime._nunchaku_transformer_class_for_repo."
            )
        try:
            import nunchaku  # type: ignore
        except ImportError as exc:
            return None, (
                f"Nunchaku import failed ({exc}). Install nunchaku>=1.2.1 "
                "from the Setup page."
            )
        cls = getattr(nunchaku, cls_name, None)
        if cls is None:
            return None, (
                f"{cls_name} not in installed nunchaku — upgrade via the "
                "Setup page to use this Nunchaku variant."
            )

        try:
            from huggingface_hub import snapshot_download  # type: ignore
            local_dir = snapshot_download(
                repo_id=nunchaku_repo,
                local_files_only=True,
            )
            kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
            if nunchaku_file:
                # Some Nunchaku snapshots ship multiple precision tiers
                # under one repo (e.g. svdq-int4 vs svdq-fp4). When the
                # variant pins a specific filename, pass it through.
                kwargs["filename"] = nunchaku_file
            transformer = cls.from_pretrained(local_dir, **kwargs)
            note = (
                f"Nunchaku SVDQuant transformer loaded from {nunchaku_repo}"
                + (f"/{nunchaku_file}" if nunchaku_file else "")
                + " (CUDA INT4 — ~3× over NF4)."
            )
            return transformer, note
        except Exception as exc:  # noqa: BLE001 — fall through to NF4
            return None, (
                f"Nunchaku load failed ({type(exc).__name__}: {exc}) — "
                "falling back to NF4 / int8wo / bf16."
            )

    def _maybe_enable_fp8_layerwise(
        self,
        pipeline: Any,
        repo: str,
        torch: Any,
    ) -> str | None:
        """FU-024: call ``transformer.enable_layerwise_casting`` with the
        family-correct fp8 dtype. Caller has already gated to CUDA. Pre-Ada
        GPUs lack hardware fp8 support — the cast still runs but generation
        is slower than bf16, so we additionally check the compute capability
        (SM 8.9 = Ada Lovelace, SM 9.0 = Hopper, SM 10.0 = Blackwell).
        Returns a runtimeNote string, or ``None`` when the path no-ops
        cleanly.
        """
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            return "FP8 layerwise skipped: torch.cuda.get_device_capability failed."
        if (major, minor) < (8, 9):
            return (
                f"FP8 layerwise skipped: SM {major}.{minor} pre-dates Ada — "
                "hardware fp8 unavailable. Use bf16 / NF4 / Nunchaku instead."
            )
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None or not hasattr(transformer, "enable_layerwise_casting"):
            return (
                "FP8 layerwise skipped: pipeline.transformer.enable_layerwise_casting "
                "missing — pipeline is UNet-based or the diffusers version is old."
            )
        # E5M2 has wider exponent range (good for activations + outliers),
        # E4M3 has more mantissa bits (better for weights). HunyuanVideo's
        # team published their FP8 weights as E5M2; FLUX / Wan / Qwen-Image
        # / SD3 use E4M3.
        repo_lower = repo.lower()
        if "hunyuan" in repo_lower:
            storage_dtype = torch.float8_e5m2
            storage_label = "E5M2"
        else:
            storage_dtype = torch.float8_e4m3fn
            storage_label = "E4M3"
        try:
            transformer.enable_layerwise_casting(
                storage_dtype=storage_dtype,
                compute_dtype=torch.bfloat16,
            )
        except Exception as exc:
            return (
                f"FP8 layerwise enable failed ({type(exc).__name__}: {exc}) — "
                "running bf16."
            )
        return f"FP8 layerwise casting enabled ({storage_label}, compute=bf16)."


class MfluxImageEngine:
    """Native Apple Silicon FLUX runtime via the ``mflux`` package.

    Only loaded for variants that set ``runtime="mflux"`` in the
    catalog. Compared to diffusers+MPS:

      * 2-3x faster on M-series Macs (native MLX kernels vs the
        PyTorch MPS backend).
      * No fp16 black-image hazard — MLX handles precision cleanly.
      * Limited to FLUX (schnell, dev) — not a diffusers replacement.

    The engine is a quiet no-op on non-Apple platforms: ``probe()``
    reports unavailability, and the manager routes to diffusers
    automatically.
    """

    runtime_label = "mflux (MLX native)"

    def __init__(self) -> None:
        self._flux: Any = None
        self._loaded_name: str | None = None

    def probe(self) -> dict[str, Any]:
        if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
            return {
                "available": False,
                "reason": "mflux runs on Apple Silicon only.",
            }
        if importlib.util.find_spec("mflux") is None:
            return {
                "available": False,
                "reason": (
                    "mflux not installed — add it from the Setup page to "
                    "enable the native Apple Silicon FLUX runtime."
                ),
            }
        return {"available": True, "reason": None}

    def generate(self, config: ImageGenerationConfig) -> list[GeneratedImage]:
        probe = self.probe()
        if not probe["available"]:
            raise RuntimeError(probe["reason"] or "mflux unavailable")

        # Map our repo ids to the names mflux expects. Anything else
        # falls back to the diffusers path.
        flux_name = _mflux_name_for_repo(config.repo)
        if flux_name is None:
            raise RuntimeError(
                f"mflux doesn't support {config.repo} — only FLUX.1-schnell "
                "and FLUX.1-dev are available via the native MLX runtime."
            )

        import mflux  # type: ignore
        started = time.perf_counter()
        if self._flux is None or self._loaded_name != flux_name:
            self._flux = mflux.Flux1.from_name(flux_name)
            self._loaded_name = flux_name
        seed = _resolve_base_seed(config.seed)
        result_image = self._flux.generate_image(
            seed=seed,
            prompt=config.prompt,
            config=mflux.Config(
                num_inference_steps=config.steps,
                height=config.height,
                width=config.width,
                guidance=config.guidance,
            ),
        )
        elapsed = max(0.1, time.perf_counter() - started)

        pil_image = getattr(result_image, "image", result_image)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", optimize=True)
        return [
            GeneratedImage(
                seed=seed,
                bytes=buffer.getvalue(),
                extension="png",
                mimeType="image/png",
                durationSeconds=round(elapsed, 1),
                runtimeLabel=self.runtime_label,
                runtimeNote=f"MLX native FLUX ({flux_name})",
            )
        ]


def _mflux_name_for_repo(repo: str) -> str | None:
    lowered = repo.lower()
    if "flux.1-schnell" in lowered or "flux-schnell" in lowered:
        return "schnell"
    if "flux.1-dev" in lowered or "flux-dev" in lowered:
        return "dev"
    return None


class ImageRuntimeManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._placeholder = PlaceholderImageEngine()
        self._diffusers = DiffusersTextToImageEngine()
        self._mflux = MfluxImageEngine()
        # FU-008 image subset: sd.cpp engine. Wired lazily so the import
        # cost (small) is paid only when the manager is actually
        # constructed. Engine probe is cheap; full binary check happens
        # at generate time.
        from backend_service.sdcpp_image_runtime import SdCppImageEngine
        self._sdcpp = SdCppImageEngine()

    def capabilities(self) -> dict[str, Any]:
        return self._diffusers.probe().to_dict()

    def preload(self, repo: str) -> dict[str, Any]:
        with self._lock:
            status = self._diffusers.probe()
            if not status.realGenerationAvailable:
                raise RuntimeError(status.message)
            return self._diffusers.preload(repo).to_dict()

    def unload(self, repo: str | None = None) -> dict[str, Any]:
        with self._lock:
            return self._diffusers.unload(repo).to_dict()

    def generate(self, config: ImageGenerationConfig) -> tuple[list[GeneratedImage], dict[str, Any]]:
        # mflux path: Apple Silicon native FLUX via MLX. Routed only
        # when the catalog variant declared runtime="mflux". Any
        # failure (missing package, unsupported repo, runtime error)
        # falls through to the diffusers path below so the user still
        # gets an image.
        if (config.runtime or "").lower() == "mflux":
            probe = self._mflux.probe()
            if probe.get("available"):
                try:
                    images = self._mflux.generate(config)
                    status = self._diffusers.probe().to_dict()
                    status["activeEngine"] = "mflux"
                    status["message"] = "Generated via mflux (MLX native)."
                    return images, status
                except Exception as exc:
                    status = self._diffusers.probe()
                    note = (
                        f"mflux failed ({type(exc).__name__}: {exc}) — "
                        "falling back to diffusers."
                    )
                    # fall through, but annotate status later
                    _mflux_fallback_note = note
                else:
                    _mflux_fallback_note = None
            else:
                _mflux_fallback_note = probe.get("reason") or "mflux unavailable"
        else:
            _mflux_fallback_note = None

        # FU-008 image subset: sd.cpp path. Routed when the catalog
        # variant declares ``engine="sdcpp"`` (which app.py threads onto
        # ``config.runtime``). Failure modes (missing binary, unsupported
        # repo, missing GGUF, subprocess error) fall through to the
        # diffusers path below and surface a runtimeNote so the user
        # still gets an image rendered.
        if (config.runtime or "").lower() == "sdcpp":
            probe = self._sdcpp.probe()
            if probe.get("available"):
                try:
                    images = self._sdcpp.generate(config)
                    status = self._diffusers.probe().to_dict()
                    status["activeEngine"] = "sd.cpp"
                    status["message"] = "Generated via stable-diffusion.cpp subprocess."
                    return images, status
                except Exception as exc:
                    _sdcpp_fallback_note = (
                        f"sd.cpp failed ({type(exc).__name__}: {exc}) — "
                        "falling back to diffusers."
                    )
                else:
                    _sdcpp_fallback_note = None
            else:
                _sdcpp_fallback_note = probe.get("reason") or "sd.cpp unavailable"
            # Combine mflux + sdcpp fallback notes if both fired (rare but
            # possible if a variant lists ``engine="sdcpp"`` AND the user
            # has overridden the runtime selector to ``"mflux"`` somehow).
            if _sdcpp_fallback_note:
                if _mflux_fallback_note:
                    _mflux_fallback_note = (
                        f"{_mflux_fallback_note} {_sdcpp_fallback_note}"
                    )
                else:
                    _mflux_fallback_note = _sdcpp_fallback_note

        status = self._diffusers.probe()
        if status.realGenerationAvailable:
            try:
                images = self._diffusers.generate(config)
                result_status = self._diffusers.probe().to_dict()
                if _mflux_fallback_note:
                    result_status["message"] = (
                        f"{_mflux_fallback_note} {result_status.get('message', '')}".strip()
                    )
                return images, result_status
            except Exception as exc:
                if _is_cuda_torch_unavailable_error(exc):
                    raise
                fallback_note = (
                    "The diffusers runtime failed, so ChaosEngineAI fell back to the placeholder engine for this run. "
                    f"Details: {exc}"
                )
                fallback_status = ImageRuntimeStatus(
                    activeEngine="placeholder",
                    realGenerationAvailable=False,
                    device=status.device,
                    pythonExecutable=status.pythonExecutable,
                    missingDependencies=[],
                    loadedModelRepo=status.loadedModelRepo,
                    message=fallback_note,
                    # Preserve the +cpu / missing-torch warning across
                    # the demotion. Without this the Studio's "GPU
                    # acceleration not active" banner disappears the
                    # moment generation fails, leaving only "Install
                    # GPU runtime" -- which is the wrong remedy when
                    # torch IS installed (just CPU-only). Recompute
                    # rather than copying ``status.torchInstallWarning``
                    # so the message reflects current disk state, not
                    # what the probe saw at preload time.
                    torchInstallWarning=_torch_install_warning(),
                )
                return self._placeholder.generate(config, runtime_note=fallback_note), fallback_status.to_dict()

        return self._placeholder.generate(config, runtime_note=status.message), status.to_dict()
