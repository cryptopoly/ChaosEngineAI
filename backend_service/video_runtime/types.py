"""Video-runtime data classes.

Mirrors ``image_runtime/types.py``. Three frozen dataclasses every
video engine speaks:

- ``VideoRuntimeStatus`` — runtime probe payload (engine readiness,
  device, missing deps, GPU memory, torch-install warning).
- ``VideoGenerationConfig`` — request payload. Mirrors
  ``ImageGenerationConfig`` with a video-specific footprint:
  ``numFrames`` + ``fps`` + ``interpolationFactor`` + LTX refiner /
  Wan distill / NF4 / FP8 / Nunchaku / preview-VAE / scheduler /
  CFG-decay / STG-scale opt-in flags.
- ``GeneratedVideo`` — engine output (mp4 bytes + duration + dims +
  effective steps/guidance).

Extracted from ``video_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VideoRuntimeStatus:
    activeEngine: str
    realGenerationAvailable: bool
    message: str
    device: str | None = None
    # ``expectedDevice`` is the device we'll ask torch to use on the
    # next Generate click, predicted from nvidia-smi + platform checks
    # WITHOUT importing torch. Lets the Studio show "Device: cuda
    # (expected)" before anything has loaded, so users can confirm GPU
    # will be used before sinking 2+ GB of model download into it.
    # Mirrors ``ImageRuntimeStatus.expectedDevice``.
    expectedDevice: str | None = None
    pythonExecutable: str | None = None
    missingDependencies: list[str] = field(default_factory=list)
    loadedModelRepo: str | None = None
    # Total memory available to the inference device, in GB. Used by the
    # frontend safety heuristic (``assessVideoGenerationSafety``) to scale its
    # attention-budget thresholds — a 64 GB M4 Max should tolerate far more
    # frames than a 16 GB base M2, and a 24 GB RTX 4090 differs again. We
    # source this from ``backend_service.helpers.gpu.get_gpu_metrics`` which
    # already reads Apple Silicon unified memory via sysctl and NVIDIA VRAM
    # via nvidia-smi. ``None`` means we couldn't detect it — the frontend
    # falls back to its MPS-strict defaults in that case.
    deviceMemoryGb: float | None = None
    # ``torchInstallWarning`` carries a one-line warning when the installed
    # torch wheel doesn't match the host accelerator (e.g. +cpu wheel on a
    # CUDA host -- generation silently runs on CPU). Computed without
    # importing torch (we read dist-info METADATA) so the probe stays free
    # of Windows DLL-lock side effects. Frontend renders this as a loud
    # warning chip in the Studio so users don't see "Real engine ready"
    # next to "Device: cuda (expected)" while their NVIDIA GPU sits idle.
    torchInstallWarning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoGenerationConfig:
    """Shape consumed by ``DiffusersVideoEngine.generate``."""
    modelId: str
    modelName: str
    repo: str
    prompt: str
    negativePrompt: str
    width: int
    height: int
    numFrames: int
    fps: int
    guidance: float
    steps: int = 50
    seed: int | None = None
    # GGUF quantization for video DiT transformers. When set, the
    # transformer is loaded from a single .gguf file while the VAE /
    # text encoders still come from the base ``repo`` snapshot. The
    # pipeline cache keys on (repo, ggufFile) so multiple quant levels
    # can coexist without evicting each other.
    ggufRepo: str | None = None
    ggufFile: str | None = None
    # Post-processing frame interpolation. Factor of 1 means disabled;
    # 2 or 4 insert interpolated frames between each generated frame
    # and bump the reported fps by the same factor, producing smoother
    # motion at higher frame rates without generating more DiT frames
    # (which is 10-50x more expensive than interpolation).
    interpolationFactor: int = 1
    # Optional diffusion cache strategy id, e.g. "teacache". Mirrors the
    # image_runtime field — video DiTs benefit even more from timestep
    # caching (Wan2.1 720P 30% faster, HunyuanVideo up to 2.1×). When the
    # strategy has no vendored patch for this pipeline the engine swallows
    # the NotImplementedError and falls back to the stock pipeline — the
    # UI shows the "Scaffold" badge so users know why.
    cacheStrategy: str | None = None
    cacheRelL1Thresh: float | None = None
    # Optional diffusers scheduler override. ``None`` (or ``"auto"``) keeps
    # whatever scheduler the per-model defaults table picks, which in turn
    # falls back to the pipeline's baked-in default. Recognised ids match
    # ``_SCHEDULER_CLASSES`` below — anything else logs a warning and
    # leaves the pipeline scheduler untouched.
    scheduler: str | None = None
    # bitsandbytes NF4 quantization for the video DiT transformer. CUDA
    # only; ignored on MPS / CPU. Brings Wan 2.1 14B from ~28 GB bf16 to
    # ~7 GB on the RTX 4090 with negligible quality loss for video DiTs
    # (NF4 is the same scheme bitsandbytes ships for QLoRA).
    useNf4: bool = False
    # LTX-Video two-stage spatial upscale. When True and the pipeline is
    # ``LTXPipeline``, the engine runs the base sampler at the requested
    # resolution then refines through ``LTXLatentUpsamplePipeline``
    # (Lightricks/LTX-Video-0.9.5-spatial-upscaler). Frame budget grows
    # ~1.5×; the ``runtimeNote`` surfaces the substitution to users.
    enableLtxRefiner: bool = False
    # Phase E1: opt-in template-based prompt enhancement for short prompts
    # (< 25 words). See ``_enhance_prompt`` for the per-model suffixes.
    enhancePrompt: bool = True
    # FU-018: TAESD / TAEHV preview-decode VAE swap. Preview-only quality
    # knob — when True the engine swaps ``pipeline.vae`` for the matching
    # tiny VAE (taew2_2 for Wan, taeltx2_3_wide for LTX, taehv1_5 for
    # HunyuanVideo, taecogvideox for CogVideoX, taemochi for Mochi)
    # before the first denoise. Each step decodes in a fraction of the
    # wall-time. Default off — video users typically want full fidelity.
    previewVae: bool = False
    # Phase 3 / Wan2.2-Distill 4-step: catalog-pinned distilled
    # transformers. Wan 2.2 A14B is MoE with two transformer experts
    # (``transformer`` = high-noise, ``transformer_2`` = low-noise).
    # lightx2v's 4-step distillation publishes both experts as standalone
    # safetensors files; the runtime swaps both onto the pipeline at
    # build time so subsequent ``pipeline(...)`` calls run the distilled
    # 4-step schedule. Mutually exclusive with LoRA loading — when the
    # distill files are pinned, the LoRA path is skipped.
    distillTransformerRepo: str | None = None
    distillTransformerHighNoiseFile: str | None = None
    distillTransformerLowNoiseFile: str | None = None
    # ``"bf16"`` | ``"fp8_e4m3"`` | ``"int8"`` — dictates the torch dtype
    # used at load. FP8/INT8 distill weights ship pre-quantized and need
    # the corresponding torch dtype + a CUDA backend that exposes the
    # native kernel. On platforms without FP8/INT8 ops the runtime falls
    # back to bf16 dequant.
    distillTransformerPrecision: str | None = None
    # Phase E2: CFG decay schedule. Linear ramp from initial guidance_scale
    # at step 0 to 1.0 at the last step. Default-on for flow-match pipelines.
    cfgDecay: bool = True
    # Spatial-Temporal Guidance scale, consumed only by the mlx-video LTX-2
    # path. 1.0 keeps the upstream-recommended perturbed forward pass per
    # step; 0.0 disables it and saves ~33 % wall time at a mild quality
    # cost. Other runtimes ignore the value.
    stgScale: float = 1.0
    # FU-023 Nunchaku / SVDQuant: pinned by catalog variants that ship
    # CUDA INT4 SVDQuant snapshots. CUDA only — falls back when the
    # nunchaku package isn't installed or device != cuda. The video-side
    # path stays parked until upstream Nunchaku ships Wan / HunyuanVideo
    # / LTX wrappers (FLUX + Qwen-Image only as of v1.2.1) — wiring is
    # in place so adding a video variant becomes a catalog-row change.
    nunchakuRepo: str | None = None
    nunchakuFile: str | None = None
    # FU-024 FP8 layerwise casting on CUDA SM 8.9+ (Ada/Hopper/Blackwell).
    # Halves transformer VRAM by storing fp8 weights + computing in bf16
    # inside the matmul. E5M2 for HunyuanVideo, E4M3 for Wan / LTX / FLUX
    # / Qwen-Image. Default off; opt-in.
    fp8LayerwiseCasting: bool = False
    # FU-019 distill LoRAs: when the catalog variant pins a LoRA
    # (lightx2v Wan2.1 CausVid, Wan2.2-Distill-Models, FastWan), the
    # engine fuses it into the pipeline transformer at load time so
    # subsequent ``pipeline(...)`` calls run with the LoRA baked in.
    # 4-step Wan via lightx2v cuts wall-time 7-8× vs the 30-step base.
    loraRepo: str | None = None
    loraFile: str | None = None
    loraScale: float | None = None
    # Variant-declared step / CFG defaults. Used by app.py's
    # ``_generate_video_artifact`` to substitute the schema defaults
    # (50 steps, CFG 3.0) when the user hasn't moved the sliders —
    # distill LoRAs run at 4 steps CFG 1.0.
    defaultSteps: int | None = None
    cfgOverride: float | None = None


@dataclass(frozen=True)
class GeneratedVideo:
    """A single rendered mp4. Mirrors ``GeneratedImage`` from image_runtime."""
    seed: int
    bytes: bytes
    extension: str
    mimeType: str
    durationSeconds: float
    frameCount: int
    fps: int
    width: int
    height: int
    runtimeLabel: str
    runtimeNote: str | None = None
    effectiveSteps: int | None = None
    effectiveGuidance: float | None = None
