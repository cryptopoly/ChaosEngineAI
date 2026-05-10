"""Image-runtime data classes.

Three frozen dataclasses every image engine speaks:

- ``ImageRuntimeStatus`` — runtime probe payload (real-engine readiness,
  device, missing deps, GPU memory). Built by
  ``ImageRuntimeManager.status()`` and surfaced on the Discover panel.
- ``ImageGenerationConfig`` — request payload (prompt, dims, steps, plus
  every opt-in flag — GGUF / Nunchaku / preview-VAE / distill-LoRA /
  fp8 / cache strategy / sampler / CFG decay). Frozen so the cache
  layer can hash it for variant identity.
- ``GeneratedImage`` — engine output (bytes + seed + duration + label).

Extracted from ``image_runtime.py`` as part of the v0.8.0 refactor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImageRuntimeStatus:
    activeEngine: str
    realGenerationAvailable: bool
    message: str
    device: str | None = None
    # ``expectedDevice`` is the device we'll ask torch to use on the
    # next Generate click, computed from nvidia-smi + find_spec without
    # importing torch. Lets the UI show "will use cuda" before any
    # model has actually been loaded. Kept separate from ``device`` so
    # consumers can distinguish "expected at load time" from "actually
    # bound right now".
    expectedDevice: str | None = None
    pythonExecutable: str | None = None
    missingDependencies: list[str] = field(default_factory=list)
    loadedModelRepo: str | None = None
    # Total memory available to the inference device, in GB. Populated via
    # ``backend_service.helpers.gpu.get_device_vram_total_gb`` — NVIDIA VRAM
    # from nvidia-smi on CUDA, unified memory from sysctl on Apple Silicon,
    # system RAM on CPU Linux/Windows. Used by the frontend image-safety
    # heuristic (``assessImageGenerationSafety``) to scale its memory-
    # budget thresholds — a 64 GB M4 Max tolerates far more than a 16 GB
    # base M2. ``None`` means detection failed; the frontend falls back
    # to MPS-strict defaults.
    deviceMemoryGb: float | None = None
    # ``torchInstallWarning`` -- mirrors VideoRuntimeStatus. Surfaces
    # the "torch is +cpu but you have a CUDA card" / "torch missing"
    # mismatch that otherwise hides behind a misleadingly green
    # "Real engine ready" + "Device: cuda (expected)" badge pair.
    torchInstallWarning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageGenerationConfig:
    modelId: str
    modelName: str
    repo: str
    prompt: str
    negativePrompt: str
    width: int
    height: int
    steps: int
    guidance: float
    batchSize: int
    seed: int | None = None
    qualityPreset: str | None = None
    sampler: str | None = None
    # GGUF quantization: when set, the transformer is loaded from a single
    # .gguf file (e.g. city96/FLUX.1-dev-gguf / flux1-dev-Q4_K_M.gguf) while
    # the VAE and text encoders come from the base ``repo`` snapshot. The
    # pipeline cache keys on (repo, ggufFile) so multiple quant levels of
    # the same model can coexist without stomping on each other.
    ggufRepo: str | None = None
    ggufFile: str | None = None
    # Runtime selector. Default (None / "diffusers") uses the
    # cross-platform diffusers pipeline; "mflux" routes to the native
    # Apple Silicon MLX path for FLUX, which is noticeably faster on
    # M-series Macs and avoids MPS fp16 corner cases.
    runtime: str | None = None
    # Optional diffusion cache strategy id, e.g. "teacache". When set to a
    # strategy that reports ``applies_to()`` including "image", the engine
    # calls the strategy's ``apply_diffusers_hook`` before the first pipeline
    # forward. Unknown / inapplicable ids are ignored quietly — the caller
    # sees the same result as not passing anything.
    cacheStrategy: str | None = None
    # Threshold knob for TeaCache-style rel-L1 caches. ``None`` means the
    # strategy's default (0.4 for TeaCache → ~1.8× speedup). See
    # ``TeaCacheStrategy.recommended_thresholds()`` for presets.
    cacheRelL1Thresh: float | None = None
    # FU-021: CFG decay schedule, mirroring the video runtime knob. When
    # True and the model is flow-match (FLUX/SD3/Qwen-Image/Sana/HiDream),
    # the engine ramps ``guidance_scale`` linearly from the user's
    # setting at step 0 toward 1.5 (the floor that keeps
    # ``do_classifier_free_guidance`` True end-to-end). Default off:
    # image users typically want consistent CFG; turning on the knob is
    # opt-in. Non-flow-match repos (SD1.5/SDXL) ignore the flag because
    # CFG decay on UNet-based ε-prediction pipelines doesn't carry the
    # same oversaturation benefit.
    cfgDecay: bool = False
    # FU-018: TAESD / TAEHV preview-decode VAE swap. Preview-only quality
    # knob — when True the engine swaps ``pipeline.vae`` for the matching
    # tiny VAE before the first denoise so each step decodes in a fraction
    # of the wall-time. Final output goes through the same fast VAE; users
    # trade fidelity for iteration speed. Default off.
    previewVae: bool = False
    # FU-019 distill LoRAs: when the catalog variant pins a LoRA
    # (Hyper-SD FLUX, alimama FLUX.1-Turbo-Alpha, lightx2v Wan
    # CausVid), the engine fuses it into the pipeline at load time so
    # subsequent generates run at the LoRA's lower step count without
    # re-loading. ``loraRepo`` is the HF repo id, ``loraFile`` is the
    # specific weight name within that repo (LoRAs commonly ship
    # multiple step variants), ``loraScale`` is the fuse strength
    # (Hyper-SD recommends 0.125, alimama Turbo wants 1.0, lightx2v
    # CausVid wants 1.0).
    loraRepo: str | None = None
    loraFile: str | None = None
    loraScale: float | None = None
    # Variant-declared step / CFG defaults. Used by
    # ``_generate_image_artifacts`` in app.py to substitute the schema
    # defaults when the user hasn't moved the sliders — distill LoRAs
    # have very different optimal points (4-8 steps, CFG 1.0-3.5)
    # than the schema defaults (24 steps, CFG 5.5).
    defaultSteps: int | None = None
    cfgOverride: float | None = None
    # FU-023 Nunchaku / SVDQuant: 4-bit weight quantization for FLUX,
    # Qwen-Image, SD3.5, SANA, PixArt-Σ on CUDA. ~3× over NF4 on FLUX.1-dev.
    # ``nunchakuRepo`` pins the precompiled SVDQuant snapshot (e.g.
    # ``mit-han-lab/svdq-int4-flux.1-dev``); ``nunchakuFile`` is optional
    # for repos that ship multiple precision tiers. CUDA only — the helper
    # falls back to the standard transformer when the import fails or the
    # device isn't ``cuda``.
    nunchakuRepo: str | None = None
    nunchakuFile: str | None = None
    # FU-024 FP8 layerwise casting (CUDA SM 8.9+, e.g. RTX 4090 / H100).
    # When True the engine calls ``transformer.enable_layerwise_casting``
    # post-load with the family-correct fp8 dtype (E4M3 for FLUX / Wan,
    # E5M2 for HunyuanVideo). No-op on Apple Silicon, CPU, and pre-Ada
    # GPUs — the helper guards before invoking. Defaults off so users
    # opt-in once their hardware is confirmed.
    fp8LayerwiseCasting: bool = False


@dataclass(frozen=True)
class GeneratedImage:
    seed: int
    bytes: bytes
    extension: str
    mimeType: str
    durationSeconds: float
    runtimeLabel: str
    runtimeNote: str | None = None
