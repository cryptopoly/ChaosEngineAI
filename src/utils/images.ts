import type { ImageModelFamily, ImageModelVariant, ImageOutputArtifact } from "../types";
import type { ImageDiscoverTaskFilter, ImageDiscoverAccessFilter, ImageGalleryOrientationFilter } from "../types/image";
import { inferDeviceFromHostPlatform } from "./videos";

export function flattenImageVariants(families: ImageModelFamily[]): ImageModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function defaultImageVariantForFamily(family?: ImageModelFamily | null): ImageModelVariant | null {
  if (!family) return null;
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

export function findImageVariantById(families: ImageModelFamily[], variantId: string): ImageModelVariant | null {
  for (const family of families) {
    const match = family.variants.find((variant) => variant.id === variantId);
    if (match) return match;
  }
  return null;
}

export function findImageVariantByRepo(families: ImageModelFamily[], repo: string | null | undefined): ImageModelVariant | null {
  if (!repo) return null;
  for (const family of families) {
    const match = family.variants.find((variant) => variant.repo === repo);
    if (match) return match;
  }
  return null;
}

export function imageRuntimeKind(label?: string | null) {
  const lowered = (label ?? "").toLowerCase();
  if (lowered.includes("placeholder")) return "placeholder";
  if (lowered.includes("diffusers")) return "diffusers";
  return "other";
}

export function imageOrientation(width: number, height: number): Exclude<ImageGalleryOrientationFilter, "all"> {
  if (width === height) return "square";
  return width > height ? "landscape" : "portrait";
}

export function imageArtifactTimestamp(artifact: ImageOutputArtifact) {
  const timestamp = Date.parse(artifact.createdAt);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

export function imageVariantMatchesDiscoverFilters(
  variant: ImageModelVariant,
  taskFilter: ImageDiscoverTaskFilter,
  accessFilter: ImageDiscoverAccessFilter,
) {
  if (taskFilter !== "all" && !variant.taskSupport.includes(taskFilter)) {
    return false;
  }
  if (accessFilter === "open" && variant.gated === true) {
    return false;
  }
  if (accessFilter === "gated" && variant.gated !== true) {
    return false;
  }
  return true;
}

export function imageDiscoverVariantHaystack(variant: ImageModelVariant) {
  return [
    variant.name,
    variant.familyName ?? "",
    variant.provider,
    variant.repo,
    variant.runtime,
    variant.recommendedResolution,
    variant.note,
    variant.updatedLabel ?? "",
    variant.license ?? "",
    variant.pipelineTag ?? "",
    variant.downloadsLabel ?? "",
    variant.likesLabel ?? "",
    String(variant.sizeGb),
    String(variant.repoSizeGb ?? ""),
    String(variant.coreWeightsGb ?? ""),
    variant.gated ? "gated access" : "open access",
    variant.taskSupport.join(" "),
    variant.styleTags.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

export function imageDiscoverFamilyHaystack(family: ImageModelFamily) {
  return [
    family.name,
    family.provider,
    family.headline,
    family.summary,
    family.updatedLabel,
    family.badges.join(" "),
    ...family.variants.map((variant) => imageDiscoverVariantHaystack(variant)),
  ]
    .join(" ")
    .toLowerCase();
}

export function imageDiscoverVariantMatchesQuery(variant: ImageModelVariant, query: string) {
  if (!query) return true;
  return imageDiscoverVariantHaystack(variant).includes(query);
}

export function imageDiscoverFamilyMatchesQuery(family: ImageModelFamily, query: string) {
  if (!query) return true;
  return imageDiscoverFamilyHaystack(family).includes(query);
}

export interface ImageDiscoverMemoryEstimate {
  estimatedPeakGb: number;
  modelFootprintGb: number;
  resolutionLabel: string;
  label: string;
  title: string;
}

function formatImageDiscoverGb(gb: number): string {
  if (!Number.isFinite(gb) || gb <= 0) return "Unknown";
  return gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;
}

function parseRecommendedImageResolution(value: string | null | undefined): { width: number; height: number } {
  const match = /(\d{3,5})\s*[x×]\s*(\d{3,5})/i.exec(value ?? "");
  if (!match) return { width: 1024, height: 1024 };
  const width = Number.parseInt(match[1], 10);
  const height = Number.parseInt(match[2], 10);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return { width: 1024, height: 1024 };
  }
  return { width, height };
}

export function imageVariantSizeForMemoryEstimate(variant: ImageModelVariant): number {
  const candidates = [
    variant.coreWeightsGb,
    variant.sizeGb,
    variant.onDiskGb,
    variant.repoSizeGb,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "number" && Number.isFinite(candidate) && candidate > 0) {
      return candidate;
    }
  }
  return 0;
}

export function imageDiscoverMemoryEstimate(variant: ImageModelVariant): ImageDiscoverMemoryEstimate | null {
  const baseModelFootprintGb = imageVariantSizeForMemoryEstimate(variant);
  if (!(baseModelFootprintGb > 0)) return null;

  const { width, height } = parseRecommendedImageResolution(variant.recommendedResolution);
  const safety = assessImageGenerationSafety({
    width,
    height,
    device: null,
    // Discover does not know the exact target device memory yet. Use a large
    // budget so this function is only a requirement estimate; Studio still
    // computes the actual risk against the live backend/device.
    deviceMemoryGb: 512,
    baseModelFootprintGb,
    runtimeFootprintGb: variant.runtimeFootprintGb,
    runtimeFootprintMpsGb: variant.runtimeFootprintMpsGb,
    runtimeFootprintCudaGb: variant.runtimeFootprintCudaGb,
    runtimeFootprintCpuGb: variant.runtimeFootprintCpuGb,
    repo: variant.repo,
    ggufFile: variant.ggufFile,
  });
  const resolutionLabel = `${width}×${height}`;
  const estimatedPeakGb = Math.max(safety.estimatedPeakGb, safety.modelFootprintGb);
  return {
    estimatedPeakGb,
    modelFootprintGb: safety.modelFootprintGb,
    resolutionLabel,
    label: `~${formatImageDiscoverGb(estimatedPeakGb)} @ ${resolutionLabel}`,
    title: (
      `Estimated peak RAM/VRAM at ${resolutionLabel}. Includes model/text encoder residency`
      + (safety.modelFootprintGb > 0 ? ` (~${formatImageDiscoverGb(safety.modelFootprintGb)})` : "")
      + " plus a resolution-dependent attention estimate. Actual usage varies by runtime and device."
    ),
  };
}

// ---------------------------------------------------------------------------
// Generation safety heuristic
// ---------------------------------------------------------------------------
// Port of ``assessVideoGenerationSafety`` without the frame-count / temporal
// dimension. Image pipelines share the two-term memory model:
//   - Resident model footprint (weights + text encoder(s) + VAE). Dominates
//     on MPS for FLUX / HiDream / SD3.5 Large where fp16 weights are 23 GB+.
//   - Attention peak ≈ tokens² × heads × dtype where tokens = (W/8) × (H/8)
//     for the 8× spatial downsample most diffusion models use. No temporal
//     term → much smaller than video, but still enough to OOM at 2K+.
//
// Thresholds track the runtime path rather than raw model size. MPS uses
// Apple's ~75% recommended working-set ceiling; CUDA uses dedicated VRAM and
// the runtime's FLUX offload path; CPU uses system RAM with a conservative
// allowance because swapping makes "will it finish?" much less predictable.

export type ImageGenerationRiskLevel = "safe" | "caution" | "danger";
export type ImageEffectiveDevice = "mps" | "cuda" | "cpu";

export interface ImageGenerationSafety {
  riskLevel: ImageGenerationRiskLevel;
  estimatedPeakGb: number;
  modelFootprintGb: number;
  deviceMemoryGb: number;
  exceedsDevice: boolean;
  effectiveDevice: ImageEffectiveDevice;
  effectiveDeviceWasInferred: boolean;
  reason: string | null;
  suggestion: { width: number; height: number; label: string } | null;
}

const DEFAULT_MPS_MEMORY_GB = 16;
const DEFAULT_CUDA_MEMORY_GB = 12;
const DEFAULT_CPU_MEMORY_GB = 8;

// Attention slab multiplier — same rationale as video, smaller constant
// because image transformers cluster around 8-16 heads per block rather
// than the 16-32 video models use, and there's no temporal term fighting
// the budget. This keeps 1024px FLUX on a 64 GB MPS machine clear while
// still flagging 2K FLUX where attention starts competing with weights.
const IMAGE_HEAD_SLAB_MULTIPLIER = 4;

function bytesPerElement(device: ImageEffectiveDevice): number {
  return device === "cpu" ? 4 : 2;
}

function effectiveBudgetGb(totalGb: number, device: ImageEffectiveDevice): number {
  if (device === "cuda") return totalGb * 0.9;
  if (device === "mps") return totalGb * 0.75;
  return totalGb * 0.6;
}

function riskRatios(device: ImageEffectiveDevice): { caution: number; danger: number } {
  if (device === "cuda") return { caution: 0.85, danger: 1.0 };
  if (device === "mps") return { caution: 0.8, danger: 0.95 };
  return { caution: 0.75, danger: 0.95 };
}

function estimatePeakAttentionBytes(latentTokens: number, device: ImageEffectiveDevice): number {
  return latentTokens * latentTokens * bytesPerElement(device) * IMAGE_HEAD_SLAB_MULTIPLIER;
}

function isFluxRepo(repo?: string | null): boolean {
  return (repo ?? "").toLowerCase().includes("flux");
}

function runsOnCpuForMps(repo?: string | null): boolean {
  // image_runtime.py intentionally routes Qwen-Image away from MPS because
  // the naive fp16 MPS path can produce black outputs. Keep the warning and
  // capacity math aligned with the backend's actual execution device.
  return (repo ?? "").toLowerCase().includes("qwen-image");
}

function estimateResidentModelGb(
  baseGb: number,
  device: ImageEffectiveDevice,
  repo?: string | null,
  ggufFile?: string | null,
): number {
  if (!(baseGb > 0) || !Number.isFinite(baseGb)) return 0;
  const flux = isFluxRepo(repo);
  const hasExplicitQuant = Boolean(ggufFile);
  if (flux && hasExplicitQuant) {
    // GGUF image variants quantize only the FLUX transformer. Diffusers still
    // loads the base repo's text encoders + VAE, and on Apple Silicon the
    // Python process includes Metal allocator/watermark overhead. A real
    // FLUX.1 Dev Q4_K_M run on a 64 GB M-series Mac sits around 43 GB in
    // Activity Monitor despite the GGUF transformer being ~7 GB, so include
    // that fixed pipeline residency rather than showing the transformer size
    // as if it were the whole model.
    if (device === "mps") return baseGb + 34.5;
    if (device === "cuda") return baseGb + 11.5;
    return baseGb + 18;
  }
  if (flux && !hasExplicitQuant) {
    // Mirrors image_runtime.py:
    // - MPS quantizes FLUX's transformer to int8wo before pipeline load.
    // - CUDA uses model CPU offload and tries NF4 for the transformer.
    // So the accelerator working set is much smaller than fp16 disk size.
    if (device === "mps") return baseGb * 0.75;
    if (device === "cuda") return Math.min(baseGb * 0.45, 11);
  }
  const factor = device === "mps" ? 1.15 : device === "cpu" ? 1.25 : 1.05;
  return baseGb * factor;
}

function positiveRuntimeFootprint(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function runtimeFootprintForDevice(opts: {
  device: ImageEffectiveDevice;
  runtimeFootprintGb?: number | null;
  runtimeFootprintMpsGb?: number | null;
  runtimeFootprintCudaGb?: number | null;
  runtimeFootprintCpuGb?: number | null;
}): number | null {
  if (opts.device === "mps") {
    return positiveRuntimeFootprint(opts.runtimeFootprintMpsGb) ?? positiveRuntimeFootprint(opts.runtimeFootprintGb);
  }
  if (opts.device === "cuda") {
    return positiveRuntimeFootprint(opts.runtimeFootprintCudaGb) ?? positiveRuntimeFootprint(opts.runtimeFootprintGb);
  }
  return positiveRuntimeFootprint(opts.runtimeFootprintCpuGb) ?? positiveRuntimeFootprint(opts.runtimeFootprintGb);
}

/**
 * Assess whether an image generation request is likely to detonate the
 * inference device. Same shape as ``assessVideoGenerationSafety`` without
 * the frame-count dimension. See that function for the calibration story.
 *
 * Calibration points (all with ``baseModelFootprintGb`` supplied):
 * - SD 1.5 (4 GB) at 1024×1024 on 16 GB MPS: safe.
 * - SDXL (13 GB) at 1024×1024 on 16 GB MPS: danger/tight.
 * - FLUX schnell/dev standard at 1024×1024 on 64 GB MPS: safe; runtime uses int8wo.
 * - FLUX standard at 1024×1024 on 16 GB MPS: danger.
 * - FLUX standard at 2048×2048 on 64 GB MPS: danger from attention peak.
 * - HiDream (47 GB) at 1024×1024 on 64 GB MPS: caution.
 */
export function assessImageGenerationSafety(opts: {
  width: number;
  height: number;
  device: string | null | undefined;
  deviceMemoryGb?: number | null;
  baseModelFootprintGb?: number | null;
  runtimeFootprintGb?: number | null;
  runtimeFootprintMpsGb?: number | null;
  runtimeFootprintCudaGb?: number | null;
  runtimeFootprintCpuGb?: number | null;
  repo?: string | null;
  ggufFile?: string | null;
}): ImageGenerationSafety {
  const {
    width,
    height,
    device,
    deviceMemoryGb,
    baseModelFootprintGb,
    runtimeFootprintGb,
    runtimeFootprintMpsGb,
    runtimeFootprintCudaGb,
    runtimeFootprintCpuGb,
    repo,
    ggufFile,
  } = opts;

  const normalised = (device ?? "").toLowerCase();
  const isCuda = normalised.startsWith("cuda");
  const isCpu = normalised === "cpu";
  const isMps = normalised === "mps";
  let effectiveDevice: ImageEffectiveDevice = isCuda
    ? "cuda"
    : isCpu
      ? "cpu"
      : isMps
        ? "mps"
        : inferDeviceFromHostPlatform();
  if (effectiveDevice === "mps" && runsOnCpuForMps(repo)) {
    effectiveDevice = "cpu";
  }
  const effectiveDeviceWasInferred = !isCuda && !isCpu && !isMps;

  const fallback =
    effectiveDevice === "cuda"
      ? DEFAULT_CUDA_MEMORY_GB
      : effectiveDevice === "cpu"
        ? DEFAULT_CPU_MEMORY_GB
        : DEFAULT_MPS_MEMORY_GB;
  const totalMemoryGb =
    deviceMemoryGb != null && Number.isFinite(deviceMemoryGb) && deviceMemoryGb > 0
      ? deviceMemoryGb
      : fallback;
  const budgetGb = effectiveBudgetGb(totalMemoryGb, effectiveDevice);

  const baseFootprint =
    baseModelFootprintGb != null && Number.isFinite(baseModelFootprintGb) && baseModelFootprintGb > 0
      ? baseModelFootprintGb
      : 0;
  const runtimeOverrideGb = runtimeFootprintForDevice({
    device: effectiveDevice,
    runtimeFootprintGb,
    runtimeFootprintMpsGb,
    runtimeFootprintCudaGb,
    runtimeFootprintCpuGb,
  });
  const modelFootprintGb =
    runtimeOverrideGb != null
      ? runtimeOverrideGb
      : estimateResidentModelGb(baseFootprint, effectiveDevice, repo, ggufFile);

  if (
    !Number.isFinite(width)
    || !Number.isFinite(height)
    || width <= 0
    || height <= 0
  ) {
    return {
      riskLevel: "safe",
      estimatedPeakGb: 0,
      modelFootprintGb: 0,
      deviceMemoryGb: totalMemoryGb,
      exceedsDevice: false,
      effectiveDevice,
      effectiveDeviceWasInferred,
      reason: null,
      suggestion: null,
    };
  }

  const latentTokens = Math.ceil(width / 8) * Math.ceil(height / 8);
  const attentionPeakGb = estimatePeakAttentionBytes(latentTokens, effectiveDevice) / 1024 ** 3;
  const estimatedPeakGb = modelFootprintGb + attentionPeakGb;

  const { caution: cautionRatio, danger: dangerRatio } = riskRatios(effectiveDevice);
  const ratio = estimatedPeakGb / budgetGb;
  const exceedsDevice = estimatedPeakGb > budgetGb;
  const riskLevel: ImageGenerationRiskLevel =
    ratio >= dangerRatio ? "danger" : ratio >= cautionRatio ? "caution" : "safe";

  if (riskLevel === "safe") {
    return {
      riskLevel,
      estimatedPeakGb,
      modelFootprintGb,
      deviceMemoryGb: totalMemoryGb,
      exceedsDevice,
      effectiveDevice,
      effectiveDeviceWasInferred,
      reason: null,
      suggestion: null,
    };
  }

  const fmt = (g: number) => (g >= 10 ? g.toFixed(0) : g.toFixed(1));
  const platform =
    effectiveDevice === "cuda"
      ? "this GPU"
      : effectiveDevice === "cpu"
        ? "CPU generation"
        : "Apple Silicon (MPS)";

  // Model-alone-too-big short-circuit: same logic as video, no resolution
  // suggestion can recover when the weights/text encoders are the pressure
  // point. Threshold against the comfort band so we do not return bogus
  // "try smaller pixels" suggestions for a model-heavy warning.
  if (modelFootprintGb > cautionRatio * budgetGb) {
    const comfortBudgetGb = cautionRatio * budgetGb;
    const highRiskBudgetGb = dangerRatio * budgetGb;
    const reason =
      riskLevel === "danger"
        ? modelFootprintGb > budgetGb
          ? `The model needs ~${fmt(modelFootprintGb)} GB just to hold its weights + text encoder. On ${platform} with ${fmt(totalMemoryGb)} GB total, the estimated working set is ~${fmt(budgetGb)} GB, so the model alone is already over that. Even small images would likely crash the backend. Try a smaller model (SD 1.5 is ~4 GB, SDXL ~13 GB) or a machine with more memory.`
          : `The model needs ~${fmt(modelFootprintGb)} GB just to hold its weights + text encoder, and this run peaks around ~${fmt(estimatedPeakGb)} GB. On ${platform} with ${fmt(totalMemoryGb)} GB total, that is above the high-risk threshold (~${fmt(highRiskBudgetGb)} GB) and close to the estimated working set (~${fmt(budgetGb)} GB). Generation is likely to crash the backend; lower the resolution or choose a smaller model.`
        : `The model needs ~${fmt(modelFootprintGb)} GB just to hold its weights + text encoder. On ${platform} with ${fmt(totalMemoryGb)} GB total, that is above the conservative comfort target (~${fmt(comfortBudgetGb)} GB) but below the estimated working set (~${fmt(budgetGb)} GB). Generation may run slowly or fail; lower the resolution if it becomes unstable.`;
    return {
      riskLevel,
      estimatedPeakGb,
      modelFootprintGb,
      deviceMemoryGb: totalMemoryGb,
      exceedsDevice,
      effectiveDevice,
      effectiveDeviceWasInferred,
      reason,
      suggestion: null,
    };
  }

  // Suggestion loop — shrink width/height in 64-px steps (divisible-by-8 is
  // the pipeline requirement; 64 is a safer multiple that also keeps the
  // aspect ratio close to the user's intent). Cap iterations so a tiny
  // budget doesn't loop forever.
  const safeRatioTarget = cautionRatio * 0.7;
  let suggestedWidth = width;
  let suggestedHeight = height;
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const tokens = Math.ceil(suggestedWidth / 8) * Math.ceil(suggestedHeight / 8);
    const peak = modelFootprintGb + estimatePeakAttentionBytes(tokens, effectiveDevice) / 1024 ** 3;
    if (peak / budgetGb < safeRatioTarget) break;
    if (suggestedWidth > 512 || suggestedHeight > 512) {
      suggestedWidth = Math.max(512, Math.floor((suggestedWidth * 0.75) / 64) * 64);
      suggestedHeight = Math.max(512, Math.floor((suggestedHeight * 0.75) / 64) * 64);
    } else {
      break;
    }
  }

  const breakdown =
    modelFootprintGb > 0
      ? ` (model ≈ ${fmt(modelFootprintGb)} GB + attention ≈ ${fmt(attentionPeakGb)} GB)`
      : "";
  const reason =
    riskLevel === "danger"
      ? `These settings would need around ${fmt(estimatedPeakGb)} GB of peak memory${breakdown} — above what ${platform} can safely allocate (~${fmt(budgetGb)} GB of ${fmt(totalMemoryGb)} GB total). Generation is likely to crash the backend.`
      : `These settings need around ${fmt(estimatedPeakGb)} GB of peak memory${breakdown} — close to the safe limit on ${platform} (~${fmt(budgetGb)} GB of ${fmt(totalMemoryGb)} GB total). Generation may run slowly or fail.`;

  return {
    riskLevel,
    estimatedPeakGb,
    modelFootprintGb,
    deviceMemoryGb: totalMemoryGb,
    exceedsDevice,
    effectiveDevice,
    effectiveDeviceWasInferred,
    reason,
    suggestion: {
      width: suggestedWidth,
      height: suggestedHeight,
      label: `${suggestedWidth}×${suggestedHeight}`,
    },
  };
}
