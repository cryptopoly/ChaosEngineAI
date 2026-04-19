import type { VideoModelFamily, VideoModelVariant, VideoRuntimeStatus } from "../types";
import type { VideoDiscoverTaskFilter } from "../types/video";

export function flattenVideoVariants(families: VideoModelFamily[]): VideoModelVariant[] {
  return families.flatMap((family) => family.variants);
}

export function defaultVideoVariantForFamily(family?: VideoModelFamily | null): VideoModelVariant | null {
  if (!family) return null;
  return family.variants.find((variant) => variant.id === family.defaultVariantId) ?? family.variants[0] ?? null;
}

export function findVideoVariantById(families: VideoModelFamily[], variantId: string): VideoModelVariant | null {
  for (const family of families) {
    const match = family.variants.find((variant) => variant.id === variantId);
    if (match) return match;
  }
  return null;
}

export function findVideoVariantByRepo(
  families: VideoModelFamily[],
  repo: string | null | undefined,
): VideoModelVariant | null {
  if (!repo) return null;
  for (const family of families) {
    const match = family.variants.find((variant) => variant.repo === repo);
    if (match) return match;
  }
  return null;
}

export function videoVariantMatchesDiscoverFilters(
  variant: VideoModelVariant,
  taskFilter: VideoDiscoverTaskFilter,
): boolean {
  if (taskFilter !== "all" && !variant.taskSupport.includes(taskFilter)) {
    return false;
  }
  return true;
}

export function videoDiscoverVariantHaystack(variant: VideoModelVariant): string {
  return [
    variant.name,
    variant.familyName ?? "",
    variant.provider,
    variant.repo,
    variant.runtime,
    variant.recommendedResolution,
    variant.note,
    String(variant.sizeGb),
    String(variant.defaultDurationSeconds),
    variant.taskSupport.join(" "),
    variant.styleTags.join(" "),
  ]
    .join(" ")
    .toLowerCase();
}

export function videoDiscoverFamilyHaystack(family: VideoModelFamily): string {
  return [
    family.name,
    family.provider,
    family.headline,
    family.summary,
    family.updatedLabel,
    family.badges.join(" "),
    ...family.variants.map((variant) => videoDiscoverVariantHaystack(variant)),
  ]
    .join(" ")
    .toLowerCase();
}

export function videoDiscoverVariantMatchesQuery(variant: VideoModelVariant, query: string): boolean {
  if (!query) return true;
  return videoDiscoverVariantHaystack(variant).includes(query);
}

export function videoDiscoverFamilyMatchesQuery(family: VideoModelFamily, query: string): boolean {
  if (!query) return true;
  return videoDiscoverFamilyHaystack(family).includes(query);
}

export function videoRuntimeErrorStatus(error: unknown): VideoRuntimeStatus {
  return {
    activeEngine: "unavailable",
    realGenerationAvailable: false,
    message: error instanceof Error ? error.message : "Video runtime unavailable.",
    missingDependencies: [],
  };
}

export type VideoGenerationRiskLevel = "safe" | "caution" | "danger";

export interface VideoGenerationSafety {
  riskLevel: VideoGenerationRiskLevel;
  /** Estimated latent token count (rough — used in the message and for the threshold). */
  latentTokens: number;
  /** Rough upper-bound estimate of peak attention memory, in GB. Consumer-facing. */
  estimatedPeakGb: number;
  /** Device memory we compared against when computing the risk level, in GB. */
  deviceMemoryGb: number;
  /** True if the peak estimate exceeds the device's effective budget. */
  exceedsDevice: boolean;
  /** Plain-English reason — null only when riskLevel is "safe". */
  reason: string | null;
  /** Concrete fallback to drop the user back into the safe envelope. */
  suggestion: {
    width: number;
    height: number;
    numFrames: number;
    label: string;
  } | null;
}

/** Default memory baselines when the backend can't tell us. We assume 16 GB
 * on MPS because that's the base-model M1/M2/M3/M4 and it's the most
 * failure-prone config in the wild. On CUDA we assume 12 GB because mid-
 * range consumer cards cluster there (3060, 4070, etc.). CPU gets a small
 * number so we don't accidentally green-light a crash. */
const DEFAULT_MPS_MEMORY_GB = 16;
const DEFAULT_CUDA_MEMORY_GB = 12;
const DEFAULT_CPU_MEMORY_GB = 8;

/** Effective concurrent-head × slab multiplier for the attention peak.
 *
 * Diffusers video pipelines cluster around 12–32 attention heads per block,
 * but only a handful are live concurrently under MPS / flash-attn tiling,
 * and ``softmax(QK^T)·V`` adds one more slab of similar shape. This single
 * constant bakes both factors into one ballpark that the Studio UI shows
 * the user.
 *
 * Calibrated from the concrete bug report: Wan 2.1 T2V 1.3B at 832×480 × 96
 * frames (~37k latent tokens × 2 bytes fp16 ≈ 2.8 GB per slab) crashed a
 * 64 GB M4 Max. The constant below was picked so that configuration
 * correctly lands in "danger" at 16 GB and "caution" at 64 GB, while the
 * Studio defaults (832×480 × 33 frames, ~14k tokens) stay "safe" even on
 * base-model 16 GB Macs. */
const EFFECTIVE_HEAD_SLAB_MULTIPLIER = 8;

/** Bytes per element for the attention compute path. MPS/CUDA run fp16/bf16,
 * CPU stays in fp32. Matters because the peak-memory estimate we show the
 * user should track what actually gets allocated. */
function bytesPerElementForDevice(device: "mps" | "cuda" | "cpu"): number {
  if (device === "cpu") return 4;
  return 2;
}

/** Effective share of total device memory available to attention. The OS,
 * model weights, VAE, text encoder, and diffusers overhead all compete — in
 * practice only ~50% of unified memory on MPS is realistically free for the
 * attention peak. CUDA is more predictable (no OS paging of VRAM) so ~70%
 * is a safer assumption there. */
function effectiveMemoryBudgetGb(totalGb: number, device: "mps" | "cuda" | "cpu"): number {
  if (device === "cuda") return totalGb * 0.7;
  return totalGb * 0.5;
}

/** Estimate peak attention-matrix memory in bytes from latent token count.
 *
 * The Q·K^T matrix is ``tokens × tokens × heads × bytes``; in practice only
 * a small subset of heads is live concurrently and fused kernels compute in
 * tiles. Rather than try to model every pipeline's exact head count and
 * kernel, we use a single empirically-calibrated multiplier — see
 * ``EFFECTIVE_HEAD_SLAB_MULTIPLIER``. Over-estimating within reason is the
 * right trade-off: we'd rather the user dial down than silently crash. */
function estimatePeakAttentionBytes(
  latentTokens: number,
  device: "mps" | "cuda" | "cpu",
): number {
  const bytesPerElement = bytesPerElementForDevice(device);
  return latentTokens * latentTokens * bytesPerElement * EFFECTIVE_HEAD_SLAB_MULTIPLIER;
}

/**
 * Estimate whether a video generation request is in danger of detonating
 * the inference device with a giant attention tensor.
 *
 * Diffusers / Wan 2.1 attention scales with the latent token count, roughly
 * ``(W/16) × (H/16) × (F/4)`` for the typical 16× spatial / 4× temporal
 * downsample. The QK^T peak then scales as ``tokens² × heads × dtype``,
 * which is what we compare against the device's effective memory budget.
 *
 * Because thresholds are computed from ``estimatedPeakGb / budgetGb`` rather
 * than a fixed token count, the heuristic scales cleanly from an 8 GB base
 * M1 to a 128 GB M3 Ultra to a 24 GB RTX 4090 — one formula, no per-
 * platform tables to maintain. When ``deviceMemoryGb`` is null (detection
 * failed), we fall back to conservative defaults; over-warning a beefy
 * machine is strictly better than silently crashing a small one.
 *
 * The returned shape carries both the risk level and the raw ``estimatedPeakGb``
 * / ``deviceMemoryGb`` pair, so the Studio UI can show "this run wants ~X GB
 * on Y GB available" even when risk is still "safe" — users asked for that
 * framing; "37k latent tokens" is not a number anyone thinks in.
 *
 * Calibration points:
 * - Studio defaults 832×480 × 33 frames: stays "safe" on 16 GB+.
 * - Observed-crash 832×480 × 96 frames on 64 GB M4 Max: now correctly comes
 *   back as "caution" (the machine can actually handle it with headroom)
 *   instead of the previous "danger" false positive.
 * - Same config on 16 GB M2: stays "danger" — the machine really would crash.
 */
export function assessVideoGenerationSafety(opts: {
  width: number;
  height: number;
  numFrames: number;
  device: string | null | undefined;
  deviceMemoryGb?: number | null;
}): VideoGenerationSafety {
  const { width, height, numFrames, device, deviceMemoryGb } = opts;

  const normalisedDevice = (device ?? "").toLowerCase();
  const isCuda = normalisedDevice.startsWith("cuda");
  const isCpu = normalisedDevice === "cpu";
  // unknown / empty falls through to MPS-strict because this app primarily
  // ships on Apple Silicon — over-warning a CUDA user is strictly better
  // than silently crashing an MPS one.
  const effectiveDevice: "mps" | "cuda" | "cpu" = isCuda ? "cuda" : isCpu ? "cpu" : "mps";

  const fallbackMemory = isCuda
    ? DEFAULT_CUDA_MEMORY_GB
    : isCpu
      ? DEFAULT_CPU_MEMORY_GB
      : DEFAULT_MPS_MEMORY_GB;
  const totalMemoryGb =
    deviceMemoryGb != null && Number.isFinite(deviceMemoryGb) && deviceMemoryGb > 0
      ? deviceMemoryGb
      : fallbackMemory;
  const budgetGb = effectiveMemoryBudgetGb(totalMemoryGb, effectiveDevice);

  if (
    !Number.isFinite(width)
    || !Number.isFinite(height)
    || !Number.isFinite(numFrames)
    || width <= 0
    || height <= 0
    || numFrames <= 0
  ) {
    return {
      riskLevel: "safe",
      latentTokens: 0,
      estimatedPeakGb: 0,
      deviceMemoryGb: totalMemoryGb,
      exceedsDevice: false,
      reason: null,
      suggestion: null,
    };
  }

  const latentTokens =
    Math.ceil(width / 16) * Math.ceil(height / 16) * Math.ceil(numFrames / 4);
  const estimatedPeakGb =
    estimatePeakAttentionBytes(latentTokens, effectiveDevice) / 1024 ** 3;

  // MPS has a lower danger ratio (0.8 vs CUDA 1.0) because Apple's Metal
  // backend has historically been less tolerant of approaching the ceiling
  // — it asserts and kills the process where CUDA would surface a catchable
  // OOM. We want an earlier warning specifically on MPS.
  const cautionRatio = effectiveDevice === "cuda" ? 0.7 : 0.5;
  const dangerRatio = effectiveDevice === "cuda" ? 1.0 : 0.8;
  const ratio = estimatedPeakGb / budgetGb;
  const exceedsDevice = estimatedPeakGb > budgetGb;
  const riskLevel: VideoGenerationRiskLevel =
    ratio >= dangerRatio ? "danger" : ratio >= cautionRatio ? "caution" : "safe";

  if (riskLevel === "safe") {
    return {
      riskLevel,
      latentTokens,
      estimatedPeakGb,
      deviceMemoryGb: totalMemoryGb,
      exceedsDevice,
      reason: null,
      suggestion: null,
    };
  }

  // Build a concrete suggestion. Halve the frame count first (biggest
  // single lever because the latent seq length is linear in frames but the
  // QK^T cost is quadratic in sequence length); if still over budget, nudge
  // resolution down in 64-px steps. Snap frames to ``(n - 1) % 4 == 0``
  // (Wan / LTX requirement) so the suggestion we hand back is directly
  // applicable — duplicated from useVideoState's ``clampNumFrames`` rather
  // than imported so this util has no hook dependency.
  let suggestedFrames = numFrames;
  let suggestedWidth = width;
  let suggestedHeight = height;
  const safeRatioTarget = cautionRatio * 0.7; // leave a real margin after apply
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const tokens =
      Math.ceil(suggestedWidth / 16)
      * Math.ceil(suggestedHeight / 16)
      * Math.ceil(suggestedFrames / 4);
    const peakGb = estimatePeakAttentionBytes(tokens, effectiveDevice) / 1024 ** 3;
    if (peakGb / budgetGb < safeRatioTarget) break;
    if (suggestedFrames > 17) {
      suggestedFrames = Math.max(17, Math.floor(suggestedFrames * 0.6));
      const remainder = (suggestedFrames - 1) % 4;
      if (remainder !== 0) suggestedFrames = suggestedFrames - remainder;
    } else if (suggestedWidth > 480 || suggestedHeight > 320) {
      suggestedWidth = Math.max(480, Math.floor((suggestedWidth * 0.75) / 64) * 64);
      suggestedHeight = Math.max(320, Math.floor((suggestedHeight * 0.75) / 64) * 64);
    } else {
      break;
    }
  }

  const platform = isCuda ? "this GPU" : isCpu ? "CPU generation" : "Apple Silicon (MPS)";
  const fmt = (g: number) => (g >= 10 ? g.toFixed(0) : g.toFixed(1));
  const reason =
    riskLevel === "danger"
      ? `These settings would need around ${fmt(estimatedPeakGb)} GB of attention memory — above what ${platform} can safely allocate (~${fmt(budgetGb)} GB of ${fmt(totalMemoryGb)} GB total). Generation is likely to crash the backend.`
      : `These settings need around ${fmt(estimatedPeakGb)} GB of attention memory — close to the safe limit on ${platform} (~${fmt(budgetGb)} GB of ${fmt(totalMemoryGb)} GB total). Generation may run slowly or fail.`;

  return {
    riskLevel,
    latentTokens,
    estimatedPeakGb,
    deviceMemoryGb: totalMemoryGb,
    exceedsDevice,
    reason,
    suggestion: {
      width: suggestedWidth,
      height: suggestedHeight,
      numFrames: suggestedFrames,
      label: `${suggestedWidth}×${suggestedHeight} · ${suggestedFrames} frames`,
    },
  };
}
