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

/** WebKit's ``fetch()`` (Safari + macOS Tauri's WKWebView) produces the
 * literal string ``"Load failed"`` when it can't reach the server — for
 * example after the Python sidecar dies from an MPS OOM. Chromium produces
 * ``"Failed to fetch"``. Both bubble up unchanged through ``fetchJson`` and
 * land in the runtime status message verbatim, where they read to the user
 * as a Diffusers / video runtime problem rather than a backend transport
 * problem. We translate them here so the Studio shows actionable copy
 * (mentioning Restart Backend) instead of two cryptic words. */
function isFetchTransportError(message: string): boolean {
  const trimmed = message.trim().toLowerCase();
  return (
    trimmed === "load failed"
    || trimmed === "failed to fetch"
    || trimmed === "networkerror when attempting to fetch resource."
    || trimmed.startsWith("networkerror")
  );
}

/** Timeout strings from ``fetchJson`` — the video runtime probe has a 30s
 * cap and will re-throw as ``"Request to /api/video/runtime timed out
 * after 30s"`` when the backend is too slow to answer (e.g. first-boot
 * torch import on Windows with a cold disk). */
function isFetchTimeoutError(message: string): boolean {
  return /\btimed out after\b/i.test(message);
}

export function videoRuntimeErrorStatus(error: unknown): VideoRuntimeStatus {
  const rawMessage = error instanceof Error ? error.message : "";
  let message: string;
  if (isFetchTransportError(rawMessage)) {
    // Don't say "Backend is not responding" — the global BACKEND ONLINE
    // pill is driven from the ``/api/health`` probe and may well still be
    // green while the video runtime probe fails (common during a backend
    // restart, or the first probe of a sidecar's life when torch is
    // importing). Name the specific subsystem so the UI isn't internally
    // contradictory.
    message = "Video runtime did not respond — the backend is likely still loading PyTorch. "
      + "First boot can take 30–60 seconds on Windows or Linux. This page will retry automatically; "
      + "use Restart Backend if it persists for more than a minute.";
  } else if (isFetchTimeoutError(rawMessage)) {
    message = "Video runtime probe timed out after 30 seconds — PyTorch is still importing in the backend. "
      + "On Windows/Linux with a cold disk this can take up to ~60 seconds on first boot. "
      + "This page will retry automatically; use Restart Backend if it persists for more than a minute.";
  } else {
    message = rawMessage || "Video runtime unavailable.";
  }
  return {
    activeEngine: "unavailable",
    realGenerationAvailable: false,
    message,
    missingDependencies: [],
  };
}

export type VideoGenerationRiskLevel = "safe" | "caution" | "danger";

export type VideoEffectiveDevice = "mps" | "cuda" | "cpu";

export interface VideoGenerationSafety {
  riskLevel: VideoGenerationRiskLevel;
  /** Estimated latent token count (rough — used in the message and for the threshold). */
  latentTokens: number;
  /** Rough upper-bound estimate of peak memory for this request, in GB. The
   * sum of the resident model footprint (weights + text encoder + VAE) and
   * the attention-matrix peak. Consumer-facing. */
  estimatedPeakGb: number;
  /** Resident memory estimate for the model itself (weights + text encoder +
   * VAE, adjusted for device-specific allocator overhead). Zero when the
   * caller didn't pass ``baseModelFootprintGb`` — used to show a breakdown
   * in the Studio capacity line. */
  modelFootprintGb: number;
  /** Device memory we compared against when computing the risk level, in GB. */
  deviceMemoryGb: number;
  /** True if the peak estimate exceeds the device's effective budget. */
  exceedsDevice: boolean;
  /** The device bucket actually used for the calculation. When the caller
   * passes ``"mps"``/``"cuda"``/``"cpu"`` we echo it back; when the caller
   * passes null/empty (backend probe failed, sidecar dead) we infer from
   * the host OS — Windows/Linux fall through to "cuda", everything else
   * to "mps". The Studio reads this so the always-visible capacity line
   * doesn't say "Apple Silicon" on a Windows RTX 4090 just because the
   * backend probe never came back. */
  effectiveDevice: VideoEffectiveDevice;
  /** True when ``effectiveDevice`` was inferred from the host OS rather
   * than supplied by the backend. Lets the Studio mark the device label
   * as a guess (consistent with how it tags the memory fallback). */
  effectiveDeviceWasInferred: boolean;
  /** Plain-English reason — null only when riskLevel is "safe". */
  reason: string | null;
  /** Concrete fallback to drop the user back into the safe envelope. Null
   * when the model's resident footprint alone exceeds the safe envelope —
   * in that case, no per-request tweak can recover and the user needs a
   * smaller model or a bigger machine. */
  suggestion: {
    width: number;
    height: number;
    numFrames: number;
    label: string;
  } | null;
}

/** Best-effort host-platform detection for the case where the backend probe
 * never came back (sidecar crashed, "Failed to fetch", first-launch race).
 * We use ``navigator.userAgentData.platform`` when available (modern
 * Chromium) and fall back to ``navigator.platform`` / ``navigator.userAgent``
 * — the WKWebView shipped inside macOS Tauri only exposes the legacy
 * fields, but they're reliable for the macOS-vs-not split.
 *
 * The bucket is intentionally coarse: we only need to decide "should the
 * memory warning say Apple Silicon or this GPU?" so anything that isn't
 * obviously macOS is treated as a CUDA host. Linux without an NVIDIA card
 * still ends up in the CUDA bucket — over-stating the headroom there is
 * less harmful than the inverse (telling a Windows RTX 4090 user that
 * their 24 GB card is "close to the safe limit on Apple Silicon"). */
export function inferDeviceFromHostPlatform(): "mps" | "cuda" {
  if (typeof navigator === "undefined") return "mps";
  // ``userAgentData`` is the modern UA-CH API; it's narrower and more
  // reliable than the deprecated ``platform`` string but isn't supported
  // by Safari / WKWebView yet — hence the layered fallback below.
  const uaData = (navigator as unknown as { userAgentData?: { platform?: string } }).userAgentData;
  const uaDataPlatform = (uaData?.platform ?? "").toLowerCase();
  if (uaDataPlatform) {
    if (uaDataPlatform.includes("mac")) return "mps";
    return "cuda";
  }
  const legacyPlatform = (navigator.platform ?? "").toLowerCase();
  const ua = (navigator.userAgent ?? "").toLowerCase();
  if (legacyPlatform.includes("mac") || ua.includes("mac os")) return "mps";
  return "cuda";
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
 * constant bakes both factors into one ballpark — the attention term in
 * the overall peak estimate.
 *
 * In practice the Wan-family OOMs on MPS are dominated by the resident
 * model footprint (transformer + UMT5-XXL text encoder + VAE) rather than
 * this attention slab, so the crash-case calibration lives in
 * ``estimateResidentModelGb``. This constant is left modest so the
 * attention-only path (when no model footprint is known) is still
 * reasonable for the Studio defaults on 16 GB base Macs. */
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
  // Apple Silicon's Metal ``recommendedMaxWorkingSetSize`` is ~75% of
  // unified memory by default — that's the actual ceiling. Earlier
  // values (0.5, then 0.65) were leaving real headroom unused; Wan 2.2
  // 5B at 22 GB resident on a 64 GB M4 Max kept tripping caution
  // (22/41.6 = 53%) even though it fits in ~33% of total memory.
  // 0.75 matches the real device limit and lets the comparable models
  // clear the caution band.
  if (device === "mps") return totalGb * 0.75;
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

/** Translate a model's on-disk size into an approximate resident-memory
 * estimate during generation.
 *
 * The on-disk size (from the catalog ``sizeGb``) packs the transformer
 * weights, text encoder (often the dominant term — UMT5-XXL is ~11 GB for
 * Wan 2.1), VAE, and CLIP in fp16. Once the pipeline is live, PyTorch's
 * allocator holds slightly more than that because of intermediate buffers
 * it doesn't aggressively free between denoising steps. On MPS the factor
 * runs ~1.4× in practice (calibrated against the Wan 2.1 1.3B crash on a
 * 64 GB M4 Max: disk = 16.4 GB → resident ≈ 23 GB → peak with attention ≈
 * 27 GB → observed 88 GB under PyTorch's 1.4× watermark). CUDA is tighter
 * because dedicated VRAM + better allocator reuse; CPU sits in between. */
function estimateResidentModelGb(
  baseFootprintGb: number,
  device: "mps" | "cuda" | "cpu",
): number {
  if (!(baseFootprintGb > 0) || !Number.isFinite(baseFootprintGb)) return 0;
  const factor = device === "mps" ? 1.4 : device === "cpu" ? 1.3 : 1.05;
  return baseFootprintGb * factor;
}

/**
 * Estimate whether a video generation request is in danger of detonating
 * the inference device. The estimate combines two memory terms:
 *
 *   1. **Resident model footprint** — weights + text encoder + VAE sitting
 *      in memory the whole time. For Wan 2.1 / HunyuanVideo this is the
 *      dominant cost on MPS, where the UMT5-XXL text encoder alone is ~11
 *      GB. Opt-in via ``baseModelFootprintGb``; the Studio passes
 *      ``selectedVariant.sizeGb`` so the warning actually reflects the
 *      real memory pressure the user is about to create.
 *
 *   2. **Attention peak** — scales with ``tokens² × heads × dtype`` where
 *      ``tokens = (W/16) × (H/16) × (F/4)`` for the typical 16× spatial /
 *      4× temporal downsample. Purely request-driven.
 *
 * Thresholds are computed from ``estimatedPeakGb / budgetGb`` so one
 * formula scales cleanly from an 8 GB base M1 to a 128 GB M3 Ultra to a 24
 * GB RTX 4090 — no per-platform tables. When ``deviceMemoryGb`` is null
 * (detection failed), we fall back to conservative defaults; over-warning
 * a beefy machine is strictly better than silently crashing a small one.
 *
 * The returned shape carries the risk level plus the raw numbers so the
 * Studio UI can show "model ≈ 23 GB, this run peak ≈ 27 GB on 32 GB
 * available" even when risk is "safe" — users asked for that framing
 * ("37k latent tokens" is not a number anyone thinks in).
 *
 * Calibration points:
 * - Studio defaults 832×480 × 33 frames (no baseFootprint): stays "safe"
 *   on 16 GB+ — preserves the attention-only heuristic when the caller
 *   doesn't know which model is loaded.
 * - Wan 2.1 T2V 1.3B (baseFootprint 16.4 GB) at 832×480 × 40 frames on a
 *   64 GB M4 Max: lands on "caution" — large but within the bumped MPS
 *   working-set budget after the catalog started carrying resident peaks.
 * - Same Wan config on 128 GB M3 Ultra: stays "safe" — the machine has
 *   real headroom for the 23 GB resident footprint + attention peak.
 * - LTX-Video (baseFootprint 2 GB) at 768×512 × 41 frames on 32 GB:
 *   stays "safe" — small model, proven to run on consumer Macs.
 */
export function assessVideoGenerationSafety(opts: {
  width: number;
  height: number;
  numFrames: number;
  device: string | null | undefined;
  deviceMemoryGb?: number | null;
  /** On-disk / fp16 size of the selected model (catalog ``sizeGb``). When
   * provided, the estimate includes the resident model footprint — crucial
   * on MPS where the text encoder + weights are the dominant cost. Leave
   * unset for the narrow "attention-only" question. */
  baseModelFootprintGb?: number | null;
  /** Resident peak (catalog ``runtimeFootprintGb``). When provided, used
   * directly — bypasses the ``sizeGb × 1.4`` heuristic. Disk size
   * overstates resident because of duplicate sharded safetensors. */
  runtimeFootprintGb?: number | null;
}): VideoGenerationSafety {
  const { width, height, numFrames, device, deviceMemoryGb, baseModelFootprintGb, runtimeFootprintGb } = opts;

  const normalisedDevice = (device ?? "").toLowerCase();
  const isCuda = normalisedDevice.startsWith("cuda");
  const isCpu = normalisedDevice === "cpu";
  const isMps = normalisedDevice === "mps";
  // When the backend hasn't told us (probe failed, sidecar dead, "Failed to
  // fetch"), we used to default to "mps" unconditionally. That was wrong on
  // Windows / Linux — a user on an RTX 4090 saw their 24 GB card warned
  // about as if it were "Apple Silicon (MPS) (~8 GB safe)". Now we infer
  // from the host OS so the fallback bucket matches the machine the user is
  // actually on. The macOS branch keeps its old behaviour (MPS-strict).
  const effectiveDevice: VideoEffectiveDevice = isCuda
    ? "cuda"
    : isCpu
      ? "cpu"
      : isMps
        ? "mps"
        : inferDeviceFromHostPlatform();
  const effectiveDeviceWasInferred = !isCuda && !isCpu && !isMps;

  const fallbackMemory =
    effectiveDevice === "cuda"
      ? DEFAULT_CUDA_MEMORY_GB
      : effectiveDevice === "cpu"
        ? DEFAULT_CPU_MEMORY_GB
        : DEFAULT_MPS_MEMORY_GB;
  const totalMemoryGb =
    deviceMemoryGb != null && Number.isFinite(deviceMemoryGb) && deviceMemoryGb > 0
      ? deviceMemoryGb
      : fallbackMemory;
  const budgetGb = effectiveMemoryBudgetGb(totalMemoryGb, effectiveDevice);

  const baseFootprint =
    baseModelFootprintGb != null
    && Number.isFinite(baseModelFootprintGb)
    && baseModelFootprintGb > 0
      ? baseModelFootprintGb
      : 0;
  // Prefer explicit runtime footprint when the catalog supplies one — it
  // already reflects resident peak. Otherwise estimate from disk size.
  const modelFootprintGb =
    runtimeFootprintGb != null
    && Number.isFinite(runtimeFootprintGb)
    && runtimeFootprintGb > 0
      ? runtimeFootprintGb
      : estimateResidentModelGb(baseFootprint, effectiveDevice);

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
      modelFootprintGb: 0,
      deviceMemoryGb: totalMemoryGb,
      exceedsDevice: false,
      effectiveDevice,
      effectiveDeviceWasInferred,
      reason: null,
      suggestion: null,
    };
  }

  const latentTokens =
    Math.ceil(width / 16) * Math.ceil(height / 16) * Math.ceil(numFrames / 4);
  const attentionPeakGb =
    estimatePeakAttentionBytes(latentTokens, effectiveDevice) / 1024 ** 3;
  const estimatedPeakGb = modelFootprintGb + attentionPeakGb;

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

  // Short-circuit the suggestion loop when the model's resident footprint
  // alone is too big for this device. No matter how small the request,
  // the weights + text encoder still have to live in memory — the right
  // answer is "pick a smaller model or run on a bigger machine", not
  // "try 480×320 × 17 frames" (which would also crash). We threshold at
  // the caution ratio rather than danger so we don't hand back bogus
  // suggestions in the caution band either.
  const safeRatioTarget = cautionRatio * 0.7; // leave a real margin after apply
  if (modelFootprintGb > cautionRatio * budgetGb) {
    const comfortBudgetGb = cautionRatio * budgetGb;
    const highRiskBudgetGb = dangerRatio * budgetGb;
    const reason =
      riskLevel === "danger"
        ? modelFootprintGb > budgetGb
          ? `The model needs ~${fmt(modelFootprintGb)} GB just to hold its model weights + text encoder. On ${platform} with ${fmt(totalMemoryGb)} GB total, the estimated working set is ~${fmt(budgetGb)} GB, so the model alone is already over that. Even the smallest clip would be likely to crash the backend. Try a smaller model (LTX-Video is ~2 GB) or a machine with more memory.`
          : `The model needs ~${fmt(modelFootprintGb)} GB just to hold its model weights + text encoder, and this run peaks around ~${fmt(estimatedPeakGb)} GB. On ${platform} with ${fmt(totalMemoryGb)} GB total, that is above the high-risk threshold (~${fmt(highRiskBudgetGb)} GB) and close to the estimated working set (~${fmt(budgetGb)} GB). Generation is likely to crash the backend; lower the settings or choose a smaller model.`
        : `The model needs ~${fmt(modelFootprintGb)} GB just to hold its model weights + text encoder. On ${platform} with ${fmt(totalMemoryGb)} GB total, that is above the conservative comfort target (~${fmt(comfortBudgetGb)} GB) but below the estimated working set (~${fmt(budgetGb)} GB). Generation may run slowly or fail; consider lowering settings if it becomes unstable.`;
    return {
      riskLevel,
      latentTokens,
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

  // Normal suggestion loop. Halve the frame count first (biggest single
  // lever because latent seq length is linear in frames but the QK^T cost
  // is quadratic in sequence length); if still over budget, nudge
  // resolution down in 64-px steps. Snap frames to ``(n - 1) % 4 == 0``
  // (Wan / LTX requirement) so the suggestion we hand back is directly
  // applicable — duplicated from useVideoState's ``clampNumFrames`` rather
  // than imported so this util has no hook dependency.
  let suggestedFrames = numFrames;
  let suggestedWidth = width;
  let suggestedHeight = height;
  for (let attempt = 0; attempt < 8; attempt += 1) {
    const tokens =
      Math.ceil(suggestedWidth / 16)
      * Math.ceil(suggestedHeight / 16)
      * Math.ceil(suggestedFrames / 4);
    const peakGb =
      modelFootprintGb
      + estimatePeakAttentionBytes(tokens, effectiveDevice) / 1024 ** 3;
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
    latentTokens,
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
      numFrames: suggestedFrames,
      label: `${suggestedWidth}×${suggestedHeight} · ${suggestedFrames} frames`,
    },
  };
}
