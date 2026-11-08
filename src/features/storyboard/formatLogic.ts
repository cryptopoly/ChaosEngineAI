/**
 * Pure rules for a project's format (D-64): aspect ratio, resolution, frame rate and the export upscale,
 * and the fit hint shown per choice. The hint reuses the Video Studio's memory estimate and the
 * catalog's per-model size and frame-rate rows; nothing here is a number of its own. No React here so
 * the rules are unit-tested directly.
 */
import type { StoryboardExportPreset, StoryboardResolution, StoryboardUpscale, VideoModelVariant } from "../../types";
import { assessVideoGenerationSafety } from "../../utils/videos";

export const ASPECTS: StoryboardExportPreset[] = ["16:9", "9:16", "1:1"];
export const RESOLUTIONS: StoryboardResolution[] = ["480p", "720p", "1080p", "1440p", "2160p"];
export const FPS_CHOICES = [24, 25, 30, 48, 50, 60] as const;
export const UPSCALES: StoryboardUpscale[] = ["none", "1440p", "2160p"];
export const DEFAULT_RESOLUTION: StoryboardResolution = "720p";
export const DEFAULT_FPS = 24;

const SHORT_SIDE: Record<StoryboardResolution, number> = { "480p": 480, "720p": 720, "1080p": 1080, "1440p": 1440, "2160p": 2160 };

/** Marketing name next to the resolution where one is in common use. */
export const RESOLUTION_ALIAS: Partial<Record<StoryboardResolution, string>> = { "2160p": "4K" };

/** Mirror of `size_for` in backend_service/storyboard/formats.py. */
export function sizeFor(aspect: StoryboardExportPreset, resolution: StoryboardResolution): { width: number; height: number } {
  const short = SHORT_SIDE[resolution];
  if (aspect === "1:1") return { width: short, height: short };
  const long = short === 480 ? 854 : Math.round((short * 16) / 9);
  return aspect === "16:9" ? { width: long, height: short } : { width: short, height: long };
}

/** The `defaults` object for a new project from the dialog's choices. */
export function newProjectDefaults(aspect: StoryboardExportPreset, resolution: StoryboardResolution, fps: number): Record<string, unknown> {
  return { ...sizeFor(aspect, resolution), resolution, fps, exportPreset: aspect };
}

/** The `edit_project` defaults for a changed aspect or resolution: the size follows the format. */
export function formatPatch(
  current: { exportPreset: StoryboardExportPreset; resolution: StoryboardResolution },
  change: { aspect?: StoryboardExportPreset; resolution?: StoryboardResolution },
): { defaults: Record<string, unknown> } {
  const aspect = change.aspect ?? current.exportPreset;
  const resolution = change.resolution ?? current.resolution;
  return { defaults: { ...sizeFor(aspect, resolution), exportPreset: aspect, resolution } };
}

/** The size an export 'Upscale' choice gives, or null when it is not larger than the film. */
export function upscaleSize(
  aspect: StoryboardExportPreset,
  upscale: StoryboardUpscale,
  film: { width: number; height: number },
): { width: number; height: number } | null {
  if (upscale === "none") return null;
  const size = sizeFor(aspect, upscale);
  return size.width * size.height > film.width * film.height ? size : null;
}

// ---- fit hints ----
export type FitLevel = "safe" | "tight" | "wontfit" | "unknown";

/** What the hint needs from the machine and the chosen video model. */
export interface VideoFitContext {
  /** The project's video model, or the first installed one when it is "Automatic". */
  model: VideoModelVariant | null;
  device: string | null;
  deviceMemoryGb: number | null;
  residentGb: number | null;
}

export interface ResolutionFit {
  level: FitLevel;
  /** The size the model would render at, when it is below the project size. */
  modelSize: { width: number; height: number } | null;
  peakGb: number | null;
}

/** The seconds of a typical shot (a new shot is 4 s); the estimate is for a clip this long. */
export const HINT_CLIP_SECONDS = 4;

function snap16(value: number): number {
  return Math.max(16, Math.floor(value / 16) * 16);
}

/**
 * The size the video model renders for a project size: the project size, or the largest size inside the
 * model's own limits (catalog `maxSide` and `maxPixels`) with the same aspect. Null when no cap applies.
 */
export function modelSizeFor(
  model: Pick<VideoModelVariant, "maxSide" | "maxPixels"> | null,
  size: { width: number; height: number },
): { width: number; height: number } | null {
  if (!model) return null;
  const long = Math.max(size.width, size.height);
  const area = size.width * size.height;
  const bySide = model.maxSide && long > model.maxSide ? model.maxSide / long : 1;
  const byArea = model.maxPixels && area > model.maxPixels ? Math.sqrt(model.maxPixels / area) : 1;
  const scale = Math.min(bySide, byArea);
  if (scale >= 1) return null;
  return { width: snap16(size.width * scale), height: snap16(size.height * scale) };
}

export function resolutionFit(
  ctx: VideoFitContext,
  aspect: StoryboardExportPreset,
  resolution: StoryboardResolution,
  fps: number,
): ResolutionFit {
  const model = ctx.model;
  if (!model) return { level: "unknown", modelSize: null, peakGb: null };
  const size = sizeFor(aspect, resolution);
  const capped = modelSizeFor(model, size);
  const run = capped ?? size;
  const safety = assessVideoGenerationSafety({
    width: run.width,
    height: run.height,
    numFrames: Math.max(8, Math.round(HINT_CLIP_SECONDS * fps)),
    device: ctx.device,
    deviceMemoryGb: ctx.deviceMemoryGb,
    baseModelFootprintGb: model.sizeGb,
    runtimeFootprintGb: model.runtimeFootprintGb,
    runtimeFootprintMpsGb: model.runtimeFootprintMpsGb,
    runtimeFootprintCudaGb: model.runtimeFootprintCudaGb,
    runtimeFootprintCpuGb: model.runtimeFootprintCpuGb,
    repo: model.repo,
    otherResidentGb: ctx.residentGb,
    skipAttentionEstimate: Boolean(model.runtime?.toLowerCase().includes("ltx-2-mlx")),
  });
  const level: FitLevel = safety.exceedsDevice ? "wontfit" : safety.riskLevel === "safe" ? "safe" : "tight";
  return { level, modelSize: capped, peakGb: safety.estimatedPeakGb };
}

/** True when the model states a frame rate and it differs from the project's. */
export function fpsNeedsConversion(model: Pick<VideoModelVariant, "nativeFps"> | null, fps: number): boolean {
  return Boolean(model?.nativeFps && model.nativeFps !== fps);
}

/** What the screens hold about the machine and the installed video models; the dialog picks the model from it. */
export interface VideoFitSource {
  variants: VideoModelVariant[];
  device: string | null;
  deviceMemoryGb: number | null;
  residentGb: number | null;
}

/** The fit context for a project's video model; "Automatic" (null) uses the first installed model. */
export function fitContext(source: VideoFitSource | undefined, videoModelId: string | null): VideoFitContext {
  const variants = source?.variants ?? [];
  const model = variants.find((variant) => variant.id === videoModelId) ?? (videoModelId ? null : (variants[0] ?? null));
  return {
    model,
    device: source?.device ?? null,
    deviceMemoryGb: source?.deviceMemoryGb ?? null,
    residentGb: source?.residentGb ?? null,
  };
}
