/**
 * Pure rules of the Upscale tab (FU-617): which sizes can be asked for, what the quality rows' fit verdicts mean
 * for the choice, which buttons a job's state shows, and the progress and time-left numbers. The backend
 * (``routes/video_upscale.py``) decides in the end; these rules only spare the user a request it would refuse.
 */
import type { VideoUpscaleAddonStatus, VideoUpscaleJob, VideoUpscaleModel, VideoUpscaleRequest, VideoUpscaleState } from "../../api/videoUpscale";

/** The largest output side the backend makes (``video_upscale/engine.py`` ``MAX_SIDE``). */
export const MAX_SIDE = 4096;

export type SizeChoice = "2x" | "4x" | "1080p" | "1440p" | "2160p";

export const SIZE_CHOICES: SizeChoice[] = ["2x", "4x", "1080p", "1440p", "2160p"];

const TARGET_HEIGHTS: Record<string, number> = { "1080p": 1080, "1440p": 1440, "2160p": 2160 };

export interface VideoSize {
  width: number;
  height: number;
}

/** The nearest even whole number, at least 2 (the backend's ``even``: yuv420p needs even sides). */
export function even(value: number): number {
  return Math.max(2, Math.round(value / 2) * 2);
}

export function isScaleChoice(choice: SizeChoice): boolean {
  return choice === "2x" || choice === "4x";
}

/** The output size of a choice for a source, or null when the source size is unknown. */
export function outputSize(choice: SizeChoice, source: VideoSize | null): VideoSize | null {
  if (!source || source.width <= 0 || source.height <= 0) return null;
  if (isScaleChoice(choice)) {
    const scale = choice === "4x" ? 4 : 2;
    return { width: even(source.width * scale), height: even(source.height * scale) };
  }
  const height = TARGET_HEIGHTS[choice];
  return { width: even((source.width * height) / source.height), height };
}

export type SizeBlock = "limit" | "needsSize" | "notBigger";

export interface SizeOption {
  choice: SizeChoice;
  /** The size it would make, when the source size is known. */
  size: VideoSize | null;
  /** Why it cannot be picked, or null. */
  block: SizeBlock | null;
}

/**
 * Whether a size can be asked for. A scale (2x, 4x) is always offered when the source size is unknown: the
 * backend then checks it and its message is shown. A target height needs the source size (for the width),
 * must be taller than the source, and no side may pass ``MAX_SIDE``.
 */
export function sizeOption(choice: SizeChoice, source: VideoSize | null): SizeOption {
  const size = outputSize(choice, source);
  if (!size) return { choice, size: null, block: isScaleChoice(choice) ? null : "needsSize" };
  if (Math.max(size.width, size.height) > MAX_SIDE) return { choice, size, block: "limit" };
  if (!isScaleChoice(choice) && source && size.height <= source.height) return { choice, size, block: "notBigger" };
  return { choice, size, block: null };
}

/** The first size that can be picked: 2x unless it is blocked. */
export function defaultSizeChoice(source: VideoSize | null): SizeChoice {
  return SIZE_CHOICES.find((choice) => sizeOption(choice, source).block === null) ?? "2x";
}

/** The size part of the start request, or null when the choice is blocked. */
export function sizeRequest(choice: SizeChoice, source: VideoSize | null): Pick<VideoUpscaleRequest, "scale" | "width" | "height"> | null {
  const option = sizeOption(choice, source);
  if (option.block) return null;
  if (isScaleChoice(choice)) return { scale: choice === "4x" ? 4 : 2 };
  return option.size ? { width: option.size.width, height: option.size.height } : null;
}

export type QualityVerdict = "fits" | "tight" | "wont-fit" | "unknown" | "none";

export interface QualityOption {
  id: string;
  verdict: QualityVerdict;
  /** False for a row that will not fit (the backend refuses it) or is not downloaded. */
  selectable: boolean;
  installed: boolean;
}

/** What a quality row's fit verdict means for the choice. ``unknown`` stays selectable: it is only "not measured". */
export function qualityOption(model: VideoUpscaleModel): QualityOption {
  const verdict: QualityVerdict = model.fit?.status ?? "none";
  return { id: model.id, verdict, installed: model.installed, selectable: model.installed && verdict !== "wont-fit" };
}

/** The row to start on: the current one when it can be picked, else the catalogue default, else the first one. */
export function defaultModelId(models: VideoUpscaleModel[], current: string | null): string | null {
  const ok = models.filter((model) => qualityOption(model).selectable);
  if (current && ok.some((model) => model.id === current)) return current;
  return (ok.find((model) => model.default) ?? ok[0] ?? null)?.id ?? null;
}

export interface JobButtons {
  cancel: boolean;
  resume: boolean;
  reveal: boolean;
  open: boolean;
  /** Clear the card and pick another video. */
  dismiss: boolean;
}

export function isLive(state: VideoUpscaleState): boolean {
  return state === "queued" || state === "running";
}

/** The buttons a job card shows for its state. */
export function jobButtons(job: Pick<VideoUpscaleJob, "state" | "resumable">): JobButtons {
  const live = isLive(job.state);
  const done = job.state === "done";
  return {
    cancel: live,
    resume: !live && !done && job.resumable,
    reveal: done,
    open: done,
    dismiss: !live,
  };
}

/** Seconds left from the measured rate, or null while there is no rate yet (or nothing left to do). */
export function etaSeconds(job: Pick<VideoUpscaleJob, "state" | "framesDone" | "framesTotal" | "framesPerSecond">): number | null {
  if (job.state !== "running" || !job.framesPerSecond || job.framesPerSecond <= 0 || job.framesTotal <= 0) return null;
  const left = Math.max(0, job.framesTotal - job.framesDone);
  return left > 0 ? left / job.framesPerSecond : null;
}

export interface EtaParts {
  unit: "seconds" | "minutes" | "hours";
  value: number;
  /** Minutes past the hour, for the "hours" unit. */
  minutes?: number;
}

/** Time left in the unit a person reads: seconds under a minute, minutes under an hour, then hours and minutes. */
export function etaParts(seconds: number): EtaParts {
  const s = Math.max(1, Math.round(seconds));
  if (s < 60) return { unit: "seconds", value: s };
  const m = Math.round(s / 60);
  if (m < 60) return { unit: "minutes", value: m };
  return { unit: "hours", value: Math.floor(m / 60), minutes: m % 60 };
}

/** Whole percent done, 0 to 100. */
export function percentDone(job: Pick<VideoUpscaleJob, "fraction" | "framesDone" | "framesTotal">): number {
  const fraction = job.framesTotal > 0 ? job.framesDone / job.framesTotal : job.fraction;
  return Math.max(0, Math.min(100, Math.floor(fraction * 100)));
}

/** The install step 1 to 3 (packages, weights, check), or 0 when nothing is installing. */
export function installStep(phase: string): number {
  return phase === "packages" ? 1 : phase === "weights" ? 2 : phase === "check" ? 3 : 0;
}

/** About how many MB the install downloads: the weights still missing (the two packages are well under 1 MB). */
export function installSizeMb(status: Pick<VideoUpscaleAddonStatus, "models">): number {
  return Math.max(1, Math.round(status.models.reduce((sum, model) => sum + (model.installed ? 0 : model.downloadMb), 0)));
}

/** The install card shows until the add-on and the default model are in, and while an install runs or failed. */
export function showInstallCard(status: VideoUpscaleAddonStatus | null): boolean {
  if (!status) return false;
  return !status.installed || status.job.running || status.job.phase === "error";
}

/** Catalogue rows the add-on has not downloaded yet. */
export function missingModelIds(status: Pick<VideoUpscaleAddonStatus, "models">): string[] {
  return status.models.filter((model) => !model.installed).map((model) => model.id);
}

/** The file name of a local path, Windows or POSIX. */
export function fileNameOf(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}
