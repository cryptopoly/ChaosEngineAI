import type React from "react";


export const KNOWN_INSTALLABLE_VIDEO_DEPS: ReadonlySet<string> = new Set([
  "imageio",
  "imageio-ffmpeg",
  "tiktoken",
  "sentencepiece",
  "protobuf",
  "ftfy",
]);


// Repos the mlx-video Apple Silicon engine supports natively. Mirrors
// _SUPPORTED_REPOS in backend_service/mlx_video_runtime.py — kept here
// so the Studio can decide when to surface the mlx-video chip without
// an extra capabilities round-trip. See FU-009 in CLAUDE.md.
//
// Today: LTX-2 prince-canuma pre-converted MLX repos only. Wan2.1/2.2
// require an explicit ``mlx_video.models.wan_2.convert`` step on raw HF
// weights (no pre-converted MLX repo today) — until that conversion is
// bundled, Wan paths use diffusers MPS.
export const MLX_VIDEO_SUPPORTED_REPOS: ReadonlySet<string> = new Set([
  "prince-canuma/LTX-2-distilled",
  "prince-canuma/LTX-2-dev",
  "prince-canuma/LTX-2.3-distilled",
  "prince-canuma/LTX-2.3-dev",
]);


export function isLtx2DistilledRepo(repo: string | null | undefined): boolean {
  return !!repo && repo.toLowerCase().startsWith("prince-canuma/ltx-2") && repo.toLowerCase().endsWith("-distilled");
}


// Quality presets: common starting points for the denoising step count.
// Frames are deliberately not part of the preset — frame count controls
// clip LENGTH, not image quality, and bundling it into "Draft/High/Max"
// confused users into thinking shorter clips were lower quality. Guidance
// is also omitted because the parent hook sets it per-model (LTX wants 3,
// Hunyuan wants 6, others 5) and presets shouldn't overwrite that.
export type VideoQualityPreset = "draft" | "standard" | "high" | "max";

export const QUALITY_PRESETS: Record<
  VideoQualityPreset,
  { label: string; sub: string; steps: number }
> = {
  draft: { label: "Draft", sub: "20 steps", steps: 20 },
  standard: { label: "Standard", sub: "30 steps", steps: 30 },
  high: { label: "High", sub: "40 steps", steps: 40 },
  max: { label: "Max", sub: "50 steps", steps: 50 },
};


// Aspect-ratio presets. Concrete resolutions rather than "apply ratio to
// current base" so clicking a pill has zero surprises. Values chosen to
// be safe across LTX / Wan / HunyuanVideo — they're all divisible by 8
// (diffusers requirement) and under the largest-tested resolutions the
// families ship with.
export type VideoAspectRatio = "1:1" | "4:3" | "16:9" | "9:16" | "21:9";

export const ASPECT_RATIOS: Record<
  VideoAspectRatio,
  { width: number; height: number }
> = {
  "1:1": { width: 512, height: 512 },
  "4:3": { width: 640, height: 480 },
  "16:9": { width: 768, height: 432 },
  "9:16": { width: 432, height: 768 },
  "21:9": { width: 1024, height: 440 },
};


// Numeric input handling that tolerates transient empty states during editing.
// The naive pattern ``onChange={e => setValue(Number(e.target.value) || fallback)}``
// treats an empty string as ``0`` and snaps back to the fallback — which means
// the user can never delete the last digit of a value (they see the default
// reappear). Instead we carry ``NaN`` as "user is mid-edit / field is empty",
// render it as "" in the input, and on blur snap to the fallback if still
// invalid. ``handleVideoGenerate`` + ``clampNumFrames`` defend against any
// ``NaN`` that slips through to the payload.
export function onNumericChange(
  event: React.ChangeEvent<HTMLInputElement>,
  setter: (value: number) => void,
): void {
  const raw = event.target.value;
  if (raw === "") {
    setter(Number.NaN);
    return;
  }
  const parsed = Number(raw);
  if (Number.isFinite(parsed)) setter(parsed);
}


export function onNumericBlur(
  current: number,
  setter: (value: number) => void,
  fallback: number,
  minimum: number = 1,
): void {
  if (!Number.isFinite(current) || current < minimum) setter(fallback);
}


export function displayNumber(value: number): number | string {
  return Number.isFinite(value) ? value : "";
}
