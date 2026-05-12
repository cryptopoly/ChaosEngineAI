/**
 * Defaults + per-family overrides for the video Studio.
 *
 * Lives outside ``useVideoState`` so the constants don't fall back into
 * the hook closure on every re-render. Values are conservative on
 * purpose — the first out-of-box generate must complete on Apple Silicon
 * unified memory rather than detonating Metal with a giant attention
 * tensor (regression: Wan 2.1 1.3B at 832×480 × 96 frames × 50 steps
 * blew up MPS during initial testing).
 */

export const MAX_VIDEO_SEED = 2147483647;

export const DEFAULT_VIDEO_NUM_FRAMES = 33;
export const DEFAULT_VIDEO_FPS = 24;
export const DEFAULT_VIDEO_STEPS = 30;
export const DEFAULT_VIDEO_GUIDANCE = 5.0;


// Baseline negative prompt that's generic enough to apply across every
// open-source video model we ship (LTX, Wan, HunyuanVideo, Mochi). With
// the field blank the models render without any corrective signal and
// produce noticeably worse geometry/anatomy — especially on LTX, which
// has the weakest priors. Users can edit or clear this if they have a
// model-specific preference.
export const DEFAULT_VIDEO_NEGATIVE_PROMPT =
  "worst quality, low quality, blurry, distorted, deformed, bad anatomy, "
  + "watermark, text, logo, static, frozen frame, jittery, flickering";


// Per-family recommended CFG. LTX dev pipelines commonly use CFG 3.0;
// the distilled LTX-2 MLX path ignores CFG entirely, but keeping the
// visible slider at 3.0 avoids over-guiding when the user switches to
// dev. HunyuanVideo benefits from stronger guidance. Everything else
// stays on the generic default.
export function recommendedGuidanceForRepo(repo: string | null | undefined): number {
  if (!repo) return DEFAULT_VIDEO_GUIDANCE;
  const lowered = repo.toLowerCase();
  if (lowered.includes("ltx")) return 3.0;
  if (lowered.includes("hunyuan")) return 6.0;
  return DEFAULT_VIDEO_GUIDANCE;
}


// Wan-family pipelines require ``(num_frames - 1) % 4 == 0``. We round to
// the nearest valid value so the user can type any frame count and we still
// hand the backend something it can run.
export function clampNumFrames(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_VIDEO_NUM_FRAMES;
  const clamped = Math.max(1, Math.min(257, Math.round(value)));
  // Snap to the nearest n where (n - 1) % 4 == 0 (i.e. 1, 5, 9, 13, ...)
  const remainder = (clamped - 1) % 4;
  if (remainder === 0) return clamped;
  const down = clamped - remainder;
  const up = down + 4;
  return up - clamped < clamped - down ? up : down;
}


/** Parse "832x480" (or similar) into [width, height], falling back to defaults. */
export function parseRecommendedResolution(
  value: string | null | undefined,
  defaultWidth: number,
  defaultHeight: number,
): [number, number] {
  if (!value) return [defaultWidth, defaultHeight];
  const match = String(value).trim().match(/^(\d+)\s*[xX×]\s*(\d+)/);
  if (!match) return [defaultWidth, defaultHeight];
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return [defaultWidth, defaultHeight];
  if (width < 256 || width > 2048 || height < 256 || height > 2048) {
    return [defaultWidth, defaultHeight];
  }
  return [width, height];
}
