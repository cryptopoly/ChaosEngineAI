/**
 * Studio time estimates scaled from a row's catalog figure to the size,
 * frames and steps asked for (FU-165). Mirrors
 * `backend_service/helpers/time_estimate.py`; both are checked against
 * `tests/fixtures/time_estimate_cases.json` so they cannot drift.
 *
 * seconds = catalog x (fixed + (1 - fixed) x steps/refSteps x w(T)/w(Tref)),
 * w(T) = T + T^2/quad, T = W/16 x H/16 x F/4 (images: F = 1).
 */

export interface EstimateRef {
  width: number;
  height: number;
  frames: number;
  steps: number;
  /** Share of the catalog time that does not grow with the work; measured per row. */
  fixed?: number;
  quad?: number;
  /** The engine runs a set schedule: the steps setting does not change the time. */
  ignoresSteps?: boolean;
}

export const VIDEO_FIXED = 0.04;
export const DEFAULT_QUAD = 53300;
export const IMAGE_FIXED = 0.6;
export const MFLUX_FIXED = 0.1;

export function defaultFixed(kind: "image" | "video", engine?: string | null): number {
  if (kind === "video") return VIDEO_FIXED;
  if (engine === "mflux") return MFLUX_FIXED;
  return IMAGE_FIXED;
}

function positive(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function round1(value: number): number {
  return Math.floor(value * 10 + 0.5) / 10;
}

function work(width: number, height: number, frames: number, quad: number): number {
  const tokens = (width / 16) * (height / 16) * (frames / 4);
  return tokens + (tokens * tokens) / quad;
}

/** Seconds for this run, or null when the row has no catalog time. Never throws. */
export function scaledSeconds(
  catalogSeconds: unknown,
  ref: unknown,
  width: unknown,
  height: unknown,
  frames: unknown = 1,
  steps: unknown = null,
  fixed: number | null = null,
  quad: number | null = null,
): number | null {
  const catalog = positive(catalogSeconds);
  if (catalog === null) return null;
  if (!ref || typeof ref !== "object") return round1(catalog);
  const r = ref as Partial<EstimateRef>;
  const refW = positive(r.width);
  const refH = positive(r.height);
  const refF = positive(r.frames);
  const refS = positive(r.steps);
  const w = positive(width);
  const h = positive(height);
  if (refW === null || refH === null || refF === null || refS === null || w === null || h === null) {
    return round1(catalog);
  }
  const f = positive(frames) ?? refF;
  const s = r.ignoresSteps ? refS : positive(steps) ?? refS;
  const share = Math.min(positive(r.fixed) ?? positive(fixed) ?? VIDEO_FIXED, 1);
  const q = positive(r.quad) ?? positive(quad) ?? DEFAULT_QUAD;
  const ratio = (s / refS) * (work(w, h, f, q) / work(refW, refH, refF, q));
  const seconds = catalog * (share + (1 - share) * ratio);
  return Number.isFinite(seconds) ? round1(seconds) : round1(catalog);
}

/** A catalog row's time for these settings, with the engine default share when the row has none. */
export function variantSeconds(
  variant: { estimatedGenerationSeconds?: number | null; estimateRef?: EstimateRef | null; engine?: string | null } | null | undefined,
  kind: "image" | "video",
  settings: { width: number; height: number; frames?: number | null; steps?: number | null },
): number | null {
  if (!variant) return null;
  return scaledSeconds(
    variant.estimatedGenerationSeconds,
    variant.estimateRef,
    settings.width,
    settings.height,
    kind === "image" ? 1 : settings.frames ?? null,
    settings.steps ?? null,
    defaultFixed(kind, variant.engine),
  );
}
