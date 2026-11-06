/**
 * Standard size terms for the Studios: aspect ratios and megapixels for images, named resolutions
 * (HD, Full HD, 4K ...) for video. Each button resolves to a width and height; a button that would
 * go past the model's own limits is reported as unavailable so the Studio can grey it out.
 */

export interface Aspect {
  id: string;
  label: string;
  w: number;
  h: number;
}

export interface SizeLimit {
  maxSide?: number | null;
  maxPixels?: number | null;
  minSide?: number | null;
}

export const IMAGE_ASPECTS: Aspect[] = [
  { id: "1:1", label: "1:1", w: 1, h: 1 },
  { id: "3:2", label: "3:2", w: 3, h: 2 },
  { id: "2:3", label: "2:3", w: 2, h: 3 },
  { id: "4:3", label: "4:3", w: 4, h: 3 },
  { id: "3:4", label: "3:4", w: 3, h: 4 },
  { id: "16:9", label: "16:9", w: 16, h: 9 },
  { id: "9:16", label: "9:16", w: 9, h: 16 },
  { id: "21:9", label: "21:9", w: 21, h: 9 },
];

export const IMAGE_MEGAPIXELS: Array<{ id: string; label: string; mp: number }> = [
  { id: "0.5", label: "0.5 MP", mp: 0.5 },
  { id: "1", label: "1 MP", mp: 1 },
  { id: "2", label: "2 MP", mp: 2 },
  { id: "4", label: "4 MP", mp: 4 },
  { id: "8", label: "8 MP", mp: 8 },
];

/** Video resolutions by the short edge, named the way people ask for them. */
export const VIDEO_RESOLUTIONS: Array<{ id: string; label: string; sub: string; shortEdge: number }> = [
  { id: "sd", label: "SD", sub: "480p", shortEdge: 480 },
  { id: "hd", label: "HD", sub: "720p", shortEdge: 720 },
  { id: "fhd", label: "Full HD", sub: "1080p", shortEdge: 1080 },
  { id: "qhd", label: "2K", sub: "1440p", shortEdge: 1440 },
  { id: "uhd", label: "4K", sub: "2160p", shortEdge: 2160 },
  { id: "8k", label: "8K", sub: "4320p", shortEdge: 4320 },
];

export const VIDEO_ASPECTS: Aspect[] = [
  { id: "16:9", label: "16:9", w: 16, h: 9 },
  { id: "9:16", label: "9:16", w: 9, h: 16 },
  { id: "1:1", label: "1:1", w: 1, h: 1 },
  { id: "4:3", label: "4:3", w: 4, h: 3 },
  { id: "21:9", label: "21:9", w: 21, h: 9 },
];

const STEP = 64;

function snap(value: number, step = STEP): number {
  return Math.max(step, Math.round(value / step) * step);
}

/** Width and height for an aspect at a pixel count, snapped to multiples of 64. */
export function sizeForPixels(aspect: Aspect, pixels: number): { width: number; height: number } {
  const ratio = aspect.w / aspect.h;
  const width = snap(Math.sqrt(pixels * ratio));
  const height = snap(width / ratio);
  return { width, height };
}

/** Width and height for an aspect when the short edge is ``shortEdge`` pixels. */
export function sizeForShortEdge(aspect: Aspect, shortEdge: number): { width: number; height: number } {
  const portraitOrSquare = aspect.h >= aspect.w;
  const ratio = aspect.w / aspect.h;
  if (portraitOrSquare) {
    const width = snap(shortEdge);
    return { width, height: snap(width / ratio) };
  }
  const height = snap(shortEdge);
  return { width: snap(height * ratio), height };
}

export function fitsLimit(size: { width: number; height: number }, limit: SizeLimit): boolean {
  const maxSide = limit.maxSide && limit.maxSide > 0 ? limit.maxSide : 4096;
  if (Math.max(size.width, size.height) > maxSide) return false;
  if (limit.maxPixels && limit.maxPixels > 0 && size.width * size.height > limit.maxPixels) return false;
  if (limit.minSide && Math.min(size.width, size.height) < limit.minSide) return false;
  return true;
}

/** The aspect closest to a width and height (by ratio), or null when none is within 4 %. */
export function nearestAspect(width: number, height: number, aspects: Aspect[]): Aspect | null {
  if (!width || !height) return null;
  const ratio = width / height;
  let best: Aspect | null = null;
  let bestGap = Infinity;
  for (const aspect of aspects) {
    const gap = Math.abs(Math.log(ratio / (aspect.w / aspect.h)));
    if (gap < bestGap) {
      best = aspect;
      bestGap = gap;
    }
  }
  return bestGap <= 0.04 ? best : null;
}

/** The megapixel option closest to the current size (within 25 %), or null. */
export function nearestMegapixels(width: number, height: number): (typeof IMAGE_MEGAPIXELS)[number] | null {
  const mp = (width * height) / 1_000_000;
  let best: (typeof IMAGE_MEGAPIXELS)[number] | null = null;
  let bestGap = Infinity;
  for (const option of IMAGE_MEGAPIXELS) {
    const gap = Math.abs(Math.log(mp / option.mp));
    if (gap < bestGap) {
      best = option;
      bestGap = gap;
    }
  }
  return bestGap <= 0.25 ? best : null;
}

/** The video resolution whose short edge matches the current size, or null. */
export function nearestVideoResolution(width: number, height: number): (typeof VIDEO_RESOLUTIONS)[number] | null {
  const shortEdge = Math.min(width, height);
  return VIDEO_RESOLUTIONS.find((item) => Math.abs(item.shortEdge - shortEdge) <= 32) ?? null;
}
