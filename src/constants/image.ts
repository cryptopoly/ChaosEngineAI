import type { ImageQualityPreset } from "../types";

export const IMAGE_RATIO_PRESETS = [
  { id: "square", label: "Square", hint: "1024 x 1024", width: 1024, height: 1024 },
  { id: "portrait", label: "Portrait", hint: "832 x 1216", width: 832, height: 1216 },
  { id: "landscape", label: "Landscape", hint: "1216 x 832", width: 1216, height: 832 },
  { id: "wide", label: "Wide", hint: "1344 x 768", width: 1344, height: 768 },
] as const;

export const IMAGE_QUALITY_PRESETS: Array<{
  id: ImageQualityPreset;
  label: string;
  hint: string;
  steps: number;
  guidance: number;
}> = [
  { id: "fast", label: "Fast", hint: "Quick drafts", steps: 12, guidance: 4.5 },
  { id: "balanced", label: "Balanced", hint: "Best default", steps: 24, guidance: 6 },
  { id: "quality", label: "High Quality", hint: "Slower final pass", steps: 36, guidance: 7 },
];
