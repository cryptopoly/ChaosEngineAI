import type { ImageQualityPreset, ImageSamplerId } from "../types";

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

// Sampler choices for SD1.5 / SDXL / SD2. Flow-matching pipelines (FLUX,
// SD3, Qwen-Image, Sana, HiDream) ship locked schedulers and the backend
// ignores this field for them — the dropdown hides via isFlowMatchingRepo.
export const IMAGE_SAMPLERS: Array<{
  id: ImageSamplerId;
  label: string;
  hint: string;
}> = [
  { id: "default", label: "Model default", hint: "Whatever the model shipped with" },
  { id: "dpmpp_2m", label: "DPM++ 2M", hint: "Fast, balanced — 20-30 steps" },
  { id: "dpmpp_2m_karras", label: "DPM++ 2M Karras", hint: "Smoother at low steps" },
  { id: "dpmpp_sde", label: "DPM++ SDE", hint: "Slightly noisier, creative" },
  { id: "euler", label: "Euler", hint: "Classic, reliable at 25+ steps" },
  { id: "euler_a", label: "Euler ancestral", hint: "Creative, non-deterministic" },
  { id: "ddim", label: "DDIM", hint: "Deterministic, slower" },
  { id: "unipc", label: "UniPC", hint: "Fast at low step counts" },
];

const FLOW_MATCHING_TOKENS = ["flux", "stable-diffusion-3", "sd3", "qwen-image", "sana", "hidream"];

export function isFlowMatchingRepo(repo: string | null | undefined): boolean {
  if (!repo) return false;
  const lowered = repo.toLowerCase();
  return FLOW_MATCHING_TOKENS.some((token) => lowered.includes(token));
}
