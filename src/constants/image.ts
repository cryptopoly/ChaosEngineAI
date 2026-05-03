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
  // FU-020: Align Your Steps. NVIDIA-published 10-step schedules that
  // preserve more detail than Karras / Euler at low step counts. SD1.5
  // and SDXL each get their own array because the optimal timestep
  // distribution differs between the two models. Flow-match pipelines
  // (FLUX, SD3, Qwen, Sana, HiDream) hide the sampler dropdown
  // entirely via ``isFlowMatchingRepo`` — AYS doesn't apply there.
  {
    id: "ays_dpmpp_2m_sd15",
    label: "AYS DPM++ 2M (SD1.5)",
    hint: "10-step Align Your Steps · pick for SD1.5 only",
  },
  {
    id: "ays_dpmpp_2m_sdxl",
    label: "AYS DPM++ 2M (SDXL)",
    hint: "10-step Align Your Steps · pick for SDXL only",
  },
];

// FU-015 + TeaCache. Diffusion cache strategies the Studios surface to
// the user. ``"none"`` keeps the stock pipeline (default — no
// behavioural change for existing users). ``"fbcache"`` is the
// cross-platform recommendation backed by diffusers 0.36's
// ``apply_first_block_cache`` hook (works on macOS / Windows / Linux,
// any DiT pipeline). ``"teacache"`` is the calibrated TeaCache port
// for FLUX / Hunyuan / LTX / CogVideoX / Mochi.
import type { ImageCacheStrategyId } from "../types";

export const IMAGE_CACHE_STRATEGIES: Array<{
  id: ImageCacheStrategyId;
  label: string;
  hint: string;
}> = [
  {
    id: "none",
    label: "Off",
    hint: "Stock pipeline — no caching",
  },
  {
    id: "fbcache",
    label: "First Block Cache",
    hint: "1.5–2× speedup on DiTs · cross-platform",
  },
  {
    id: "teacache",
    label: "TeaCache",
    hint: "Calibrated for FLUX / Hunyuan / LTX / CogVideoX / Mochi",
  },
];

export const IMAGE_CACHE_STRATEGY_DEFAULT_THRESH: Record<ImageCacheStrategyId, number> = {
  none: 0,
  fbcache: 0.12,
  teacache: 0.4,
};

// Video DiTs are slightly more sensitive to caching drift than image
// DiTs (temporal consistency tightens the budget) so the FBCache
// default is lower for video. TeaCache calibration tables are
// per-model so its threshold default is the same value users see in
// the image side.
export const VIDEO_CACHE_STRATEGY_DEFAULT_THRESH: Record<ImageCacheStrategyId, number> = {
  none: 0,
  fbcache: 0.08,
  teacache: 0.4,
};

const FLOW_MATCHING_TOKENS = ["flux", "stable-diffusion-3", "sd3", "qwen-image", "sana", "hidream"];

export function isFlowMatchingRepo(repo: string | null | undefined): boolean {
  if (!repo) return false;
  const lowered = repo.toLowerCase();
  return FLOW_MATCHING_TOKENS.some((token) => lowered.includes(token));
}

// FU-015: image cache strategy gates. Mirrors the video-side filter
// added to VideoStudioTab — keeps the dropdown honest about what the
// backend will actually apply.
//
//   - FLUX family (FLUX.1 / FLUX.2 / FLUX.2-Klein / FLUX.2-Turbo /
//     community FLUX fine-tunes): both First Block Cache and TeaCache
//     apply. TeaCache's vendored forward
//     (``cache_compression/_teacache_patches/flux.py``) is calibrated
//     against the upstream FLUX FluxTransformer2DModel.
//   - Other DiT pipelines (SD3.5, Qwen-Image, Sana, HiDream, Z-Image,
//     FLUX.2 community variants, ERNIE-Image, GLM-Image, Nucleus-Image):
//     First Block Cache applies via the diffusers 0.36 generic hook.
//     TeaCache patches don't cover these pipelines yet — hide it from
//     the dropdown so users don't pick a strategy the backend will
//     swallow with a runtimeNote.
//   - UNet-based pipelines (SDXL base / refiner, SD1.5, SD2): neither
//     strategy applies because both attach to ``pipeline.transformer``
//     which UNets don't have. Hide both rows; backend gracefully
//     no-ops with a runtimeNote anyway.
const FLUX_FAMILY_TOKENS = ["flux"];
const UNET_IMAGE_TOKENS = [
  "stable-diffusion-xl",
  "sdxl",
  "sd_xl",
  "stable-diffusion-v1-5",
  "stable-diffusion-1-5",
  "sd-1-5",
  "sd_1_5",
  "stable-diffusion-2",
  "sd-2-",
];

export function isFluxFamilyRepo(repo: string | null | undefined): boolean {
  if (!repo) return false;
  const lowered = repo.toLowerCase();
  return FLUX_FAMILY_TOKENS.some((token) => lowered.includes(token));
}

export function isUnetImageRepo(repo: string | null | undefined): boolean {
  if (!repo) return false;
  const lowered = repo.toLowerCase();
  return UNET_IMAGE_TOKENS.some((token) => lowered.includes(token));
}

/** Return the image cache strategies that actually apply to this repo.
 *
 * UNet pipelines get only the "Off" entry; the dropdown is effectively
 * disabled. FLUX family pipelines get all three. Every other DiT
 * pipeline gets Off + First Block Cache only — TeaCache calibration
 * exists for FLUX only on the image side. */
export function imageCacheStrategiesForRepo(
  repo: string | null | undefined,
): typeof IMAGE_CACHE_STRATEGIES {
  if (isUnetImageRepo(repo)) {
    return IMAGE_CACHE_STRATEGIES.filter((s) => s.id === "none");
  }
  if (isFluxFamilyRepo(repo)) {
    return IMAGE_CACHE_STRATEGIES;
  }
  return IMAGE_CACHE_STRATEGIES.filter((s) => s.id !== "teacache");
}
