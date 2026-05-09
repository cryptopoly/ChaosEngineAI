export type ImageGalleryRuntimeFilter = "all" | "diffusers" | "placeholder" | "warning";
export type ImageGalleryOrientationFilter = "all" | "square" | "portrait" | "landscape";
export type ImageGallerySort = "newest" | "oldest";
export type ImageDiscoverTaskFilter = "all" | "txt2img" | "img2img" | "inpaint";
export type ImageDiscoverAccessFilter = "all" | "open" | "gated";
/** Discover sort axis. ``release`` = most recently released first (prefers
 * the curated releaseDate, falls back to HF createdAt). ``size`` and ``ram``
 * sort largest first using the same metadata that powers the row labels.
 * ``likes`` = HF stars/hearts desc. ``downloads`` = HF downloads desc.
 * Row views also use name/provider/tasks/status for table headers.
 * Variants without the relevant metadata sort to the bottom. */
export type DiscoverSort =
  | "name"
  | "provider"
  | "tasks"
  | "release"
  | "size"
  | "ram"
  | "likes"
  | "downloads"
  | "status";

export type ImageModelTask = "txt2img" | "img2img" | "inpaint";
export type ImageQualityPreset = "fast" | "balanced" | "quality";
export type ImageSamplerId =
  | "default"
  | "dpmpp_2m"
  | "dpmpp_2m_karras"
  | "dpmpp_sde"
  | "euler"
  | "euler_a"
  | "ddim"
  | "unipc"
  // FU-020: Align Your Steps schedules. Wins meaningful detail at
  // 7-10 step counts on SD1.5 / SDXL where Karras / Euler look soft.
  // Flow-match families (FLUX, SD3, Qwen, Sana, HiDream) keep the
  // sampler dropdown hidden — backend ignores the flag for them.
  | "ays_dpmpp_2m_sd15"
  | "ays_dpmpp_2m_sdxl";

// FU-015 + TeaCache. UI-facing strategy id surface — must match the
// keys of ``cache_compression`` in the backend. Default ``"none"`` keeps
// the stock pipeline; ``"fbcache"`` is the cross-platform recommendation
// for DiT pipelines (FLUX, SD3, Wan, Hunyuan, LTX, CogVideoX, Mochi).
export type ImageCacheStrategyId = "none" | "fbcache" | "teacache";

export interface ImageModelVariant {
  id: string;
  familyId: string;
  name: string;
  provider: string;
  repo: string;
  ggufRepo?: string | null;
  ggufFile?: string | null;
  link: string;
  runtime: string;
  styleTags: string[];
  taskSupport: ImageModelTask[];
  sizeGb: number;
  /** Resident peak memory at runtime. Useful when on-disk / quantized
   * transformer size materially understates the full pipeline footprint
   * (for example FLUX GGUF: GGUF covers the transformer only, while T5/CLIP,
   * VAE, and runtime buffers still dominate the Python process). */
  runtimeFootprintGb?: number;
  runtimeFootprintMpsGb?: number;
  runtimeFootprintCudaGb?: number;
  runtimeFootprintCpuGb?: number;
  recommendedResolution: string;
  note: string;
  availableLocally: boolean;
  hasLocalData?: boolean;
  estimatedGenerationSeconds: number | null;
  downloads?: number | null;
  likes?: number | null;
  downloadsLabel?: string | null;
  likesLabel?: string | null;
  lastModified?: string | null;
  updatedLabel?: string | null;
  license?: string | null;
  gated?: boolean;
  pipelineTag?: string | null;
  repoSizeBytes?: number | null;
  repoSizeGb?: number | null;
  coreWeightsBytes?: number | null;
  coreWeightsGb?: number | null;
  onDiskBytes?: number | null;
  onDiskGb?: number | null;
  metadataWarning?: string | null;
  source?: "curated" | "latest" | "experimental";
  familyName?: string | null;
  /** Absolute path to the local HF snapshot, when something is on disk. */
  localPath?: string | null;
  releaseDate?: string | null;
  createdAt?: string | null;
  releaseLabel?: string | null;
}

export interface ImageModelFamily {
  id: string;
  name: string;
  provider: string;
  headline: string;
  summary: string;
  updatedLabel: string;
  badges: string[];
  defaultVariantId: string;
  variants: ImageModelVariant[];
}

export interface ImageCatalogResponse {
  families: ImageModelFamily[];
  latest: ImageModelVariant[];
}

export interface ImageOutputArtifact {
  artifactId: string;
  modelId: string;
  modelName: string;
  prompt: string;
  negativePrompt?: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
  createdAt: string;
  durationSeconds: number;
  previewUrl: string;
  imagePath?: string | null;
  metadataPath?: string | null;
  runtimeLabel?: string | null;
  runtimeNote?: string | null;
  qualityPreset?: ImageQualityPreset | null;
  draftMode?: boolean | null;
}

export interface ImageGenerationPayload {
  modelId: string;
  prompt: string;
  negativePrompt?: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed?: number | null;
  batchSize?: number;
  qualityPreset?: ImageQualityPreset;
  draftMode?: boolean;
  sampler?: ImageSamplerId | null;
  /** FU-015: diffusion cache strategy id ("fbcache" / "teacache" /
   * unset / "none"). Reserved id "none" maps to no header on the
   * payload — the backend treats missing/empty/"none" identically. */
  cacheStrategy?: ImageCacheStrategyId | null;
  /** Threshold knob for caching strategies. Lower = stricter
   * (less speedup, less quality drift). Default unset → strategy
   * default (FBCache 0.12, TeaCache 0.4). */
  cacheRelL1Thresh?: number | null;
  /** FU-021: opt-in CFG decay schedule for flow-match image models
   * (FLUX, SD3, Qwen, Sana, HiDream). Default off — image users
   * typically want consistent CFG. Backend gates non-flow-match
   * repos automatically. */
  cfgDecay?: boolean;
  /** FU-018: TAESD preview-decode VAE swap. Preview-only quality
   * knob — when on, the engine swaps ``pipeline.vae`` for the
   * matching tiny VAE for the duration of the run. Default off. */
  previewVae?: boolean;
  /** FU-024: FP8 layerwise casting (CUDA SM 8.9+ Ada/Hopper/Blackwell).
   * Halves transformer VRAM. No-op on non-CUDA / pre-Ada GPUs. */
  fp8LayerwiseCasting?: boolean;
}

export interface ImageRuntimeStatus {
  activeEngine: string;
  realGenerationAvailable: boolean;
  message: string;
  // Actual device bound to the currently-loaded model (null when nothing
  // is loaded). Populated by ``_detect_device`` after torch import.
  device?: string | null;
  // Best-effort prediction of what the device will be on the next
  // Generate click, computed without importing torch. Useful for
  // surfacing "will use CUDA" BEFORE the user clicks generate.
  expectedDevice?: string | null;
  pythonExecutable?: string | null;
  missingDependencies?: string[];
  loadedModelRepo?: string | null;
  /** Total memory available to the inference device, in GB. Feeds the
   * image-safety heuristic (``assessImageGenerationSafety``) so large
   * models are flagged before a user clicks Generate on a tight machine.
   * Parallel to ``VideoRuntimeStatus.deviceMemoryGb`` — same semantics. */
  deviceMemoryGb?: number | null;
  /** Mirror of ``VideoRuntimeStatus.torchInstallWarning`` -- one-line
   * warning when the torch wheel doesn't match the host accelerator. */
  torchInstallWarning?: string | null;
}

export interface ImageGenerationResponse {
  artifacts: ImageOutputArtifact[];
  outputs: ImageOutputArtifact[];
  runtime?: ImageRuntimeStatus;
}
