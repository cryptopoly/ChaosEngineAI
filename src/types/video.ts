export type VideoDiscoverTaskFilter = "all" | "txt2video" | "img2video" | "video2video";

// 2026-05-16 (FU-026 follow-up): taylorseer + pab exposed in the UI
// alongside the original fbcache + teacache. magcache + fastercache
// stay backend-only (CLI / API) until calibration / differentiation
// stories land.
export type VideoCacheStrategyId =
  | "none"
  | "fbcache"
  | "teacache"
  | "taylorseer"
  | "pab";

export type VideoModelTask = "txt2video" | "img2video" | "video2video";

export interface VideoModelVariant {
  id: string;
  familyId: string;
  name: string;
  provider: string;
  repo: string;
  ggufRepo?: string | null;
  ggufFile?: string | null;
  textEncoderRepo?: string | null;
  link: string;
  runtime: string;
  styleTags: string[];
  taskSupport: VideoModelTask[];
  sizeGb: number;
  /** Resident peak memory at runtime (transformer + text encoder + VAE
   * during the heaviest phase, typically text encoding). When present, the
   * safety estimator uses this instead of multiplying ``sizeGb`` by a fudge
   * factor — disk size routinely overstates resident because of duplicate
   * sharded safetensors and tokenizer caches. ``undefined`` falls back to
   * the legacy ``sizeGb × 1.4`` heuristic. */
  runtimeFootprintGb?: number;
  runtimeFootprintMpsGb?: number;
  runtimeFootprintCudaGb?: number;
  runtimeFootprintCpuGb?: number;
  recommendedResolution: string;
  defaultDurationSeconds: number;
  note: string;
  availableLocally: boolean;
  hasLocalData?: boolean;
  localDataRepos?: string[];
  primaryLocalRepo?: string | null;
  localStatusReason?: string | null;
  estimatedGenerationSeconds: number | null;
  onDiskBytes?: number | null;
  onDiskGb?: number | null;
  familyName?: string | null;
  /** Absolute path to the local HF snapshot, when something is on disk. */
  localPath?: string | null;
  releaseDate?: string | null;
  releaseLabel?: string | null;
  /** Live Hugging Face metadata fetched by the backend in parallel when the
   * catalog loads. All optional — repos whose fetch times out at probe time
   * render without these fields rather than blocking the page. */
  downloads?: number | null;
  likes?: number | null;
  downloadsLabel?: string | null;
  likesLabel?: string | null;
  lastModified?: string | null;
  updatedLabel?: string | null;
  createdAt?: string | null;
  pipelineTag?: string | null;
  license?: string | null;
  /** Total HF repo size summed from all siblings — the worst-case download
   * if allow_patterns doesn't filter out auxiliary checkpoints. Bigger than
   * ``coreWeightsBytes`` when the repo ships legacy / non-diffusers blobs
   * alongside the diffusers tree. */
  repoSizeBytes?: number | null;
  repoSizeGb?: number | null;
  /** Size of just the model weight files (safetensors / bin / gguf).
   * Closer to what the diffusers allow-pattern download actually pulls. */
  coreWeightsBytes?: number | null;
  coreWeightsGb?: number | null;
  /** Optional Fast-preview swap target. When set, the Studio shows a
   * Fast preview toggle that submits this sibling's variant id instead
   * — typically pointing a "dev" variant at its "distilled" sibling so
   * the same prompt + seed renders in a fraction of the time. */
  fastPreviewSiblingId?: string | null;
}

export interface VideoModelFamily {
  id: string;
  name: string;
  provider: string;
  headline: string;
  summary: string;
  updatedLabel: string;
  badges: string[];
  defaultVariantId: string;
  variants: VideoModelVariant[];
}

export interface VideoCatalogResponse {
  families: VideoModelFamily[];
  latest: VideoModelVariant[];
}

export interface VideoRuntimeStatus {
  activeEngine: string;
  realGenerationAvailable: boolean;
  message: string;
  device?: string | null;
  /** Predicted device for the next Generate click, computed without
   * importing torch. Lets the UI show "Device: cuda (expected)" before
   * any model has been loaded. Mirrors ImageRuntimeStatus.expectedDevice. */
  expectedDevice?: string | null;
  pythonExecutable?: string | null;
  missingDependencies?: string[];
  loadedModelRepo?: string | null;
  /** Total device memory in GB — used by the video-gen safety heuristic to
   * scale attention-budget thresholds per hardware capability. Nullable
   * because detection can fail (unsupported platform, nvidia-smi absent on a
   * non-CUDA Linux box, etc.); consumers treat null as "stay conservative". */
  deviceMemoryGb?: number | null;
  /** One-line warning when the installed torch wheel doesn't match the host
   * accelerator (e.g. +cpu wheel on a CUDA host -- generation silently
   * falls back to CPU). Computed without importing torch by reading the
   * dist-info METADATA. Frontend renders this as a loud red chip in the
   * Studio so users don't see "Real engine ready" while their NVIDIA GPU
   * sits idle. ``null`` when everything looks fine. */
  torchInstallWarning?: string | null;
}

export interface VideoOutputArtifact {
  artifactId: string;
  modelId: string;
  modelName: string;
  prompt: string;
  negativePrompt?: string | null;
  width: number;
  height: number;
  numFrames: number;
  fps: number;
  steps: number;
  guidance: number;
  seed: number;
  createdAt: string;
  durationSeconds: number;
  clipDurationSeconds: number;
  videoPath?: string | null;
  metadataPath?: string | null;
  videoMimeType?: string | null;
  videoExtension?: string | null;
  runtimeLabel?: string | null;
  runtimeNote?: string | null;
}

export interface VideoGenerationPayload {
  modelId: string;
  prompt: string;
  negativePrompt?: string;
  width: number;
  height: number;
  numFrames: number;
  fps: number;
  steps: number;
  guidance: number;
  seed?: number | null;
  useNf4?: boolean;
  enableLtxRefiner?: boolean;
  enhancePrompt?: boolean;
  cfgDecay?: boolean;
  stgScale?: number;
  /** FU-018: TAESD/TAEHV preview-decode VAE swap. Preview-only
   * quality knob; default off (video users typically want full
   * fidelity). */
  previewVae?: boolean;
  /** FU-024: FP8 layerwise casting (CUDA SM 8.9+ Ada/Hopper/Blackwell).
   * Halves transformer VRAM. No-op on Apple Silicon / CPU / pre-Ada. */
  fp8LayerwiseCasting?: boolean;
  /** FU-015: cache strategy id ("fbcache" / "teacache" / "none"). */
  cacheStrategy?: VideoCacheStrategyId | null;
  /** Optional caching threshold override; null uses strategy default. */
  cacheRelL1Thresh?: number | null;
}

export interface VideoGenerationResponse {
  artifact: VideoOutputArtifact;
  outputs: VideoOutputArtifact[];
  runtime?: VideoRuntimeStatus;
}

export interface VideoGenerationCachePayload {
  cacheStrategy?: VideoCacheStrategyId | null;
  cacheRelL1Thresh?: number | null;
}
