import type { SystemStats, Recommendation } from "./types/system";
export type {
  TabId,
  SidebarGroupId,
  SidebarMode,
  SystemStats,
  Recommendation,
} from "./types/system";

import type {
  LaunchPreferences,
  LibraryItem,
  ModelDirectorySetting,
  ModelFamily,
} from "./types/models";
export type {
  LaunchPreferences,
  LibraryItem,
  ModelDirectorySetting,
  ModelFamily,
  ModelLaunchMode,
  ModelVariant,
} from "./types/models";

import type { AppSettings } from "./types/settings";
export type {
  AppSettings,
  RemoteProvider,
  SettingsUpdateResponse,
  StrategyInstallLog,
  StrategyInstallLogStep,
  UpdateSettingsPayload,
} from "./types/settings";

import type { ChatMessage, ChatSession } from "./types/chat";
export type {
  ChatMessage,
  ChatMessageVariant,
  ChatPanicSignal,
  ChatReasoningEffort,
  ChatSession,
  ChatStreamPhase,
  ChatThermalWarning,
  ChatThinkingMode,
  CitationInfo,
  CreateSessionResponse,
  GeneratePayload,
  GenerateResponse,
  SamplerOverrides,
  SessionDocument,
  TokenLogprob,
  ToolCallInfo,
  ToolRenderAs,
  UpdateSessionPayload,
} from "./types/chat";

import type { RuntimeStatus, ServerStatus } from "./types/server";
export type {
  LoadModelActionResult,
  LoadedModel,
  ModelCapabilities,
  ModelLoadingState,
  NativeBackendStatus,
  OrphanedWorker,
  RuntimeStatus,
  ServerStatus,
  WarmModel,
} from "./types/server";

/**
 * Phase 3.5: per-turn host telemetry snapshot. Captured at stream
 * finalisation so the values reflect the load the turn generated,
 * not idle baseline. Any field can be null when the underlying
 * sampler is unavailable on this OS.
 */
export interface PerfTelemetry {
  cpuPercent?: number | null;
  gpuPercent?: number | null;
  thermalState?: "nominal" | "moderate" | "critical" | null;
  availableMemoryGb?: number | null;
}

export interface GenerationMetrics {
  finishReason: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  tokS: number;
  responseSeconds?: number | null;
  /** Phase 3.5: host telemetry sampled at turn finalisation. */
  perfTelemetry?: PerfTelemetry | null;
  /**
   * Phase 3.1: DDTree accepted-span overlay data. `acceptedSpans` is
   * a run-length-encoded list over `acceptedTokenText` describing
   * which character ranges came from accepted draft tokens vs
   * verifier-decoded tokens. Only populated when speculative
   * decoding ran (DFLASH path).
   */
  acceptedSpans?: Array<{ start: number; length: number; accepted: boolean }> | null;
  acceptedTokenText?: string | null;
  /** Time-to-first-token in seconds (Phase 2.0). Time from generation start
   * to the moment the model produced its first reasoning or text token.
   * Useful for diagnosing slow prompt-eval phases on long contexts. */
  ttftSeconds?: number | null;
  runtimeNote: string | null;
  dflashAcceptanceRate?: number | null;
  model?: string | null;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  backend?: string | null;
  engineLabel?: string | null;
  cacheLabel?: string | null;
  cacheStrategy?: string | null;
  cacheBits?: number | null;
  fp16Layers?: number | null;
  fusedAttention?: boolean | null;
  fitModelInMemory?: boolean | null;
  requestedCacheLabel?: string | null;
  requestedCacheStrategy?: string | null;
  requestedCacheBits?: number | null;
  requestedFp16Layers?: number | null;
  requestedFitModelInMemory?: boolean | null;
  requestedSpeculativeDecoding?: boolean | null;
  requestedTreeBudget?: number | null;
  speculativeDecoding?: boolean | null;
  dflashDraftModel?: string | null;
  treeBudget?: number | null;
  modelSource?: string | null;
  modelPath?: string | null;
  contextTokens?: number | null;
  generatedAt?: string | null;
}

export interface BenchmarkResult {
  id: string;
  mode?: BenchmarkMode;
  label: string;
  model: string;
  modelRef?: string | null;
  backend: string;
  engineLabel: string;
  source: string;
  measuredAt: string;
  bits: number;
  fp16Layers: number;
  cacheStrategy: string;
  cacheLabel: string;
  cacheGb: number;
  baselineCacheGb: number;
  compression: number;
  tokS: number;
  quality: number;
  responseSeconds: number;
  loadSeconds: number;
  totalSeconds: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  contextTokens: number;
  maxTokens: number;
  notes?: string | null;
  // Perplexity mode
  perplexity?: number | null;
  perplexityStdError?: number | null;
  perplexityDataset?: string | null;
  perplexityNumSamples?: number | null;
  evalTokensPerSecond?: number | null;
  evalSeconds?: number | null;
  // Task accuracy mode
  taskName?: string | null;
  taskAccuracy?: number | null;
  taskCorrect?: number | null;
  taskTotal?: number | null;
  taskNumShots?: number | null;
}

export interface LogEntry {
  ts: string;
  source: string;
  level: string;
  message: string;
}

export interface ActivityItem {
  time: string;
  title: string;
  detail: string;
}

export interface PreviewMetrics {
  bits: number;
  fp16Layers: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  hiddenSize: number;
  contextTokens: number;
  paramsB: number;
  baselineCacheGb: number;
  optimizedCacheGb: number;
  compressionRatio: number;
  estimatedTokS: number;
  speedRatio: number;
  qualityPercent: number;
  diskSizeGb: number;
  summary: string;
}

export interface WorkspaceData {
  system: SystemStats;
  recommendation: Recommendation;
  featuredModels: ModelFamily[];
  library: LibraryItem[];
  libraryStatus?: "scanning" | "ready";
  settings: AppSettings;
  chatSessions: ChatSession[];
  runtime: RuntimeStatus;
  server: ServerStatus;
  benchmarks: BenchmarkResult[];
  logs: LogEntry[];
  activity: ActivityItem[];
  preview: PreviewMetrics;
  quickActions: string[];
}

export interface LoadModelPayload {
  modelRef: string;
  modelName?: string;
  canonicalRepo?: string;
  source?: string;
  backend?: string;
  path?: string;
  cacheBits?: number;
  fp16Layers?: number;
  fusedAttention?: boolean;
  cacheStrategy?: string;
  fitModelInMemory?: boolean;
  contextTokens?: number;
  speculativeDecoding?: boolean;
  /** FU-002: TriAttention MLX kv_budget. Backend defaults to 2048
   * when omitted; only consulted when ``cacheStrategy === "triattention"``. */
  kvBudget?: number;
}


export interface ConvertModelPayload {
  modelRef?: string;
  path?: string;
  hfRepo?: string;
  outputPath?: string;
  quantize?: boolean;
  qBits?: number;
  qGroupSize?: number;
  dtype?: string;
}

export interface ConversionResult {
  sourceRef?: string | null;
  sourcePath?: string | null;
  sourceLabel: string;
  sourceFormat?: string | null;
  sourceSizeGb?: number | null;
  hfRepo: string;
  outputPath: string;
  outputSizeGb?: number | null;
  quantize: boolean;
  qBits: number;
  qGroupSize?: number;
  dtype: string;
  paramsB?: number | null;
  contextWindow?: string | null;
  architecture?: string | null;
  estimatedTokS?: number | null;
  baselineCacheGb?: number | null;
  optimizedCacheGb?: number | null;
  compressionRatio?: number | null;
  qualityPercent?: number | null;
  ggufMetadata?: {
    architecture?: string | null;
    baseModelRepo?: string | null;
    chatTemplate?: string | null;
    contextLength?: number | null;
    name?: string | null;
    quantization?: string | null;
  } | null;
  log: string;
}

export interface ConvertModelResponse {
  conversion: ConversionResult;
  library: LibraryItem[];
  runtime: RuntimeStatus;
}

export type BenchmarkMode = "throughput" | "perplexity" | "task_accuracy";

export interface BenchmarkRunPayload {
  mode?: BenchmarkMode;
  modelRef?: string;
  modelName?: string;
  source?: string;
  backend?: string;
  path?: string;
  label?: string;
  prompt?: string;
  cacheBits: number;
  fp16Layers: number;
  fusedAttention: boolean;
  cacheStrategy: string;
  fitModelInMemory: boolean;
  speculativeDecoding: boolean;
  treeBudget: number;
  /** FU-002: TriAttention MLX kv_budget. Defaults to 2048 server-side. */
  kvBudget: number;
  contextTokens: number;
  maxTokens: number;
  temperature: number;
  // Perplexity mode
  perplexityDataset?: string;
  perplexityNumSamples?: number;
  perplexitySeqLength?: number;
  perplexityBatchSize?: number;
  // Task accuracy mode
  taskName?: string;
  taskLimit?: number;
  taskNumShots?: number;
}

export interface BenchmarkRunResponse {
  result: BenchmarkResult;
  benchmarks: BenchmarkResult[];
  runtime: RuntimeStatus;
}

export interface TauriBackendInfo {
  apiBase: string;
  apiToken?: string | null;
  port: number;
  managedByTauri: boolean;
  processRunning?: boolean;
  started: boolean;
  startupError?: string | null;
  workspaceRoot?: string | null;
  pythonExecutable?: string | null;
  logPath?: string | null;
  launcherMode?: string | null;
}

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
export type VideoCacheStrategyId = "none" | "fbcache" | "teacache";

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

export interface VideoGenerationCachePayload {
  cacheStrategy?: VideoCacheStrategyId | null;
  cacheRelL1Thresh?: number | null;
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

export type { GenerationProgressSnapshot } from "./types/progress";

export type { HubModel, HubFile, HubFileListResponse } from "./types/hub";
