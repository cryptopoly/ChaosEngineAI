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

export type {
  ImageCacheStrategyId,
  ImageCatalogResponse,
  ImageGenerationPayload,
  ImageGenerationResponse,
  ImageModelFamily,
  ImageModelTask,
  ImageModelVariant,
  ImageOutputArtifact,
  ImageQualityPreset,
  ImageRuntimeStatus,
  ImageSamplerId,
} from "./types/image";

export type {
  VideoCacheStrategyId,
  VideoCatalogResponse,
  VideoGenerationCachePayload,
  VideoGenerationPayload,
  VideoGenerationResponse,
  VideoModelFamily,
  VideoModelTask,
  VideoModelVariant,
  VideoOutputArtifact,
  VideoRuntimeStatus,
} from "./types/video";


export type { GenerationProgressSnapshot } from "./types/progress";

export type { HubModel, HubFile, HubFileListResponse } from "./types/hub";
