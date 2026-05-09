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
import type {
  BenchmarkResult,
} from "./types/benchmarks";
export type {
  BenchmarkMode,
  BenchmarkResult,
  BenchmarkRunPayload,
  BenchmarkRunResponse,
  GenerationMetrics,
  PerfTelemetry,
} from "./types/benchmarks";

import type {
  ActivityItem,
  LogEntry,
  PreviewMetrics,
} from "./types/observability";
export type {
  ActivityItem,
  LogEntry,
  PreviewMetrics,
} from "./types/observability";

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
