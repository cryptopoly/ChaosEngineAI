export interface ModelLoadingState {
  modelName: string;
  modelRef?: string;
  stage: string;
  elapsedSeconds: number;
  progress: number | null;
  progressPercent?: number | null;
  progressPhase?: string | null;
  progressMessage?: string | null;
  recentLogLines?: string[];
}

export interface OrphanedWorker {
  pid: number;
  kind: "mlx_worker" | "llama_server";
  label: string;
  action: string;
  detectedAt: string;
}

export interface ServerStatus {
  status: "running" | "idle";
  baseUrl: string;
  localhostUrl?: string;
  lanUrls?: string[];
  bindHost?: string;
  remoteAccessActive?: boolean;
  port: number;
  activeConnections: number;
  concurrentRequests: number;
  requestsServed: number;
  loadedModelName: string | null;
  loading: ModelLoadingState | null;
  recentOrphanedWorkers?: OrphanedWorker[];
  logTail: string[];
}

/**
 * Phase 2.11: typed capability declarations for the loaded model.
 *
 * Resolved by the backend from the curated catalog (with a heuristic
 * fallback for non-catalog models). The frontend uses these to gate
 * composer affordances — image attach hides when !supportsVision, the
 * Tools toggle hides when !supportsTools, etc. — and to render capability
 * badges next to the model picker.
 */
export interface ModelCapabilities {
  supportsVision: boolean;
  supportsTools: boolean;
  supportsReasoning: boolean;
  supportsCoding: boolean;
  supportsAgents: boolean;
  supportsAudio: boolean;
  supportsVideo: boolean;
  supportsMultilingual: boolean;
  /** Free-form tags from the catalog ("reasoning", "vision", etc.). */
  tags: string[];
}

export interface LoadedModel {
  ref: string;
  name: string;
  canonicalRepo?: string | null;
  backend: string;
  source: string;
  engine: string;
  cacheBits: number;
  fp16Layers: number;
  fusedAttention: boolean;
  cacheStrategy: string;
  fitModelInMemory: boolean;
  contextTokens: number;
  loadedAt: string;
  path: string | null;
  runtimeTarget?: string | null;
  runtimeNote: string | null;
  speculativeDecoding: boolean;
  dflashDraftModel?: string | null;
  treeBudget: number;
  /** Phase 2.11: capability declarations (vision / tools / reasoning / etc.) */
  capabilities?: ModelCapabilities | null;
}

export interface WarmModel {
  ref: string;
  name: string;
  engine: string;
  warm: boolean;
  active: boolean;
}

export interface RuntimeStatus {
  state: "idle" | "loaded";
  engine: string;
  engineLabel: string;
  loadedModel: LoadedModel | null;
  warmModels?: WarmModel[];
  supportsGeneration: boolean;
  serverReady: boolean;
  activeRequests: number;
  requestsServed: number;
  runtimeNote: string | null;
  nativeBackends?: NativeBackendStatus;
}

export interface LoadModelActionResult {
  ok: boolean;
  runtime?: RuntimeStatus;
  error?: string;
}

export interface NativeBackendStatus {
  pythonExecutable: string;
  mlxAvailable: boolean;
  mlxLmAvailable: boolean;
  mlxUsable: boolean;
  mlxVersion?: string | null;
  mlxLmVersion?: string | null;
  mlxMessage?: string | null;
  ggufAvailable: boolean;
  llamaCliPath?: string | null;
  llamaServerPath?: string | null;
  llamaServerTurboPath?: string | null;
  converterAvailable: boolean;
  probing?: boolean;
}
