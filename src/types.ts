export type TabId =
  | "dashboard"
  | "online-models"
  | "my-models"
  | "image-discover"
  | "image-models"
  | "image-studio"
  | "image-gallery"
  | "video-models"
  | "video-discover"
  | "video-studio"
  | "video-gallery"
  | "conversion"
  | "chat"
  | "server"
  | "benchmarks"
  | "benchmark-history"
  | "finetuning"
  | "prompt-library"
  | "plugins"
  | "logs"
  | "settings";

export type SidebarGroupId =
  | "models"
  | "images"
  | "video"
  | "benchmarks"
  | "tools";

export type SidebarMode = "collapsible" | "tabs";

export interface SystemStats {
  platform: string;
  arch: string;
  hardwareSummary: string;
  backendLabel: string;
  appVersion: string;
  availableCacheStrategies: Array<{
    id: string;
    name: string;
    available: boolean;
    bitRange: number[] | null;
    defaultBits: number | null;
    supportsFp16Layers: boolean;
    availabilityBadge?: string | null;
    availabilityTone?: string | null;
    availabilityReason?: string | null;
    requiredLlamaBinary?: string | null;
    appliesTo?: string[];
  }>;
  llamaServerTurboPath?: string | null;
  mlxAvailable: boolean;
  mlxLmAvailable: boolean;
  totalMemoryGb: number;
  availableMemoryGb: number;
  usedMemoryGb: number;
  swapUsedGb: number;
  cpuUtilizationPercent: number;
  gpuUtilizationPercent: number | null;
  spareHeadroomGb: number;
  dflash?: {
    available: boolean;
    mlxAvailable: boolean;
    vllmAvailable: boolean;
    ddtreeAvailable?: boolean;
    supportedModels: string[];
  };
  runningLlmProcesses: Array<{
    pid: number;
    name: string;
    owner?: string;
    modelName?: string | null;
    modelStatus?: "active" | "warm" | null;
    kind?: "mlx_worker" | "llama_server" | "backend" | "other";
    memoryGb: number;
    cpuPercent: number;
  }>;
  compressedMemoryGb?: number;
  memoryPressurePercent?: number;
  swapTotalGb?: number;
  diskFreeGb?: number;
  diskUsedGb?: number;
  diskTotalGb?: number;
  diskPath?: string;
  battery?: {
    percent: number;
    powerSource: "AC" | "Battery";
    charging: boolean;
  } | null;
  uptimeMinutes: number;
}

export interface Recommendation {
  title: string;
  detail: string;
  targetModel: string;
  cacheLabel: string;
  headroomPercent: number;
}

export type ModelLaunchMode = "direct" | "convert";

export interface ModelVariant {
  id: string;
  familyId: string;
  name: string;
  repo: string;
  ggufRepo?: string | null;
  ggufFile?: string | null;
  link: string;
  paramsB: number;
  sizeGb: number;
  format: string;
  quantization: string;
  capabilities: string[];
  note: string;
  contextWindow: string;
  estimatedMemoryGb: number | null;
  estimatedCompressedMemoryGb: number | null;
  availableLocally: boolean;
  launchMode: ModelLaunchMode;
  backend: "mlx" | "llama.cpp" | "auto";
  maxContext?: number | null;
  releaseDate?: string | null;
  releaseLabel?: string | null;
}

export interface ModelFamily {
  id: string;
  name: string;
  provider: string;
  headline: string;
  summary: string;
  description: string;
  updatedLabel: string;
  popularityLabel: string;
  likesLabel: string;
  badges: string[];
  capabilities: string[];
  defaultVariantId: string;
  variants: ModelVariant[];
  readme: string[];
}

export interface LibraryItem {
  name: string;
  path: string;
  format: string;
  sourceKind?: string | null;
  quantization?: string | null;
  backend?: string | null;
  modelType?: string | null;
  sizeGb: number;
  lastModified: string;
  actions: string[];
  directoryId?: string;
  directoryLabel?: string;
  directoryPath?: string;
  maxContext?: number | null;
  broken?: boolean;
  brokenReason?: string | null;
}

export interface ModelDirectorySetting {
  id: string;
  label: string;
  path: string;
  enabled: boolean;
  source: "default" | "user";
  exists?: boolean;
  modelCount?: number;
}

export interface LaunchPreferences {
  contextTokens: number;
  maxTokens: number;
  temperature: number;
  cacheBits: number;
  fp16Layers: number;
  fusedAttention: boolean;
  cacheStrategy: string;
  fitModelInMemory: boolean;
  speculativeDecoding: boolean;
  treeBudget: number;
}

export interface StrategyInstallLogStep {
  id: string;
  label: string;
  command: string;
  status: "running" | "success" | "failed";
  output: string;
}

export interface StrategyInstallLog {
  strategyId: string;
  label: string;
  status: "running" | "success" | "failed";
  startedAt: string;
  finishedAt?: string | null;
  steps: StrategyInstallLogStep[];
}

export interface RemoteProvider {
  id: string;
  label: string;
  apiBase: string;
  model: string;
  hasApiKey?: boolean;
  apiKeyMasked?: string;
  apiKey?: string;
}

export interface AppSettings {
  modelDirectories: ModelDirectorySetting[];
  preferredServerPort: number;
  allowRemoteConnections: boolean;
  // When false, the backend disables bearer-token enforcement so external
  // clients (OpenWebUI, curl, another desktop app) can hit /api and /v1
  // endpoints without a token. Default true.
  requireApiAuth: boolean;
  autoStartServer: boolean;
  launchPreferences: LaunchPreferences;
  remoteProviders?: RemoteProvider[];
  huggingFaceToken?: string | null;
  hasHuggingFaceToken?: boolean;
  dataDirectory?: string;
  // Empty string means "use the default under dataDirectory". A non-empty
  // value redirects new image / video artifacts to a custom folder (e.g. an
  // external SSD or a cloud-synced delivery folder).
  imageOutputsDirectory?: string;
  videoOutputsDirectory?: string;
}

export interface SettingsUpdateResponse {
  settings: AppSettings;
  restartRequired?: boolean;
  migrationSummary?: {
    copied: string[];
    skipped: string[];
    from: string;
    to: string;
  };
}

/**
 * Phase 2.8: rendering hint for tool-call output. Tools that opt in
 * to structured output set this on their result so the UI knows
 * whether to render a table, code block, markdown body, image, or a
 * chart. Tools that don't override `execute_structured` send `null`
 * and the frontend falls back to the legacy collapsible-JSON view.
 */
export type ToolRenderAs = "table" | "code" | "markdown" | "image" | "chart" | "json";

export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  elapsed: number;
  /** Phase 2.8: rendering hint. Null/undefined → JSON fallback. */
  renderAs?: ToolRenderAs | null;
  /** Phase 2.8: structured payload matching the renderAs shape. */
  data?: Record<string, unknown> | null;
}

export interface CitationInfo {
  docId: string;
  docName: string;
  chunkIndex: number;
  page?: number | null;
  preview: string;
}

export type ChatStreamPhase = "prompt_eval" | "generating";

export interface ChatPanicSignal {
  /** User-visible panic message from the backend. */
  message: string;
  /** Available RAM (GB) sampled at panic emission. */
  availableGb?: number;
  /** Combined memory pressure percentage at panic emission. */
  pressurePercent?: number;
}

export interface ChatThermalWarning {
  /** Reported thermal state from backend ("moderate" | "critical"). */
  state: "moderate" | "critical";
  /** User-visible thermal message from backend. */
  message: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  images?: string[];
  reasoning?: string;
  reasoningDone?: boolean | null;
  metrics?: GenerationMetrics | null;
  toolCalls?: ToolCallInfo[];
  citations?: CitationInfo[];
  /**
   * Live phase tracker for the streaming assistant message (Phase 2.0).
   * Cleared once the message finalises via the backend's done event. Used
   * to render an explicit prompt-processing indicator before the first
   * token arrives instead of a blank flashing cursor.
   */
  streamPhase?: ChatStreamPhase | null;
  /**
   * Phase 2.0.5-G: panic signal emitted mid-stream when system memory
   * crosses critical thresholds. Renders a non-blocking warning banner
   * so the user can decide whether to cancel before the host wedges.
   */
  panic?: ChatPanicSignal | null;
  /**
   * Phase 2.0.5-I: thermal pressure warning emitted mid-stream when
   * the host is throttling. Renders a non-blocking warning banner.
   */
  thermalWarning?: ChatThermalWarning | null;
}

export interface SessionDocument {
  id: string;
  filename: string;
  originalName: string;
  sizeBytes: number;
  chunkCount: number;
  uploadedAt: string;
}

export interface ChatSession {
  id: string;
  title: string;
  documents?: SessionDocument[];
  updatedAt: string;
  pinned?: boolean;
  model: string;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  modelSource?: string | null;
  modelPath?: string | null;
  modelBackend?: string | null;
  thinkingMode?: ChatThinkingMode | null;
  reasoningEffort?: ChatReasoningEffort | null;
  cacheLabel: string;
  cacheStrategy?: string | null;
  cacheBits?: number | null;
  fp16Layers?: number | null;
  fusedAttention?: boolean | null;
  fitModelInMemory?: boolean | null;
  contextTokens?: number | null;
  speculativeDecoding?: boolean | null;
  dflashDraftModel?: string | null;
  treeBudget?: number | null;
  /**
   * Phase 2.4: when this session was forked from another, the source
   * session's id. Sidebar reads this to render a fork-relationship
   * hint and the future merge/diff features key off it.
   */
  parentSessionId?: string | null;
  /** Phase 2.4: index of the last message copied from the parent. */
  forkedAtMessageIndex?: number | null;
  messages: ChatMessage[];
}

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

export interface GenerationMetrics {
  finishReason: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  tokS: number;
  responseSeconds?: number | null;
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
}

export interface CreateSessionResponse {
  session: ChatSession;
}

export type ChatThinkingMode = "off" | "auto";
export type ChatReasoningEffort = "low" | "medium" | "high";

export interface UpdateSessionPayload {
  title?: string;
  model?: string | null;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  modelSource?: string | null;
  modelPath?: string | null;
  modelBackend?: string | null;
  thinkingMode?: ChatThinkingMode | null;
  reasoningEffort?: ChatReasoningEffort | null;
  pinned?: boolean | null;
  cacheStrategy?: string | null;
  cacheBits?: number | null;
  fp16Layers?: number | null;
  fusedAttention?: boolean | null;
  fitModelInMemory?: boolean | null;
  contextTokens?: number | null;
  speculativeDecoding?: boolean | null;
  dflashDraftModel?: string | null;
  treeBudget?: number | null;
  messages?: ChatMessage[];
}

export interface GeneratePayload {
  sessionId?: string;
  title?: string;
  prompt: string;
  images?: string[];
  modelRef?: string;
  modelName?: string;
  canonicalRepo?: string;
  source?: string;
  path?: string;
  backend?: string;
  thinkingMode?: ChatThinkingMode;
  reasoningEffort?: ChatReasoningEffort;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  // Phase 2.2: full sampler chain. None means "use backend default".
  // llama-server applies all of these natively; mlx-lm uses what its
  // make_sampler signature supports (top_p, top_k, min_p) and silently
  // ignores the rest.
  topP?: number;
  topK?: number;
  minP?: number;
  repeatPenalty?: number;
  seed?: number;
  mirostatMode?: 0 | 1 | 2;
  mirostatTau?: number;
  mirostatEta?: number;
  jsonSchema?: Record<string, unknown>;
  cacheBits?: number;
  fp16Layers?: number;
  fusedAttention?: boolean;
  cacheStrategy?: string;
  fitModelInMemory?: boolean;
  contextTokens?: number;
  speculativeDecoding?: boolean;
  // Agent tool-use
  enableTools?: boolean;
  availableTools?: string[];
  /**
   * Phase 2.12: when true, the model selectors in this payload override
   * the loaded runtime for THIS turn only — the session's stored
   * `modelRef` / `model` / `modelSource` etc. are not updated, so the
   * thread reverts to its default model on the next plain message.
   */
  oneTurnOverride?: boolean;
}

/**
 * Phase 2.2: per-thread sampler override blob. Stored in localStorage
 * keyed by session id. useChat reads it when assembling stream payloads;
 * the SamplerPanel writes it back when the user adjusts a slider.
 */
export interface SamplerOverrides {
  topP?: number | null;
  topK?: number | null;
  minP?: number | null;
  repeatPenalty?: number | null;
  seed?: number | null;
  mirostatMode?: 0 | 1 | 2 | null;
  mirostatTau?: number | null;
  mirostatEta?: number | null;
}

export interface GenerateResponse {
  session: ChatSession;
  assistant: ChatMessage;
  runtime: RuntimeStatus;
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
  | "unipc";

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
}

export interface ImageGenerationResponse {
  artifacts: ImageOutputArtifact[];
  outputs: ImageOutputArtifact[];
  runtime?: ImageRuntimeStatus;
}

/**
 * Live snapshot of the in-flight image / video generation as published by the
 * backend ProgressTracker. ``active=false`` means nothing is running (or the
 * runtime hasn't published a phase yet) and the UI should fall back to its
 * client-side estimates.
 */
export interface GenerationProgressSnapshot {
  kind: "image" | "video";
  active: boolean;
  phase: "idle" | "loading" | "encoding" | "diffusing" | "decoding" | "saving";
  message: string;
  step: number;
  totalSteps: number;
  startedAt: number;
  updatedAt: number;
  elapsedSeconds: number;
  runLabel: string | null;
}

export interface HubModel {
  id: string;
  repo: string;
  name: string;
  provider: string;
  link: string;
  format: string;
  tags: string[];
  downloads: number;
  likes: number;
  downloadsLabel: string;
  likesLabel: string;
  lastModified?: string | null;
  updatedLabel?: string | null;
  createdAt?: string | null;
  releaseLabel?: string | null;
  availableLocally: boolean;
  launchMode: string;
  backend: string;
}

export interface HubFile {
  path: string;
  sizeBytes: number;
  sizeGb: number;
  kind: "weight" | "vision_projector" | "config" | "tokenizer" | "readme" | "template" | "other";
}

export interface HubFileListResponse {
  repo: string;
  files: HubFile[];
  totalSizeBytes: number;
  totalSizeGb: number;
  license: string | null;
  tags: string[];
  pipelineTag: string | null;
  lastModified: string | null;
  warning?: string | null;
}

export interface UpdateSettingsPayload {
  modelDirectories?: ModelDirectorySetting[];
  preferredServerPort?: number;
  allowRemoteConnections?: boolean;
  requireApiAuth?: boolean;
  autoStartServer?: boolean;
  launchPreferences?: LaunchPreferences;
  remoteProviders?: Array<{ id: string; label: string; apiBase: string; apiKey: string; model: string }>;
  huggingFaceToken?: string | null;
  dataDirectory?: string | null;
  imageOutputsDirectory?: string | null;
  videoOutputsDirectory?: string | null;
}
