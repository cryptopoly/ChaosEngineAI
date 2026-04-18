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
  autoStartServer: boolean;
  launchPreferences: LaunchPreferences;
  remoteProviders?: RemoteProvider[];
  huggingFaceToken?: string | null;
  hasHuggingFaceToken?: boolean;
  dataDirectory?: string;
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

export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  result: string;
  elapsed: number;
}

export interface CitationInfo {
  docId: string;
  docName: string;
  chunkIndex: number;
  page?: number | null;
  preview: string;
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
}

export interface GenerationMetrics {
  finishReason: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  tokS: number;
  responseSeconds?: number | null;
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

export interface UpdateSessionPayload {
  title?: string;
  model?: string | null;
  modelRef?: string | null;
  canonicalRepo?: string | null;
  modelSource?: string | null;
  modelPath?: string | null;
  modelBackend?: string | null;
  thinkingMode?: ChatThinkingMode | null;
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
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
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

export interface ImageModelVariant {
  id: string;
  familyId: string;
  name: string;
  provider: string;
  repo: string;
  link: string;
  runtime: string;
  styleTags: string[];
  taskSupport: ImageModelTask[];
  sizeGb: number;
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
  metadataWarning?: string | null;
  source?: "curated" | "latest" | "experimental";
  familyName?: string | null;
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
  link: string;
  runtime: string;
  styleTags: string[];
  taskSupport: VideoModelTask[];
  sizeGb: number;
  recommendedResolution: string;
  defaultDurationSeconds: number;
  note: string;
  availableLocally: boolean;
  hasLocalData?: boolean;
  estimatedGenerationSeconds: number | null;
  familyName?: string | null;
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
  pythonExecutable?: string | null;
  missingDependencies?: string[];
  loadedModelRepo?: string | null;
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
}

export interface ImageRuntimeStatus {
  activeEngine: string;
  realGenerationAvailable: boolean;
  message: string;
  device?: string | null;
  pythonExecutable?: string | null;
  missingDependencies?: string[];
  loadedModelRepo?: string | null;
}

export interface ImageGenerationResponse {
  artifacts: ImageOutputArtifact[];
  outputs: ImageOutputArtifact[];
  runtime?: ImageRuntimeStatus;
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
  autoStartServer?: boolean;
  launchPreferences?: LaunchPreferences;
  remoteProviders?: Array<{ id: string; label: string; apiBase: string; apiKey: string; model: string }>;
  huggingFaceToken?: string | null;
  dataDirectory?: string | null;
}
