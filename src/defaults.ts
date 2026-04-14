/**
 * Empty / zero-value defaults for React hook initial state.
 *
 * These are NOT mock data — they are the honest "nothing loaded yet" values
 * that hooks hold until the first successful backend response arrives.
 * Production code should never import from mockData.ts (test-only).
 */

import type {
  AppSettings,
  LaunchPreferences,
  PreviewMetrics,
  WorkspaceData,
} from "./types";

export const emptyLaunchPreferences: LaunchPreferences = {
  contextTokens: 8192,
  maxTokens: 512,
  temperature: 0.7,
  cacheBits: 0,
  fp16Layers: 4,
  fusedAttention: false,
  cacheStrategy: "native",
  fitModelInMemory: true,
  speculativeDecoding: false,
};

export const emptySettings: AppSettings = {
  modelDirectories: [],
  preferredServerPort: 8876,
  allowRemoteConnections: false,
  autoStartServer: false,
  launchPreferences: emptyLaunchPreferences,
  remoteProviders: [],
  huggingFaceToken: null,
  hasHuggingFaceToken: false,
  dataDirectory: "",
};

export const emptyPreview: PreviewMetrics = {
  bits: 0,
  fp16Layers: 0,
  numLayers: 0,
  numHeads: 0,
  hiddenSize: 0,
  contextTokens: 0,
  paramsB: 0,
  baselineCacheGb: 0,
  optimizedCacheGb: 0,
  compressionRatio: 0,
  estimatedTokS: 0,
  speedRatio: 0,
  qualityPercent: 0,
  diskSizeGb: 0,
  summary: "",
};

export const emptyWorkspace: WorkspaceData = {
  system: {
    platform: "",
    arch: "",
    hardwareSummary: "",
    backendLabel: "",
    appVersion: "",
    availableCacheStrategies: [],
    mlxAvailable: false,
    mlxLmAvailable: false,
    totalMemoryGb: 0,
    availableMemoryGb: 0,
    usedMemoryGb: 0,
    swapUsedGb: 0,
    cpuUtilizationPercent: 0,
    gpuUtilizationPercent: null,
    spareHeadroomGb: 0,
    runningLlmProcesses: [],
    uptimeMinutes: 0,
  },
  recommendation: {
    title: "",
    detail: "",
    targetModel: "",
    cacheLabel: "",
    headroomPercent: 0,
  },
  featuredModels: [],
  library: [],
  settings: emptySettings,
  chatSessions: [],
  runtime: {
    state: "idle",
    engine: "",
    engineLabel: "Connecting...",
    loadedModel: null,
    supportsGeneration: false,
    serverReady: false,
    activeRequests: 0,
    requestsServed: 0,
    runtimeNote: null,
  },
  server: {
    status: "idle",
    baseUrl: "http://127.0.0.1:8876/v1",
    port: 8876,
    activeConnections: 0,
    concurrentRequests: 0,
    requestsServed: 0,
    loadedModelName: null,
    loading: null,
    logTail: [],
  },
  benchmarks: [],
  logs: [],
  activity: [],
  preview: emptyPreview,
  quickActions: [],
};
