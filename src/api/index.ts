import { invoke, isTauri } from "@tauri-apps/api/core";
import type {
  AppSettings,
  BenchmarkRunPayload,
  BenchmarkRunResponse,
  ChatSession,
  ConvertModelPayload,
  ConvertModelResponse,
  CreateSessionResponse,
  GenerationProgressSnapshot,
  GeneratePayload,
  GenerateResponse,
  HubFileListResponse,
  HubModel,
  ImageCatalogResponse,
  ImageGenerationPayload,
  ImageGenerationResponse,
  ImageOutputArtifact,
  ImageRuntimeStatus,
  LibraryItem,
  LoadModelPayload,
  ModelFamily,
  PreviewMetrics,
  RuntimeStatus,
  SettingsUpdateResponse,
  TauriBackendInfo,
  UpdateSettingsPayload,
  UpdateSessionPayload,
  VideoCatalogResponse,
  VideoGenerationPayload,
  VideoGenerationResponse,
  VideoOutputArtifact,
  VideoRuntimeStatus,
  WorkspaceData,
} from "../types";

const DEFAULT_API_BASE = (import.meta.env.VITE_CHAOSENGINE_API_BASE as string | undefined) ?? "http://127.0.0.1:8876";
const CONFIGURED_API_TOKEN = ((import.meta.env.VITE_CHAOSENGINE_API_TOKEN as string | undefined) ?? "").trim() || null;
let apiBasePromise: Promise<string> | null = null;
let apiTokenPromise: Promise<string | null> | null = null;
let tauriBackendInfoPromise: Promise<TauriBackendInfo | null> | null = null;

function resetBackendRuntimeCache() {
  apiBasePromise = null;
  apiTokenPromise = null;
  tauriBackendInfoPromise = null;
}

export async function getTauriBackendInfo(force = false): Promise<TauriBackendInfo | null> {
  if (!isTauri()) {
    return null;
  }
  if (force) {
    tauriBackendInfoPromise = null;
  }
  if (!tauriBackendInfoPromise) {
    tauriBackendInfoPromise = invoke<TauriBackendInfo>("backend_runtime_info").catch(() => null);
  }
  return tauriBackendInfoPromise;
}

export async function resolveApiBase(): Promise<string> {
  if (import.meta.env.VITE_CHAOSENGINE_API_BASE) {
    return DEFAULT_API_BASE;
  }
  if (!apiBasePromise) {
    apiBasePromise = getTauriBackendInfo().then((info) => info?.apiBase ?? DEFAULT_API_BASE);
  }
  return apiBasePromise;
}

async function fetchSessionToken(apiBase: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`${apiBase}/api/auth/session`, {
      signal: controller.signal,
    });
    if (!response.ok) {
      return null;
    }
    const payload = (await response.json()) as { apiToken?: unknown };
    return typeof payload.apiToken === "string" ? payload.apiToken : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function resolveApiToken(force = false): Promise<string | null> {
  if (CONFIGURED_API_TOKEN) {
    return CONFIGURED_API_TOKEN;
  }
  if (force) {
    apiTokenPromise = null;
    if (isTauri()) {
      tauriBackendInfoPromise = null;
    }
  }
  if (!apiTokenPromise) {
    const attempt: { self: Promise<string | null> | null } = { self: null };
    attempt.self = (async () => {
      const apiBase = await resolveApiBase();
      if (force) {
        const fresh = await fetchSessionToken(apiBase);
        if (fresh) return fresh;
      }

      const info = await getTauriBackendInfo(force);
      if (info?.apiToken) {
        return info.apiToken;
      }
      const token = await fetchSessionToken(apiBase);
      // Don't cache a negative result. If the token fetch failed (backend
      // still starting, transient network error), leave the cache empty
      // so the next caller can try again. Caching null here poisons every
      // subsequent request until the user quits the app.
      if (token === null && apiTokenPromise === attempt.self) {
        apiTokenPromise = null;
      }
      return token;
    })();
    apiTokenPromise = attempt.self;
  }
  return apiTokenPromise;
}

function withAuthHeaders(headers: HeadersInit | undefined, apiToken: string | null): Headers {
  const merged = new Headers(headers ?? {});
  if (apiToken) {
    merged.set("Authorization", `Bearer ${apiToken}`);
  }
  return merged;
}

export async function readErrorDetail(response: Response, fallback: string): Promise<string> {
  let detail = fallback;
  try {
    const errorBody = await response.json();
    // FU-042: prefer the ``localized`` field from the FastAPI
    // ``localized_detail(...)`` envelope when present.  Envelope shape:
    // ``{detail: {message, localized, locale, errorKey?}}``.  When the
    // backend route hasn't been migrated yet, ``detail`` is still a
    // plain string and the legacy branch below handles it.
    if (
      errorBody?.detail
      && typeof errorBody.detail === "object"
      && !Array.isArray(errorBody.detail)
    ) {
      const env = errorBody.detail as Record<string, unknown>;
      if (typeof env.localized === "string" && env.localized) {
        detail = env.localized;
      } else if (typeof env.message === "string" && env.message) {
        detail = env.message;
      } else {
        detail = JSON.stringify(errorBody.detail);
      }
    } else if (errorBody?.detail) {
      detail = typeof errorBody.detail === "string" ? errorBody.detail : JSON.stringify(errorBody.detail);
    } else if (errorBody?.error) {
      detail = typeof errorBody.error === "string" ? errorBody.error : JSON.stringify(errorBody.error);
    } else if (errorBody?.message) {
      detail = typeof errorBody.message === "string" ? errorBody.message : JSON.stringify(errorBody.message);
    }
  } catch {
    try {
      const text = await response.text();
      if (text) {
        detail = text.slice(0, 500);
      }
    } catch {
      // ignore non-JSON/non-text error responses
    }
  }
  return detail;
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
  options: { includeAuth?: boolean; retryUnauthorized?: boolean } = {},
): Promise<Response> {
  const { includeAuth = true, retryUnauthorized = true } = options;
  const apiBase = await resolveApiBase();
  const apiToken = includeAuth ? await resolveApiToken() : null;
  const response = await fetch(`${apiBase}${path}`, {
    ...init,
    headers: withAuthHeaders(init.headers, apiToken),
  });
  if (includeAuth && retryUnauthorized && response.status === 401) {
    resetBackendRuntimeCache();
    const retryBase = await resolveApiBase();
    // force=true makes resolveApiToken bypass Rust's potentially stale
    // cache and re-read from /api/auth/session directly.
    const retryToken = await resolveApiToken(true);
    return await fetch(`${retryBase}${path}`, {
      ...init,
      headers: withAuthHeaders(init.headers, retryToken),
    });
  }
  return response;
}

export async function fetchJson<T>(
  path: string,
  timeoutMs = 15000,
  options: { includeAuth?: boolean } = {},
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await apiFetch(path, { signal: controller.signal }, options);
    if (!response.ok) {
      const detail = await readErrorDetail(response, `Request failed with status ${response.status}`);
      throw new Error(detail);
    }
    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error(`Request to ${path} timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function postJson<T>(path: string, body?: object, timeoutMs: number | null = 120000, externalSignal?: AbortSignal): Promise<T> {
  return await sendJson<T>("POST", path, body, timeoutMs, externalSignal);
}

export async function patchJson<T>(path: string, body?: object, timeoutMs: number | null = 120000, externalSignal?: AbortSignal): Promise<T> {
  return await sendJson<T>("PATCH", path, body, timeoutMs, externalSignal);
}

export async function deleteJson<T>(path: string, body?: object, timeoutMs: number | null = 120000, externalSignal?: AbortSignal): Promise<T> {
  return await sendJson<T>("DELETE", path, body, timeoutMs, externalSignal);
}

async function sendJson<T>(method: "POST" | "PATCH" | "DELETE", path: string, body?: object, timeoutMs: number | null = 120000, externalSignal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  // `timeoutMs: null` (or 0) means no client-side timeout — used for
  // model loads where the backend drives its own long ceiling and we
  // never want the frontend to give up ahead of it.
  const timer =
    timeoutMs && timeoutMs > 0
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
  // Chain caller-supplied `externalSignal` into the internal controller
  // so callers can cancel long-running requests (e.g. an in-flight
  // benchmark) without losing the timeout-vs-user-abort distinction.
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort();
    } else {
      externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
    }
  }
  let response: Response;
  try {
    response = await apiFetch(path, {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (err) {
    if (timer) clearTimeout(timer);
    if (err instanceof DOMException && err.name === "AbortError") {
      if (externalSignal?.aborted) throw new Error("Request cancelled");
      if (timer) throw new Error(`Request to ${path} timed out after ${Math.round((timeoutMs ?? 0) / 1000)}s`);
    }
    throw err;
  }
  if (!response.ok) {
    const detail = await readErrorDetail(response, `Request failed with status ${response.status}`);
    throw new Error(detail);
  }
  if (timer) clearTimeout(timer);
  return (await response.json()) as T;
}

export async function getWorkspace(): Promise<WorkspaceData> {
  return await fetchJson<WorkspaceData>("/api/workspace");
}

export async function checkBackend(): Promise<boolean> {
  try {
    await fetchJson("/api/health", 15000, { includeAuth: false });
    return true;
  } catch {
    try {
      await fetchJson("/api/auth/session", 5000, { includeAuth: false });
      return true;
    } catch {
      return false;
    }
  }
}

export interface GpuStatus {
  platform: string;
  nvidiaGpuDetected: boolean;
  torchImported: boolean;
  torchCudaAvailable: boolean;
  torchMpsAvailable: boolean;
  cpuFallbackWarning: boolean;
  recommendation: string | null;
}

export async function getGpuStatus(): Promise<GpuStatus> {
  return await fetchJson<GpuStatus>("/api/system/gpu-status", 15000, { includeAuth: false });
}

export async function getSettings(): Promise<AppSettings> {
  const result = await fetchJson<{ settings: AppSettings }>("/api/settings");
  return result.settings;
}

export async function updateSettings(payload: UpdateSettingsPayload): Promise<SettingsUpdateResponse> {
  return await patchJson<SettingsUpdateResponse>("/api/settings", payload);
}













export async function getCachePreview(options: {
  bits: number;
  fp16Layers: number;
  numLayers: number;
  numHeads: number;
  numKvHeads?: number;
  hiddenSize: number;
  contextTokens: number;
  paramsB: number;
  strategy?: string;
}): Promise<PreviewMetrics> {
  const search = new URLSearchParams({
    bits: String(options.bits),
    fp16_layers: String(options.fp16Layers),
    num_layers: String(options.numLayers),
    num_heads: String(options.numHeads),
    hidden_size: String(options.hiddenSize),
    context_tokens: String(options.contextTokens),
    params_b: String(options.paramsB),
  });
  if (options.numKvHeads && options.numKvHeads > 0) {
    search.set("num_kv_heads", String(options.numKvHeads));
  }
  if (options.strategy) {
    search.set("strategy", options.strategy);
  }

  try {
    return await fetchJson<PreviewMetrics>(`/api/cache/preview?${search.toString()}`);
  } catch {
    return {
      bits: options.bits,
      fp16Layers: options.fp16Layers,
      numLayers: options.numLayers,
      numHeads: options.numHeads,
      numKvHeads: options.numKvHeads ?? options.numHeads,
      hiddenSize: options.hiddenSize,
      contextTokens: options.contextTokens,
      paramsB: options.paramsB,
      baselineCacheGb: 0,
      optimizedCacheGb: 0,
      compressionRatio: 0,
      estimatedTokS: 0,
      speedRatio: 0,
      qualityPercent: 0,
      diskSizeGb: 0,
      summary: "Cache preview unavailable \u2014 connect the backend to calculate machine-specific estimates.",
    };
  }
}

































export async function openHtmlChallengeFile(path: string): Promise<void> {
  await postJson<{ opened: string }>("/api/chat/html-challenges/open-file", { path });
}



export async function shutdownServer(): Promise<void> {
  await postJson<{ status: string }>("/api/server/shutdown");
}


export async function stopManagedBackend(): Promise<TauriBackendInfo | null> {
  if (!isTauri()) {
    return null;
  }
  resetBackendRuntimeCache();
  const info = await invoke<TauriBackendInfo>("stop_backend_sidecar").catch(() => null);
  tauriBackendInfoPromise = Promise.resolve(info);
  apiBasePromise = Promise.resolve(info?.apiBase ?? DEFAULT_API_BASE);
  apiTokenPromise = Promise.resolve(info?.apiToken ?? null);
  return info;
}

export async function restartManagedBackend(): Promise<TauriBackendInfo | null> {
  if (!isTauri()) {
    return null;
  }
  resetBackendRuntimeCache();
  const info = await invoke<TauriBackendInfo>("restart_backend_sidecar").catch(() => null);
  tauriBackendInfoPromise = Promise.resolve(info);
  apiBasePromise = Promise.resolve(info?.apiBase ?? DEFAULT_API_BASE);
  apiTokenPromise = Promise.resolve(info?.apiToken ?? null);
  return info;
}

// ---------------------------------------------------------------------------
// Domain re-exports — extracted in v0.8.0 Phase 2 to keep this facade thin.
// ---------------------------------------------------------------------------

export {
  cancelImageDownload,
  cancelImageGeneration,
  deleteImageDownload,
  deleteImageOutput,
  downloadImageModel,
  generateImage,
  getImageCatalog,
  getImageDownloadStatus,
  getImageGenerationProgress,
  getImageOutputs,
  getImageRuntime,
  preloadImageModel,
  unloadImageModel,
} from "./image";

export {
  cancelVideoDownload,
  cancelVideoGeneration,
  deleteVideoDownload,
  deleteVideoOutput,
  downloadVideoModel,
  fetchVideoOutputBlobUrl,
  generateVideo,
  getLongLiveRuntime,
  getMlxVideoRuntime,
  getVideoCatalog,
  getVideoDownloadStatus,
  getVideoGenerationProgress,
  getVideoOutputs,
  getVideoRuntime,
  preloadVideoModel,
  unloadVideoModel,
} from "./video";

export {
  checkTurboUpdate,
  enhancePromptViaLLM,
  fetchGpuBundleInfo,
  getGpuBundleStatus,
  getLongLiveInstallStatus,
  getWanInstallStatus,
  getWanInventory,
  installCudaTorch,
  installPipPackage,
  installSystemPackage,
  refreshCapabilities,
  startGpuBundleInstall,
  startLongLiveInstall,
  startWanInstall,
} from "./setup";
export type {
  CudaTorchInstallAttempt,
  CudaTorchInstallResult,
  GpuBundleAttempt,
  GpuBundleInfo,
  GpuBundleJobState,
  GpuBundlePackage,
  InstallResult,
  LongLiveAttempt,
  LongLiveJobState,
  PromptEnhanceResult,
  TurboUpdateInfo,
  WanConvertStatusFields,
  WanInstallAttempt,
  WanInstallJobState,
  WanInventory,
  WanInventoryItem,
} from "./setup";

export {
  fetchDiagnosticsLogTail,
  fetchDiagnosticsSnapshot,
  getModelMoveStatus,
  getStorageSettings,
  reextractRuntime,
  startModelMove,
  updateHfCachePath,
} from "./admin";
export type {
  DiagnosticsLogTail,
  DiagnosticsSnapshot,
  ModelMoveJobState,
  ReextractRuntimeResult,
  StorageSettingsSnapshot,
  UpdateStoragePathResult,
} from "./admin";

export {
  addMessageVariant,
  cancelChatGeneration,
  createSession,
  delveMessage,
  deleteSession,
  deleteSessionDocument,
  forkChatSession,
  generateChat,
  generateChatStream,
  listSessionDocuments,
  updateSession,
  uploadSessionDocument,
} from "./chat";
export type {
  ChatStreamPhase,
  SessionDocument,
  StreamCallbacks,
} from "./chat";

export {
  cancelDownload,
  convertModel,
  deleteModelDownload,
  deleteModelPath,
  downloadModel,
  getDownloadStatus,
  listHubFiles,
  loadModel,
  revealModelPath,
  runBenchmark,
  searchHubModels,
  searchModels,
  unloadModel,
} from "./models";
export type {
  DeleteDownloadResult,
  DownloadStatus,
  SearchResults,
} from "./models";
