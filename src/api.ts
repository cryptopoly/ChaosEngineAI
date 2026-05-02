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
} from "./types";

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

async function readErrorDetail(response: Response, fallback: string): Promise<string> {
  let detail = fallback;
  try {
    const errorBody = await response.json();
    if (errorBody?.detail) {
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

async function postJson<T>(path: string, body?: object, timeoutMs: number | null = 120000): Promise<T> {
  return await sendJson<T>("POST", path, body, timeoutMs);
}

async function patchJson<T>(path: string, body?: object, timeoutMs: number | null = 120000): Promise<T> {
  return await sendJson<T>("PATCH", path, body, timeoutMs);
}

async function deleteJson<T>(path: string, body?: object, timeoutMs: number | null = 120000): Promise<T> {
  return await sendJson<T>("DELETE", path, body, timeoutMs);
}

async function sendJson<T>(method: "POST" | "PATCH" | "DELETE", path: string, body?: object, timeoutMs: number | null = 120000): Promise<T> {
  const controller = new AbortController();
  // `timeoutMs: null` (or 0) means no client-side timeout — used for
  // model loads where the backend drives its own long ceiling and we
  // never want the frontend to give up ahead of it.
  const timer =
    timeoutMs && timeoutMs > 0
      ? setTimeout(() => controller.abort(), timeoutMs)
      : null;
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
    if (err instanceof DOMException && err.name === "AbortError" && timer) {
      throw new Error(`Request to ${path} timed out after ${Math.round((timeoutMs ?? 0) / 1000)}s`);
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
    return false;
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

export interface SearchResults {
  families: ModelFamily[];
  hubModels: HubModel[];
}

export async function searchModels(query: string): Promise<SearchResults> {
  const result = await fetchJson<{ results: ModelFamily[]; hubResults?: HubModel[] }>(
    `/api/models/search?q=${encodeURIComponent(query)}`,
    60000,
  );
  return { families: result.results, hubModels: result.hubResults ?? [] };
}

export async function searchHubModels(query: string): Promise<HubModel[]> {
  const result = await fetchJson<{ results: HubModel[] }>(
    `/api/models/hub-search?q=${encodeURIComponent(query)}`,
    60000,
  );
  return result.results ?? [];
}

export async function getImageCatalog(): Promise<ImageCatalogResponse> {
  return await fetchJson<ImageCatalogResponse>("/api/images/catalog", 25000);
}

export async function getImageOutputs(): Promise<ImageOutputArtifact[]> {
  const result = await fetchJson<{ outputs: ImageOutputArtifact[] }>("/api/images/outputs");
  return result.outputs;
}

export async function getImageRuntime(): Promise<ImageRuntimeStatus> {
  const result = await fetchJson<{ runtime: ImageRuntimeStatus }>("/api/images/runtime");
  return result.runtime;
}

/**
 * Polled by ImageGenerationModal while the bar is visible to override the
 * client-side phase estimates with the runtime's actual phase / step count.
 * Short timeout — if the backend is busy with the generation it can still
 * answer this lightweight read in well under a second.
 */
export async function getImageGenerationProgress(): Promise<GenerationProgressSnapshot> {
  const result = await fetchJson<{ progress: GenerationProgressSnapshot }>(
    "/api/images/progress",
    5000,
  );
  return result.progress;
}

export async function getVideoCatalog(): Promise<VideoCatalogResponse> {
  return await fetchJson<VideoCatalogResponse>("/api/video/catalog", 25000);
}

export async function getVideoRuntime(): Promise<VideoRuntimeStatus> {
  // 30s rather than the 15s default — the first call of a sidecar's life
  // imports torch and (on Windows/Linux) shells out to nvidia-smi, both of
  // which can take several seconds on cold disks. Backend caches the VRAM
  // total after the first probe so subsequent calls are fast, but the
  // initial one needs the headroom.
  const result = await fetchJson<{ runtime: VideoRuntimeStatus }>("/api/video/runtime", 30000);
  return result.runtime;
}

export async function getLongLiveRuntime(): Promise<VideoRuntimeStatus> {
  // LongLive probe is separate from the diffusers video runtime — it
  // checks the isolated install marker at ~/.chaosengine/longlive rather
  // than torch/diffusers on the host Python. Surfaces an install action
  // in the Studio when the LongLive variant is selected but not yet set up.
  const result = await fetchJson<{ runtime: VideoRuntimeStatus }>("/api/video/longlive", 30000);
  return result.runtime;
}

export async function getMlxVideoRuntime(): Promise<VideoRuntimeStatus> {
  // mlx-video probe (FU-009). Separate from the diffusers video runtime
  // so Apple Silicon users get a dedicated "Install mlx-video" affordance
  // on the Studio without mixing it into the diffusers/torch state. The
  // probe returns activeEngine="mlx-video" with realGenerationAvailable=
  // false on non-Apple platforms — the Studio hides the chip in that
  // case (platform mismatch, not a missing-package state).
  const result = await fetchJson<{ runtime: VideoRuntimeStatus }>("/api/video/mlx-runtime", 30000);
  return result.runtime;
}

/** Mirror of ``getImageGenerationProgress`` for the video runtime. */
export async function getVideoGenerationProgress(): Promise<GenerationProgressSnapshot> {
  const result = await fetchJson<{ progress: GenerationProgressSnapshot }>(
    "/api/video/progress",
    5000,
  );
  return result.progress;
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

export async function loadModel(payload: LoadModelPayload): Promise<RuntimeStatus> {
  // NO client-side timeout on model loads — the backend has its own
  // MLX_LOAD_TIMEOUT_SECONDS=1800 ceiling. We never want the frontend to
  // give up ahead of the backend and leave the user staring at a false
  // "timed out" while the worker is still happily loading weights.
  const result = await postJson<{ runtime: RuntimeStatus }>("/api/models/load", payload, null);
  return result.runtime;
}

export async function unloadModel(ref?: string): Promise<RuntimeStatus> {
  const result = await postJson<{ runtime: RuntimeStatus }>(
    "/api/models/unload",
    ref ? { ref } : undefined,
  );
  return result.runtime;
}

export async function createSession(title?: string): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>("/api/chat/sessions", { title });
  return result.session;
}

/**
 * Phase 2.5: generate a sibling variant for an assistant message
 * using a different (currently-loaded) model. Returns the updated
 * session payload with `messages[messageIndex].variants` populated.
 */
export async function addMessageVariant(
  sessionId: string,
  payload: {
    messageIndex: number;
    modelRef: string;
    modelName: string;
    canonicalRepo?: string | null;
    source?: string;
    path?: string;
    backend?: string;
    maxTokens?: number;
    temperature?: number;
  },
): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/variants`,
    payload,
    300000,
  );
  return result.session;
}

/**
 * Phase 2.4: fork an existing thread at a specific message index.
 * Returns the new session, which the caller swaps active to so the
 * user can continue divergently. Parent linkage is preserved on
 * `parentSessionId` + `forkedAtMessageIndex`.
 */
export async function forkChatSession(
  sourceSessionId: string,
  forkAtMessageIndex: number,
  title?: string,
): Promise<ChatSession> {
  const result = await postJson<CreateSessionResponse>(
    `/api/chat/sessions/${encodeURIComponent(sourceSessionId)}/fork`,
    { forkAtMessageIndex, title },
  );
  return result.session;
}

export async function updateSession(sessionId: string, payload: UpdateSessionPayload): Promise<ChatSession> {
  const result = await patchJson<CreateSessionResponse>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, payload);
  return result.session;
}

export async function generateChat(payload: GeneratePayload): Promise<GenerateResponse> {
  return await postJson<GenerateResponse>("/api/chat/generate", payload, 300000);
}

export type ChatStreamPhase = "prompt_eval" | "generating";

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onReasoning?: (reasoning: string) => void;
  onReasoningDone?: () => void;
  onCancelled?: () => void;
  /**
   * Phase transition signal (Phase 2.0). Backend emits `prompt_eval`
   * immediately when generation begins, then `generating` (with a
   * `ttftSeconds` measurement) the moment the model produces its first
   * token or reasoning fragment. Use this to render an explicit
   * "Processing prompt..." indicator instead of a blank flashing cursor.
   */
  onPhase?: (phase: ChatStreamPhase, ttftSeconds?: number) => void;
  /**
   * Phase 2.0.5-G: mid-stream panic signal. Backend emits at most once
   * per turn when memory crosses critical floors (free < 0.5 GB OR
   * pressure > 96%). Stream continues; user decides whether to cancel.
   */
  onPanic?: (signal: { message: string; availableGb?: number; pressurePercent?: number }) => void;
  /**
   * Phase 2.0.5-I: mid-stream thermal warning. Backend emits when host
   * is actively thermally throttling. Stream continues.
   */
  onThermalWarning?: (signal: { state: "moderate" | "critical"; message: string }) => void;
  onDone: (response: GenerateResponse) => void;
  onError: (error: string) => void;
}

/**
 * Ask the backend to cancel an in-flight chat generation. The streaming loop
 * checks this flag between events and stops within ~one tick, persisting
 * whatever output has accumulated. Safe to call when no generation is active.
 */
export async function cancelChatGeneration(sessionId: string): Promise<{ sessionId: string; cancelled: boolean; wasActive: boolean }> {
  return await postJson<{ sessionId: string; cancelled: boolean; wasActive: boolean }>(
    `/api/chat/generate/${encodeURIComponent(sessionId)}/cancel`,
    {},
    10000,
  );
}

export async function generateChatStream(
  payload: GeneratePayload,
  callbacks: StreamCallbacks,
  abortSignal?: AbortController,
): Promise<void> {
  const controller = abortSignal ?? new AbortController();
  const timer = setTimeout(() => controller.abort(), 300000);

  try {
    const response = await apiFetch("/api/chat/generate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const errorBody = await response.json();
        if (errorBody?.detail) {
          detail = typeof errorBody.detail === "string"
            ? errorBody.detail
            : JSON.stringify(errorBody.detail);
        }
      } catch { /* ignore */ }
      callbacks.onError(detail);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError("Streaming not supported");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          if (event.error) {
            const errDetail = typeof event.error === "string"
              ? event.error
              : event.error?.detail ?? event.error?.message ?? JSON.stringify(event.error);
            callbacks.onError(errDetail);
            return;
          }
          if (event.token) {
            callbacks.onToken(event.token);
          }
          if (event.reasoning) {
            callbacks.onReasoning?.(event.reasoning);
          }
          if (event.reasoningDone) {
            callbacks.onReasoningDone?.();
          }
          if (event.cancelled) {
            callbacks.onCancelled?.();
          }
          if (event.phase === "prompt_eval" || event.phase === "generating") {
            const ttft = typeof event.ttftSeconds === "number" ? event.ttftSeconds : undefined;
            callbacks.onPhase?.(event.phase, ttft);
          }
          if (event.panic === true && typeof event.message === "string") {
            callbacks.onPanic?.({
              message: event.message,
              availableGb: typeof event.availableGb === "number" ? event.availableGb : undefined,
              pressurePercent: typeof event.pressurePercent === "number" ? event.pressurePercent : undefined,
            });
          }
          if (event.thermalWarning === true && typeof event.message === "string"
              && (event.state === "moderate" || event.state === "critical")) {
            callbacks.onThermalWarning?.({
              state: event.state,
              message: event.message,
            });
          }
          if (event.done) {
            callbacks.onDone({
              session: event.session,
              assistant: event.assistant,
              runtime: event.runtime,
            });
          }
        } catch {
          // Malformed JSON chunk, skip
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      callbacks.onError("Streaming timed out");
    } else {
      callbacks.onError(err instanceof Error ? err.message : "Unknown streaming error");
    }
  } finally {
    clearTimeout(timer);
  }
}

export interface SessionDocument {
  id: string;
  filename: string;
  originalName: string;
  sizeBytes: number;
  chunkCount: number;
  uploadedAt: string;
}

export async function uploadSessionDocument(sessionId: string, file: File): Promise<SessionDocument> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Upload failed with status ${response.status}`));
  }
  const result = await response.json();
  return result.document;
}

export async function listSessionDocuments(sessionId: string): Promise<SessionDocument[]> {
  const result = await fetchJson<{ documents: SessionDocument[] }>(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents`);
  return result.documents;
}

export async function deleteSessionDocument(sessionId: string, docId: string): Promise<void> {
  const response = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Delete failed with status ${response.status}`));
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  const response = await apiFetch(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(await readErrorDetail(response, `Delete failed with status ${response.status}`));
  }
}

export interface DownloadStatus {
  repo: string;
  state: "downloading" | "completed" | "failed" | "cancelled";
  progress: number;
  downloadedGb: number;
  totalGb: number | null;
  error: string | null;
}

export interface DeleteDownloadResult {
  repo: string;
  state: "deleted" | "not_found";
}

export async function downloadModel(repo: string): Promise<DownloadStatus> {
  const result = await postJson<{ download: DownloadStatus }>("/api/models/download", { repo });
  return result.download;
}

export async function getDownloadStatus(): Promise<DownloadStatus[]> {
  const result = await fetchJson<{ downloads: DownloadStatus[] }>("/api/models/download/status");
  return result.downloads;
}

export async function cancelDownload(repo: string): Promise<DownloadStatus> {
  const result = await postJson<{ download: DownloadStatus }>("/api/models/download/cancel", { repo });
  return result.download;
}

export async function deleteModelDownload(repo: string): Promise<DeleteDownloadResult> {
  const result = await postJson<{ result: DeleteDownloadResult }>("/api/models/download/delete", { repo });
  return result.result;
}

export async function downloadImageModel(repo: string): Promise<DownloadStatus> {
  const result = await postJson<{ download: DownloadStatus }>("/api/images/download", { repo });
  return result.download;
}

export async function getImageDownloadStatus(): Promise<DownloadStatus[]> {
  const result = await fetchJson<{ downloads: DownloadStatus[] }>("/api/images/download/status");
  return result.downloads;
}

export async function cancelImageDownload(repo: string): Promise<DownloadStatus> {
  const result = await postJson<{ download: DownloadStatus }>("/api/images/download/cancel", { repo });
  return result.download;
}

export async function deleteImageDownload(repo: string): Promise<DeleteDownloadResult> {
  const result = await postJson<{ result: DeleteDownloadResult }>("/api/images/download/delete", { repo });
  return result.result;
}

export async function preloadImageModel(modelId: string): Promise<ImageRuntimeStatus> {
  const result = await postJson<{ runtime: ImageRuntimeStatus }>("/api/images/preload", { modelId }, null);
  return result.runtime;
}

export async function unloadImageModel(modelId?: string): Promise<ImageRuntimeStatus> {
  const result = await postJson<{ runtime: ImageRuntimeStatus }>(
    "/api/images/unload",
    modelId ? { modelId } : undefined,
  );
  return result.runtime;
}

export async function downloadVideoModel(repo: string, modelId?: string): Promise<DownloadStatus> {
  const result = await postJson<{ download: DownloadStatus }>("/api/video/download", { repo, modelId });
  return result.download;
}

export async function getVideoDownloadStatus(): Promise<DownloadStatus[]> {
  const result = await fetchJson<{ downloads: DownloadStatus[] }>("/api/video/download/status");
  return result.downloads;
}

export async function cancelVideoDownload(repo: string): Promise<DownloadStatus> {
  const result = await postJson<{ download: DownloadStatus }>("/api/video/download/cancel", { repo });
  return result.download;
}

export async function deleteVideoDownload(repo: string): Promise<DeleteDownloadResult> {
  const result = await postJson<{ result: DeleteDownloadResult }>("/api/video/download/delete", { repo });
  return result.result;
}

export async function preloadVideoModel(modelId: string): Promise<VideoRuntimeStatus> {
  const result = await postJson<{ runtime: VideoRuntimeStatus }>("/api/video/preload", { modelId }, null);
  return result.runtime;
}

export async function unloadVideoModel(modelId?: string): Promise<VideoRuntimeStatus> {
  const result = await postJson<{ runtime: VideoRuntimeStatus }>(
    "/api/video/unload",
    modelId ? { modelId } : undefined,
  );
  return result.runtime;
}

export async function generateVideo(payload: VideoGenerationPayload): Promise<VideoGenerationResponse> {
  // No client timeout — video generation legitimately takes minutes on consumer hardware.
  return await postJson<VideoGenerationResponse>("/api/video/generate", payload, null);
}

export async function cancelVideoGeneration(): Promise<{ cancelled: boolean }> {
  // 10s timeout — the endpoint just sets a flag and returns, no wait.
  return await postJson<{ cancelled: boolean }>("/api/video/cancel", {}, 10000);
}

export async function getVideoOutputs(): Promise<VideoOutputArtifact[]> {
  const result = await fetchJson<{ outputs: VideoOutputArtifact[] }>("/api/video/outputs");
  return result.outputs;
}

export async function deleteVideoOutput(
  artifactId: string,
): Promise<{ deleted: string; outputs: VideoOutputArtifact[] }> {
  return await deleteJson<{ deleted: string; outputs: VideoOutputArtifact[] }>(
    `/api/video/outputs/${encodeURIComponent(artifactId)}`,
  );
}

/**
 * Fetch a saved mp4 as a blob URL that an HTML5 <video> element can play.
 *
 * The backend auth middleware only reads the token from the ``Authorization``
 * or ``x-chaosengine-token`` headers, so we can't just point a <video src> at
 * the file endpoint directly. Fetching the bytes ourselves and handing back
 * an object URL keeps auth clean and works even for clips > 25MB. Callers
 * are responsible for calling ``URL.revokeObjectURL`` when the component
 * unmounts.
 */
export async function fetchVideoOutputBlobUrl(artifactId: string): Promise<string> {
  const response = await apiFetch(
    `/api/video/outputs/${encodeURIComponent(artifactId)}/file`,
    { method: "GET" },
  );
  if (!response.ok) {
    throw new Error(`Failed to load video (${response.status} ${response.statusText})`);
  }
  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

export async function generateImage(payload: ImageGenerationPayload): Promise<ImageGenerationResponse> {
  return await postJson<ImageGenerationResponse>("/api/images/generate", payload, null);
}

export async function cancelImageGeneration(): Promise<{ cancelled: boolean }> {
  return await postJson<{ cancelled: boolean }>("/api/images/cancel", {}, 10000);
}

export async function deleteImageOutput(artifactId: string): Promise<{ deleted: string; outputs: ImageOutputArtifact[] }> {
  return await deleteJson<{ deleted: string; outputs: ImageOutputArtifact[] }>(
    `/api/images/outputs/${encodeURIComponent(artifactId)}`,
  );
}

export async function convertModel(payload: ConvertModelPayload): Promise<ConvertModelResponse> {
  // No client-side timeout — conversion can legitimately take 10+ min for
  // large models on a cold cache. Backend has its own 3600s subprocess cap.
  return await postJson<ConvertModelResponse>("/api/models/convert", payload, null);
}

export async function runBenchmark(payload: BenchmarkRunPayload): Promise<BenchmarkRunResponse> {
  // No client-side timeout — a benchmark on a cold 70B model legitimately
  // takes >120s (cold load + prompt processing + N-token generation +
  // measurement). The backend has its own per-phase ceilings.
  return await postJson<BenchmarkRunResponse>("/api/benchmarks/run", payload, null);
}

export async function revealModelPath(path: string): Promise<void> {
  await postJson<{ revealed: string }>("/api/models/reveal", { path });
}

export async function deleteModelPath(path: string): Promise<{ deleted: string; library: LibraryItem[] }> {
  return await postJson<{ deleted: string; library: LibraryItem[] }>(
    "/api/models/delete",
    { path },
  );
}

export async function listHubFiles(repo: string): Promise<HubFileListResponse> {
  return await fetchJson<HubFileListResponse>(`/api/models/hub-files?repo=${encodeURIComponent(repo)}`, 15000);
}

export async function shutdownServer(): Promise<void> {
  await postJson<{ status: string }>("/api/server/shutdown");
}

// ------------------------------------------------------------------
// Setup / install
// ------------------------------------------------------------------

export interface InstallResult {
  ok: boolean;
  output: string;
  capabilities: Record<string, unknown>;
}

export async function installPipPackage(packageName: string): Promise<InstallResult> {
  return await postJson<InstallResult>("/api/setup/install-package", { package: packageName }, 360000);
}

export async function installSystemPackage(packageName: string): Promise<InstallResult> {
  return await postJson<InstallResult>("/api/setup/install-system-package", { package: packageName }, 660000);
}

export interface CudaTorchInstallAttempt {
  indexUrl: string;
  ok: boolean;
  output: string;
}

export interface CudaTorchInstallResult {
  ok: boolean;
  output: string;
  indexUrl: string | null;
  attempts: CudaTorchInstallAttempt[];
  requiresRestart: boolean;
  pythonExecutable: string;
  pythonVersion: string | null;
  noWheelForPython: boolean;
  targetDir?: string;
  capabilities: Record<string, unknown>;
}

export async function installCudaTorch(): Promise<CudaTorchInstallResult> {
  // 15 minute timeout — torch CUDA wheels are ~2.5 GB, and the endpoint
  // walks up to four CUDA indexes before giving up.
  return await postJson<CudaTorchInstallResult>("/api/setup/install-cuda-torch", {}, 900000);
}

export interface GpuBundlePackage {
  label: string;
  spec: string;
}

export interface GpuBundleInfo {
  targetDir: string | null;
  approxDownloadBytes: number;
  requiredFreeBytes: number;
  freeBytes: number | null;
  packages: GpuBundlePackage[];
}

export interface GpuBundleAttempt {
  indexUrl?: string;
  package?: string;
  phase?: string;
  ok: boolean;
  output: string;
}

export interface GpuBundleJobState {
  id: string;
  // Lifecycle: idle (no run yet) -> preflight -> downloading -> verifying -> done | error
  phase: "idle" | "preflight" | "downloading" | "verifying" | "done" | "error";
  message: string;
  packageCurrent: string | null;
  packageIndex: number;
  packageTotal: number;
  percent: number;
  targetDir: string | null;
  indexUrlUsed: string | null;
  pythonVersion: string | null;
  noWheelForPython: boolean;
  cudaVerified: boolean | null;
  requiresRestart: boolean;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  attempts: GpuBundleAttempt[];
  done: boolean;
}

export async function fetchGpuBundleInfo(): Promise<GpuBundleInfo> {
  return await fetchJson<GpuBundleInfo>("/api/setup/gpu-bundle-info", 15000);
}

export async function startGpuBundleInstall(): Promise<GpuBundleJobState> {
  // Returns quickly — the install runs in a backend background thread.
  // Poll ``getGpuBundleStatus`` to follow progress.
  return await postJson<GpuBundleJobState>("/api/setup/install-gpu-bundle", {}, 15000);
}

export async function getGpuBundleStatus(): Promise<GpuBundleJobState> {
  return await fetchJson<GpuBundleJobState>("/api/setup/install-gpu-bundle/status", 10000);
}

// LongLive async install — same job pattern as the GPU bundle. The
// backend installer takes 10-20 minutes (pip ~30 packages, optional
// flash-attn build, ~8 GB of HF weights) so we cannot run it through
// the synchronous ``/api/setup/install-system-package`` route.
//
// ``LongLiveJobState`` is shaped as a subset of ``GpuBundleJobState``
// so the same ``InstallLogPanel`` component can render either job.
// LongLive doesn't have a CUDA-index walk or wheel-availability check,
// so the LongLive-specific fields just default to neutral values.
export interface LongLiveAttempt {
  phase?: string;
  package?: string;
  // Always undefined for LongLive — the field exists in the type only
  // so the shared ``InstallLogPanel`` can read it on the discriminated
  // union without a per-job branch. Cheap to carry, expensive to fork
  // the panel just to drop one optional property.
  indexUrl?: string;
  ok: boolean;
  output: string;
}

export interface LongLiveJobState {
  id: string;
  phase: "idle" | "preflight" | "downloading" | "verifying" | "done" | "error";
  message: string;
  packageCurrent: string | null;
  packageIndex: number;
  packageTotal: number;
  percent: number;
  targetDir: string | null;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  attempts: LongLiveAttempt[];
  done: boolean;
}

export async function startLongLiveInstall(): Promise<LongLiveJobState> {
  // Returns quickly — install runs in a backend daemon thread.
  // Poll ``getLongLiveInstallStatus`` to follow progress.
  return await postJson<LongLiveJobState>("/api/setup/install-longlive", {}, 15000);
}

export async function getLongLiveInstallStatus(): Promise<LongLiveJobState> {
  return await fetchJson<LongLiveJobState>("/api/setup/install-longlive/status", 10000);
}

// --- Diagnostics ---------------------------------------------------
//
// Surfaced in Settings → Diagnostics. The snapshot is a structured dump
// of everything we'd otherwise ask users to gather by PowerShell: OS,
// hardware, runtime paths, GPU state, env vars, log tail. The frontend
// pretty-prints it as Markdown for one-click clipboard sharing.

export interface DiagnosticsSnapshot {
  generatedAt: number;
  app: {
    appVersion: string;
    workspaceRoot: string;
    logCount: number;
    activeRequests: number;
    requestsServed: number;
  };
  os: Record<string, unknown>;
  hardware: {
    cpu: Record<string, unknown>;
    memory: Record<string, number | null | undefined>;
    swap: Record<string, number | null | undefined>;
    disks: Array<Record<string, unknown>>;
    gpu: Record<string, unknown>;
    error?: string;
  };
  python: {
    executable: string;
    version: string | null;
    versionTuple: number[];
    implementation: string;
    prefix: string;
    basePrefix: string;
    platform: string;
    sysPath: string[];
    cwd: string | null;
  };
  runtime: {
    engineName: string | null;
    engineLabel: string | null;
    loadedModel: Record<string, unknown> | null;
    warmPoolCount: number | null;
    llamaServerPath: string | null;
    llamaServerTurboPath: string | null;
    llamaCliPath: string | null;
  };
  gpu: {
    torchFindSpec: boolean;
    diffusersFindSpec: boolean;
    accelerateFindSpec: boolean;
    transformersFindSpec: boolean;
    imageioFindSpec: boolean;
    ffmpegFindSpec: boolean;
    sentencepieceFindSpec: boolean;
    tiktokenFindSpec: boolean;
    protobufFindSpec: boolean;
    ftfyFindSpec: boolean;
    torchSubprocess: Record<string, unknown> | null;
  };
  extras: {
    path: string;
    exists: boolean;
    freeBytes: number | null;
    sizeBytes: number | null;
    topLevelEntries: string[];
    error?: string;
  };
  environment: Record<string, string | null>;
  logs: {
    path: string | null;
    tailLines: string[];
  };
}

export interface DiagnosticsLogTail {
  path: string | null;
  lines: string[];
  lineCount: number;
}

export interface ReextractRuntimeResult {
  path: string | null;
  deleted: boolean;
  error: string | null;
}

export async function fetchDiagnosticsSnapshot(): Promise<DiagnosticsSnapshot> {
  // 60s timeout — the snapshot fires a torch-probe subprocess and disk
  // scans, which on a slow NTFS volume can add a few seconds. Plenty of
  // headroom beyond the typical ~500ms.
  return await fetchJson<DiagnosticsSnapshot>("/api/diagnostics/snapshot", 60000);
}

export async function fetchDiagnosticsLogTail(lines = 200): Promise<DiagnosticsLogTail> {
  return await fetchJson<DiagnosticsLogTail>(
    `/api/diagnostics/log-tail?lines=${encodeURIComponent(lines)}`,
    15000,
  );
}

export async function reextractRuntime(): Promise<ReextractRuntimeResult> {
  return await postJson<ReextractRuntimeResult>("/api/diagnostics/reextract-runtime", {}, 30000);
}

export interface TurboUpdateInfo {
  installed: boolean;
  installedCommit: string | null;
  remoteCommit: string | null;
  updateAvailable: boolean;
  branch: string | null;
  buildDate: string | null;
}

export async function checkTurboUpdate(): Promise<TurboUpdateInfo> {
  return await fetchJson<TurboUpdateInfo>("/api/setup/turbo-update-check", 20000);
}

export interface ModelMoveJobState {
  id: string;
  phase: "idle" | "preflight" | "copying" | "cleanup" | "done" | "error";
  message: string;
  sourcePath: string | null;
  destinationPath: string | null;
  bytesTotal: number;
  bytesCopied: number;
  percent: number;
  filesTotal: number;
  filesCopied: number;
  currentEntry: string | null;
  error: string | null;
  startedAt: number;
  finishedAt: number;
  done: boolean;
}

export interface StorageSettingsSnapshot {
  configuredPath: string;
  effectivePath: string;
  effectiveHubPath: string;
  defaultPath: string;
  currentHubSizeBytes: number;
  currentFreeBytes: number | null;
  moveJob: ModelMoveJobState;
}

export interface UpdateStoragePathResult {
  configuredPath: string;
  effectivePath: string;
  restartRequired: boolean;
}

export async function getStorageSettings(): Promise<StorageSettingsSnapshot> {
  return await fetchJson<StorageSettingsSnapshot>("/api/settings/storage", 15000);
}

export async function updateHfCachePath(path: string): Promise<UpdateStoragePathResult> {
  return await postJson<UpdateStoragePathResult>(
    "/api/settings/storage",
    { hfCachePath: path },
    20000,
  );
}

export async function startModelMove(
  destinationPath: string,
  deleteSourceAfter = true,
): Promise<ModelMoveJobState> {
  // No client-side timeout — the move worker runs in a background thread,
  // the POST itself returns immediately with the initial state. Status is
  // polled via getModelMoveStatus. 30s is plenty for the spawn handshake.
  return await postJson<ModelMoveJobState>(
    "/api/settings/storage/move",
    { destinationPath, deleteSourceAfter },
    30000,
  );
}

export async function getModelMoveStatus(): Promise<ModelMoveJobState> {
  return await fetchJson<ModelMoveJobState>("/api/settings/storage/move/status", 10000);
}

export async function refreshCapabilities(): Promise<Record<string, unknown>> {
  const result = await postJson<{ capabilities: Record<string, unknown> }>("/api/setup/refresh-capabilities");
  return result.capabilities;
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
