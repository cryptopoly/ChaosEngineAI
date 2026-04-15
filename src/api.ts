import { invoke, isTauri } from "@tauri-apps/api/core";
import type {
  AppSettings,
  BenchmarkRunPayload,
  BenchmarkRunResponse,
  ChatSession,
  ConvertModelPayload,
  ConvertModelResponse,
  CreateSessionResponse,
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
  WorkspaceData,
} from "./types";

const DEFAULT_API_BASE = (import.meta.env.VITE_CHAOSENGINE_API_BASE as string | undefined) ?? "http://127.0.0.1:8876";
let apiBasePromise: Promise<string> | null = null;
let tauriBackendInfoPromise: Promise<TauriBackendInfo | null> | null = null;

function resetBackendRuntimeCache() {
  apiBasePromise = null;
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

async function fetchJson<T>(path: string, timeoutMs = 15000): Promise<T> {
  const apiBase = await resolveApiBase();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${apiBase}${path}`, { signal: controller.signal });
    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const errorBody = await response.json();
        if (errorBody?.detail) {
          detail = typeof errorBody.detail === "string" ? errorBody.detail : JSON.stringify(errorBody.detail);
        }
      } catch {
        // ignore non-JSON error responses
      }
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
  const apiBase = await resolveApiBase();
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
    response = await fetch(`${apiBase}${path}`, {
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
    let detail = `Request failed with status ${response.status}`;
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
      // response body was not JSON; try plain text
      try {
        const text = await response.text();
        if (text) {
          detail = text.slice(0, 500);
        }
      } catch {
        // ignore – keep the status-only message
      }
    }
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
    await fetchJson("/api/health");
    return true;
  } catch {
    return false;
  }
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

export async function getCachePreview(options: {
  bits: number;
  fp16Layers: number;
  numLayers: number;
  numHeads: number;
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

export async function updateSession(sessionId: string, payload: UpdateSessionPayload): Promise<ChatSession> {
  const result = await patchJson<CreateSessionResponse>(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, payload);
  return result.session;
}

export async function generateChat(payload: GeneratePayload): Promise<GenerateResponse> {
  return await postJson<GenerateResponse>("/api/chat/generate", payload, 300000);
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onReasoning?: (reasoning: string) => void;
  onReasoningDone?: () => void;
  onDone: (response: GenerateResponse) => void;
  onError: (error: string) => void;
}

export async function generateChatStream(
  payload: GeneratePayload,
  callbacks: StreamCallbacks,
  abortSignal?: AbortController,
): Promise<void> {
  const apiBase = await resolveApiBase();
  const controller = abortSignal ?? new AbortController();
  const timer = setTimeout(() => controller.abort(), 300000);

  try {
    const response = await fetch(`${apiBase}/api/chat/generate/stream`, {
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
  const apiBase = await resolveApiBase();
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${apiBase}/api/chat/sessions/${encodeURIComponent(sessionId)}/documents`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    let detail = `Upload failed with status ${response.status}`;
    try {
      const err = await response.json();
      if (err?.detail) detail = String(err.detail);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  const result = await response.json();
  return result.document;
}

export async function listSessionDocuments(sessionId: string): Promise<SessionDocument[]> {
  const result = await fetchJson<{ documents: SessionDocument[] }>(`/api/chat/sessions/${encodeURIComponent(sessionId)}/documents`);
  return result.documents;
}

export async function deleteSessionDocument(sessionId: string, docId: string): Promise<void> {
  const apiBase = await resolveApiBase();
  await fetch(`${apiBase}/api/chat/sessions/${encodeURIComponent(sessionId)}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  const apiBase = await resolveApiBase();
  await fetch(`${apiBase}/api/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
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

export async function generateImage(payload: ImageGenerationPayload): Promise<ImageGenerationResponse> {
  return await postJson<ImageGenerationResponse>("/api/images/generate", payload, null);
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
  return info;
}
