import type { AppSettings, RuntimeStatus, TauriBackendInfo, WorkspaceData } from "../types";
import type { SettingsDraft } from "../types/chat";

export function serverOriginFromBase(baseUrl: string) {
  return baseUrl.replace(/\/v1\/?$/, "");
}

// Background refreshes can fire before the Tauri sidecar has bound its
// port (cold start on Windows is especially slow: extract embedded
// runtime, spawn Python, import FastAPI + torch). A transient fetch
// failure in that window is not a user-facing error — callers should
// swallow it and rely on the next poll/interaction to recover.
export function isTransientNetworkError(err: unknown): boolean {
  if (err instanceof TypeError) {
    // Browser-level fetch failures ("Failed to fetch", "Network request
    // failed", "NetworkError when attempting to fetch resource", etc.)
    // all surface as TypeError.
    return true;
  }
  if (err instanceof Error) {
    const msg = err.message.toLowerCase();
    if (msg.includes("failed to fetch")) return true;
    if (msg.includes("timed out after")) return true;
    if (msg.includes("networkerror")) return true;
    if (msg.includes("load failed")) return true; // Safari
  }
  return false;
}

export function serverBaseFromOrigin(origin: string) {
  return `${origin.replace(/\/$/, "")}/v1`;
}

export function syncRuntime(current: WorkspaceData, runtime: RuntimeStatus): WorkspaceData {
  return {
    ...current,
    runtime,
    server: {
      ...current.server,
      status: runtime.serverReady ? "running" : "idle",
      activeConnections: runtime.activeRequests,
      concurrentRequests: runtime.activeRequests,
      requestsServed: runtime.requestsServed,
      loadedModelName: runtime.loadedModel?.name ?? null,
      loading: runtime.loadedModel ? null : current.server.loading,
    },
  };
}

export function syncStoppedBackend(current: WorkspaceData, runtimeInfo: TauriBackendInfo | null): WorkspaceData {
  const origin = runtimeInfo?.apiBase ?? serverOriginFromBase(current.server.localhostUrl ?? current.server.baseUrl);
  const localhostUrl = serverBaseFromOrigin(origin);
  return {
    ...current,
    runtime: {
      ...current.runtime,
      state: "idle",
      loadedModel: null,
      supportsGeneration: false,
      serverReady: false,
      activeRequests: 0,
      runtimeNote: "The local API service is stopped.",
    },
    server: {
      ...current.server,
      status: "idle",
      baseUrl: localhostUrl,
      localhostUrl,
      activeConnections: 0,
      concurrentRequests: 0,
      loadedModelName: null,
      loading: null,
      remoteAccessActive: false,
      logTail: [
        "API service stopped.",
        "Use Restart to bring the local API back online.",
        ...(runtimeInfo?.startupError ? [runtimeInfo.startupError] : []),
      ].slice(0, 3),
    },
  };
}

export function settingsDraftFromWorkspace(settings: AppSettings): SettingsDraft {
  return {
    modelDirectories: settings?.modelDirectories ?? [],
    preferredServerPort: settings?.preferredServerPort ?? 8876,
    allowRemoteConnections: settings?.allowRemoteConnections ?? false,
    requireApiAuth: settings?.requireApiAuth ?? true,
    autoStartServer: settings?.autoStartServer ?? false,
    remoteProviders: settings?.remoteProviders ?? [],
    huggingFaceToken: "",
    hasHuggingFaceToken: settings?.hasHuggingFaceToken ?? false,
    huggingFaceTokenMasked: settings?.huggingFaceToken ?? "",
    dataDirectory: settings?.dataDirectory ?? "",
    imageOutputsDirectory: settings?.imageOutputsDirectory ?? "",
    videoOutputsDirectory: settings?.videoOutputsDirectory ?? "",
  };
}
