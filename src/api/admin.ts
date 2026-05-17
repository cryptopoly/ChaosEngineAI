/**
 * Admin / diagnostics / storage API endpoints.
 *
 * Two clusters surfaced in Settings:
 *
 * - **Diagnostics** — structured snapshot dump (OS, hardware, runtime
 *   paths, GPU state, env vars, log tail) for one-click clipboard
 *   sharing in support threads. Plus the runtime re-extraction action.
 * - **Storage** — HF cache path config + the background move job that
 *   relocates ``~/.cache/huggingface`` to a different drive.
 *
 * Re-exported from ``./index`` so existing
 * ``import { fetchDiagnosticsSnapshot } from "../api"`` paths keep
 * working.
 *
 * Extracted from ``api.ts`` as part of the v0.8.0 refactor.
 */

import { fetchJson, postJson } from "./index";

// ---------------------------------------------------------------------------
// Diagnostics snapshot
// ---------------------------------------------------------------------------

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

export interface StorageTopEntry {
  path: string;
  repoLabel: string;
  sizeBytes: number;
  sizeGb: number | null;
  sourceKind: string;
  lastModified: number;
}

export interface StorageTopResponse {
  entries: StorageTopEntry[];
  totalBytes: number;
  totalGb: number | null;
  scannedDirectories: string[];
}

export async function fetchStorageTop(limit = 20): Promise<StorageTopResponse> {
  // FU-055: walks every enabled model dir → can take a few seconds on big
  // HF caches (997 GB across 14 repos on the dev box ≈ 4 s with cycle
  // protection). 30 s timeout is enough headroom for a TB-scale scan.
  return await fetchJson<StorageTopResponse>(
    `/api/diagnostics/storage-top?limit=${encodeURIComponent(limit)}`,
    30000,
  );
}

// ---------------------------------------------------------------------------
// HF cache storage settings + background model-move job
// ---------------------------------------------------------------------------

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
