/**
 * Models / search / downloads / lifecycle API endpoints.
 *
 * Search (catalog + HF hub passthrough), load / unload of the active
 * inference engine, text-model download cluster, conversion, benchmark
 * runs, library admin (reveal in Finder, delete from disk, list hub
 * files for a given repo).
 *
 * Re-exported from ``./index`` so existing
 * ``import { loadModel, downloadModel } from "../api"`` paths keep
 * working.
 *
 * Extracted from ``api.ts`` as part of the v0.8.0 refactor.
 */

import { deleteJson, fetchJson, postJson } from "./index";
import type {
  GenerationProgressSnapshot,
  BenchmarkResult,
  BenchmarkRunPayload,
  BenchmarkRunResponse,
  ConvertModelPayload,
  ConvertModelResponse,
  HubFileListResponse,
  HubModel,
  LibraryItem,
  LoadModelPayload,
  ModelFamily,
  RuntimeStatus,
} from "../types";

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

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

export async function stopModelProcess(pid: number): Promise<{ pid: number; action: string }> {
  return await postJson<{ pid: number; action: string }>("/api/models/processes/stop", { pid });
}

export async function stopIdleModelProcesses(): Promise<{ warmEvicted: number; orphansStopped: number }> {
  return await postJson<{ warmEvicted: number; orphansStopped: number }>("/api/models/processes/stop-idle");
}

// ---------------------------------------------------------------------------
// Downloads (text models — image / video have their own clusters)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Conversion + benchmarks + library admin
// ---------------------------------------------------------------------------

export async function convertModel(payload: ConvertModelPayload): Promise<ConvertModelResponse> {
  // No client-side timeout — conversion can legitimately take 10+ min for
  // large models on a cold cache. Backend has its own 3600s subprocess cap.
  return await postJson<ConvertModelResponse>("/api/models/convert", payload, null);
}

/** Ask the backend to stop the conversion in progress; the waiting convert request then fails with "Conversion cancelled". */
export async function cancelConversion(): Promise<{ cancelled: boolean }> {
  return await postJson<{ cancelled: boolean }>("/api/models/convert/cancel", undefined, 30000);
}

export async function runBenchmark(
  payload: BenchmarkRunPayload,
  options?: { signal?: AbortSignal },
): Promise<BenchmarkRunResponse> {
  // No client-side timeout — a benchmark on a cold 70B model legitimately
  // takes >120s (cold load + prompt processing + N-token generation +
  // measurement). The backend has its own per-phase ceilings.
  // Cancel goes through `cancelBenchmark`; this request then answers
  // `cancelled: true`. `options.signal` only stops the wait when the
  // backend cannot be reached to confirm the cancel.
  return await postJson<BenchmarkRunResponse>("/api/benchmarks/run", payload, null, options?.signal);
}

/** What the running benchmark is doing (phase, items done of total). ``active`` is false when none runs. */
export async function getBenchmarkProgress(): Promise<GenerationProgressSnapshot> {
  const result = await fetchJson<{ progress: GenerationProgressSnapshot }>("/api/benchmarks/progress", 5000);
  return result.progress;
}

/** What the running conversion is doing (phase, files downloaded of total, bytes written). */
export async function getConversionProgress(): Promise<GenerationProgressSnapshot> {
  const result = await fetchJson<{ progress: GenerationProgressSnapshot }>("/api/models/convert/progress", 5000);
  return result.progress;
}

/** Ask the backend to stop the benchmark in progress; the waiting run request then answers ``cancelled: true``. */
export async function cancelBenchmark(): Promise<{ cancelled: boolean }> {
  return await postJson<{ cancelled: boolean }>("/api/benchmarks/cancel", undefined, 60000);
}

/** Remove one saved run from the benchmark history; answers with the runs that are left. */
export async function deleteBenchmark(runId: string): Promise<{ deleted: string; benchmarks: BenchmarkResult[] }> {
  return await deleteJson<{ deleted: string; benchmarks: BenchmarkResult[] }>(
    `/api/benchmarks/${encodeURIComponent(runId)}`,
    undefined,
    30000,
  );
}

export async function revealModelPath(path: string): Promise<void> {
  await postJson<{ revealed: string }>("/api/models/reveal", { path });
}

export async function openMediaFile(path: string): Promise<void> {
  await postJson<{ opened: string }>("/api/files/open", { path });
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

// ---------------------------------------------------------------------------
// Run any Hugging Face model (#5)
// ---------------------------------------------------------------------------

export interface ResolvedHfModel {
  repo: string;
  ref: string;
  label: string;
  backend: "llama.cpp" | "mlx" | "vllm" | "unknown";
  ggufFile: string | null;
  contextTokens: number;
  capabilities: { text: boolean; vision: boolean };
  sizeBytes: number;
  totalSizeGb: number;
  family: string;
  custom: boolean;
  warnings: string[];
}

export async function resolveHfModel(repo: string, file?: string): Promise<ResolvedHfModel> {
  const result = await postJson<{ resolved: ResolvedHfModel }>(
    "/api/models/resolve-hf",
    { repo, file: file ?? null },
    20000,
  );
  return result.resolved;
}

// ---------------------------------------------------------------------------
// Import models already on disk from another local-AI app, no re-download (#4)
// ---------------------------------------------------------------------------

export interface ImportableModel {
  name: string;
  repo: string;
  source: "ollama" | "lmstudio";
  path: string;
  sizeBytes: number;
  sizeGb: number;
  format: string;
}

export interface ImportScanResult {
  ollama: { available: boolean; dir: string | null; models: ImportableModel[] };
  lmstudio: { available: boolean; dirs: string[]; models: ImportableModel[] };
}

export async function scanImportableModels(): Promise<ImportScanResult> {
  return await fetchJson<ImportScanResult>("/api/models/import/scan", 20000);
}

export async function importModel(payload: {
  source: string;
  path: string;
  name: string;
  repo?: string;
}): Promise<{ imported: { importedPath: string; alreadyImported: boolean }; repo: string; name: string; source: string }> {
  return await postJson("/api/models/import", payload, 30000);
}
