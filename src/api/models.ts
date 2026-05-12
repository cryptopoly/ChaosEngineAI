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

import { fetchJson, postJson } from "./index";
import type {
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

export async function runBenchmark(
  payload: BenchmarkRunPayload,
  options?: { signal?: AbortSignal },
): Promise<BenchmarkRunResponse> {
  // No client-side timeout — a benchmark on a cold 70B model legitimately
  // takes >120s (cold load + prompt processing + N-token generation +
  // measurement). The backend has its own per-phase ceilings.
  // `options.signal` enables the Cancel button on the running-benchmark
  // modal to abort the in-flight request without waiting for the backend
  // to finish.
  return await postJson<BenchmarkRunResponse>("/api/benchmarks/run", payload, null, options?.signal);
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
