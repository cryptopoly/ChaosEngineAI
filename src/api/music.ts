/**
 * Music API client: catalog, runtime, estimate, progress, cancel, preload and unload, generate, the Music
 * Library, the music pack install, settings and weight downloads (`/api/music/*`).
 *
 * Errors keep the backend's plain `detail` sentence as the message (`MusicApiError`), so the Studio can show
 * it as it is. The audio route needs the bearer token, so a track is played through a blob URL
 * (`fetchMusicAudioBlob`), the same way Voice and the Storyboard do.
 */

import { apiFetch, readErrorDetail, updateSettings } from "./index";
import type { DownloadStatus, DeleteDownloadResult } from "./index";
import type {
  MusicCatalogResponse,
  MusicEstimateResponse,
  MusicGeneratePayload,
  MusicGenerateResponse,
  MusicInstallJob,
  MusicLibraryItem,
  MusicLibraryListResponse,
  MusicProgressSnapshot,
  MusicRuntimeStatus,
  MusicSettings,
  MusicSoundPayload,
} from "../types";

export class MusicApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "MusicApiError";
    this.status = status;
  }
}

async function request<T>(
  method: "GET" | "POST" | "DELETE",
  path: string,
  body?: object,
  timeoutMs: number | null = 60000,
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const timer = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : null;
  signal?.addEventListener("abort", () => controller.abort(), { once: true });
  try {
    const response = await apiFetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new MusicApiError(
        await readErrorDetail(response, `Request failed with status ${response.status}`),
        response.status,
      );
    }
    return (await response.json()) as T;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new MusicApiError(`Request to ${path} timed out or was stopped`, 0);
    }
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// Catalog, runtime, estimate, progress

export async function getMusicCatalog(): Promise<MusicCatalogResponse> {
  return await request<MusicCatalogResponse>("GET", "/api/music/catalog", undefined, 25000);
}

export async function getMusicRuntime(): Promise<MusicRuntimeStatus> {
  return (await request<{ runtime: MusicRuntimeStatus }>("GET", "/api/music/runtime", undefined, 20000)).runtime;
}

export async function estimateMusicRun(
  modelId: string | null,
  seconds: number,
  variations: 1 | 2 | 4,
  signal?: AbortSignal,
  /** Sound rows: the chosen diffusion steps; left out = the model's default. */
  steps?: number | null,
): Promise<MusicEstimateResponse> {
  const body = steps == null ? { modelId, seconds, variations } : { modelId, seconds, variations, steps };
  return await request<MusicEstimateResponse>("POST", "/api/music/estimate", body, 10000, signal);
}

export async function getMusicProgress(): Promise<MusicProgressSnapshot> {
  return (await request<{ progress: MusicProgressSnapshot }>("GET", "/api/music/progress", undefined, 5000)).progress;
}

export async function cancelMusicGeneration(): Promise<boolean> {
  return (await request<{ cancelled: boolean }>("POST", "/api/music/cancel", {}, 10000)).cancelled;
}

export async function preloadMusicModel(modelId: string | null): Promise<MusicRuntimeStatus> {
  return (await request<{ runtime: MusicRuntimeStatus }>("POST", "/api/music/preload", { modelId }, null)).runtime;
}

export async function unloadMusicModel(): Promise<MusicRuntimeStatus> {
  return (await request<{ runtime: MusicRuntimeStatus }>("POST", "/api/music/unload", {}, 30000)).runtime;
}

/** Makes the track(s); resolves when the run is finished (a long call), rejects with `detail` otherwise. */
export async function generateMusic(payload: MusicGeneratePayload): Promise<MusicGenerateResponse> {
  return await request<MusicGenerateResponse>("POST", "/api/music/generate", payload, 30 * 60 * 1000);
}

/** Makes the sound effect(s) of the Sounds mode; resolves when the run is finished. */
export async function generateSound(payload: MusicSoundPayload): Promise<MusicGenerateResponse> {
  return await request<MusicGenerateResponse>("POST", "/api/music/sound/generate", payload, 30 * 60 * 1000);
}

// The sounds pack (Stable Audio Open runs in its own environment)

export async function startSoundInstall(): Promise<MusicInstallJob> {
  return await request<MusicInstallJob>("POST", "/api/music/sound/install", {}, 20000);
}

export async function getSoundInstallStatus(): Promise<MusicInstallJob> {
  return await request<MusicInstallJob>("GET", "/api/music/sound/install/status", undefined, 10000);
}

export async function cancelSoundInstall(): Promise<MusicInstallJob> {
  return await request<MusicInstallJob>("POST", "/api/music/sound/install/cancel", {}, 10000);
}

// Library

export async function getMusicLibrary(query = "", tag = ""): Promise<MusicLibraryListResponse> {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (tag) params.set("tag", tag);
  const suffix = params.toString() ? `?${params}` : "";
  return await request<MusicLibraryListResponse>("GET", `/api/music/library${suffix}`, undefined, 20000);
}

export async function getMusicLibraryItem(id: string): Promise<MusicLibraryItem> {
  return await request<MusicLibraryItem>("GET", `/api/music/library/${encodeURIComponent(id)}`);
}

export async function updateMusicLibraryItem(
  id: string,
  patch: { title?: string; tags?: string[] },
): Promise<MusicLibraryItem> {
  return await request<MusicLibraryItem>("POST", `/api/music/library/${encodeURIComponent(id)}`, patch);
}

export async function deleteMusicLibraryItems(ids: string[]): Promise<string[]> {
  return (await request<{ deleted: string[] }>("POST", "/api/music/library/delete", { ids })).deleted;
}

export function musicAudioPath(id: string): string {
  return `/api/music/library/${encodeURIComponent(id)}/audio`;
}

/** A track's audio file as a Blob (the route needs the bearer token, so an `<audio src>` cannot reach it). */
export async function fetchMusicAudioBlob(id: string): Promise<Blob> {
  const response = await apiFetch(musicAudioPath(id));
  if (!response.ok) {
    throw new MusicApiError(await readErrorDetail(response, `Loading audio failed with status ${response.status}`), response.status);
  }
  return await response.blob();
}

// Install and settings

export async function startMusicInstall(): Promise<MusicInstallJob> {
  return await request<MusicInstallJob>("POST", "/api/music/install", {});
}

export async function getMusicInstallStatus(): Promise<MusicInstallJob> {
  return await request<MusicInstallJob>("GET", "/api/music/install/status", undefined, 15000);
}

export async function cancelMusicInstall(): Promise<MusicInstallJob> {
  return await request<MusicInstallJob>("POST", "/api/music/install/cancel", {});
}

export async function getMusicSettings(): Promise<MusicSettings> {
  return await request<MusicSettings>("GET", "/api/music/settings");
}

export async function saveMusicSettings(patch: { idleUnloadMinutes?: number }): Promise<MusicSettings> {
  return await request<MusicSettings>("POST", "/api/music/settings", patch);
}

/** The AI-tag switch (D-34) lives in the app settings (`musicEmbedTag`), not in the music settings file. */
export async function saveMusicEmbedTag(on: boolean): Promise<void> {
  await updateSettings({ musicEmbedTag: on });
}

// Weight downloads (the shared downloader, music repos only)

export async function downloadMusicModel(repo: string): Promise<DownloadStatus> {
  return (await request<{ download: DownloadStatus }>("POST", "/api/music/download", { repo })).download;
}

export async function getMusicDownloadStatus(): Promise<DownloadStatus[]> {
  return (await request<{ downloads: DownloadStatus[] }>("GET", "/api/music/download/status")).downloads;
}

export async function cancelMusicDownload(repo: string): Promise<DownloadStatus> {
  return (await request<{ download: DownloadStatus }>("POST", "/api/music/download/cancel", { repo })).download;
}

export async function deleteMusicDownload(repo: string): Promise<DeleteDownloadResult> {
  return (await request<{ result: DeleteDownloadResult }>("POST", "/api/music/download/delete", { repo })).result;
}
