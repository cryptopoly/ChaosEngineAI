/**
 * Video-domain API endpoints.
 *
 * Mirrors ``./image`` for the video runtime: catalog, three runtime
 * probes (diffusers / LongLive / mlx-video), progress polling,
 * downloads, preload + unload, generate + cancel, outputs CRUD +
 * blob-URL fetcher for the saved mp4 viewer.
 *
 * Re-exported from ``./index`` so existing
 * ``import { generateVideo } from "../api"`` paths keep working.
 *
 * Extracted from ``api.ts`` as part of the v0.8.0 refactor.
 */

import { apiFetch, deleteJson, fetchJson, postJson } from "./index";
import type { DeleteDownloadResult, DownloadStatus } from "./index";
import type {
  GenerationProgressSnapshot,
  VideoCatalogResponse,
  VideoGenerationPayload,
  VideoGenerationResponse,
  VideoOutputArtifact,
  VideoRuntimeStatus,
} from "../types";

// ---------------------------------------------------------------------------
// Catalog + runtime probes
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Preload / unload / generate / outputs
// ---------------------------------------------------------------------------

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
