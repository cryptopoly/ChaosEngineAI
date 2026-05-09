/**
 * Image-domain API endpoints.
 *
 * Catalog / runtime / outputs / progress polling, downloads, preload +
 * unload, generate + cancel + delete-output. All thirteen functions
 * proxy to ``/api/images/*`` on the Python backend; the shared
 * ``apiFetch`` / ``fetchJson`` / ``postJson`` / ``deleteJson`` helpers
 * are imported from the package facade in ``./index``.
 *
 * Re-exported from ``backend_service/api/index.ts`` so existing
 * ``import { generateImage } from "../api"`` paths keep working.
 *
 * Extracted from ``api.ts`` as part of the v0.8.0 refactor.
 */

import { deleteJson, fetchJson, postJson } from "./index";
import type { DeleteDownloadResult, DownloadStatus } from "./index";
import type {
  GenerationProgressSnapshot,
  ImageCatalogResponse,
  ImageGenerationPayload,
  ImageGenerationResponse,
  ImageOutputArtifact,
  ImageRuntimeStatus,
} from "../types";

// ---------------------------------------------------------------------------
// Catalog + runtime + outputs
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Preload / unload / generate
// ---------------------------------------------------------------------------

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

export async function cancelImageGeneration(): Promise<{ cancelled: boolean }> {
  return await postJson<{ cancelled: boolean }>("/api/images/cancel", {}, 10000);
}

export async function deleteImageOutput(artifactId: string): Promise<{ deleted: string; outputs: ImageOutputArtifact[] }> {
  return await deleteJson<{ deleted: string; outputs: ImageOutputArtifact[] }>(
    `/api/images/outputs/${encodeURIComponent(artifactId)}`,
  );
}
