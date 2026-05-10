/**
 * Image-download lifecycle handlers.
 *
 * Three small async helpers pulled out of ``useImageState`` so the hook
 * can stay focused on selection/runtime/generation state. Each takes the
 * setters it needs as parameters and runs the API call inside a
 * try/catch, surfacing failures through ``setError``:
 *
 * * ``startImageDownload`` — POST ``/api/images/download``, optimistically
 *   marks the row as pending so the UI button switches to "Downloading..."
 *   before the backend's first status poll lands.
 * * ``cancelImageDownloadById`` — POST cancel; backend returns the new
 *   ``DownloadStatus`` row which we splice into ``activeImageDownloads``.
 * * ``deleteImageDownloadById`` — DELETE the cached weights, then refetch
 *   the download statuses + image data so any catalogue badges flip
 *   from ``"Installed"`` back to ``"Available"``.
 *
 * Extracted from ``src/hooks/useImageState.ts`` as part of the v0.8.0
 * Phase 2c-5 refactor. The hook keeps thin one-liner wrappers that
 * close over the setters so the call-sites stay readable.
 */

import {
  cancelImageDownload,
  deleteImageDownload,
  downloadImageModel,
  getImageDownloadStatus,
} from "../../api";
import type { DownloadStatus } from "../../api";
import {
  buildDownloadStatusMap,
  failedDownloadStatus,
  pendingDownloadStatus,
} from "../../utils";

type DownloadMap = Record<string, DownloadStatus>;
type DownloadMapSetter = (
  updater: (prev: DownloadMap) => DownloadMap,
) => void;
type DownloadMapDirectSetter = (next: DownloadMap) => void;
type ErrorSetter = (msg: string | null) => void;

interface DownloadDeps {
  setActiveImageDownloads: DownloadMapSetter & DownloadMapDirectSetter;
  setError: ErrorSetter;
  refreshImageData: () => Promise<void> | void;
}

export async function startImageDownload(
  repo: string,
  deps: DownloadDeps,
): Promise<void> {
  const { setActiveImageDownloads, setError, refreshImageData } = deps;
  try {
    setActiveImageDownloads((prev) => ({
      ...prev,
      [repo]: pendingDownloadStatus(repo, prev[repo]),
    }));
    const download = await downloadImageModel(repo);
    setActiveImageDownloads((prev) => ({ ...prev, [repo]: download }));
    void refreshImageData();
  } catch (err) {
    setError(err instanceof Error ? err.message : "Image download failed");
    setActiveImageDownloads((prev) => ({
      ...prev,
      [repo]: failedDownloadStatus(repo, String(err)),
    }));
  }
}

export async function cancelImageDownloadById(
  repo: string,
  deps: Pick<DownloadDeps, "setActiveImageDownloads" | "setError">,
): Promise<void> {
  const { setActiveImageDownloads, setError } = deps;
  try {
    const download = await cancelImageDownload(repo);
    setActiveImageDownloads((prev) => ({ ...prev, [repo]: download }));
  } catch (err) {
    setError(err instanceof Error ? err.message : "Could not pause image download");
  }
}

export async function deleteImageDownloadById(
  repo: string,
  deps: DownloadDeps,
): Promise<void> {
  const { setActiveImageDownloads, setError, refreshImageData } = deps;
  try {
    await deleteImageDownload(repo);
    const statuses = await getImageDownloadStatus();
    setActiveImageDownloads(buildDownloadStatusMap(statuses));
    await refreshImageData();
  } catch (err) {
    setError(err instanceof Error ? err.message : "Could not delete image download");
  }
}
