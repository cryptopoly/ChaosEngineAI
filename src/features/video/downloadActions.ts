/**
 * Video download lifecycle actions.
 *
 * Pulled out of ``useVideoState`` so the hook stays focused on
 * orchestration. Each handler takes its dependencies as kwargs (no
 * closure over hook locals) which keeps them testable in isolation.
 *
 * Also re-homes two pure helpers (``pickVideoDownloadStatus``,
 * ``buildVariantAwareDownloadMap``) used to merge the backend's
 * per-repo download statuses into the variant-keyed map the UI reads.
 *
 * Extracted as part of the v0.8.0 Phase 2c-6 refactor.
 */

import {
  cancelVideoDownload,
  deleteVideoDownload,
  downloadVideoModel,
  getVideoDownloadStatus,
} from "../../api";
import type { DownloadStatus } from "../../api";
import {
  buildDownloadStatusMap,
  failedDownloadStatus,
  isTransientNetworkError,
  pendingDownloadStatus,
  videoDownloadRepos,
} from "../../utils";
import type { VideoModelVariant } from "../../types";


type SetActiveDownloads = (
  updater: (
    prev: Record<string, DownloadStatus>,
  ) => Record<string, DownloadStatus>,
) => void;


export function pickVideoDownloadStatus(
  statuses: DownloadStatus[],
): DownloadStatus | undefined {
  return (
    statuses.find((status) => status.state === "downloading")
    ?? statuses.find((status) => status.state === "failed")
    ?? statuses.find((status) => status.state === "cancelled")
    ?? statuses.find((status) => status.state === "completed")
  );
}


export function buildVariantAwareDownloadMap(
  statuses: DownloadStatus[],
  previous: Record<string, DownloadStatus>,
  knownVideoDownloadVariants: VideoModelVariant[],
): Record<string, DownloadStatus> {
  const repoMap = buildDownloadStatusMap(statuses);
  const next: Record<string, DownloadStatus> = { ...repoMap };
  for (const variant of knownVideoDownloadVariants) {
    if (!previous[variant.id]) continue;
    const variantStatuses = videoDownloadRepos(variant)
      .map((repo) => repoMap[repo])
      .filter((status): status is DownloadStatus => Boolean(status));
    next[variant.id] = pickVideoDownloadStatus(variantStatuses) ?? previous[variant.id];
  }
  return next;
}


export async function handleVideoDownload(
  repo: string,
  modelId: string | undefined,
  deps: {
    setActiveVideoDownloads: SetActiveDownloads;
    setError: (msg: string | null) => void;
    refreshVideoData: () => Promise<unknown>;
  },
): Promise<void> {
  const { setActiveVideoDownloads, setError, refreshVideoData } = deps;
  const activeKey = modelId ?? repo;
  try {
    setActiveVideoDownloads((prev) => ({
      ...prev,
      [activeKey]: pendingDownloadStatus(repo, prev[activeKey] ?? prev[repo]),
    }));
    const download = await downloadVideoModel(repo, modelId);
    setActiveVideoDownloads((prev) => ({
      ...prev,
      [activeKey]: download,
      [download.repo]: download,
    }));
    void refreshVideoData();
  } catch (err) {
    if (isTransientNetworkError(err)) {
      setError("Backend is restarting or temporarily unreachable. Try the download again when it is online.");
      setActiveVideoDownloads((prev) => {
        const next = { ...prev };
        delete next[activeKey];
        return next;
      });
      return;
    }
    setError(err instanceof Error ? err.message : "Video download failed");
    setActiveVideoDownloads((prev) => ({
      ...prev,
      [activeKey]: failedDownloadStatus(repo, String(err)),
    }));
  }
}


export async function handleCancelVideoDownload(
  repo: string,
  deps: {
    setActiveVideoDownloads: SetActiveDownloads;
    setError: (msg: string | null) => void;
  },
): Promise<void> {
  const { setActiveVideoDownloads, setError } = deps;
  try {
    const download = await cancelVideoDownload(repo);
    setActiveVideoDownloads((prev) => ({ ...prev, [repo]: download }));
  } catch (err) {
    setError(err instanceof Error ? err.message : "Could not pause video download");
  }
}


export async function handleDeleteVideoDownload(
  repo: string,
  deps: {
    setActiveVideoDownloads: (next: Record<string, DownloadStatus>) => void;
    setError: (msg: string | null) => void;
    refreshVideoData: () => Promise<unknown>;
  },
): Promise<void> {
  const { setActiveVideoDownloads, setError, refreshVideoData } = deps;
  try {
    await deleteVideoDownload(repo);
    const statuses = await getVideoDownloadStatus();
    setActiveVideoDownloads(buildDownloadStatusMap(statuses));
    await refreshVideoData();
  } catch (err) {
    setError(err instanceof Error ? err.message : "Could not delete video download");
  }
}
