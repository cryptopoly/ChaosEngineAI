/**
 * One status ladder for every Models page.
 *
 * Replaces the per-tab ladders (``imageVariantStatus``, ``imageStatus``,
 * ``videoVariantStatus``, ``videoStatus`` and the inline chat ones) so a
 * row shows the same status, buttons and filter bucket wherever it is.
 *
 * Order, first match wins:
 *   1. an active transfer: downloading (or a runtime install in progress),
 *      then paused. A running or paused transfer needs its Pause / Resume
 *      controls even when files are already on disk.
 *   2. installed: the backend validated the files, the download completed,
 *      or a runtime probe says the model is ready (LongLive).
 *   3. failed download, then broken library files, then partial local
 *      data: "needs attention".
 *   4. available: nothing on disk.
 */

import type { DownloadStatus } from "../../api";
import type { ModelStatusKind } from "../ModelActionIcons";
import type {
  ModelAvailabilityFilter,
  ModelEntry,
  ModelInstallState,
  ModelStatusBucket,
} from "./modelEntry";

export interface ModelStatusContext {
  /** The row's entry in the area's active-download map. */
  download?: DownloadStatus | null;
  /** A runtime install is running for this row (LongLive). */
  installing?: boolean;
}

export interface ResolvedModelStatus {
  state: ModelInstallState;
  bucket: ModelStatusBucket;
  /** Icon for ``StatusIcon``. */
  chip: ModelStatusKind;
  /** Installed and in memory right now. */
  loaded: boolean;
  /** The download the status came from, if any. */
  download: DownloadStatus | null;
}

const BUCKET_OF: Record<ModelInstallState, ModelStatusBucket> = {
  installed: "installed",
  available: "available",
  downloading: "downloading",
  paused: "downloading",
  failed: "attention",
  partial: "attention",
  broken: "attention",
};

const CHIP_OF: Record<ModelInstallState, ModelStatusKind> = {
  installed: "installed",
  available: "incomplete",
  downloading: "downloading",
  paused: "paused",
  failed: "failed",
  partial: "incomplete",
  broken: "failed",
};

function resolveState(
  entry: Pick<ModelEntry, "availableLocally" | "hasLocalData" | "installedByRuntime" | "broken">,
  ctx: ModelStatusContext,
): ModelInstallState {
  const download = ctx.download ?? null;
  if (ctx.installing || download?.state === "downloading") return "downloading";
  if (download?.state === "cancelled") return "paused";
  if (entry.installedByRuntime === true || entry.availableLocally || download?.state === "completed") return "installed";
  if (download?.state === "failed") return "failed";
  if (entry.broken) return "broken";
  if (entry.hasLocalData) return "partial";
  return "available";
}

export function resolveModelStatus(
  entry: Pick<ModelEntry, "availableLocally" | "hasLocalData" | "installedByRuntime" | "broken" | "loaded">,
  ctx: ModelStatusContext = {},
): ResolvedModelStatus {
  const state = resolveState(entry, ctx);
  const loaded = state === "installed" && Boolean(entry.loaded);
  return {
    state,
    bucket: BUCKET_OF[state],
    chip: loaded ? "loaded" : CHIP_OF[state],
    loaded,
    download: ctx.download ?? null,
  };
}

export function bucketForState(state: ModelInstallState): ModelStatusBucket {
  return BUCKET_OF[state];
}

export function matchesAvailability(bucket: ModelStatusBucket, filter: ModelAvailabilityFilter): boolean {
  return filter === "all" || bucket === filter;
}

export type ModelBucketCounts = Record<ModelAvailabilityFilter, number>;

export function countBuckets(statuses: Iterable<ResolvedModelStatus>): ModelBucketCounts {
  const counts: ModelBucketCounts = { all: 0, installed: 0, available: 0, downloading: 0, attention: 0 };
  for (const status of statuses) {
    counts.all += 1;
    counts[status.bucket] += 1;
  }
  return counts;
}

/** Sort rank for the Status column: installed first, then transfers,
 * then rows that need attention, then available ones. */
export function statusSortRank(status: ResolvedModelStatus): number {
  switch (status.state) {
    case "installed":
      return status.loaded ? 0 : 1;
    case "downloading":
      return 2;
    case "paused":
      return 3;
    case "failed":
      return 4;
    case "broken":
      return 5;
    case "partial":
      return 6;
    default:
      return 7;
  }
}

/** A row the host cannot run is hidden unless something of it is on disk
 * or in flight: an installed (or partial) model stays visible, marked
 * "Cannot run on this machine", so it can still be deleted. */
export function isVisibleOnHost(entry: Pick<ModelEntry, "hostCompatible">, status: ResolvedModelStatus): boolean {
  return entry.hostCompatible || status.bucket !== "available";
}
