/**
 * One row of a Models page, whatever the area.
 *
 * Chat, Image and Video each turn their own catalog / library data into
 * ``ModelEntry`` objects with an adapter (``adapters/image.ts`` today;
 * ``adapters/video.ts`` and ``adapters/chat.ts`` next). Everything the
 * shared parts need (status, filters, sort, card, download controls,
 * delete confirm, fit badge) reads only these fields; the area keeps the
 * original object in ``source`` for its own columns.
 */

import type { AcceleratorId } from "../acceleratorCatalog";
import type { Fit } from "./fit";

export type ModelsArea = "chat" | "image" | "video";

/** What is on disk / in flight for one row. ``loaded`` is not a state: it
 * is a flag on an installed row (``ResolvedModelStatus.loaded``). */
export type ModelInstallState =
  | "installed"
  | "available"
  | "downloading"
  | "paused"
  | "failed"
  | "partial"
  | "broken";

/** The four buckets of the availability filter. */
export type ModelStatusBucket = "installed" | "available" | "downloading" | "attention";

export type ModelAvailabilityFilter = "all" | ModelStatusBucket;

export const MODEL_AVAILABILITY_FILTERS: readonly ModelAvailabilityFilter[] = [
  "all",
  "installed",
  "available",
  "downloading",
  "attention",
];

export interface ModelSharedDownload {
  /** Other catalog rows that use the same download. */
  count: number;
  /** The download they share (repo id). */
  key: string;
  /** Display names of those rows, for the delete confirm and the tooltip. */
  siblingNames: string[];
}

export interface ModelMemoryEstimate {
  /** "~18 GB @ 1024×1024"; ``null`` while the size is unknown. */
  label: string | null;
  title?: string | null;
  /** Peak GB, for the RAM sort. */
  peakGb?: number | null;
}

export interface ModelEntry<TSource = unknown> {
  /** Stable row key (catalog variant id, or library path for chat rows
   * with no catalog match). */
  id: string;
  area: ModelsArea;
  name: string;
  provider: string;
  familyId?: string | null;
  familyName?: string | null;
  /** Long description; shown in the expandable detail, not the subtitle. */
  note?: string | null;
  /** Hugging Face repo id (may be empty for local-only chat rows). */
  repo: string;
  /** Model card URL. */
  link?: string | null;
  /** Key into the area's active-download map (repo for chat and image,
   * variant id first for video). */
  downloadKey: string;
  /** What the area's delete handler receives (repo for image, the
   * delete repo for video, a library path for chat). */
  deleteTarget: string;
  /** Validated as complete on disk by the backend. */
  availableLocally: boolean;
  /** Something is on disk, complete or not. */
  hasLocalData: boolean;
  /** Installed according to a runtime probe rather than the catalog
   * (LongLive). ``true`` promotes the row to installed; anything else
   * leaves the decision to the catalog fields. */
  installedByRuntime?: boolean | null;
  /** The runtime has this model in memory right now. */
  loaded?: boolean;
  /** Library scan says the files are broken (chat). */
  broken?: { reason: string } | null;
  localPath?: string | null;
  /** False when the row's engine cannot run on this machine. */
  hostCompatible: boolean;
  /** Tracked upstream only: nothing in the app can run it yet (FU-061). */
  trackedOnly?: boolean;
  gated?: boolean | null;
  license?: string | null;
  runtime?: string | null;
  tasks?: string[];
  styleTags?: string[];
  capabilities?: string[];
  /** Host-aware accelerators that apply to this row. */
  accelerators?: AcceleratorId[];
  sizeGb?: number | null;
  sizeLabel?: string | null;
  sizeTitle?: string | null;
  memory?: ModelMemoryEstimate | null;
  /** ISO-ish date used to sort by release; "" when unknown. */
  releaseSortKey: string;
  releaseLabel?: string | null;
  likes?: number | null;
  downloads?: number | null;
  likesLabel?: string | null;
  downloadsLabel?: string | null;
  fit?: Fit | null;
  /** The family's best fit for this machine. */
  bestFit?: boolean;
  shared?: ModelSharedDownload | null;
  /** The area's original object. */
  source: TSource;
}
