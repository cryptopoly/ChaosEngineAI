/**
 * "Will this model run on this machine?" as sent by the backend
 * (``backend_service/helpers/fit.py``): a ``fit`` object on every chat,
 * image and video catalog variant and a ``bestFitVariantId`` on every
 * family.
 *
 * The real payload types live in ``src/types/models.ts`` (``ModelFit``);
 * this file keeps a structurally identical copy so the Models page can be
 * built before the catalog types carry the field. Every read is defensive:
 * an older backend sends nothing, and a malformed object is ignored rather
 * than rendered.
 */

export type FitStatus = "fits" | "tight" | "wont-fit" | "unknown";

export type FitBasis = "measured" | "estimated";

export interface Fit {
  /** The verdict. Read this, never the sign of ``headroomPct``: a
   * ``wont-fit`` model between 92 % and 100 % of usable memory still has
   * a small positive headroom. ``unknown`` means the row cannot run on
   * this kind of hardware at all (``reason`` says why). */
  status: FitStatus;
  /** Memory the model needs, GB. ``null`` when ``status`` is ``unknown``. */
  needGb: number | null;
  /** Memory this machine can give a model, GB. */
  usableGb: number;
  /** Share of usable memory left after the model, negative when it needs
   * more than the machine has. ``null`` when ``status`` is ``unknown``. */
  headroomPct: number | null;
  /** One line that explains the number ("Weights 15.9 GB + 2.5 GB cache"). */
  reason: string;
  /** ``measured`` only when the number comes from a run on this machine. */
  basis: FitBasis;
  /** Set by an area's adapter when ``unknown`` means "gated, no access yet", not "cannot run here". */
  accessBlocked?: boolean;
}

const FIT_STATUSES: readonly FitStatus[] = ["fits", "tight", "wont-fit", "unknown"];

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Parse an unknown value into a ``Fit``; ``null`` when it is missing or
 * not shaped like one. */
export function readFit(value: unknown): Fit | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const status = raw.status;
  if (typeof status !== "string" || !(FIT_STATUSES as readonly string[]).includes(status)) return null;
  const usableGb = finiteOrNull(raw.usableGb);
  return {
    status: status as FitStatus,
    needGb: finiteOrNull(raw.needGb),
    usableGb: usableGb ?? 0,
    headroomPct: finiteOrNull(raw.headroomPct),
    reason: typeof raw.reason === "string" ? raw.reason : "",
    basis: raw.basis === "measured" ? "measured" : "estimated",
  };
}

/** The ``fit`` field of a catalog variant, read defensively. */
export function readVariantFit(variant: unknown): Fit | null {
  if (!variant || typeof variant !== "object") return null;
  return readFit((variant as { fit?: unknown }).fit);
}

/**
 * The memory estimate shown beside the fit chip. When the backend gave a verdict with a number, that number
 * replaces the page's own heuristic, so "~67 GB" can never sit beside "Fits" on a 24 GB card: the chip is
 * judged on ``needGb`` (measured where a measurement exists), and the column must say the same figure.
 */
export function memoryAgreeingWithFit<T extends { label: string | null; title?: string | null; peakGb?: number | null }>(
  memory: T | null,
  fit: Fit | null,
): T | null {
  if (!memory || !memory.label || !fit || fit.status === "unknown" || fit.needGb === null || !(fit.needGb > 0)) return memory;
  const figure = fit.needGb >= 10 ? fit.needGb.toFixed(0) : fit.needGb.toFixed(1);
  const [, resolution] = memory.label.split(" @ ");
  return {
    ...memory,
    label: resolution ? `~${figure} GB @ ${resolution}` : `~${figure} GB`,
    title: fit.reason || memory.title,
    peakGb: fit.needGb,
  };
}

/** The family's ``bestFitVariantId``, or ``null``. */
export function readBestFitVariantId(family: unknown): string | null {
  if (!family || typeof family !== "object") return null;
  const value = (family as { bestFitVariantId?: unknown }).bestFitVariantId;
  return typeof value === "string" && value.length > 0 ? value : null;
}

/**
 * FU-472, one verdict per row: a Studio "may struggle" (caution) warning is dropped when the backend's verdict
 * for this row on this machine is ``fits`` or ``tight`` and the run is no bigger than the row's recommended
 * size (and, for video, its default clip), which is what the verdict judged. FLUX 3 Klein and LTX-2.3 were
 * measured working on a 24 GB card and marked Tight on Models while the Studio said they "may fail". A larger
 * run, or a danger-level estimate, keeps its warning.
 */
export function cautionCoveredByFit(
  fit: Fit | null | undefined,
  run: { width: number; height: number; frames?: number | null },
  row: { recommendedResolution?: string | null; defaultFrames?: number | null },
): boolean {
  if (!fit || (fit.status !== "fits" && fit.status !== "tight")) return false;
  const match = /(\d{3,5})\s*[x×]\s*(\d{3,5})/i.exec(row.recommendedResolution ?? "");
  if (!match) return false;
  const recommendedPixels = Number(match[1]) * Number(match[2]);
  if (!(run.width > 0 && run.height > 0) || run.width * run.height > recommendedPixels) return false;
  if (row.defaultFrames != null && run.frames != null && run.frames > row.defaultFrames) return false;
  return true;
}

/** Sort rank for a fit verdict: fits first, then tight, then unknown,
 * then won't fit; rows without a verdict last. */
export function fitSortRank(fit: Fit | null | undefined): number {
  if (!fit) return 4;
  if (fit.status === "fits") return 0;
  if (fit.status === "tight") return 1;
  if (fit.status === "unknown") return 2;
  return 3;
}
