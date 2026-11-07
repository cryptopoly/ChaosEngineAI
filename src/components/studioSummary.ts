/**
 * Small pure helpers the Studios share for the value summaries on their
 * Options / Expert disclosures and for the basis chip of the run estimate.
 */
import type { EstimateBasis } from "./RunEstimateStrip";
import type { RecommendationRow } from "./recommendedSettings";

export const SUMMARY_SEPARATOR = " · ";

/** Joins the parts that are set; `fallback` (e.g. "Model defaults") when none is. */
export function summaryLine(parts: Array<string | false | null | undefined>, fallback?: string): string | undefined {
  const set = parts.filter((part): part is string => typeof part === "string" && part.length > 0);
  return set.length > 0 ? set.join(SUMMARY_SEPARATOR) : fallback;
}

/**
 * Basis chip for the run estimate, from the same rows the recommended-settings
 * panel shows: measured when any row was timed on a machine like this one,
 * estimated when there are rows or a time estimate, none when nothing is known.
 */
export function runEstimateBasis(rows: readonly RecommendationRow[], hasSeconds: boolean): EstimateBasis | null {
  if (rows.some((row) => row.basis === "measured")) return "measured";
  if (rows.length > 0 || hasSeconds) return "estimated";
  return null;
}
