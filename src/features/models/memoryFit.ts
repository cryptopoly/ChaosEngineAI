import type { ModelVariant } from "../../types";

/**
 * Phase 2.14: classify whether a variant fits the current host's
 * available memory. Three buckets: comfortable / tight / over.
 *
 * Superseded by the backend ``fit`` verdict (``FitBadge``); the chat
 * Models page uses it only when the backend sends no verdict (R14).
 *
 * - comfortable: estimated memory ≤ 70% of available
 * - tight: estimated memory ≤ 100% of available
 * - over: estimated memory > available
 *
 * Returns ``unknown`` when neither size nor estimate is known. The hint
 * is optimistic on purpose: KV-cache compression can reclaim part of the
 * listed estimate, so "tight" is still a usable signal rather than a hard
 * block.
 */
export function memoryFitBucket(
  variant: Pick<ModelVariant, "estimatedMemoryGb" | "sizeGb">,
  availableMemoryGb: number | null | undefined,
): { kind: "comfortable" | "tight" | "over" | "unknown"; label: string } {
  // ``label`` is the English fallback. The render site re-resolves the
  // localized label via ``t("onlineModels.memory.<kind>Label")`` so it
  // follows the active locale.
  if (availableMemoryGb == null || availableMemoryGb <= 0) {
    return { kind: "unknown", label: "" };
  }
  const estimate = variant.estimatedMemoryGb ?? variant.sizeGb;
  if (!estimate || estimate <= 0) {
    return { kind: "unknown", label: "" };
  }
  if (estimate <= availableMemoryGb * 0.7) {
    return { kind: "comfortable", label: "Fits" };
  }
  if (estimate <= availableMemoryGb) {
    return { kind: "tight", label: "Tight" };
  }
  return { kind: "over", label: "Too big" };
}
