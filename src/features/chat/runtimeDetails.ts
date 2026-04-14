import type { GenerationMetrics } from "../../types";

export const LEGACY_MESSAGE_VALUE = "Unknown (older turn)";

function parseTreeBudget(runtimeNote: string | null | undefined): number | null {
  const match = (runtimeNote ?? "").match(/budget=(\d+)/i);
  return match ? parseInt(match[1], 10) : null;
}

function parseDraftModel(runtimeNote: string | null | undefined): string | null {
  const match = (runtimeNote ?? "").match(/draft:\s*([^\s).,]+)/i);
  return match ? match[1] : null;
}

function noteHasNoCompatibleDraft(runtimeNote: string | null | undefined): boolean {
  const note = runtimeNote ?? "";
  return (
    /requested but no compatible draft model found/i.test(note)
    || /no compatible draft model (?:is )?(?:registered|found)/i.test(note)
    || (/dflash unavailable/i.test(note) && /draft model/i.test(note))
  );
}

function cacheFellBackToNative(metrics: GenerationMetrics): boolean {
  return /fell back to native f16 cache/i.test(metrics.runtimeNote ?? "");
}

function parseLegacyCacheQuant(cacheLabel: string | null | undefined): { bits: number | null; fp16Layers: number | null } {
  const match = (cacheLabel ?? "").match(/(\d+)-bit\s+(\d+)\+\d+/i);
  if (!match) {
    return { bits: null, fp16Layers: null };
  }
  return {
    bits: parseInt(match[1], 10),
    fp16Layers: parseInt(match[2], 10),
  };
}

export function resolvedCacheLabel(metrics: GenerationMetrics): string {
  if (cacheFellBackToNative(metrics)) {
    return "Native f16 cache";
  }
  return metrics.cacheLabel ?? LEGACY_MESSAGE_VALUE;
}

export function resolvedCacheStrategy(metrics: GenerationMetrics): string {
  if (cacheFellBackToNative(metrics)) {
    return "native";
  }
  return metrics.cacheStrategy ?? LEGACY_MESSAGE_VALUE;
}

export function resolvedCacheBits(metrics: GenerationMetrics): string {
  if (cacheFellBackToNative(metrics)) {
    return "0-bit";
  }
  if (metrics.cacheBits != null) {
    return `${metrics.cacheBits}-bit`;
  }
  const parsed = parseLegacyCacheQuant(metrics.cacheLabel);
  if (parsed.bits != null) {
    return `${parsed.bits}-bit`;
  }
  if ((metrics.cacheLabel ?? "").toLowerCase().includes("f16")) {
    return "0-bit";
  }
  return LEGACY_MESSAGE_VALUE;
}

export function resolvedFp16Layers(metrics: GenerationMetrics): string {
  if (cacheFellBackToNative(metrics)) {
    return "0";
  }
  if (metrics.fp16Layers != null) {
    return String(metrics.fp16Layers);
  }
  const parsed = parseLegacyCacheQuant(metrics.cacheLabel);
  if (parsed.fp16Layers != null) {
    return String(parsed.fp16Layers);
  }
  return LEGACY_MESSAGE_VALUE;
}

export function resolvedSpeculativeMode(metrics: GenerationMetrics): string {
  const runtimeNote = metrics.runtimeNote ?? "";
  const runtimeBudget = parseTreeBudget(runtimeNote);
  if (metrics.speculativeDecoding) {
    return (metrics.treeBudget ?? 0) > 0 ? `DDTree (${metrics.treeBudget})` : "DFlash";
  }
  if (noteHasNoCompatibleDraft(runtimeNote)) {
    return "Requested, no compatible draft";
  }
  if (runtimeBudget != null) {
    return `DDTree (${runtimeBudget})`;
  }
  if (/ddtree init failed/i.test(runtimeNote)) {
    return "DDTree requested, fell back to DFlash";
  }
  if (/dflash speculative decoding/i.test(runtimeNote)) {
    return "DFlash";
  }
  return "Off";
}

export function resolvedTreeBudget(metrics: GenerationMetrics): string {
  if (metrics.treeBudget != null) {
    return String(metrics.treeBudget);
  }
  const runtimeBudget = parseTreeBudget(metrics.runtimeNote);
  if (runtimeBudget != null) {
    return String(runtimeBudget);
  }
  const specMode = resolvedSpeculativeMode(metrics);
  if (specMode === "Off" || specMode === "DFlash") {
    return "0";
  }
  return LEGACY_MESSAGE_VALUE;
}

export function resolvedDraftModel(metrics: GenerationMetrics): string | null {
  if (metrics.dflashDraftModel) {
    return metrics.dflashDraftModel.split("/").pop() ?? metrics.dflashDraftModel;
  }
  const parsed = parseDraftModel(metrics.runtimeNote);
  if (!parsed) {
    return null;
  }
  return parsed.split("/").pop() ?? parsed;
}

export function requestedCacheLabel(metrics: GenerationMetrics): string | null {
  return metrics.requestedCacheLabel ?? null;
}

export function requestedSpeculativeMode(metrics: GenerationMetrics): string | null {
  if (metrics.requestedSpeculativeDecoding == null) {
    return null;
  }
  if (!metrics.requestedSpeculativeDecoding) {
    return "Off";
  }
  return (metrics.requestedTreeBudget ?? 0) > 0 ? `DDTree (${metrics.requestedTreeBudget})` : "DFlash";
}

export function runtimeOutcomeWarning(metrics: GenerationMetrics): string | null {
  const requestedSpecMode = requestedSpeculativeMode(metrics);
  const actualSpecMode = resolvedSpeculativeMode(metrics);

  if (requestedSpecMode && requestedSpecMode !== "Off") {
    if (noteHasNoCompatibleDraft(metrics.runtimeNote)) {
      return `${requestedSpecMode} requested, no compatible draft`;
    }
    if (/ddtree init failed/i.test(metrics.runtimeNote ?? "")) {
      return `${requestedSpecMode} requested, ran DFlash`;
    }
    if (actualSpecMode !== "Off" && actualSpecMode !== requestedSpecMode) {
      return `${requestedSpecMode} requested, ran ${actualSpecMode}`;
    }
    if (actualSpecMode === "Off") {
      return `${requestedSpecMode} requested, ran standard decoding`;
    }
  }

  const requestedCache = requestedCacheLabel(metrics);
  const actualCache = resolvedCacheLabel(metrics);
  if (requestedCache && requestedCache !== actualCache) {
    return `${requestedCache} requested, ran ${actualCache}`;
  }

  return null;
}
