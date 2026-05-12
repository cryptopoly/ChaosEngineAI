/**
 * HTML Challenge tab — helpers that originally lived inline on the tab.
 *
 * Keeps the composition root in ``HtmlChallengeTab.tsx`` thin.  Anything
 * that can be expressed as a pure function of `slot` / `state` /
 * `manifest` / `option` lives here — only the orchestration root
 * (streaming, abort, run/retry/repair) stays on the tab.
 */

import { number } from "../../../utils";
import type { ChatModelOption } from "../../../types/chat";
import { sizeLabel } from "../../../utils";
import {
  buildComparePayload,
  type CompareTarget,
} from "../CompareView";
import {
  type ChallengeSlot,
  type ChallengeSlotState,
  type HtmlChallengeManifestSlot,
  type HtmlChallengeReasoningEffort,
  type HtmlChallengeStreamEvent,
  type HtmlChallengeThinkingMode,
  emptySlotState,
  formatBytes,
  formatCount,
  htmlChallengeGameKeys,
  htmlValidationForState,
  mergeMetrics,
  reasoningLabel,
} from "../htmlChallengeHelpers";

export type HtmlChallengeModelPayload = ReturnType<typeof buildComparePayload> & {
  thinkingMode: HtmlChallengeThinkingMode;
  reasoningEffort?: HtmlChallengeReasoningEffort;
  seed?: number | null;
};

export function modelKeyFromManifestSlot(
  slot: HtmlChallengeManifestSlot,
  textModelOptions: ChatModelOption[],
) {
  const ref = slot.modelRef;
  const path = slot.path ?? "";
  const canonicalRepo = slot.canonicalRepo ?? "";
  return textModelOptions.find((option) => (
    option.key === ref
    || option.modelRef === ref
    || (path && option.path === path)
    || (canonicalRepo && option.canonicalRepo === canonicalRepo)
  ))?.key ?? "";
}

export function stateFromManifestSlot(slot: HtmlChallengeManifestSlot): ChallengeSlotState {
  const metrics = slot.metrics ?? null;
  return {
    ...emptySlotState(),
    done: slot.status === "done" || Boolean(slot.filename) || Boolean(slot.error),
    error: slot.error,
    filename: slot.filename,
    filePath: slot.filePath,
    fileBytes: slot.fileBytes,
    validHtmlDocument: slot.validHtmlDocument,
    htmlValidation: slot.htmlValidation,
    tokS: metrics?.tokS ?? 0,
    promptTokens: metrics?.promptTokens ?? 0,
    completionTokens: metrics?.completionTokens ?? 0,
    totalTokens: metrics?.totalTokens ?? 0,
    responseSeconds: slot.responseSeconds ?? metrics?.responseSeconds ?? 0,
    loadSeconds: slot.loadSeconds ?? 0,
    totalSeconds: slot.totalSeconds ?? 0,
    runtimeNote: metrics?.runtimeNote ?? null,
    metrics,
  };
}

export function buildRetryModelPayload(
  slot: ChallengeSlot,
  selectedOption: ChatModelOption | null,
  manifestSlot?: HtmlChallengeManifestSlot,
): HtmlChallengeModelPayload | null {
  const withThinking = (payload: ReturnType<typeof buildComparePayload>): HtmlChallengeModelPayload => ({
    ...payload,
    thinkingMode: slot.thinkingMode,
    reasoningEffort: slot.thinkingMode === "auto" ? slot.reasoningEffort : undefined,
    seed: slot.seed,
  });
  if (selectedOption) return withThinking(buildComparePayload(selectedOption, slot.settings));
  if (!manifestSlot?.modelRef) return null;
  return withThinking({
    modelRef: manifestSlot.modelRef,
    modelName: manifestSlot.modelName,
    displayLabel: manifestSlot.displayLabel ?? manifestSlot.modelName ?? manifestSlot.modelRef,
    displayDetail: manifestSlot.displayDetail ?? "",
    format: manifestSlot.format ?? undefined,
    quantization: manifestSlot.quantization ?? undefined,
    sizeGb: manifestSlot.sizeGb ?? undefined,
    contextWindow: manifestSlot.contextWindow ?? undefined,
    canonicalRepo: manifestSlot.canonicalRepo ?? undefined,
    source: manifestSlot.source || "catalog",
    backend: manifestSlot.backend || "auto",
    path: manifestSlot.path ?? undefined,
    launch: slot.settings,
  });
}

export function isRetryableState(state: ChallengeSlotState) {
  const validation = htmlValidationForState(state);
  return Boolean(state.error || (state.done && validation?.status !== "valid"));
}

export function isRepairableState(state: ChallengeSlotState) {
  const validation = htmlValidationForState(state);
  return Boolean(
    state.done
    && !state.error
    && validation
    && validation.status !== "valid"
    && (state.html || state.filename),
  );
}

export function isPreviewGameKey(event: Pick<KeyboardEvent, "key" | "code">) {
  return htmlChallengeGameKeys.has(event.key.toLowerCase()) || event.code.toLowerCase() === "space";
}

export function isEditableKeyboardTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

export function fileActionPath(state: ChallengeSlotState, folderPath: string | undefined) {
  if (state.filePath) return state.filePath;
  if (folderPath && state.filename) return `${folderPath}/${state.filename}`;
  return "";
}

export function runtimeCacheDetail(state: ChallengeSlotState) {
  const noteMatch = state.runtimeNote?.match(/(\d+\+\d+\s+cache)/i);
  if (noteMatch?.[1]) return noteMatch[1].toLowerCase();
  const labelMatch = state.metrics?.cacheLabel?.match(/(\d+\+\d+)$/);
  return labelMatch?.[1] ? `${labelMatch[1]} cache` : "";
}

export function compactSettingsSummary(
  slot: ChallengeSlot,
  state: ChallengeSlotState,
  summarizeLaunchSettings: (settings: ChallengeSlot["settings"]) => string,
) {
  const parts = summarizeLaunchSettings(slot.settings).split(" · ");
  const cacheDetail = runtimeCacheDetail(state);
  if (cacheDetail && !parts.some((part) => part.toLowerCase().includes("cache"))) {
    parts.splice(1, 0, cacheDetail);
  }
  parts.push(reasoningLabel(slot.thinkingMode, slot.reasoningEffort));
  if (slot.seed != null) parts.push(`seed ${slot.seed}`);
  return parts.join(" · ");
}

export function slotSubtitle(state: ChallengeSlotState) {
  if (state.deleted) return "File deleted";
  if (!state.done || state.error) {
    return state.loading ? "Loading..." : state.text ? "Generating..." : "";
  }
  return [
    `${number(state.tokS)} tok/s`,
    `${number(state.responseSeconds)}s`,
    state.loadSeconds > 0 ? `Load ${number(state.loadSeconds)}s` : null,
    state.totalTokens > 0 ? `${formatCount(state.totalTokens)} tokens` : null,
    state.fileBytes ? formatBytes(state.fileBytes) : null,
  ].filter(Boolean).join(" | ");
}

export function slotBusyMessage(
  slot: ChallengeSlot,
  index: number,
  manifestSlot: HtmlChallengeManifestSlot | undefined,
  slots: ChallengeSlot[],
  slotStates: Record<CompareTarget, ChallengeSlotState>,
  manifestSlots: HtmlChallengeManifestSlot[] | undefined,
  compareTargetLabels: Record<CompareTarget, string>,
) {
  const state = slotStates[slot.id];
  if (state.loadingMessage) return state.loadingMessage;
  if (manifestSlot?.status === "loading") return "Loading model...";
  if (manifestSlot?.status === "running") return "Generating...";

  const previousPending = slots.slice(0, index).find((previousSlot) => {
    const previousState = slotStates[previousSlot.id];
    const previousManifestSlot = manifestSlots?.find((item) => item.slotId === previousSlot.id);
    if (previousState.done || previousState.deleted || previousState.error) return false;
    if (previousState.loading || previousState.text) return true;
    return previousManifestSlot?.status === "loading"
      || previousManifestSlot?.status === "running"
      || previousManifestSlot?.status === "queued";
  });
  if (previousPending) return `Waiting for ${compareTargetLabels[previousPending.id]} to finish...`;
  return index === 0 ? "Waiting..." : "Waiting to start...";
}

/**
 * Pure reducer for slot state under a streaming challenge event.
 * Returns the next slotStates record so the tab can pass it straight
 * into ``setSlotStates``.
 */
export function applySlotStreamEvent(
  current: Record<CompareTarget, ChallengeSlotState>,
  event: HtmlChallengeStreamEvent,
): Record<CompareTarget, ChallengeSlotState> {
  const target = event.model;
  if (!target) return current;
  const prev = current[target];
  let next = prev;
  if (event.loading) {
    next = { ...next, loading: true, loadingMessage: event.message, error: undefined, deleted: false };
  }
  if (event.loaded) {
    next = {
      ...next,
      loading: false,
      loadingMessage: "Generating...",
      loadSeconds: event.loadSeconds ?? next.loadSeconds,
      metrics: mergeMetrics(next.metrics, event),
    };
  }
  if (event.reasoning) {
    next = { ...next, reasoning: next.reasoning + event.reasoning, reasoningDone: false };
  }
  if (event.reasoningDone) {
    next = { ...next, reasoningDone: true };
  }
  if (event.token) {
    next = { ...next, loading: false, loadingMessage: undefined, text: next.text + event.token };
  }
  if (event.done) {
    next = {
      ...next,
      done: true,
      loading: false,
      loadingMessage: undefined,
      deleted: false,
      reasoningDone: true,
      text: event.text ?? next.text,
      html: event.html ?? next.html,
      filename: event.filename,
      filePath: event.filePath,
      fileBytes: event.fileBytes,
      validHtmlDocument: event.validHtmlDocument,
      htmlValidation: event.htmlValidation,
      tokS: event.tokS ?? 0,
      promptTokens: event.promptTokens ?? 0,
      completionTokens: event.completionTokens ?? 0,
      totalTokens: event.totalTokens ?? 0,
      responseSeconds: event.responseSeconds ?? 0,
      loadSeconds: event.loadSeconds ?? next.loadSeconds,
      totalSeconds: event.totalSeconds ?? next.totalSeconds,
      runtimeNote: event.runtimeNote ?? next.runtimeNote,
      metrics: mergeMetrics(next.metrics, event),
    };
  }
  if (event.error) {
    next = {
      ...next,
      error: event.error,
      done: true,
      loading: false,
      deleted: false,
      reasoningDone: true,
      html: "",
    };
  }
  return next === prev ? current : { ...current, [target]: next };
}

// Re-exports so the tab can import sizeLabel via this module if convenient.
export { sizeLabel };
