/**
 * HTML Challenge: shared types + pure helpers used across the tab and
 * its iframe / sidebar / history sub-views.
 *
 * Extracted from ``HtmlChallengeTab.tsx`` as part of the v0.8.0 refactor.
 * Anything that touches React state stays in the main tab — this module
 * is types + ``slot`` shape factories + format / validation helpers,
 * all pure.
 */

import { number } from "../../utils";
import type { ChatModelOption } from "../../types/chat";
import type { GenerationMetrics, LaunchPreferences } from "../../types";
import {
  cloneLaunchSettings,
  compareTargets,
  type CompareTarget,
} from "./CompareView";


// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type HtmlChallengeLayoutMode = "row" | "stacked";
export type HtmlChallengeThinkingMode = "off" | "auto";
export type HtmlChallengeReasoningEffort = "low" | "medium" | "high";
export type HtmlValidationStatus =
  | "valid"
  | "partial"
  | "script-error"
  | "blank-render"
  | "no-html";

export interface HtmlValidation {
  status: HtmlValidationStatus;
  label?: string;
  issues?: string[];
  checks?: Record<string, unknown>;
  source?: string;
  updatedAt?: string;
}

export interface ChallengeSlot {
  id: CompareTarget;
  modelKey: string;
  settings: LaunchPreferences;
  thinkingMode: HtmlChallengeThinkingMode;
  reasoningEffort: HtmlChallengeReasoningEffort;
  seed: number | null;
}

export interface ChallengeSlotState {
  text: string;
  reasoning: string;
  reasoningDone: boolean;
  loading: boolean;
  done: boolean;
  deleted: boolean;
  loadingMessage?: string;
  error?: string;
  html: string;
  filename?: string;
  filePath?: string;
  fileBytes?: number;
  validHtmlDocument?: boolean;
  htmlValidation?: HtmlValidation | null;
  tokS: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  responseSeconds: number;
  loadSeconds: number;
  totalSeconds: number;
  runtimeNote?: string | null;
  metrics: GenerationMetrics | null;
}

export interface HtmlChallengeManifestSlot {
  slotId: CompareTarget;
  label?: string;
  status: string;
  modelName: string;
  modelRef: string;
  displayLabel?: string;
  displayDetail?: string;
  format?: string | null;
  quantization?: string | null;
  sizeGb?: number | null;
  contextWindow?: string | null;
  canonicalRepo?: string | null;
  source?: string | null;
  backend?: string | null;
  path?: string | null;
  settings?: Partial<LaunchPreferences>;
  thinkingMode?: HtmlChallengeThinkingMode | null;
  reasoningEffort?: HtmlChallengeReasoningEffort | null;
  seed?: number | null;
  filename?: string;
  filePath?: string;
  fileBytes?: number;
  validHtmlDocument?: boolean;
  htmlValidation?: HtmlValidation | null;
  responseSeconds?: number;
  loadSeconds?: number;
  totalSeconds?: number;
  error?: string;
  metrics?: GenerationMetrics;
}

export interface HtmlChallengeManifest {
  id: string;
  title: string;
  prompt: string;
  thinkingMode?: HtmlChallengeThinkingMode | null;
  reasoningEffort?: HtmlChallengeReasoningEffort | null;
  createdAt: string;
  updatedAt: string;
  folderPath: string;
  settingsFilename?: string;
  settingsPath?: string;
  slots: HtmlChallengeManifestSlot[];
}

export interface HtmlChallengeStreamEvent extends Partial<GenerationMetrics> {
  challengeStarted?: boolean;
  challengeDone?: boolean;
  challenge?: HtmlChallengeManifest;
  model?: CompareTarget;
  loading?: boolean;
  loaded?: boolean;
  message?: string;
  token?: string;
  reasoning?: string;
  reasoningDone?: boolean;
  done?: boolean;
  error?: string;
  text?: string;
  html?: string;
  filename?: string;
  filePath?: string;
  fileBytes?: number;
  validHtmlDocument?: boolean;
  htmlValidation?: HtmlValidation | null;
  loadSeconds?: number;
  totalSeconds?: number;
}


// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const htmlChallengeGameKeys = new Set([
  " ",
  "enter",
  "spacebar",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
  "w",
  "a",
  "s",
  "d",
]);

export const htmlValidationLabels: Record<HtmlValidationStatus, string> = {
  valid: "Valid",
  partial: "Partial",
  "script-error": "Script error",
  "blank-render": "Blank render",
  "no-html": "No HTML",
};


// ---------------------------------------------------------------------------
// Slot shape factories
// ---------------------------------------------------------------------------

export const emptySlotState = (): ChallengeSlotState => ({
  text: "",
  reasoning: "",
  reasoningDone: true,
  loading: false,
  done: false,
  deleted: false,
  html: "",
  tokS: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  responseSeconds: 0,
  loadSeconds: 0,
  totalSeconds: 0,
  runtimeNote: null,
  metrics: null,
});

export function emptySlotStates(): Record<CompareTarget, ChallengeSlotState> {
  return {
    a: emptySlotState(),
    b: emptySlotState(),
    c: emptySlotState(),
    d: emptySlotState(),
  };
}

export function emptyStreamAtBottom(): Record<CompareTarget, boolean> {
  return { a: true, b: true, c: true, d: true };
}

export function emptyCodeViews(): Record<CompareTarget, boolean> {
  return { a: false, b: false, c: false, d: false };
}

export function emptyPreviewBackgrounds(): Record<CompareTarget, string | null> {
  return { a: null, b: null, c: null, d: null };
}

export function randomChallengeSeed() {
  // Backend pydantic field accepts [0, 2147483647], so Math.random() * 2147483648
  // covers the full int32 range when rounded down.
  return Math.floor(Math.random() * 2147483648);
}

export function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function defaultChallengeSlot(
  id: CompareTarget,
  launchSettings: LaunchPreferences,
  thinkingMode: HtmlChallengeThinkingMode = "off",
  reasoningEffort: HtmlChallengeReasoningEffort = "medium",
): ChallengeSlot {
  return {
    id,
    modelKey: "",
    settings: cloneLaunchSettings(launchSettings),
    thinkingMode,
    reasoningEffort,
    seed: randomChallengeSeed(),
  };
}

export function isTextModelOption(option: ChatModelOption) {
  const backend = (option.backend ?? "").toLowerCase();
  const format = (option.format ?? option.detail ?? "").toLowerCase();
  const label = option.label.toLowerCase();
  return backend !== ""
    && !format.includes("diffuser")
    && !label.includes("stable-diffusion")
    && !label.includes("flux")
    && !label.includes("sana");
}

export function settingsFromManifest(
  settings: Partial<LaunchPreferences> | undefined,
  fallback: LaunchPreferences,
): LaunchPreferences {
  return { ...cloneLaunchSettings(fallback), ...(settings ?? {}) };
}

export function mergeMetrics(
  current: GenerationMetrics | null,
  event: HtmlChallengeStreamEvent,
): GenerationMetrics | null {
  const keys: Array<keyof GenerationMetrics> = [
    "finishReason",
    "promptTokens",
    "completionTokens",
    "totalTokens",
    "tokS",
    "responseSeconds",
    "runtimeNote",
    "modelRef",
    "canonicalRepo",
    "backend",
    "engineLabel",
    "cacheLabel",
    "cacheStrategy",
    "cacheBits",
    "fp16Layers",
    "fusedAttention",
    "fitModelInMemory",
    "requestedCacheLabel",
    "requestedCacheStrategy",
    "requestedCacheBits",
    "requestedFp16Layers",
    "requestedFitModelInMemory",
    "requestedSpeculativeDecoding",
    "requestedTreeBudget",
    "speculativeDecoding",
    "dflashDraftModel",
    "treeBudget",
    "modelSource",
    "modelPath",
    "contextTokens",
    "generatedAt",
  ];
  let changed = false;
  const next: Record<string, unknown> = {
    finishReason: "stop",
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    tokS: 0,
    runtimeNote: null,
    ...(current ?? {}),
  };
  for (const key of keys) {
    if (!(key in event)) continue;
    changed = true;
    next[key] = event[key];
  }
  return changed ? next as unknown as GenerationMetrics : current;
}


// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

export function formatBytes(bytes?: number) {
  if (!bytes || bytes < 1) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${number(bytes / 1024)} KB`;
  return `${number(bytes / (1024 * 1024))} MB`;
}

export function formatCount(value: number) {
  return Math.round(value).toLocaleString();
}

export function formatChallengeDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function modelTitleFragments(slot: HtmlChallengeManifestSlot) {
  const fragments: string[] = [];
  for (const value of [slot.displayLabel, slot.modelName, slot.modelRef]) {
    if (!value) continue;
    const candidates = [value.trim()];
    if (value.includes("/")) {
      const parts = value.split("/");
      candidates.push(parts[parts.length - 1]?.trim() ?? "");
    }
    for (const candidate of candidates) {
      if (candidate.length >= 4 && !fragments.includes(candidate)) {
        fragments.push(candidate);
      }
    }
  }
  return fragments;
}

export function displayChallengeTitle(challenge: HtmlChallengeManifest) {
  const cleaned = challenge.title.replace(/\s+/g, " ").trim();
  const lowered = cleaned.toLowerCase();
  let earliestModelIndex: number | null = null;
  for (const slot of challenge.slots) {
    for (const fragment of modelTitleFragments(slot)) {
      const index = lowered.indexOf(fragment.toLowerCase());
      if (index > 0 && (earliestModelIndex == null || index < earliestModelIndex)) {
        earliestModelIndex = index;
      }
    }
  }
  if (earliestModelIndex == null) return cleaned;
  const candidate = cleaned
    .slice(0, earliestModelIndex)
    .replace(/(?:\s+(?:vs|versus|and))+$/i, "")
    .replace(/[\s\-–—·:|,/+&]+$/g, "")
    .trim();
  return candidate || cleaned;
}

export function challengeHistoryLabel(challenge: HtmlChallengeManifest) {
  return `${displayChallengeTitle(challenge)} · ${formatChallengeDate(challenge.createdAt)}`;
}

export function fuzzyIncludes(value: string, query: string) {
  const haystack = value.toLowerCase();
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  if (haystack.includes(needle)) return true;

  let cursor = 0;
  for (const char of needle) {
    cursor = haystack.indexOf(char, cursor);
    if (cursor < 0) return false;
    cursor += 1;
  }
  return true;
}


// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

export function isHtmlValidationStatus(value: unknown): value is HtmlValidationStatus {
  return value === "valid"
    || value === "partial"
    || value === "script-error"
    || value === "blank-render"
    || value === "no-html";
}

export function isCompareTarget(value: unknown): value is CompareTarget {
  return compareTargets.includes(value as CompareTarget);
}

export function htmlValidationForState(state: ChallengeSlotState): HtmlValidation | null {
  if (state.htmlValidation?.status) return state.htmlValidation;
  if (state.validHtmlDocument === false) {
    const status: HtmlValidationStatus = state.html ? "partial" : "no-html";
    return { status, label: htmlValidationLabels[status] };
  }
  if (state.done && !state.error) {
    const status: HtmlValidationStatus = state.html ? "valid" : "no-html";
    return { status, label: htmlValidationLabels[status] };
  }
  return null;
}

export function validationBadgeClass(status: HtmlValidationStatus) {
  if (status === "valid") return "success";
  if (status === "script-error" || status === "blank-render") return "danger";
  return "warning";
}

export function validationMessage(validation: HtmlValidation | null) {
  if (!validation) return "";
  return validation.issues?.filter(Boolean).slice(0, 3).join(" ") ?? "";
}


// ---------------------------------------------------------------------------
// Reasoning + layout
// ---------------------------------------------------------------------------

export function normalizeThinkingMode(value: unknown): HtmlChallengeThinkingMode {
  return value === "auto" ? "auto" : "off";
}

export function normalizeReasoningEffort(value: unknown): HtmlChallengeReasoningEffort {
  return value === "low" || value === "high" ? value : "medium";
}

export function reasoningLabel(mode: HtmlChallengeThinkingMode, effort: HtmlChallengeReasoningEffort) {
  return mode === "off" ? "Thinking off" : `Thinking ${effort}`;
}

export function challengeGridColumns(count: number, layoutMode: HtmlChallengeLayoutMode) {
  if (layoutMode === "stacked") {
    return `repeat(${count <= 2 ? 1 : 2}, minmax(0, 1fr))`;
  }
  return `repeat(${Math.min(Math.max(count, 2), 4)}, minmax(0, 1fr))`;
}

export function stackedLayoutLabel(count: number) {
  return count <= 2 ? "1 x 2" : "2 x 2";
}
