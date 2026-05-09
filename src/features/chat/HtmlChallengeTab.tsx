import { type KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import { apiFetch } from "../../api";
import { ModelLaunchModal } from "../../components/ModelLaunchModal";
import { Panel } from "../../components/Panel";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import type { GenerationMetrics, LaunchPreferences, PreviewMetrics, StrategyInstallLog, SystemStats } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import { number, sizeLabel } from "../../utils";
import {
  buildComparePayload,
  cloneLaunchSettings,
  compareTargetLabels,
  compareTargets,
  summarizeLaunchSettings,
  useLaunchPreview,
  type CompareTarget,
} from "./CompareView";
import {
  BrowserIcon,
  CollapseIcon,
  ExpandIcon,
  OpenFileIcon,
} from "./htmlChallengeIcons";
import {
  escapeHtmlCode,
  highlightHtmlCode,
  previewSrcDoc,
} from "./htmlChallengeMarkup";

interface HtmlChallengeTabProps {
  modelOptions: ChatModelOption[];
  launchSettings: LaunchPreferences;
  availableMemoryGb: number;
  totalMemoryGb: number;
  gpuVramTotalGb?: number | null;
  availableCacheStrategies?: SystemStats["availableCacheStrategies"];
  dflashInfo?: SystemStats["dflash"];
  turboInstalled?: boolean;
  onInstallPackage?: (strategyId: string) => void;
  installingPackage?: string | null;
  installLogs?: Record<string, StrategyInstallLog>;
  fileRevealLabel: string;
  onRevealPath: (path: string) => void;
  onOpenFilePath: (path: string) => void;
}

interface ChallengeSlot {
  id: CompareTarget;
  modelKey: string;
  settings: LaunchPreferences;
  thinkingMode: HtmlChallengeThinkingMode;
  reasoningEffort: HtmlChallengeReasoningEffort;
  seed: number | null;
}

type HtmlValidationStatus = "valid" | "partial" | "script-error" | "blank-render" | "no-html";

interface HtmlValidation {
  status: HtmlValidationStatus;
  label?: string;
  issues?: string[];
  checks?: Record<string, unknown>;
  source?: string;
  updatedAt?: string;
}

interface ChallengeSlotState {
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

interface HtmlChallengeManifestSlot {
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

interface HtmlChallengeManifest {
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

interface HtmlChallengeStreamEvent extends Partial<GenerationMetrics> {
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

type HtmlChallengeLayoutMode = "row" | "stacked";
type HtmlChallengeThinkingMode = "off" | "auto";
type HtmlChallengeReasoningEffort = "low" | "medium" | "high";
type HtmlChallengeModelPayload = ReturnType<typeof buildComparePayload> & {
  thinkingMode: HtmlChallengeThinkingMode;
  reasoningEffort?: HtmlChallengeReasoningEffort;
  seed?: number | null;
};



const htmlChallengeGameKeys = new Set([
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

const emptySlotState = (): ChallengeSlotState => ({
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

function emptySlotStates(): Record<CompareTarget, ChallengeSlotState> {
  return {
    a: emptySlotState(),
    b: emptySlotState(),
    c: emptySlotState(),
    d: emptySlotState(),
  };
}

function emptyStreamAtBottom(): Record<CompareTarget, boolean> {
  return { a: true, b: true, c: true, d: true };
}

function emptyCodeViews(): Record<CompareTarget, boolean> {
  return { a: false, b: false, c: false, d: false };
}

function emptyPreviewBackgrounds(): Record<CompareTarget, string | null> {
  return { a: null, b: null, c: null, d: null };
}

function randomChallengeSeed() {
  // Backend pydantic field accepts [0, 2147483647], so Math.random() * 2147483648
  // covers the full int32 range when rounded down.
  return Math.floor(Math.random() * 2147483648);
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function defaultChallengeSlot(
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

function isTextModelOption(option: ChatModelOption) {
  const backend = (option.backend ?? "").toLowerCase();
  const format = (option.format ?? option.detail ?? "").toLowerCase();
  const label = option.label.toLowerCase();
  return backend !== ""
    && !format.includes("diffuser")
    && !label.includes("stable-diffusion")
    && !label.includes("flux")
    && !label.includes("sana");
}

function mergeMetrics(current: GenerationMetrics | null, event: HtmlChallengeStreamEvent): GenerationMetrics | null {
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

function formatBytes(bytes?: number) {
  if (!bytes || bytes < 1) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${number(bytes / 1024)} KB`;
  return `${number(bytes / (1024 * 1024))} MB`;
}

function formatCount(value: number) {
  return Math.round(value).toLocaleString();
}

function settingsFromManifest(settings: Partial<LaunchPreferences> | undefined, fallback: LaunchPreferences): LaunchPreferences {
  return { ...cloneLaunchSettings(fallback), ...(settings ?? {}) };
}

function formatChallengeDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function modelTitleFragments(slot: HtmlChallengeManifestSlot) {
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

function displayChallengeTitle(challenge: HtmlChallengeManifest) {
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

function challengeHistoryLabel(challenge: HtmlChallengeManifest) {
  return `${displayChallengeTitle(challenge)} · ${formatChallengeDate(challenge.createdAt)}`;
}

function fuzzyIncludes(value: string, query: string) {
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

const htmlValidationLabels: Record<HtmlValidationStatus, string> = {
  valid: "Valid",
  partial: "Partial",
  "script-error": "Script error",
  "blank-render": "Blank render",
  "no-html": "No HTML",
};

function isHtmlValidationStatus(value: unknown): value is HtmlValidationStatus {
  return value === "valid"
    || value === "partial"
    || value === "script-error"
    || value === "blank-render"
    || value === "no-html";
}

function isCompareTarget(value: unknown): value is CompareTarget {
  return compareTargets.includes(value as CompareTarget);
}

function htmlValidationForState(state: ChallengeSlotState): HtmlValidation | null {
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

function validationBadgeClass(status: HtmlValidationStatus) {
  if (status === "valid") return "success";
  if (status === "script-error" || status === "blank-render") return "danger";
  return "warning";
}

function validationMessage(validation: HtmlValidation | null) {
  if (!validation) return "";
  return validation.issues?.filter(Boolean).slice(0, 3).join(" ") ?? "";
}

function normalizeThinkingMode(value: unknown): HtmlChallengeThinkingMode {
  return value === "auto" ? "auto" : "off";
}

function normalizeReasoningEffort(value: unknown): HtmlChallengeReasoningEffort {
  return value === "low" || value === "high" ? value : "medium";
}

function reasoningLabel(mode: HtmlChallengeThinkingMode, effort: HtmlChallengeReasoningEffort) {
  return mode === "off" ? "Thinking off" : `Thinking ${effort}`;
}

function challengeGridColumns(count: number, layoutMode: HtmlChallengeLayoutMode) {
  if (layoutMode === "stacked") {
    return `repeat(${count <= 2 ? 1 : 2}, minmax(0, 1fr))`;
  }
  return `repeat(${Math.min(Math.max(count, 2), 4)}, minmax(0, 1fr))`;
}

function stackedLayoutLabel(count: number) {
  return count <= 2 ? "1 x 2" : "2 x 2";
}


export function HtmlChallengeTab({
  modelOptions,
  launchSettings,
  availableMemoryGb,
  totalMemoryGb,
  gpuVramTotalGb,
  availableCacheStrategies,
  dflashInfo,
  turboInstalled,
  onInstallPackage,
  installingPackage,
  installLogs,
  fileRevealLabel,
  onRevealPath,
  onOpenFilePath,
}: HtmlChallengeTabProps) {
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [slots, setSlots] = useState<ChallengeSlot[]>(() => [
    defaultChallengeSlot("a", launchSettings),
    defaultChallengeSlot("b", launchSettings),
  ]);
  const [slotStates, setSlotStates] = useState<Record<CompareTarget, ChallengeSlotState>>(emptySlotStates);
  const [busy, setBusy] = useState(false);
  const [manifest, setManifest] = useState<HtmlChallengeManifest | null>(null);
  const [challenges, setChallenges] = useState<HtmlChallengeManifest[]>([]);
  const [selectedChallengeId, setSelectedChallengeId] = useState("");
  const [historySearch, setHistorySearch] = useState("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [loadingChallengeId, setLoadingChallengeId] = useState<string | null>(null);
  const [layoutMode, setLayoutMode] = useState<HtmlChallengeLayoutMode>("row");
  const [expandedHtmlSlot, setExpandedHtmlSlot] = useState<CompareTarget | null>(null);
  const [codeViewSlots, setCodeViewSlots] = useState<Record<CompareTarget, boolean>>(emptyCodeViews);
  const [streamAtBottom, setStreamAtBottom] = useState<Record<CompareTarget, boolean>>(emptyStreamAtBottom);
  const [pickerTarget, setPickerTarget] = useState<CompareTarget | null>(null);
  // When true, confirming the picker re-runs the slot's challenge with the
  // newly chosen model so manifest filename + metadata stay consistent.
  const [pickerAutoRetry, setPickerAutoRetry] = useState(false);
  const [pickerSearch, setPickerSearch] = useState("");
  const [pickerDraftKey, setPickerDraftKey] = useState("");
  const [pickerDraftSettings, setPickerDraftSettings] = useState<LaunchPreferences>(() => cloneLaunchSettings(launchSettings));
  const abortRef = useRef<AbortController | null>(null);
  const streamRefs = useRef<Record<CompareTarget, HTMLPreElement | null>>({
    a: null,
    b: null,
    c: null,
    d: null,
  });
  const frameRefs = useRef<Record<CompareTarget, HTMLIFrameElement | null>>({
    a: null,
    b: null,
    c: null,
    d: null,
  });
  const frameShellRefs = useRef<Record<CompareTarget, HTMLDivElement | null>>({
    a: null,
    b: null,
    c: null,
    d: null,
  });
  const activePreviewSlotRef = useRef<CompareTarget | null>(null);

  const textModelOptions = modelOptions.filter(isTextModelOption);
  const selectedBySlot = Object.fromEntries(
    slots.map((slot) => [slot.id, textModelOptions.find((option) => option.key === slot.modelKey) ?? null]),
  ) as Record<CompareTarget, ChatModelOption | null>;
  const allSelected = slots.every((slot) => selectedBySlot[slot.id] != null);
  const pickerDraftOption =
    textModelOptions.find((option) => option.key === pickerDraftKey)
    ?? (pickerTarget ? textModelOptions[0] ?? null : null);
  const pickerDraftPreview: PreviewMetrics = useLaunchPreview(pickerDraftOption, pickerDraftSettings);
  const installPackage = onInstallPackage ?? (() => {});
  const completedChallenge = Boolean(
    manifest?.slots.length
      && manifest.slots.every((slot) => slot.status === "done" || slot.status === "error"),
  );
  const completedValidChallenge = Boolean(
    manifest
      && completedChallenge
      && slots.every((slot) => htmlValidationForState(slotStates[slot.id])?.status === "valid"),
  );
  const selectedChallenge = challenges.find((challenge) => challenge.id === selectedChallengeId) ?? null;
  const visibleSlots = expandedHtmlSlot
    ? slots.filter((slot) => slot.id === expandedHtmlSlot)
    : slots;
  const filteredChallenges = challenges.filter((challenge) =>
    fuzzyIncludes(challengeHistoryLabel(challenge), historySearch),
  );
  const historyInputValue = historyOpen || historySearch
    ? historySearch
    : selectedChallenge ? challengeHistoryLabel(selectedChallenge) : "";

  useEffect(() => {
    void refreshChallengeHistory();
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    function handlePreviewValidation(event: MessageEvent) {
      const data = event.data as {
        __htmlChallengePreviewValidation?: boolean;
        slotId?: unknown;
        status?: unknown;
        message?: unknown;
      } | null;
      if (!data?.__htmlChallengePreviewValidation) return;
      if (!isCompareTarget(data.slotId) || !isHtmlValidationStatus(data.status)) return;
      const target = data.slotId;
      const status = data.status;
      if (status === "valid") return;
      if (status !== "script-error" && status !== "blank-render") return;
      const message = typeof data.message === "string" ? data.message.trim() : "";
      const validation: HtmlValidation = {
        status,
        label: htmlValidationLabels[status],
        issues: message ? [message] : [],
        source: "runtime",
        updatedAt: new Date().toISOString(),
      };
      let shouldPersist = false;
      setSlotStates((current) => {
        const previous = current[target];
        const currentValidation = htmlValidationForState(previous);
        if (currentValidation?.status === "no-html") return current;
        if (status === "blank-render" && currentValidation?.status && currentValidation.status !== "valid") {
          return current;
        }
        if (
          currentValidation?.status === validation.status
          && validationMessage(currentValidation) === validationMessage(validation)
        ) {
          return current;
        }
        shouldPersist = Boolean(manifest?.id);
        return {
          ...current,
          [target]: {
            ...previous,
            validHtmlDocument: false,
            htmlValidation: validation,
          },
        };
      });
      if (shouldPersist && manifest?.id) {
        void persistSlotValidation(manifest.id, target, validation);
      }
    }

    window.addEventListener("message", handlePreviewValidation);
    return () => window.removeEventListener("message", handlePreviewValidation);
  }, [manifest?.id]);

  // Per-slot computed background colour reported by the iframe so the frame
  // shell margin matches the page theme (no white margin on dark pages, no
  // dark margin on light pages).
  const [previewBackgrounds, setPreviewBackgrounds] = useState<Record<CompareTarget, string | null>>(
    emptyPreviewBackgrounds,
  );
  useEffect(() => {
    function handlePreviewBackground(event: MessageEvent) {
      const data = event.data as {
        __htmlChallengePreviewBackground?: boolean;
        slotId?: unknown;
        color?: unknown;
      } | null;
      if (!data?.__htmlChallengePreviewBackground) return;
      if (!isCompareTarget(data.slotId) || typeof data.color !== "string") return;
      const target = data.slotId;
      const color = data.color;
      setPreviewBackgrounds((current) => current[target] === color ? current : { ...current, [target]: color });
    }
    window.addEventListener("message", handlePreviewBackground);
    return () => window.removeEventListener("message", handlePreviewBackground);
  }, []);

  useEffect(() => {
    const handles = slots
      .filter((slot) => streamAtBottom[slot.id])
      .map((slot) => requestAnimationFrame(() => scrollStreamToBottom(slot.id)));
    return () => handles.forEach((handle) => cancelAnimationFrame(handle));
  }, [slots, slotStates, streamAtBottom]);

  useEffect(() => {
    function handleWindowPreviewKey(event: KeyboardEvent) {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (!isPreviewGameKey(event) || isEditableKeyboardTarget(event.target)) return;
      const target = expandedHtmlSlot ?? activePreviewSlotRef.current;
      if (!target) return;
      const shell = frameShellRefs.current[target];
      if (shell && event.target instanceof Node && shell.contains(event.target)) return;
      event.preventDefault();
      sendPreviewKey(target, event);
    }

    window.addEventListener("keydown", handleWindowPreviewKey);
    window.addEventListener("keyup", handleWindowPreviewKey);
    return () => {
      window.removeEventListener("keydown", handleWindowPreviewKey);
      window.removeEventListener("keyup", handleWindowPreviewKey);
    };
  }, [expandedHtmlSlot]);

  function updateSlot(slotId: CompareTarget, patch: Partial<ChallengeSlot>) {
    setSlots((current) => current.map((slot) => slot.id === slotId ? { ...slot, ...patch } : slot));
  }

  function addSlot() {
    if (busy || slots.length >= 4) return;
    const nextId = compareTargets[slots.length];
    if (!nextId) return;
    setSlots((current) => [
      ...current,
      defaultChallengeSlot(nextId, launchSettings),
    ]);
  }

  function removeLastSlot() {
    if (busy || slots.length <= 2) return;
    setSlots((current) => current.slice(0, -1));
  }

  function handleStreamScroll(target: CompareTarget) {
    const element = streamRefs.current[target];
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 32;
    setStreamAtBottom((current) => ({ ...current, [target]: atBottom }));
  }

  function scrollStreamToBottom(target: CompareTarget) {
    const element = streamRefs.current[target];
    if (!element) return;
    element.scrollTop = element.scrollHeight;
    setStreamAtBottom((current) => current[target] ? current : { ...current, [target]: true });
  }

  function attachFrameShell(target: CompareTarget, element: HTMLDivElement | null) {
    frameShellRefs.current[target] = element;
  }

  function modelKeyFromManifestSlot(slot: HtmlChallengeManifestSlot) {
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

  async function consumeChallengeStream(response: Response) {
    const reader = response.body?.getReader();
    if (!reader) return "";
    const decoder = new TextDecoder();
    let buffer = "";
    let finalChallengeId = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6)) as HtmlChallengeStreamEvent;
          if (event.challenge?.id) finalChallengeId = event.challenge.id;
          applyStreamEvent(event);
        } catch {
          // Ignore malformed chunks.
        }
      }
    }
    return finalChallengeId;
  }

  function buildRetryModelPayload(slot: ChallengeSlot, manifestSlot?: HtmlChallengeManifestSlot) {
    const option = selectedBySlot[slot.id];
    const withThinking = (payload: ReturnType<typeof buildComparePayload>): HtmlChallengeModelPayload => ({
      ...payload,
      thinkingMode: slot.thinkingMode,
      reasoningEffort: slot.thinkingMode === "auto" ? slot.reasoningEffort : undefined,
      seed: slot.seed,
    });
    if (option) return withThinking(buildComparePayload(option, slot.settings));
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

  function isRetryableState(state: ChallengeSlotState) {
    const validation = htmlValidationForState(state);
    return Boolean(state.error || (state.done && validation?.status !== "valid"));
  }

  function isRepairableState(state: ChallengeSlotState) {
    const validation = htmlValidationForState(state);
    return Boolean(
      state.done
      && !state.error
      && validation
      && validation.status !== "valid"
      && (state.html || state.filename),
    );
  }

  function markPreviewActive(target: CompareTarget) {
    activePreviewSlotRef.current = target;
  }

  function focusPreviewFrame(target: CompareTarget) {
    markPreviewActive(target);
    const frame = frameRefs.current[target];
    if (!frame) return;
    frame.focus();
    try {
      frame.contentWindow?.focus();
    } catch {
      // Sandboxed frames can reject focus in some WebView builds.
    }
  }

  function isPreviewGameKey(event: Pick<KeyboardEvent, "key" | "code">) {
    return htmlChallengeGameKeys.has(event.key.toLowerCase()) || event.code.toLowerCase() === "space";
  }

  function isEditableKeyboardTarget(target: EventTarget | null) {
    if (!(target instanceof HTMLElement)) return false;
    if (target.isContentEditable) return true;
    const tag = target.tagName.toLowerCase();
    return tag === "input" || tag === "textarea" || tag === "select";
  }

  function sendPreviewKey(target: CompareTarget, event: KeyboardEvent | ReactKeyboardEvent<HTMLElement>) {
    const frame = frameRefs.current[target];
    if (!frame?.contentWindow) return;
    const keyCodes = event as unknown as { keyCode?: number; which?: number };
    frame.contentWindow.postMessage({
      __htmlChallengePreviewKey: true,
      type: event.type,
      key: event.key,
      code: event.code,
      keyCode: keyCodes.keyCode ?? 0,
      which: keyCodes.which ?? keyCodes.keyCode ?? 0,
      repeat: event.repeat,
      altKey: event.altKey,
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
      shiftKey: event.shiftKey,
    }, "*");
  }

  function forwardPreviewKey(target: CompareTarget, event: ReactKeyboardEvent<HTMLElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    markPreviewActive(target);
    if (isPreviewGameKey(event)) {
      event.preventDefault();
    }
    sendPreviewKey(target, event);
  }

  function newChallenge() {
    if (busy) return;
    setTitle("");
    setPrompt("");
    setManifest(null);
    setExpandedHtmlSlot(null);
    setSelectedChallengeId("");
    setHistorySearch("");
    setHistoryOpen(false);
    setCodeViewSlots(emptyCodeViews());
    setPreviewBackgrounds(emptyPreviewBackgrounds());
    setSlotStates(emptySlotStates());
    setStreamAtBottom(emptyStreamAtBottom());
    setSlots([
      defaultChallengeSlot("a", launchSettings),
      defaultChallengeSlot("b", launchSettings),
    ]);
  }

  function usePromptInNewChallenge() {
    if (busy || !manifest) return;
    setManifest(null);
    setExpandedHtmlSlot(null);
    setSelectedChallengeId("");
    setHistorySearch("");
    setHistoryOpen(false);
    setCodeViewSlots(emptyCodeViews());
    setPreviewBackgrounds(emptyPreviewBackgrounds());
    setSlotStates(emptySlotStates());
    setStreamAtBottom(emptyStreamAtBottom());
    setSlots((current) => current.map((slot) => ({
      ...slot,
      settings: cloneLaunchSettings(slot.settings),
      seed: slot.seed ?? randomChallengeSeed(),
    })));
  }

  async function refreshChallengeHistory(selectId?: string) {
    try {
      const response = await apiFetch("/api/chat/html-challenges");
      if (!response.ok) return [];
      const payload = await response.json() as { challenges?: HtmlChallengeManifest[] };
      const nextChallenges = payload.challenges ?? [];
      setChallenges(nextChallenges);
      setSelectedChallengeId((current) => {
        if (selectId) return selectId;
        return current && nextChallenges.some((challenge) => challenge.id === current)
          ? current
          : "";
      });
      return nextChallenges;
    } catch {
      return [];
    }
  }

  async function readResponseDetail(response: Response, fallback: string) {
    try {
      const payload = await response.json();
      if (payload?.detail) return String(payload.detail);
    } catch {
      try {
        const text = await response.text();
        if (text.trim()) return text.trim();
      } catch {
        // Ignore unreadable error bodies.
      }
    }
      return fallback;
  }

  async function deleteChallengeById(challengeId: string, label?: string) {
    if (busy || !challengeId) return;
    const display = label
      || (manifest?.id === challengeId ? displayChallengeTitle(manifest) : "")
      || (selectedChallenge?.id === challengeId ? displayChallengeTitle(selectedChallenge) : "")
      || (challenges.find((c) => c.id === challengeId) ? displayChallengeTitle(challenges.find((c) => c.id === challengeId)!) : "this challenge");
    const confirmed = window.confirm(`Move "${display}" to the .trash folder? You can restore it from disk if needed.`);
    if (!confirmed) return;
    const response = await apiFetch(`/api/chat/html-challenges/${encodeURIComponent(challengeId)}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      const message = await readResponseDetail(response, "Delete challenge failed.");
      const firstSlot = slots[0]?.id ?? "a";
      setSlotStates((current) => ({
        ...current,
        [firstSlot]: { ...current[firstSlot], error: message, done: true },
      }));
      return;
    }
    if (manifest?.id === challengeId || selectedChallengeId === challengeId) {
      newChallenge();
    }
    await refreshChallengeHistory();
  }

  async function persistSlotValidation(challengeId: string, target: CompareTarget, validation: HtmlValidation) {
    try {
      const response = await apiFetch(
        `/api/chat/html-challenges/${encodeURIComponent(challengeId)}/slots/${encodeURIComponent(target)}/validation`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            status: validation.status,
            message: validationMessage(validation),
            issues: validation.issues ?? [],
            source: validation.source ?? "runtime",
          }),
        },
      );
      if (!response.ok) return;
      const payload = await response.json() as { challenge?: HtmlChallengeManifest };
      if (payload.challenge) {
        setManifest(payload.challenge);
      }
    } catch {
      // Runtime preview validation is best-effort; the local card already shows it.
    }
  }

  function stateFromManifestSlot(slot: HtmlChallengeManifestSlot): ChallengeSlotState {
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

  async function loadChallenge(challengeId: string) {
    if (busy || !challengeId) return;
    setLoadingChallengeId(challengeId);
    try {
      const response = await apiFetch(`/api/chat/html-challenges/${encodeURIComponent(challengeId)}`);
      if (!response.ok) return;
      const payload = await response.json() as { challenge?: HtmlChallengeManifest };
      const challenge = payload.challenge;
      if (!challenge) return;

      const manifestSlots = challenge.slots
        .filter((slot) => compareTargets.includes(slot.slotId))
        .slice(0, 4);
      const nextSlots = manifestSlots.length >= 2
        ? manifestSlots.map((slot) => ({
          id: slot.slotId,
          modelKey: modelKeyFromManifestSlot(slot),
          settings: settingsFromManifest(slot.settings, launchSettings),
          thinkingMode: normalizeThinkingMode(slot.thinkingMode ?? challenge.thinkingMode),
          reasoningEffort: normalizeReasoningEffort(slot.reasoningEffort ?? challenge.reasoningEffort),
          seed: typeof slot.seed === "number" ? slot.seed : null,
        }))
        : [
          defaultChallengeSlot("a", launchSettings),
          defaultChallengeSlot("b", launchSettings),
        ];
      const nextStates = emptySlotStates();

      for (const slot of manifestSlots) {
        const nextState = stateFromManifestSlot(slot);
        if (slot.filename) {
          const fileResponse = await apiFetch(
            `/api/chat/html-challenges/${encodeURIComponent(challenge.id)}/files/${encodeURIComponent(slot.slotId)}`,
          );
          if (fileResponse.ok) {
            nextState.html = await fileResponse.text();
            nextState.done = true;
            nextState.deleted = false;
          } else if (fileResponse.status === 404 || fileResponse.status === 410) {
            nextState.done = true;
            nextState.deleted = true;
          } else {
            nextState.done = true;
            nextState.error = await readResponseDetail(fileResponse, "Could not load saved HTML.");
          }
        }
        nextStates[slot.slotId] = nextState;
      }

      setTitle(displayChallengeTitle(challenge));
      setPrompt(challenge.prompt);
      setSlots(nextSlots);
      setSlotStates(nextStates);
      setStreamAtBottom(emptyStreamAtBottom());
      setManifest(challenge);
      setExpandedHtmlSlot(null);
      setCodeViewSlots(emptyCodeViews());
    setPreviewBackgrounds(emptyPreviewBackgrounds());
      setSelectedChallengeId(challenge.id);
      setHistorySearch("");
      setHistoryOpen(false);
    } finally {
      setLoadingChallengeId(null);
    }
  }

  function openPicker(target: CompareTarget) {
    const slot = slots.find((item) => item.id === target);
    setPickerSearch("");
    setPickerDraftKey(slot?.modelKey ?? "");
    setPickerDraftSettings(cloneLaunchSettings(slot?.settings ?? launchSettings));
    setPickerAutoRetry(false);
    setPickerTarget(target);
  }

  // Used after a challenge has run (manifest exists). Confirming the picker
  // automatically re-runs that slot so filename + metadata + rendered HTML
  // all reflect the newly-chosen model — no orphan files or stale labels.
  function openPickerForChangeModel(target: CompareTarget) {
    const slot = slots.find((item) => item.id === target);
    setPickerSearch("");
    setPickerDraftKey(slot?.modelKey ?? "");
    setPickerDraftSettings(cloneLaunchSettings(slot?.settings ?? launchSettings));
    setPickerAutoRetry(true);
    setPickerTarget(target);
  }

  function applyStreamEvent(event: HtmlChallengeStreamEvent) {
    if (event.challenge) {
      setManifest(event.challenge);
      setSelectedChallengeId(event.challenge.id);
      setTitle(displayChallengeTitle(event.challenge));
    }
    if (event.challengeDone) {
      setBusy(false);
      return;
    }
    const target = event.model;
    if (!target) return;
    setSlotStates((current) => {
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
      return { ...current, [target]: next };
    });
  }

  async function runChallenge() {
    if (!title.trim() || !prompt.trim() || !allSelected) return;
    setBusy(true);
    setManifest(null);
    setSlotStates(emptySlotStates());
    setStreamAtBottom(emptyStreamAtBottom());
    setCodeViewSlots(emptyCodeViews());
    setPreviewBackgrounds(emptyPreviewBackgrounds());
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let finalChallengeId = "";
      const response = await apiFetch("/api/chat/html-challenges", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title.trim(),
          prompt: prompt.trim(),
          models: slots.map((slot) => buildRetryModelPayload(slot)!),
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        const firstSlot = slots[0]?.id ?? "a";
        setSlotStates((current) => ({
          ...current,
          [firstSlot]: { ...current[firstSlot], error: detail?.detail ?? "Challenge failed" },
        }));
        setBusy(false);
        return;
      }
      finalChallengeId = await consumeChallengeStream(response);
      await refreshChallengeHistory(finalChallengeId || undefined);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const firstSlot = slots[0]?.id ?? "a";
      setSlotStates((current) => ({
        ...current,
        [firstSlot]: { ...current[firstSlot], error: String(err), done: true },
      }));
    } finally {
      setBusy(false);
    }
  }

  async function retryChallengeSlot(slot: ChallengeSlot, overridePayload?: HtmlChallengeModelPayload) {
    const challengeId = manifest?.id;
    const manifestSlot = manifest?.slots.find((item) => item.slotId === slot.id);
    const modelPayload = overridePayload ?? buildRetryModelPayload(slot, manifestSlot);
    if (busy || !challengeId || !modelPayload) return;

    setBusy(true);
    setStreamAtBottom((current) => ({ ...current, [slot.id]: true }));
    setSlotStates((current) => ({
      ...current,
      [slot.id]: {
        ...emptySlotState(),
        loading: true,
        loadingMessage: "Queued retry...",
      },
    }));

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await apiFetch(
        `/api/chat/html-challenges/${encodeURIComponent(challengeId)}/slots/${encodeURIComponent(slot.id)}/retry`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: modelPayload,
          }),
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        setSlotStates((current) => ({
          ...current,
          [slot.id]: {
            ...current[slot.id],
            error: detail?.detail ?? "Retry failed",
            done: true,
            loading: false,
          },
        }));
        return;
      }
      const finalChallengeId = await consumeChallengeStream(response);
      await refreshChallengeHistory(finalChallengeId || challengeId);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setSlotStates((current) => ({
        ...current,
        [slot.id]: {
          ...current[slot.id],
          error: String(err),
          done: true,
          loading: false,
        },
      }));
    } finally {
      setBusy(false);
    }
  }

  async function repairChallengeSlot(slot: ChallengeSlot, mode: "continue" | "repair") {
    const challengeId = manifest?.id;
    const manifestSlot = manifest?.slots.find((item) => item.slotId === slot.id);
    const modelPayload = buildRetryModelPayload(slot, manifestSlot);
    if (busy || !challengeId || !modelPayload) return;

    setBusy(true);
    setStreamAtBottom((current) => ({ ...current, [slot.id]: true }));
    setSlotStates((current) => ({
      ...current,
      [slot.id]: {
        ...emptySlotState(),
        loading: true,
        loadingMessage: mode === "continue" ? "Queued continuation..." : "Queued repair...",
      },
    }));

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await apiFetch(
        `/api/chat/html-challenges/${encodeURIComponent(challengeId)}/slots/${encodeURIComponent(slot.id)}/repair`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode,
            model: modelPayload,
          }),
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        setSlotStates((current) => ({
          ...current,
          [slot.id]: {
            ...current[slot.id],
            error: detail?.detail ?? "Repair failed",
            done: true,
            loading: false,
          },
        }));
        return;
      }
      const finalChallengeId = await consumeChallengeStream(response);
      await refreshChallengeHistory(finalChallengeId || challengeId);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setSlotStates((current) => ({
        ...current,
        [slot.id]: {
          ...current[slot.id],
          error: String(err),
          done: true,
          loading: false,
        },
      }));
    } finally {
      setBusy(false);
    }
  }

  function cancelChallenge() {
    abortRef.current?.abort();
    setBusy(false);
  }

  function updateSlotThinking(
    slotId: CompareTarget,
    thinkingMode: HtmlChallengeThinkingMode,
    reasoningEffort?: HtmlChallengeReasoningEffort,
  ) {
    updateSlot(slotId, {
      thinkingMode,
      reasoningEffort: reasoningEffort ?? slots.find((slot) => slot.id === slotId)?.reasoningEffort ?? "medium",
    });
  }

  function updateSlotTemperature(slotId: CompareTarget, value: number) {
    updateSlot(slotId, {
      settings: {
        ...(slots.find((slot) => slot.id === slotId)?.settings ?? cloneLaunchSettings(launchSettings)),
        temperature: clampNumber(value, 0, 2),
      },
    });
  }

  function updateSlotSeed(slotId: CompareTarget, value: number | null) {
    updateSlot(slotId, {
      seed: value == null ? null : Math.round(clampNumber(value, 0, 2147483647)),
    });
  }

  function renderModelCard(slot: ChallengeSlot) {
    const option = selectedBySlot[slot.id];
    const manifestSlot = manifest?.slots.find((item) => item.slotId === slot.id);
    const label = option?.label ?? manifestSlot?.displayLabel ?? manifestSlot?.modelName ?? "Select a model";
    const format = option?.format ?? manifestSlot?.format ?? "";
    const quantization = option?.quantization ?? manifestSlot?.quantization ?? "";
    const sizeGb = typeof option?.sizeGb === "number"
      ? option.sizeGb
      : typeof manifestSlot?.sizeGb === "number" ? manifestSlot.sizeGb : null;
    const contextWindow = option?.contextWindow ?? manifestSlot?.contextWindow ?? "";
    const thinkingValue = slot.thinkingMode === "auto" ? slot.reasoningEffort : "off";
    return (
      <div key={slot.id}>
        <span className="eyebrow">{compareTargetLabels[slot.id]}</span>
        <div className="model-selected-card model-selected-card--compact">
          <div className="model-selected-info">
            <div className="html-challenge-slot-headline">
              <strong className="html-challenge-slot-name" title={label}>{label}</strong>
              <div className="model-selected-meta html-challenge-slot-badges">
                {format ? <span className="badge muted">{format}</span> : null}
                {quantization ? <span className="badge muted">{quantization}</span> : null}
                {sizeGb ? <span className="badge muted">{sizeLabel(sizeGb)}</span> : null}
                {contextWindow ? <span className="badge muted">{contextWindow}</span> : null}
              </div>
            </div>
            <small className="muted-text">{summarizeLaunchSettings(slot.settings)}</small>
            {!completedChallenge ? (
              <div className="html-challenge-slot-sampler-row">
                <label>
                  <span>Thinking</span>
                  <select
                    className="text-input"
                    value={thinkingValue}
                    disabled={busy}
                    onChange={(event) => {
                      const next = event.target.value;
                      if (next === "off") updateSlotThinking(slot.id, "off");
                      else updateSlotThinking(slot.id, "auto", next as HtmlChallengeReasoningEffort);
                    }}
                  >
                    <option value="off">Off</option>
                    <option value="low">Low</option>
                    <option value="medium">Med</option>
                    <option value="high">High</option>
                  </select>
                </label>
                <label>
                  <span>Temp</span>
                  <input
                    className="text-input"
                    type="number"
                    min={0}
                    max={2}
                    step={0.05}
                    value={slot.settings.temperature}
                    disabled={busy}
                    onChange={(event) => {
                      const parsed = parseFloat(event.target.value);
                      if (Number.isFinite(parsed)) updateSlotTemperature(slot.id, parsed);
                    }}
                  />
                </label>
                <label className="html-challenge-seed-field">
                  <span>Seed</span>
                  <div className="html-challenge-seed-field-controls">
                    <input
                      className="text-input"
                      type="number"
                      min={0}
                      max={2147483647}
                      step={1}
                      placeholder="random"
                      value={slot.seed ?? ""}
                      disabled={busy}
                      onChange={(event) => {
                        const raw = event.target.value;
                        if (!raw) {
                          updateSlotSeed(slot.id, null);
                          return;
                        }
                        const parsed = parseInt(raw, 10);
                        if (Number.isFinite(parsed)) updateSlotSeed(slot.id, parsed);
                      }}
                    />
                    <button
                      className="secondary-button"
                      type="button"
                      disabled={busy}
                      onClick={() => updateSlotSeed(slot.id, randomChallengeSeed())}
                    >
                      Randomize
                    </button>
                  </div>
                </label>
              </div>
            ) : null}
          </div>
          {!completedChallenge ? (
            <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
              {slots.length > 2 && slot.id === slots[slots.length - 1]?.id ? (
                <button className="secondary-button" type="button" disabled={busy} onClick={removeLastSlot}>
                  Remove
                </button>
              ) : null}
              <button className="secondary-button" type="button" disabled={busy} onClick={() => openPicker(slot.id)}>
                {option || manifestSlot ? "Change" : "Select"}
              </button>
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  function fileActionPath(state: ChallengeSlotState) {
    if (state.filePath) return state.filePath;
    if (manifest?.folderPath && state.filename) return `${manifest.folderPath}/${state.filename}`;
    return "";
  }

  function toggleCodeView(slotId: CompareTarget) {
    setCodeViewSlots((current) => ({ ...current, [slotId]: !current[slotId] }));
  }

  function runtimeCacheDetail(state: ChallengeSlotState) {
    const noteMatch = state.runtimeNote?.match(/(\d+\+\d+\s+cache)/i);
    if (noteMatch?.[1]) return noteMatch[1].toLowerCase();
    const labelMatch = state.metrics?.cacheLabel?.match(/(\d+\+\d+)$/);
    return labelMatch?.[1] ? `${labelMatch[1]} cache` : "";
  }

  function compactSettingsSummary(slot: ChallengeSlot, state: ChallengeSlotState) {
    const parts = summarizeLaunchSettings(slot.settings).split(" · ");
    const cacheDetail = runtimeCacheDetail(state);
    if (cacheDetail && !parts.some((part) => part.toLowerCase().includes("cache"))) {
      parts.splice(1, 0, cacheDetail);
    }
    parts.push(reasoningLabel(slot.thinkingMode, slot.reasoningEffort));
    if (slot.seed != null) parts.push(`seed ${slot.seed}`);
    return parts.join(" · ");
  }

  function slotSubtitle(state: ChallengeSlotState) {
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

  function slotBusyMessage(slot: ChallengeSlot, index: number, manifestSlot?: HtmlChallengeManifestSlot) {
    const state = slotStates[slot.id];
    if (state.loadingMessage) return state.loadingMessage;
    if (manifestSlot?.status === "loading") return "Loading model...";
    if (manifestSlot?.status === "running") return "Generating...";

    const previousPending = slots.slice(0, index).find((previousSlot) => {
      const previousState = slotStates[previousSlot.id];
      const previousManifestSlot = manifest?.slots.find((item) => item.slotId === previousSlot.id);
      if (previousState.done || previousState.deleted || previousState.error) return false;
      if (previousState.loading || previousState.text) return true;
      return previousManifestSlot?.status === "loading"
        || previousManifestSlot?.status === "running"
        || previousManifestSlot?.status === "queued";
    });
    if (previousPending) return `Waiting for ${compareTargetLabels[previousPending.id]} to finish...`;
    return index === 0 ? "Waiting..." : "Waiting to start...";
  }

  function renderValidationBadge(state: ChallengeSlotState) {
    const validation = htmlValidationForState(state);
    if (!validation) return null;
    const status = validation.status;
    const message = validationMessage(validation);
    return (
      <span
        className={`badge ${validationBadgeClass(status)}`}
        title={message || validation.label || htmlValidationLabels[status]}
      >
        {validation.label || htmlValidationLabels[status]}
      </span>
    );
  }

  function renderFileActions(slot: ChallengeSlot, state: ChallengeSlotState) {
    const actionPath = fileActionPath(state);
    const validationBadge = renderValidationBadge(state);
    const validation = htmlValidationForState(state);
    const canExpand = Boolean(state.html && validation?.status !== "no-html");
    const canViewCode = Boolean(state.html && validation?.status !== "no-html");
    const isCodeView = codeViewSlots[slot.id];
    if (!state.filename && !actionPath && !validationBadge && !canExpand) return null;
    const isExpanded = expandedHtmlSlot === slot.id;
    return (
      <div className="html-challenge-file-row">
        {state.filename ? <span className="badge success">{state.filename}</span> : null}
        {validationBadge}
        {actionPath ? (
          <>
            <button
              className="secondary-button html-challenge-icon-button"
              type="button"
              aria-label={fileRevealLabel}
              title={fileRevealLabel}
              onClick={() => onRevealPath(actionPath)}
            >
              <OpenFileIcon />
            </button>
            <button
              className="secondary-button html-challenge-icon-button"
              type="button"
              aria-label="Open in default browser"
              title="Open in default browser"
              onClick={() => onOpenFilePath(actionPath)}
            >
              <BrowserIcon />
            </button>
          </>
        ) : null}
        {canExpand ? (
          <button
            className="secondary-button html-challenge-icon-button"
            type="button"
            aria-label={isExpanded ? "Collapse preview" : "Expand preview"}
            title={isExpanded ? "Collapse preview" : "Expand preview"}
            onClick={() => setExpandedHtmlSlot((current) => current === slot.id ? null : slot.id)}
          >
            {isExpanded ? <CollapseIcon /> : <ExpandIcon />}
          </button>
        ) : null}
        {canViewCode ? (
          <button
            className="secondary-button html-challenge-code-toggle"
            type="button"
            onClick={() => toggleCodeView(slot.id)}
          >
            {isCodeView ? "View Render" : "View HTML Code"}
          </button>
        ) : null}
      </div>
    );
  }

  function renderChallengeSlot(slot: ChallengeSlot, index: number) {
    const state = slotStates[slot.id];
    const option = selectedBySlot[slot.id];
    const manifestSlot = manifest?.slots.find((item) => item.slotId === slot.id);
    const modelLabel = option?.label ?? manifestSlot?.displayLabel ?? manifestSlot?.modelName ?? "";
    const subtitle = slotSubtitle(state) || manifestSlot?.status || "";
    const waitingLabel = slotBusyMessage(slot, index, manifestSlot);
    const showLatestButton = !streamAtBottom[slot.id] && Boolean(state.text) && !state.html;
    const validation = htmlValidationForState(state);
    const retryable = isRetryableState(state);
    const repairable = isRepairableState(state);
    const retryPayload = retryable ? buildRetryModelPayload(slot, manifestSlot) : null;
    const isExpanded = expandedHtmlSlot === slot.id;
    // Show Change Model on every manifested slot (any status) so a wrongly
    // picked or retried-into-the-wrong-model slot can be swapped without
    // first having to fail / repair. Confirming the picker auto-retries to
    // keep filename + manifest + rendered HTML consistent.
    const canChangeModel = Boolean(manifest && manifestSlot);
    const panelActions = isExpanded || showLatestButton || retryable || canChangeModel ? (
      <div className="html-challenge-panel-actions">
        {isExpanded ? (
          <button className="secondary-button" type="button" onClick={() => setExpandedHtmlSlot(null)}>
            Collapse
          </button>
        ) : null}
        {showLatestButton ? (
          <button className="secondary-button" type="button" onClick={() => scrollStreamToBottom(slot.id)}>
            Latest
          </button>
        ) : null}
        {canChangeModel ? (
          <button className="secondary-button" type="button" disabled={busy} onClick={() => openPickerForChangeModel(slot.id)}>
            Change Model
          </button>
        ) : null}
        {retryable ? (
          <>
            {repairable ? (
              <>
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy || !manifest?.id || !retryPayload}
                  onClick={() => void repairChallengeSlot(slot, "continue")}
                >
                  Continue HTML
                </button>
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy || !manifest?.id || !retryPayload}
                  onClick={() => void repairChallengeSlot(slot, "repair")}
                >
                  Repair HTML
                </button>
              </>
            ) : null}
            <button
              className="secondary-button"
              type="button"
              disabled={busy || !manifest?.id || !retryPayload}
              onClick={() => void retryChallengeSlot(slot)}
            >
              Retry
            </button>
          </>
        ) : null}
      </div>
    ) : null;

    return (
      <Panel
        title={compareTargetLabels[slot.id]}
        subtitle={subtitle}
        className={`html-challenge-preview-panel${isExpanded ? " html-challenge-preview-panel--expanded" : ""}`}
        actions={panelActions}
      >
        <div className="html-challenge-panel-body">
          {modelLabel ? (
            <div className="html-challenge-meta">
              <strong>{modelLabel}</strong>
              <span className="html-challenge-settings-summary">{compactSettingsSummary(slot, state)}</span>
            </div>
          ) : null}
          <ReasoningPanel
            className="html-challenge-reasoning"
            text={state.reasoning}
            streaming={!state.reasoningDone}
          />
          {state.error ? (
            <p style={{ color: "#f87171" }}>{state.error}</p>
          ) : state.deleted ? (
            <div className="html-challenge-deleted">
              <strong>File deleted</strong>
              <span>{state.filename ?? "The saved HTML file"} is no longer in this challenge folder.</span>
              <div className="html-challenge-file-row">
                {state.filename ? <span className="badge warning">{state.filename}</span> : null}
                {manifest?.folderPath ? (
                  <button
                    className="secondary-button html-challenge-file-button"
                    type="button"
                    onClick={() => onRevealPath(manifest.folderPath)}
                  >
                    Open Folder
                  </button>
                ) : null}
              </div>
            </div>
          ) : state.done && !state.html ? (
            <div className="html-challenge-empty-result">
              <strong>No HTML output</strong>
              <span>{validationMessage(validation) || "This model finished without a renderable HTML page."}</span>
              {renderFileActions(slot, state)}
            </div>
          ) : state.html && validation?.status === "no-html" ? (
            <div className="html-challenge-empty-result">
              <strong>No HTML output</strong>
              <span>{validationMessage(validation) || "The saved output did not contain a standalone HTML document."}</span>
              {renderFileActions(slot, state)}
            </div>
          ) : state.html ? (
            <>
              {renderFileActions(slot, state)}
              {codeViewSlots[slot.id] ? (
                <pre className="html-challenge-stream html-challenge-code-view">
                  <code dangerouslySetInnerHTML={{ __html: highlightHtmlCode(state.html) }} />
                </pre>
              ) : (
                <div
                  ref={(element) => attachFrameShell(slot.id, element)}
                  className={`html-challenge-frame-shell${expandedHtmlSlot === slot.id ? " html-challenge-frame-shell--expanded" : ""}`}
                  style={previewBackgrounds[slot.id] ? { background: previewBackgrounds[slot.id] as string } : undefined}
                  tabIndex={0}
                  onPointerEnter={() => markPreviewActive(slot.id)}
                  onPointerDownCapture={() => markPreviewActive(slot.id)}
                  onMouseDownCapture={() => focusPreviewFrame(slot.id)}
                  onKeyDown={(event) => forwardPreviewKey(slot.id, event)}
                  onKeyUp={(event) => forwardPreviewKey(slot.id, event)}
                >
                  {/* Iframe fills the shell directly so the page sees the
                      actual rendered pixel size as its viewport — the same
                      way a desktop browser hands its window to a page on
                      resize. No fixed 1280x720 stage, no transform scaling. */}
                  <iframe
                    ref={(element) => { frameRefs.current[slot.id] = element; }}
                    className="html-challenge-frame"
                    title={`${compareTargetLabels[slot.id]} HTML preview`}
                    srcDoc={previewSrcDoc(state.html, slot.id)}
                    sandbox="allow-scripts"
                    scrolling="yes"
                    tabIndex={0}
                    /* `allowTransparency` is a legacy WebKit attribute that
                       lets the iframe's default document canvas show the
                       frame's CSS background instead of opaque white. Pair
                       it with `:where(html, body) { background: transparent }`
                       inside the doc and `background: transparent` on the
                       iframe element so the frame-shell colour shows through
                       any region the model HTML doesn't paint. */
                    // @ts-expect-error legacy attribute, still honored by WebKit
                    allowtransparency="true"
                    onFocus={() => markPreviewActive(slot.id)}
                  />
                </div>
              )}
            </>
          ) : state.text ? (
            <pre
              ref={(element) => { streamRefs.current[slot.id] = element; }}
              className="html-challenge-stream"
              onScroll={() => handleStreamScroll(slot.id)}
            >
              <code dangerouslySetInnerHTML={{ __html: highlightHtmlCode(state.text) }} />
            </pre>
          ) : state.loading ? (
            <p className="muted-text">{state.loadingMessage ?? "Loading model..."}</p>
          ) : busy ? (
            <p className="muted-text">{waitingLabel}</p>
          ) : null}
        </div>
      </Panel>
    );
  }

  function shouldRenderChallengeSlot(slot: ChallengeSlot) {
    const state = slotStates[slot.id];
    return Boolean(
      manifest
        || busy
        || state.loading
        || state.text
        || state.reasoning
        || state.done
        || state.deleted
        || state.error
        || state.html
        || state.filename
        || state.filePath,
    );
  }

  function renderChallengeCard(slot: ChallengeSlot, index: number) {
    return (
      <div key={slot.id} className="html-challenge-card-stack">
        {!busy && !completedChallenge ? renderModelCard(slot) : null}
        {shouldRenderChallengeSlot(slot) ? renderChallengeSlot(slot, index) : null}
      </div>
    );
  }

  return (
    <div className="html-challenge-layout">
      {!expandedHtmlSlot ? (
        <section className="panel html-challenge-setup-panel html-challenge-setup-panel--compact">
          <div className="html-challenge-setup-actions">
            {challenges.length > 0 ? (
              <div className="html-challenge-history-row">
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy || (!manifest && !selectedChallengeId)}
                  onClick={newChallenge}
                >
                  New Challenge
                </button>
                <div className="html-challenge-history-combobox">
                  <input
                    className="text-input html-challenge-history-input"
                    type="search"
                    value={historyInputValue}
                    placeholder="Search previous challenges..."
                    disabled={busy || Boolean(loadingChallengeId)}
                    onFocus={() => {
                      setHistoryOpen(true);
                      setHistorySearch("");
                    }}
                    onChange={(event) => {
                      setHistorySearch(event.target.value);
                      setHistoryOpen(true);
                    }}
                    onBlur={() => {
                      window.setTimeout(() => {
                        setHistoryOpen(false);
                        setHistorySearch("");
                      }, 120);
                    }}
                  />
                  {historyOpen && !busy && !loadingChallengeId ? (
                    <div className="html-challenge-history-menu" role="listbox">
                      {filteredChallenges.map((challenge) => (
                        <div
                          key={challenge.id}
                          role="option"
                          aria-selected={challenge.id === selectedChallengeId}
                          className={`html-challenge-history-option${challenge.id === selectedChallengeId ? " active" : ""}`}
                        >
                          <button
                            type="button"
                            className="html-challenge-history-option-main"
                            onMouseDown={(event) => {
                              event.preventDefault();
                              void loadChallenge(challenge.id);
                            }}
                          >
                            <span>{displayChallengeTitle(challenge)}</span>
                            <small>{formatChallengeDate(challenge.createdAt)}</small>
                          </button>
                          <button
                            type="button"
                            className="html-challenge-history-option-delete"
                            aria-label={`Move "${displayChallengeTitle(challenge)}" to trash`}
                            title="Move to trash"
                            disabled={busy}
                            onMouseDown={(event) => {
                              event.preventDefault();
                              event.stopPropagation();
                              void deleteChallengeById(challenge.id, displayChallengeTitle(challenge));
                            }}
                          >
                            ×
                          </button>
                        </div>
                      ))}
                      {filteredChallenges.length === 0 ? (
                        <p className="html-challenge-history-empty">No matching challenges.</p>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
            <div className="html-challenge-setup-actions-spacer" />
            <div className="html-challenge-layout-toggle" aria-label="HTML challenge layout">
              <button
                className={layoutMode === "row" ? "active" : ""}
                type="button"
                onClick={() => setLayoutMode("row")}
              >
                Row
              </button>
              <button
                className={layoutMode === "stacked" ? "active" : ""}
                type="button"
                onClick={() => setLayoutMode("stacked")}
              >
                {stackedLayoutLabel(slots.length)}
              </button>
            </div>
            {manifest?.folderPath ? (
              <button
                className="secondary-button"
                type="button"
                onClick={() => onRevealPath(manifest.folderPath)}
              >
                Open Folder
              </button>
            ) : null}
            {manifest?.settingsPath ? (
              <button
                className="secondary-button"
                type="button"
                onClick={() => onOpenFilePath(manifest.settingsPath!)}
              >
                Open Settings
              </button>
            ) : null}
            {!completedChallenge ? (
              <button className="secondary-button" type="button" onClick={addSlot} disabled={busy || slots.length >= 4}>
                Add model
              </button>
            ) : null}
            {busy ? (
              <button className="secondary-button" type="button" onClick={cancelChallenge}>Cancel</button>
            ) : completedValidChallenge ? (
              <button
                className="primary-button"
                type="button"
                onClick={usePromptInNewChallenge}
              >
                Use Prompt in New Challenge
              </button>
            ) : (
              <button
                className="primary-button"
                type="button"
                onClick={() => void runChallenge()}
                disabled={!title.trim() || !prompt.trim() || !allSelected}
              >
                {manifest ? "Run New Challenge" : "Run Challenge"}
              </button>
            )}
          </div>
          <div className="html-challenge-controls">
            <input
              className="text-input"
              type="text"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder="Challenge title"
              disabled={busy}
            />
            <textarea
              className="text-input html-challenge-prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Prompt all selected models with the same webpage challenge..."
              disabled={busy}
            />
          </div>
        </section>
      ) : null}

      <div
        className={`html-challenge-grid html-challenge-grid--${expandedHtmlSlot ? "expanded" : layoutMode}`}
        style={{ gridTemplateColumns: expandedHtmlSlot ? "minmax(0, 1fr)" : challengeGridColumns(slots.length, layoutMode) }}
      >
        {visibleSlots.map((slot, index) => renderChallengeCard(slot, index))}
      </div>

      <ModelLaunchModal
        open={pickerTarget != null}
        title={pickerTarget ? `Select ${compareTargetLabels[pickerTarget]}` : "Select Model"}
        confirmLabel={pickerTarget ? `Use for ${compareTargetLabels[pickerTarget]}` : "Use model"}
        selectedKey={pickerDraftKey}
        collapseOnOpen={Boolean(pickerDraftKey)}
        search={pickerSearch}
        options={textModelOptions}
        settings={pickerDraftSettings}
        preview={pickerDraftPreview}
        availableMemoryGb={availableMemoryGb}
        totalMemoryGb={totalMemoryGb}
        gpuVramTotalGb={gpuVramTotalGb}
        availableCacheStrategies={availableCacheStrategies}
        dflashInfo={dflashInfo}
        installingPackage={installingPackage ?? null}
        installLogs={installLogs}
        turboInstalled={turboInstalled}
        onSelectedKeyChange={setPickerDraftKey}
        onSearchChange={setPickerSearch}
        onSettingChange={(key, value) => {
          setPickerDraftSettings((current) => ({ ...current, [key]: value }));
        }}
        onConfirm={(selectedKey) => {
          if (pickerTarget) {
            const target = pickerTarget;
            const newSettings = cloneLaunchSettings(pickerDraftSettings);
            const slot = slots.find((item) => item.id === target);
            updateSlot(target, {
              modelKey: selectedKey,
              settings: newSettings,
            });
            if (pickerAutoRetry && slot && manifest?.id) {
              const option = textModelOptions.find((item) => item.key === selectedKey);
              if (option) {
                const payload: HtmlChallengeModelPayload = {
                  ...buildComparePayload(option, newSettings),
                  thinkingMode: slot.thinkingMode,
                  reasoningEffort: slot.thinkingMode === "auto" ? slot.reasoningEffort : undefined,
                  seed: slot.seed,
                };
                // Run after state flush so the slot's modelKey reflects the
                // newly chosen option in any subsequent UI reads.
                window.setTimeout(() => {
                  void retryChallengeSlot({ ...slot, modelKey: selectedKey, settings: newSettings }, payload);
                }, 0);
              }
            }
          }
          setPickerSearch("");
          setPickerAutoRetry(false);
          setPickerTarget(null);
        }}
        onClose={() => {
          setPickerSearch("");
          setPickerAutoRetry(false);
          setPickerTarget(null);
        }}
        onInstallPackage={installPackage}
      />
    </div>
  );
}
