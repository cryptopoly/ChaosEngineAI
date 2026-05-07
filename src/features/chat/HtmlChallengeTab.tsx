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
  filename?: string;
  filePath?: string;
  fileBytes?: number;
  validHtmlDocument?: boolean;
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
  loadSeconds?: number;
  totalSeconds?: number;
}

type HtmlChallengeLayoutMode = "row" | "stacked";
type HtmlChallengeThinkingMode = "off" | "auto";
type HtmlChallengeReasoningEffort = "low" | "medium" | "high";

const htmlChallengePreviewKeyBridge = `
<script>
(function () {
  window.addEventListener("message", function (event) {
    var data = event.data;
    if (!data || data.__htmlChallengePreviewKey !== true) return;
    var init = {
      key: data.key,
      code: data.code,
      keyCode: data.keyCode,
      which: data.which,
      bubbles: true,
      cancelable: true,
      repeat: Boolean(data.repeat),
      altKey: Boolean(data.altKey),
      ctrlKey: Boolean(data.ctrlKey),
      metaKey: Boolean(data.metaKey),
      shiftKey: Boolean(data.shiftKey)
    };
    var targets = [window, document, document.activeElement || document.body, document.body];
    targets.forEach(function (target) {
      if (!target || typeof target.dispatchEvent !== "function") return;
      try {
        target.dispatchEvent(new KeyboardEvent(data.type || "keydown", init));
      } catch (_) {}
    });
  });
})();
</script>`;

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

function previewSrcDoc(html: string) {
  const csp = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline';">`;
  const injection = `${csp}${htmlChallengePreviewKeyBridge}`;
  if (/<head[^>]*>/i.test(html)) {
    return html.replace(/<head([^>]*)>/i, `<head$1>${injection}`);
  }
  return `${injection}${html}`;
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

function challengeHistoryLabel(challenge: HtmlChallengeManifest) {
  return `${challenge.title} · ${formatChallengeDate(challenge.createdAt)}`;
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

function OpenFileIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <path d="M4 6.5A2.5 2.5 0 0 1 6.5 4H10l2 2h5.5A2.5 2.5 0 0 1 20 8.5v9A2.5 2.5 0 0 1 17.5 20h-11A2.5 2.5 0 0 1 4 17.5z" />
      <path d="M8 13h8" />
      <path d="m13 10 3 3-3 3" />
    </svg>
  );
}

function BrowserIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" />
      <path d="M3.6 9h16.8" />
      <path d="M3.6 15h16.8" />
      <path d="M12 3a13.5 13.5 0 0 1 0 18" />
      <path d="M12 3a13.5 13.5 0 0 0 0 18" />
    </svg>
  );
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
    { id: "a", modelKey: "", settings: cloneLaunchSettings(launchSettings) },
    { id: "b", modelKey: "", settings: cloneLaunchSettings(launchSettings) },
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
  const [thinkingMode, setThinkingMode] = useState<HtmlChallengeThinkingMode>("off");
  const [reasoningEffort, setReasoningEffort] = useState<HtmlChallengeReasoningEffort>("medium");
  const [streamAtBottom, setStreamAtBottom] = useState<Record<CompareTarget, boolean>>(emptyStreamAtBottom);
  const [pickerTarget, setPickerTarget] = useState<CompareTarget | null>(null);
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
  const selectedChallenge = challenges.find((challenge) => challenge.id === selectedChallengeId) ?? null;
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
    const handles = slots
      .filter((slot) => streamAtBottom[slot.id])
      .map((slot) => requestAnimationFrame(() => scrollStreamToBottom(slot.id)));
    return () => handles.forEach((handle) => cancelAnimationFrame(handle));
  }, [slots, slotStates, streamAtBottom]);

  function updateSlot(slotId: CompareTarget, patch: Partial<ChallengeSlot>) {
    setSlots((current) => current.map((slot) => slot.id === slotId ? { ...slot, ...patch } : slot));
  }

  function addSlot() {
    if (busy || slots.length >= 4) return;
    const nextId = compareTargets[slots.length];
    if (!nextId) return;
    setSlots((current) => [
      ...current,
      { id: nextId, modelKey: "", settings: cloneLaunchSettings(launchSettings) },
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
    if (option) return buildComparePayload(option, slot.settings);
    if (!manifestSlot?.modelRef) return null;
    return {
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
    } satisfies ReturnType<typeof buildComparePayload>;
  }

  function isRetryableState(state: ChallengeSlotState) {
    return Boolean(state.error || (state.done && (!state.html || state.validHtmlDocument === false)));
  }

  function focusPreviewFrame(target: CompareTarget) {
    const frame = frameRefs.current[target];
    if (!frame) return;
    frame.focus();
    try {
      frame.contentWindow?.focus();
    } catch {
      // Sandboxed frames can reject focus in some WebView builds.
    }
  }

  function forwardPreviewKey(target: CompareTarget, event: ReactKeyboardEvent<HTMLElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const frame = frameRefs.current[target];
    if (!frame?.contentWindow) return;
    const key = event.key.toLowerCase();
    if (htmlChallengeGameKeys.has(key)) {
      event.preventDefault();
    }
    frame.contentWindow.postMessage({
      __htmlChallengePreviewKey: true,
      type: event.type,
      key: event.key,
      code: event.code,
      keyCode: event.keyCode,
      which: event.which,
      repeat: event.repeat,
      altKey: event.altKey,
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
      shiftKey: event.shiftKey,
    }, "*");
  }

  function newChallenge() {
    if (busy) return;
    setTitle("");
    setPrompt("");
    setThinkingMode("off");
    setReasoningEffort("medium");
    setManifest(null);
    setSelectedChallengeId("");
    setHistorySearch("");
    setHistoryOpen(false);
    setSlotStates(emptySlotStates());
    setStreamAtBottom(emptyStreamAtBottom());
    setSlots([
      { id: "a", modelKey: "", settings: cloneLaunchSettings(launchSettings) },
      { id: "b", modelKey: "", settings: cloneLaunchSettings(launchSettings) },
    ]);
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
        }))
        : [
          { id: "a" as const, modelKey: "", settings: cloneLaunchSettings(launchSettings) },
          { id: "b" as const, modelKey: "", settings: cloneLaunchSettings(launchSettings) },
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

      setTitle(challenge.title);
      setPrompt(challenge.prompt);
      setThinkingMode(challenge.thinkingMode === "auto" ? "auto" : "off");
      setReasoningEffort(
        challenge.reasoningEffort === "low" || challenge.reasoningEffort === "high"
          ? challenge.reasoningEffort
          : "medium",
      );
      setSlots(nextSlots);
      setSlotStates(nextStates);
      setStreamAtBottom(emptyStreamAtBottom());
      setManifest(challenge);
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
    setPickerTarget(target);
  }

  function applyStreamEvent(event: HtmlChallengeStreamEvent) {
    if (event.challenge) {
      setManifest(event.challenge);
      setSelectedChallengeId(event.challenge.id);
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
        next = { ...next, loading: false, text: next.text + event.token };
      }
      if (event.done) {
        next = {
          ...next,
          done: true,
          loading: false,
          deleted: false,
          reasoningDone: true,
          text: event.text ?? next.text,
          html: event.html ?? next.html,
          filename: event.filename,
          filePath: event.filePath,
          fileBytes: event.fileBytes,
          validHtmlDocument: event.validHtmlDocument,
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
          thinkingMode,
          reasoningEffort: thinkingMode === "auto" ? reasoningEffort : undefined,
          models: slots.map((slot) => buildComparePayload(selectedBySlot[slot.id]!, slot.settings)),
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

  async function retryChallengeSlot(slot: ChallengeSlot) {
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
            thinkingMode,
            reasoningEffort: thinkingMode === "auto" ? reasoningEffort : undefined,
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

  function cancelChallenge() {
    abortRef.current?.abort();
    setBusy(false);
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
    return (
      <div key={slot.id}>
        <span className="eyebrow">{compareTargetLabels[slot.id]}</span>
        <div className="model-selected-card" style={{ minHeight: 104 }}>
          <div className="model-selected-info">
            <strong>{label}</strong>
            <div className="model-selected-meta">
              {format ? <span className="badge muted">{format}</span> : null}
              {quantization ? <span className="badge muted">{quantization}</span> : null}
              {sizeGb ? <span className="badge muted">{sizeLabel(sizeGb)}</span> : null}
              {contextWindow ? <span className="badge muted">{contextWindow}</span> : null}
            </div>
            <small className="muted-text">{summarizeLaunchSettings(slot.settings)}</small>
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
    parts.push(reasoningLabel(thinkingMode, reasoningEffort));
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

  function renderFileActions(state: ChallengeSlotState) {
    const actionPath = fileActionPath(state);
    if (!state.filename && !actionPath) return null;
    return (
      <div className="html-challenge-file-row">
        {state.filename ? <span className="badge success">{state.filename}</span> : null}
        {state.validHtmlDocument === false ? <span className="badge warning">HTML rescued</span> : null}
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
      </div>
    );
  }

  function renderChallengeSlot(slot: ChallengeSlot, index: number) {
    const state = slotStates[slot.id];
    const option = selectedBySlot[slot.id];
    const manifestSlot = manifest?.slots.find((item) => item.slotId === slot.id);
    const modelLabel = option?.label ?? manifestSlot?.displayLabel ?? manifestSlot?.modelName ?? "";
    const subtitle = slotSubtitle(state) || manifestSlot?.status || "";
    const waitingLabel = index === 0 ? "Waiting..." : `Waiting for ${compareTargetLabels[slots[index - 1]?.id ?? "a"]} to finish...`;
    const showLatestButton = !streamAtBottom[slot.id] && Boolean(state.text) && !state.html;
    const retryable = isRetryableState(state);
    const retryPayload = retryable ? buildRetryModelPayload(slot, manifestSlot) : null;
    const panelActions = showLatestButton || retryable ? (
      <div className="html-challenge-panel-actions">
        {showLatestButton ? (
          <button className="secondary-button" type="button" onClick={() => scrollStreamToBottom(slot.id)}>
            Latest
          </button>
        ) : null}
        {retryable ? (
          <>
            <button className="secondary-button" type="button" disabled={busy} onClick={() => openPicker(slot.id)}>
              Change Model
            </button>
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
        className="html-challenge-preview-panel"
        actions={panelActions}
      >
        <div className="html-challenge-panel-body">
          {modelLabel ? (
            <div className="html-challenge-meta">
              <strong>{modelLabel}</strong>
              <span className="html-challenge-settings-summary">{compactSettingsSummary(slot, state)}</span>
            </div>
          ) : null}
          <ReasoningPanel text={state.reasoning} streaming={!state.reasoningDone} />
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
              <span>This model finished without a renderable HTML page.</span>
            </div>
          ) : state.html ? (
            <>
              {renderFileActions(state)}
              <div
                className="html-challenge-frame-shell"
                tabIndex={0}
                onMouseDownCapture={() => focusPreviewFrame(slot.id)}
                onKeyDown={(event) => forwardPreviewKey(slot.id, event)}
                onKeyUp={(event) => forwardPreviewKey(slot.id, event)}
              >
                <iframe
                  ref={(element) => { frameRefs.current[slot.id] = element; }}
                  className="html-challenge-frame"
                  title={`${compareTargetLabels[slot.id]} HTML preview`}
                  srcDoc={previewSrcDoc(state.html)}
                  sandbox="allow-scripts"
                  tabIndex={0}
                  onFocus={() => focusPreviewFrame(slot.id)}
                />
              </div>
            </>
          ) : state.text ? (
            <pre
              ref={(element) => { streamRefs.current[slot.id] = element; }}
              className="html-challenge-stream"
              onScroll={() => handleStreamScroll(slot.id)}
            >
              {state.text}
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

  function renderChallengeCard(slot: ChallengeSlot, index: number) {
    return (
      <div key={slot.id} className="html-challenge-card-stack">
        {!completedChallenge ? renderModelCard(slot) : null}
        {renderChallengeSlot(slot, index)}
      </div>
    );
  }

  return (
    <div className="html-challenge-layout">
      <Panel
        title="HTML Challenge"
        subtitle={manifest?.folderPath ?? "Create a shareable webpage comparison"}
        actions={
          <>
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
          </>
        }
      >
        <div className="html-challenge-controls">
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
                      <button
                        key={challenge.id}
                        type="button"
                        role="option"
                        aria-selected={challenge.id === selectedChallengeId}
                        className={`html-challenge-history-option${challenge.id === selectedChallengeId ? " active" : ""}`}
                        onMouseDown={(event) => {
                          event.preventDefault();
                          void loadChallenge(challenge.id);
                        }}
                      >
                        <span>{challenge.title}</span>
                        <small>{formatChallengeDate(challenge.createdAt)}</small>
                      </button>
                    ))}
                    {filteredChallenges.length === 0 ? (
                      <p className="html-challenge-history-empty">No matching challenges.</p>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          ) : null}
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
          <div className="html-challenge-thinking-row">
            <span className="composer-mode-label">Thinking</span>
            <div className="thread-mode-toggle composer-thinking-toggle" role="group" aria-label="HTML Challenge thinking mode">
              <button
                type="button"
                className={`thread-mode-button${thinkingMode === "off" ? " thread-mode-button--active" : ""}`}
                disabled={busy}
                onClick={() => setThinkingMode("off")}
                title="Ask the model for direct output and suppress reasoning capture"
              >
                Off
              </button>
              {(["low", "medium", "high"] as HtmlChallengeReasoningEffort[]).map((effort) => (
                <button
                  key={effort}
                  type="button"
                  className={`thread-mode-button${thinkingMode === "auto" && reasoningEffort === effort ? " thread-mode-button--active" : ""}`}
                  disabled={busy}
                  onClick={() => {
                    setThinkingMode("auto");
                    setReasoningEffort(effort);
                  }}
                  title={`${effort[0].toUpperCase()}${effort.slice(1)} reasoning effort`}
                >
                  {effort === "medium" ? "Med" : effort[0].toUpperCase() + effort.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
            {busy ? (
              <button className="secondary-button" type="button" onClick={cancelChallenge}>Cancel</button>
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
        </div>
      </Panel>

      <div
        className={`html-challenge-grid html-challenge-grid--${layoutMode}`}
        style={{ gridTemplateColumns: challengeGridColumns(slots.length, layoutMode) }}
      >
        {slots.map((slot, index) => renderChallengeCard(slot, index))}
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
            updateSlot(pickerTarget, {
              modelKey: selectedKey,
              settings: cloneLaunchSettings(pickerDraftSettings),
            });
          }
          setPickerSearch("");
          setPickerTarget(null);
        }}
        onClose={() => {
          setPickerSearch("");
          setPickerTarget(null);
        }}
        onInstallPackage={installPackage}
      />
    </div>
  );
}
