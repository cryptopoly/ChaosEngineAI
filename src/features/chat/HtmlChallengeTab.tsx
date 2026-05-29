import { type KeyboardEvent as ReactKeyboardEvent, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { apiFetch } from "../../api";
import type { MtplxJobState } from "../../api";
import type { LaunchPreferences, StrategyInstallLog, SystemStats } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import {
  buildComparePayload,
  cloneLaunchSettings,
  compareTargetLabels,
  compareTargets,
  modelUsesMtplx,
  summarizeLaunchSettings,
  type CompareTarget,
} from "./CompareView";
import {
  type ChallengeSlot,
  type ChallengeSlotState,
  type HtmlChallengeLayoutMode,
  type HtmlChallengeManifest,
  type HtmlChallengeStreamEvent,
  type HtmlChallengeReasoningEffort,
  type HtmlChallengeThinkingMode,
  type HtmlValidation,
  challengeGridColumns,
  clampNumber,
  defaultChallengeSlot,
  type HtmlChallengeManifestSlot,
  displayChallengeTitle,
  emptyCodeViews,
  emptyPreviewBackgrounds,
  emptySlotState,
  emptySlotStates,
  emptyStreamAtBottom,
  htmlValidationForState,
  htmlValidationLabels,
  isCompareTarget,
  isHtmlValidationStatus,
  isTextModelOption,
  mergeMetrics,
  normalizeReasoningEffort,
  normalizeThinkingMode,
  randomChallengeSeed,
  settingsFromManifest,
  stackedLayoutLabel,
  validationMessage,
} from "./htmlChallengeHelpers";
import {
  type HtmlChallengeModelPayload,
  applySlotStreamEvent,
  buildRetryModelPayload,
  compactSettingsSummary,
  isEditableKeyboardTarget,
  isPreviewGameKey,
  isRepairableState,
  isRetryableState,
  modelKeyFromManifestSlot,
  slotBusyMessage,
  slotSubtitle,
  stateFromManifestSlot,
} from "./html_challenge/htmlChallengeTabHelpers";
import {
  deleteChallenge,
  fetchChallenge,
  fetchChallengeFile,
  fetchChallengeList,
  patchSlotValidation,
  readResponseDetail,
} from "./html_challenge/challengeApi";
import { ChallengeHistoryCombobox } from "./html_challenge/ChallengeHistoryCombobox";
import { ChallengeModelCard } from "./html_challenge/ChallengeModelCard";
import { ChallengePickerModal } from "./html_challenge/ChallengePickerModal";
import { ChallengePromptLibraryModal } from "./html_challenge/ChallengePromptLibraryModal";
import { ChallengeSetupPanel } from "./html_challenge/ChallengeSetupPanel";
import { ChallengeSlotPanel } from "./html_challenge/ChallengeSlotPanel";

interface HtmlChallengeTabProps {
  modelOptions: ChatModelOption[];
  launchSettings: LaunchPreferences;
  availableMemoryGb: number;
  totalMemoryGb: number;
  gpuVramTotalGb?: number | null;
  availableCacheStrategies?: SystemStats["availableCacheStrategies"];
  dflashInfo?: SystemStats["dflash"];
  turboInstalled?: boolean;
  mtplxSystemInfo?: SystemStats["mtplx"];
  onInstallMtplx?: () => void;
  installingMtplx?: boolean;
  mtplxJob?: MtplxJobState | null;
  /** FU-056 follow-up: hide MTPLX block on non-Apple-Silicon hosts. */
  isAppleSilicon?: boolean;
  onInstallPackage?: (strategyId: string) => void;
  installingPackage?: string | null;
  installLogs?: Record<string, StrategyInstallLog>;
  fileRevealLabel: string;
  onRevealPath: (path: string) => void;
  onOpenFilePath: (path: string) => void;
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
  mtplxSystemInfo,
  onInstallMtplx,
  installingMtplx,
  mtplxJob,
  isAppleSilicon = false,
  onInstallPackage,
  installingPackage,
  installLogs,
  fileRevealLabel,
  onRevealPath,
  onOpenFilePath,
}: HtmlChallengeTabProps) {
  const { t } = useTranslation("chat");
  const [title, setTitle] = useState("");
  const [prompt, setPrompt] = useState("");
  const [promptLibraryOpen, setPromptLibraryOpen] = useState(false);
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
  const [pickerInitialKey, setPickerInitialKey] = useState("");
  const [pickerInitialSettings, setPickerInitialSettings] = useState<LaunchPreferences>(() => cloneLaunchSettings(launchSettings));
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
  // FU-036: timestamp of the most recent programmatic scroll-to-bottom for
  // each slot. ``handleStreamScroll`` ignores scroll events fired in the
  // immediate window after one (the browser fires ``scroll`` for both
  // user wheel input AND ``element.scrollTop = …`` writes; without the
  // guard the post-write event re-flips ``streamAtBottom`` to true even
  // when the user just scrolled away).
  const lastProgrammaticScrollRef = useRef<Record<string, number>>({});

  const textModelOptions = modelOptions.filter(isTextModelOption);
  const selectedBySlot = Object.fromEntries(
    slots.map((slot) => [slot.id, textModelOptions.find((option) => option.key === slot.modelKey) ?? null]),
  ) as Record<CompareTarget, ChatModelOption | null>;
  const allSelected = slots.every((slot) => selectedBySlot[slot.id] != null);
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
    // FU-036 (2026-05-10): re-measure scroll position inside the rAF
    // before yanking back to bottom. ``setStreamAtBottom`` from the
    // ``onScroll`` handler is async, so a streaming chunk that arrives
    // a few ms after the user wheel-scrolls would otherwise see the
    // stale ``streamAtBottom[slot.id] === true`` from the previous
    // render and snap the box back down — felt as a jerk that fought
    // the user's scroll. Re-measuring against the live DOM closes the
    // race; if the user has moved away in the gap, we drop tracking
    // for that slot instead of stomping their scroll position.
    const handles = slots
      .filter((slot) => streamAtBottom[slot.id])
      .map((slot) => requestAnimationFrame(() => {
        const element = streamRefs.current[slot.id];
        if (!element) return;
        const stillNearBottom =
          element.scrollHeight - element.scrollTop - element.clientHeight < 32;
        if (stillNearBottom) {
          scrollStreamToBottom(slot.id);
        } else {
          setStreamAtBottom((current) => current[slot.id]
            ? { ...current, [slot.id]: false }
            : current);
        }
      }));
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
    // Ignore the scroll event the browser fires immediately after our
    // own ``element.scrollTop = …`` write. Without this guard, the
    // post-write event re-flipped ``streamAtBottom`` true and the next
    // chunk would yank the box back even when the user had since
    // scrolled away.
    const lastProgrammatic = lastProgrammaticScrollRef.current[target] ?? 0;
    if (performance.now() - lastProgrammatic < 80) return;
    const element = streamRefs.current[target];
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 32;
    setStreamAtBottom((current) => ({ ...current, [target]: atBottom }));
  }

  function scrollStreamToBottom(target: CompareTarget) {
    const element = streamRefs.current[target];
    if (!element) return;
    lastProgrammaticScrollRef.current[target] = performance.now();
    element.scrollTop = element.scrollHeight;
    setStreamAtBottom((current) => current[target] ? current : { ...current, [target]: true });
  }

  function attachStream(target: CompareTarget, element: HTMLPreElement | null) {
    streamRefs.current[target] = element;
  }

  function attachFrame(target: CompareTarget, element: HTMLIFrameElement | null) {
    frameRefs.current[target] = element;
  }

  function attachFrameShell(target: CompareTarget, element: HTMLDivElement | null) {
    frameShellRefs.current[target] = element;
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

  function retryPayloadForSlot(slot: ChallengeSlot, manifestSlot?: HtmlChallengeManifestSlot) {
    return buildRetryModelPayload(slot, selectedBySlot[slot.id], manifestSlot);
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
    const nextChallenges = await fetchChallengeList();
    setChallenges(nextChallenges);
    setSelectedChallengeId((current) => {
      if (selectId) return selectId;
      return current && nextChallenges.some((challenge) => challenge.id === current)
        ? current
        : "";
    });
    return nextChallenges;
  }

  async function deleteChallengeById(challengeId: string, label?: string) {
    if (busy || !challengeId) return;
    const display = label
      || (manifest?.id === challengeId ? displayChallengeTitle(manifest) : "")
      || (selectedChallenge?.id === challengeId ? displayChallengeTitle(selectedChallenge) : "")
      || (challenges.find((c) => c.id === challengeId)
        ? displayChallengeTitle(challenges.find((c) => c.id === challengeId)!)
        : t("htmlChallenge.thisChallengeFallback", { defaultValue: "this challenge" }));
    const confirmed = window.confirm(t("htmlChallenge.deleteConfirm", {
      defaultValue: "Move \"{title}\" to the .trash folder? You can restore it from disk if needed.",
      title: display,
    }));
    if (!confirmed) return;
    const result = await deleteChallenge(challengeId);
    if (!result.ok) {
      const firstSlot = slots[0]?.id ?? "a";
      setSlotStates((current) => ({
        ...current,
        [firstSlot]: {
          ...current[firstSlot],
          error: result.error ?? t("htmlChallenge.errors.deleteFailed", { defaultValue: "Delete challenge failed." }),
          done: true,
        },
      }));
      return;
    }
    if (manifest?.id === challengeId || selectedChallengeId === challengeId) {
      newChallenge();
    }
    await refreshChallengeHistory();
  }

  async function persistSlotValidation(challengeId: string, target: CompareTarget, validation: HtmlValidation) {
    const updated = await patchSlotValidation(challengeId, target, validation);
    if (updated) setManifest(updated);
  }

  async function loadChallenge(challengeId: string) {
    if (busy || !challengeId) return;
    setLoadingChallengeId(challengeId);
    try {
      const challenge = await fetchChallenge(challengeId);
      if (!challenge) return;

      const manifestSlots = challenge.slots
        .filter((slot) => compareTargets.includes(slot.slotId))
        .slice(0, 4);
      const nextSlots = manifestSlots.length >= 2
        ? manifestSlots.map((slot) => ({
          id: slot.slotId,
          modelKey: modelKeyFromManifestSlot(slot, textModelOptions),
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
          const fileResult = await fetchChallengeFile(challenge.id, slot.slotId);
          nextState.done = true;
          if (fileResult.status === "ok") {
            nextState.html = fileResult.html ?? "";
            nextState.deleted = false;
          } else if (fileResult.status === "deleted") {
            nextState.deleted = true;
          } else {
            nextState.error = fileResult.error;
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
    setPickerInitialKey(slot?.modelKey ?? "");
    setPickerInitialSettings(cloneLaunchSettings(slot?.settings ?? launchSettings));
    setPickerAutoRetry(false);
    setPickerTarget(target);
  }

  // Used after a challenge has run (manifest exists). Confirming the picker
  // automatically re-runs that slot so filename + metadata + rendered HTML
  // all reflect the newly-chosen model — no orphan files or stale labels.
  function openPickerForChangeModel(target: CompareTarget) {
    const slot = slots.find((item) => item.id === target);
    setPickerInitialKey(slot?.modelKey ?? "");
    setPickerInitialSettings(cloneLaunchSettings(slot?.settings ?? launchSettings));
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
          loadingMessage: t("htmlChallenge.status.generating", { defaultValue: "Generating..." }),
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
          models: slots.map((slot) => retryPayloadForSlot(slot)!),
        }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        const firstSlot = slots[0]?.id ?? "a";
        setSlotStates((current) => ({
          ...current,
          [firstSlot]: {
            ...current[firstSlot],
            error: detail?.detail ?? t("htmlChallenge.errors.challengeFailed", { defaultValue: "Challenge failed" }),
          },
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
    const modelPayload = overridePayload ?? retryPayloadForSlot(slot, manifestSlot);
    if (busy || !challengeId || !modelPayload) return;

    setBusy(true);
    setStreamAtBottom((current) => ({ ...current, [slot.id]: true }));
    setSlotStates((current) => ({
      ...current,
      [slot.id]: {
        ...emptySlotState(),
        loading: true,
        loadingMessage: t("htmlChallenge.status.queuedRetry", { defaultValue: "Queued retry..." }),
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
            error: detail?.detail ?? t("htmlChallenge.errors.retryFailed", { defaultValue: "Retry failed" }),
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
    const modelPayload = retryPayloadForSlot(slot, manifestSlot);
    if (busy || !challengeId || !modelPayload) return;

    setBusy(true);
    setStreamAtBottom((current) => ({ ...current, [slot.id]: true }));
    setSlotStates((current) => ({
      ...current,
      [slot.id]: {
        ...emptySlotState(),
        loading: true,
        loadingMessage: mode === "continue"
          ? t("htmlChallenge.status.queuedContinuation", { defaultValue: "Queued continuation..." })
          : t("htmlChallenge.status.queuedRepair", { defaultValue: "Queued repair..." }),
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
            error: detail?.detail ?? t("htmlChallenge.errors.repairFailed", { defaultValue: "Repair failed" }),
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

  function toggleCodeView(slotId: CompareTarget) {
    setCodeViewSlots((current) => ({ ...current, [slotId]: !current[slotId] }));
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
    const state = slotStates[slot.id];
    const option = selectedBySlot[slot.id];
    const manifestSlot = manifest?.slots.find((item) => item.slotId === slot.id);
    const modelLabel = option?.label ?? manifestSlot?.displayLabel ?? manifestSlot?.modelName ?? "";
    const subtitle = slotSubtitle(state) || manifestSlot?.status || "";
    const waitingLabel = slotBusyMessage(slot, index, manifestSlot, slots, slotStates, manifest?.slots, compareTargetLabels);
    const showLatestButton = !streamAtBottom[slot.id] && Boolean(state.text) && !state.html;
    const retryable = isRetryableState(state);
    const repairable = isRepairableState(state);
    const retryPayload = retryable ? retryPayloadForSlot(slot, manifestSlot) : null;
    const isExpanded = expandedHtmlSlot === slot.id;
    // Show Change Model on every manifested slot (any status) so a wrongly
    // picked or retried-into-the-wrong-model slot can be swapped without
    // first having to fail / repair. Confirming the picker auto-retries to
    // keep filename + manifest + rendered HTML consistent.
    const canChangeModel = Boolean(manifest && manifestSlot);
    // Bind the option + mtplx context so the summary label reflects which
    // engine the backend will actually route to.
    const summarizeForSlot = (settings: ChallengeSlot["settings"]) =>
      summarizeLaunchSettings(settings, { usesMtplx: modelUsesMtplx(option, settings, mtplxSystemInfo) });
    const settingsSummary = compactSettingsSummary(slot, state, summarizeForSlot);

    return (
      <div key={slot.id} className="html-challenge-card-stack">
        {!busy && !completedChallenge ? (
          <ChallengeModelCard
            slot={slot}
            option={option}
            manifestSlot={manifestSlot}
            busy={busy}
            completedChallenge={completedChallenge}
            isLastSlot={slot.id === slots[slots.length - 1]?.id}
            canRemove={slots.length > 2}
            summary={summarizeForSlot(slot.settings)}
            onUpdateThinking={updateSlotThinking}
            onUpdateTemperature={updateSlotTemperature}
            onUpdateSeed={updateSlotSeed}
            onRemoveLastSlot={removeLastSlot}
            onOpenPicker={openPicker}
          />
        ) : null}
        {shouldRenderChallengeSlot(slot) ? (
          <ChallengeSlotPanel
            slot={slot}
            state={state}
            manifest={manifest}
            manifestSlot={manifestSlot}
            modelLabel={modelLabel}
            subtitle={subtitle}
            waitingLabel={waitingLabel}
            busy={busy}
            isExpanded={isExpanded}
            showLatestButton={showLatestButton}
            retryable={retryable}
            repairable={repairable}
            hasRetryPayload={Boolean(retryPayload)}
            canChangeModel={canChangeModel}
            isCodeView={codeViewSlots[slot.id]}
            previewBackground={previewBackgrounds[slot.id]}
            fileRevealLabel={fileRevealLabel}
            settingsSummary={settingsSummary}
            onSetExpanded={setExpandedHtmlSlot}
            onScrollStreamToBottom={scrollStreamToBottom}
            onToggleCodeView={toggleCodeView}
            onChangeModel={openPickerForChangeModel}
            onRetrySlot={() => void retryChallengeSlot(slot)}
            onRepairSlot={(mode) => void repairChallengeSlot(slot, mode)}
            onRevealPath={onRevealPath}
            onOpenFilePath={onOpenFilePath}
            onAttachStream={attachStream}
            onAttachFrame={attachFrame}
            onAttachFrameShell={attachFrameShell}
            onStreamScroll={handleStreamScroll}
            onMarkPreviewActive={markPreviewActive}
            onFocusPreviewFrame={focusPreviewFrame}
            onForwardPreviewKey={forwardPreviewKey}
          />
        ) : null}
      </div>
    );
  }

  return (
    <div className="html-challenge-layout">
      {!expandedHtmlSlot ? (
        <section className="panel html-challenge-setup-panel html-challenge-setup-panel--compact">
          <div className="html-challenge-setup-actions">
            <button
              className="secondary-button"
              type="button"
              disabled={busy}
              onClick={() => setPromptLibraryOpen(true)}
            >
              {t("htmlChallenge.actions.browsePrompts", { defaultValue: "Prompt library" })}
            </button>
            {challenges.length > 0 ? (
              <div className="html-challenge-history-row">
                <button
                  className="secondary-button"
                  type="button"
                  disabled={busy || (!manifest && !selectedChallengeId)}
                  onClick={newChallenge}
                >
                  {t("htmlChallenge.actions.newChallenge", { defaultValue: "New Challenge" })}
                </button>
                <ChallengeHistoryCombobox
                  challenges={challenges}
                  selectedChallengeId={selectedChallengeId}
                  historySearch={historySearch}
                  historyOpen={historyOpen}
                  busy={busy}
                  loadingChallengeId={loadingChallengeId}
                  onHistorySearchChange={setHistorySearch}
                  onHistoryOpenChange={setHistoryOpen}
                  onLoadChallenge={(id) => void loadChallenge(id)}
                  onDeleteChallenge={(id, label) => void deleteChallengeById(id, label)}
                />
              </div>
            ) : null}
            <div className="html-challenge-setup-actions-spacer" />
            <div className="html-challenge-layout-toggle" aria-label={t("htmlChallenge.layoutToggleAria", { defaultValue: "HTML challenge layout" })}>
              <button
                className={layoutMode === "row" ? "active" : ""}
                type="button"
                onClick={() => setLayoutMode("row")}
              >
                {t("htmlChallenge.layoutRow", { defaultValue: "Row" })}
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
                {t("htmlChallenge.actions.openFolder", { defaultValue: "Open Folder" })}
              </button>
            ) : null}
            {manifest?.settingsPath ? (
              <button
                className="secondary-button"
                type="button"
                onClick={() => onOpenFilePath(manifest.settingsPath!)}
              >
                {t("htmlChallenge.actions.openSettings", { defaultValue: "Open Settings" })}
              </button>
            ) : null}
            {!completedChallenge ? (
              <button className="secondary-button" type="button" onClick={addSlot} disabled={busy || slots.length >= 4}>
                {t("htmlChallenge.actions.addModel", { defaultValue: "Add model" })}
              </button>
            ) : null}
            {busy ? (
              <button className="secondary-button" type="button" onClick={cancelChallenge}>
                {t("htmlChallenge.actions.cancel", { defaultValue: "Cancel" })}
              </button>
            ) : completedValidChallenge ? (
              <button
                className="primary-button"
                type="button"
                onClick={usePromptInNewChallenge}
              >
                {t("htmlChallenge.actions.usePromptInNewChallenge", { defaultValue: "Use Prompt in New Challenge" })}
              </button>
            ) : (
              <button
                className="primary-button"
                type="button"
                onClick={() => void runChallenge()}
                disabled={!title.trim() || !prompt.trim() || !allSelected}
              >
                {manifest
                  ? t("htmlChallenge.actions.runNewChallenge", { defaultValue: "Run New Challenge" })
                  : t("htmlChallenge.actions.runChallenge", { defaultValue: "Run Challenge" })}
              </button>
            )}
          </div>
          <div className="html-challenge-controls">
            <input
              className="text-input"
              type="text"
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              placeholder={t("htmlChallenge.titlePlaceholder", { defaultValue: "Challenge title" })}
              disabled={busy}
            />
            <textarea
              className="text-input html-challenge-prompt"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder={t("htmlChallenge.promptPlaceholder", { defaultValue: "Prompt all selected models with the same webpage challenge..." })}
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

      <ChallengePickerModal
        target={pickerTarget}
        initialKey={pickerInitialKey}
        initialSettings={pickerInitialSettings}
        textModelOptions={textModelOptions}
        availableMemoryGb={availableMemoryGb}
        totalMemoryGb={totalMemoryGb}
        gpuVramTotalGb={gpuVramTotalGb}
        availableCacheStrategies={availableCacheStrategies}
        dflashInfo={dflashInfo}
        installingPackage={installingPackage ?? null}
        installLogs={installLogs}
        turboInstalled={turboInstalled}
        mtplxSystemInfo={mtplxSystemInfo}
        onInstallMtplx={onInstallMtplx}
        installingMtplx={installingMtplx}
        mtplxJob={mtplxJob}
        isAppleSilicon={isAppleSilicon}
        onConfirm={(selectedKey, newSettings) => {
          if (pickerTarget) {
            const target = pickerTarget;
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
          setPickerAutoRetry(false);
          setPickerTarget(null);
        }}
        onClose={() => {
          setPickerAutoRetry(false);
          setPickerTarget(null);
        }}
        onInstallPackage={installPackage}
      />

      <ChallengePromptLibraryModal
        open={promptLibraryOpen}
        onSelect={(entry) => {
          setTitle(entry.title);
          setPrompt(entry.prompt);
          setPromptLibraryOpen(false);
        }}
        onClose={() => setPromptLibraryOpen(false)}
      />
    </div>
  );
}
