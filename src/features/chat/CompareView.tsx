import { useEffect, useRef, useState } from "react";
import { RichMarkdown } from "../../components/RichMarkdown";
import { apiFetch, getCachePreview } from "../../api";
import { ModelLaunchModal } from "../../components/ModelLaunchModal";
import { Panel } from "../../components/Panel";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import { emptyPreview } from "../../defaults";
import type { GenerationMetrics, LaunchPreferences, PreviewMetrics, StrategyInstallLog, SystemStats } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import {
  detectBitsPerWeight,
  estimateArchFromParams,
  estimateParamsBFromDisk,
  number,
  sizeLabel,
} from "../../utils";
import {
  requestedSpeculativeMode,
  resolvedDraftModel,
  resolvedSpeculativeMode,
  runtimeOutcomeWarning,
} from "./runtimeDetails";

export const compareTargets = ["a", "b", "c", "d"] as const;
export type CompareTarget = typeof compareTargets[number];

export const compareTargetLabels: Record<CompareTarget, string> = {
  a: "Model A",
  b: "Model B",
  c: "Model C",
  d: "Model D",
};

interface CompareSlot {
  id: CompareTarget;
  modelKey: string;
  settings: LaunchPreferences;
}

interface CompareModelState {
  text: string;
  reasoning: string;
  reasoningDone: boolean;
  done: boolean;
  loading: boolean;
  loadingMessage?: string;
  appliedSummary?: string;
  runtimeNote?: string;
  tokS: number;
  promptTokens: number;
  completionTokens: number;
  responseSeconds: number;
  loadSeconds: number;
  totalSeconds: number;
  metrics: GenerationMetrics | null;
  error?: string;
}

interface CompareViewProps {
  modelOptions: ChatModelOption[];
  onBack: () => void;
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
}

interface CompareStreamEvent extends Partial<GenerationMetrics> {
  model?: CompareTarget;
  loading?: boolean;
  loaded?: boolean;
  message?: string;
  token?: string;
  text?: string;
  done?: boolean;
  error?: string;
  appliedSummary?: string;
  allDone?: boolean;
  reasoning?: string;
  reasoningDone?: boolean;
  loadSeconds?: number;
  totalSeconds?: number;
}

const emptyModelState = (): CompareModelState => ({
  text: "",
  reasoning: "",
  reasoningDone: true,
  done: false,
  loading: false,
  tokS: 0,
  promptTokens: 0,
  completionTokens: 0,
  responseSeconds: 0,
  loadSeconds: 0,
  totalSeconds: 0,
  metrics: null,
});

function emptyModelStates(): Record<CompareTarget, CompareModelState> {
  return {
    a: emptyModelState(),
    b: emptyModelState(),
    c: emptyModelState(),
    d: emptyModelState(),
  };
}

function emptyAtBottom(): Record<CompareTarget, boolean> {
  return { a: true, b: true, c: true, d: true };
}

const compareMetricKeys = [
  "finishReason",
  "promptTokens",
  "completionTokens",
  "totalTokens",
  "tokS",
  "responseSeconds",
  "runtimeNote",
  "dflashAcceptanceRate",
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
] as const;

function defaultCompareMetrics(): GenerationMetrics {
  return {
    finishReason: "stop",
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    tokS: 0,
    runtimeNote: null,
  };
}

function mergeCompareMetrics(
  current: GenerationMetrics | null,
  event: CompareStreamEvent,
): GenerationMetrics | null {
  let hasPatch = false;
  const next: Record<string, unknown> = { ...(current ?? defaultCompareMetrics()) };

  for (const key of compareMetricKeys) {
    if (!(key in event)) continue;
    hasPatch = true;
    next[key] = event[key];
  }

  return hasPatch ? next as unknown as GenerationMetrics : current;
}

export function cloneLaunchSettings(settings: LaunchPreferences): LaunchPreferences {
  return { ...settings };
}

function formatTokenSetting(value: number) {
  if (value >= 1024) return `${Math.round(value / 1024)}K`;
  return String(value);
}

export function summarizeLaunchSettings(settings: LaunchPreferences) {
  const cacheLabel = settings.cacheStrategy === "native"
    ? "Native f16"
    : `${settings.cacheStrategy} ${settings.cacheBits}-bit`;
  const speculativeLabel = settings.speculativeDecoding
    ? settings.treeBudget > 0 ? `DDTree ${settings.treeBudget}` : "DFlash"
    : null;
  return [
    cacheLabel,
    `${formatTokenSetting(settings.contextTokens)} ctx`,
    `${formatTokenSetting(settings.maxTokens)} max`,
    `temp ${number(settings.temperature)}`,
    settings.fusedAttention ? "Fused attention" : null,
    speculativeLabel,
  ].filter(Boolean).join(" · ");
}

function estimatePreviewShape(option: ChatModelOption | null) {
  if (!option) return null;
  let paramsB = option.paramsB ?? 0;
  if (!paramsB && option.sizeGb) {
    const bitsPerWeight = detectBitsPerWeight(`${option.label} ${option.format ?? ""} ${option.quantization ?? ""}`);
    paramsB = estimateParamsBFromDisk(option.sizeGb, bitsPerWeight);
  }
  if (!paramsB) return null;
  return { paramsB, ...estimateArchFromParams(paramsB) };
}

export function useLaunchPreview(option: ChatModelOption | null, settings: LaunchPreferences) {
  const [preview, setPreview] = useState<PreviewMetrics>(emptyPreview);

  useEffect(() => {
    const shape = estimatePreviewShape(option);
    if (!shape) {
      setPreview({
        ...emptyPreview,
        bits: settings.cacheBits,
        fp16Layers: settings.fp16Layers,
        contextTokens: settings.contextTokens,
        summary: option ? "Cache preview unavailable for this model." : "",
      });
      return;
    }

    let cancelled = false;
    const timeout = window.setTimeout(() => {
      void (async () => {
        const nextPreview = await getCachePreview({
          bits: settings.cacheBits,
          fp16Layers: settings.fp16Layers,
          numLayers: shape.numLayers,
          numHeads: shape.numHeads,
          numKvHeads: shape.numKvHeads,
          hiddenSize: shape.hiddenSize,
          contextTokens: settings.contextTokens,
          paramsB: shape.paramsB,
          strategy: settings.cacheStrategy,
        });
        if (!cancelled) setPreview(nextPreview);
      })();
    }, 220);

    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [
    option?.key,
    option?.label,
    option?.format,
    option?.quantization,
    option?.paramsB,
    option?.sizeGb,
    settings.cacheBits,
    settings.fp16Layers,
    settings.contextTokens,
    settings.cacheStrategy,
  ]);

  return preview;
}

export function buildComparePayload(option: ChatModelOption, settings: LaunchPreferences) {
  return {
    modelRef: option.modelRef,
    modelName: option.model,
    displayLabel: option.label,
    displayDetail: option.detail,
    format: option.format ?? undefined,
    quantization: option.quantization ?? undefined,
    sizeGb: option.sizeGb ?? undefined,
    contextWindow: option.contextWindow ?? undefined,
    canonicalRepo: option.canonicalRepo ?? undefined,
    source: option.source,
    path: option.path ?? undefined,
    backend: option.backend,
    launch: settings,
  };
}

export function gridColumns(count: number) {
  return `repeat(${Math.min(Math.max(count, 2), 4)}, minmax(220px, 1fr))`;
}

export function CompareView({
  modelOptions,
  onBack,
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
}: CompareViewProps) {
  const [slots, setSlots] = useState<CompareSlot[]>(() => [
    { id: "a", modelKey: "", settings: cloneLaunchSettings(launchSettings) },
    { id: "b", modelKey: "", settings: cloneLaunchSettings(launchSettings) },
  ]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [modelStates, setModelStates] = useState<Record<CompareTarget, CompareModelState>>(emptyModelStates);
  const [pickerTarget, setPickerTarget] = useState<CompareTarget | null>(null);
  const [pickerSearch, setPickerSearch] = useState("");
  const [pickerDraftKey, setPickerDraftKey] = useState("");
  const [pickerDraftSettings, setPickerDraftSettings] = useState<LaunchPreferences>(() => cloneLaunchSettings(launchSettings));
  const [resultAtBottom, setResultAtBottom] = useState<Record<CompareTarget, boolean>>(emptyAtBottom);
  const abortRef = useRef<AbortController | null>(null);
  const resultRefs = useRef<Record<CompareTarget, HTMLDivElement | null>>({
    a: null,
    b: null,
    c: null,
    d: null,
  });

  const textModelOptions = modelOptions.filter((option) => {
    const backend = (option.backend ?? "").toLowerCase();
    const format = (option.format ?? option.detail ?? "").toLowerCase();
    const label = option.label.toLowerCase();
    return backend !== ""
      && !format.includes("diffuser")
      && !label.includes("stable-diffusion")
      && !label.includes("flux")
      && !label.includes("sana");
  });

  const selectedBySlot = Object.fromEntries(
    slots.map((slot) => [slot.id, textModelOptions.find((option) => option.key === slot.modelKey) ?? null]),
  ) as Record<CompareTarget, ChatModelOption | null>;
  const pickerDraftOption =
    textModelOptions.find((option) => option.key === pickerDraftKey)
    ?? (pickerTarget ? textModelOptions[0] ?? null : null);
  const pickerDraftPreview = useLaunchPreview(pickerDraftOption, pickerDraftSettings);
  const installPackage = onInstallPackage ?? (() => {});
  const allSelected = slots.every((slot) => selectedBySlot[slot.id] != null);
  const duplicateSelected = (() => {
    const keys = slots.map((slot) => slot.modelKey).filter(Boolean);
    return new Set(keys).size < keys.length;
  })();

  useEffect(() => {
    const handles = slots
      .filter((slot) => resultAtBottom[slot.id])
      .map((slot) => requestAnimationFrame(() => scrollResultToBottom(slot.id)));
    return () => handles.forEach((handle) => cancelAnimationFrame(handle));
    // modelStates changes on each streamed token; that is intentional so
    // panels auto-scroll only while their latest-content lock is active.
  }, [slots, modelStates, resultAtBottom]);

  function updateSlot(slotId: CompareTarget, patch: Partial<CompareSlot>) {
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

  function handleResultScroll(target: CompareTarget) {
    const element = resultRefs.current[target];
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 32;
    setResultAtBottom((current) => ({ ...current, [target]: atBottom }));
  }

  function scrollResultToBottom(target: CompareTarget) {
    const element = resultRefs.current[target];
    if (!element) return;
    element.scrollTop = element.scrollHeight;
    setResultAtBottom((current) => current[target] ? current : { ...current, [target]: true });
  }

  function applyStreamEvent(event: CompareStreamEvent) {
    if (event.allDone) {
      setBusy(false);
      return;
    }
    const target = event.model;
    if (!target) return;
    setModelStates((current) => {
      const prev = current[target];
      let next: CompareModelState = prev;
      if (event.reasoning) {
        next = { ...next, reasoning: next.reasoning + event.reasoning, reasoningDone: false };
      }
      if (event.reasoningDone) {
        next = { ...next, reasoningDone: true };
      }
      if (event.loading) {
        next = { ...next, loading: true, loadingMessage: event.message };
      }
      if (event.loaded) {
        next = {
          ...next,
          loading: false,
          loadSeconds: event.loadSeconds ?? next.loadSeconds,
          appliedSummary: event.appliedSummary ?? next.appliedSummary,
          runtimeNote: event.runtimeNote ?? next.runtimeNote,
          metrics: mergeCompareMetrics(next.metrics, event),
        };
      }
      if (event.token) {
        next = { ...next, loading: false, text: next.text + event.token };
      }
      if (event.done) {
        next = {
          ...next,
          done: true,
          loading: false,
          reasoningDone: true,
          tokS: event.tokS ?? 0,
          promptTokens: event.promptTokens ?? 0,
          completionTokens: event.completionTokens ?? 0,
          responseSeconds: event.responseSeconds ?? 0,
          loadSeconds: event.loadSeconds ?? next.loadSeconds,
          totalSeconds: event.totalSeconds ?? next.totalSeconds,
          appliedSummary: event.appliedSummary ?? next.appliedSummary,
          runtimeNote: event.runtimeNote ?? next.runtimeNote,
          metrics: mergeCompareMetrics(next.metrics, event),
        };
      }
      if (event.error) {
        next = { ...next, error: event.error, done: true, loading: false, reasoningDone: true };
      }
      return { ...current, [target]: next };
    });
  }

  async function handleCompare() {
    if (!prompt.trim() || !allSelected) return;

    setBusy(true);
    setResultAtBottom(emptyAtBottom());
    setModelStates(emptyModelStates());

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await apiFetch("/api/chat/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim(),
          models: slots.map((slot) => buildComparePayload(selectedBySlot[slot.id]!, slot.settings)),
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        const firstSlot = slots[0]?.id ?? "a";
        setModelStates((current) => ({
          ...current,
          [firstSlot]: { ...current[firstSlot], error: err?.detail ?? "Request failed" },
        }));
        setBusy(false);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            applyStreamEvent(JSON.parse(line.slice(6)) as CompareStreamEvent);
          } catch {
            // Skip malformed chunks.
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const firstSlot = slots[0]?.id ?? "a";
      setModelStates((current) => ({
        ...current,
        [firstSlot]: { ...current[firstSlot], error: String(err) },
      }));
    } finally {
      setBusy(false);
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
    setBusy(false);
  }

  function openPicker(target: CompareTarget) {
    const slot = slots.find((item) => item.id === target);
    setPickerSearch("");
    setPickerDraftKey(slot?.modelKey ?? "");
    setPickerDraftSettings(cloneLaunchSettings(slot?.settings ?? launchSettings));
    setPickerTarget(target);
  }

  function renderModelCard(slot: CompareSlot) {
    const option = selectedBySlot[slot.id];
    return (
      <div key={slot.id}>
        <span className="eyebrow">{compareTargetLabels[slot.id]}</span>
        <div className="model-selected-card" style={{ minHeight: 104 }}>
          <div className="model-selected-info">
            <strong>{option?.label ?? "Select a model"}</strong>
            <div className="model-selected-meta">
              {option?.format ? <span className="badge muted">{option.format}</span> : null}
              {option?.quantization ? <span className="badge muted">{option.quantization}</span> : null}
              {option?.sizeGb ? <span className="badge muted">{sizeLabel(option.sizeGb)}</span> : null}
              {option?.contextWindow ? <span className="badge muted">{option.contextWindow}</span> : null}
            </div>
            <small className="muted-text">{summarizeLaunchSettings(slot.settings)}</small>
          </div>
          <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
            {slots.length > 2 && slot.id === slots[slots.length - 1]?.id ? (
              <button
                className="secondary-button"
                type="button"
                onClick={removeLastSlot}
                disabled={busy}
                title={`Remove ${compareTargetLabels[slot.id]}`}
              >
                Remove
              </button>
            ) : null}
            <button
              className="secondary-button"
              type="button"
              onClick={() => openPicker(slot.id)}
              disabled={busy}
            >
              {option ? "Change" : "Select"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  function renderResultPanel(slot: CompareSlot, index: number) {
    const option = selectedBySlot[slot.id];
    const settings = slot.settings;
    const modelState = modelStates[slot.id];
    const atBottom = resultAtBottom[slot.id];
    const metrics = modelState.metrics;
    const actualSpeculativeMode = metrics ? resolvedSpeculativeMode(metrics) : null;
    const requestedSpecMode = metrics ? requestedSpeculativeMode(metrics) : null;
    const draftModel = metrics ? resolvedDraftModel(metrics) : null;
    const runtimeWarning = metrics ? runtimeOutcomeWarning(metrics) : null;
    const speculativeActive = actualSpeculativeMode != null
      && actualSpeculativeMode !== "Off"
      && actualSpeculativeMode !== "Requested, no compatible draft";
    const speculativeSummary = speculativeActive
      ? [
          `Speculative: ${actualSpeculativeMode}`,
          draftModel ? `draft ${draftModel}` : null,
          metrics?.dflashAcceptanceRate != null ? `${number(metrics.dflashAcceptanceRate)} avg accepted` : null,
        ].filter(Boolean).join(" · ")
      : requestedSpecMode && requestedSpecMode !== "Off" && runtimeWarning
        ? `Speculative: ${requestedSpecMode}`
        : null;
    const showLatestButton = !atBottom && (
      Boolean(modelState.text)
      || Boolean(modelState.reasoning)
      || modelState.loading
      || modelState.done
    );
    const subtitle = modelState.done
      ? `${number(modelState.tokS)} tok/s | ${number(modelState.responseSeconds)}s`
      : modelState.loading ? "Loading..." : modelState.text ? "Generating..." : "";
    const waitingLabel = index === 0 ? "Waiting..." : `Waiting for ${compareTargetLabels[slots[index - 1]?.id ?? "a"]} to finish...`;

    return (
      <Panel
        key={slot.id}
        title={compareTargetLabels[slot.id]}
        subtitle={subtitle}
        actions={showLatestButton ? (
          <button className="secondary-button" type="button" onClick={() => scrollResultToBottom(slot.id)}>
            Latest
          </button>
        ) : null}
      >
        <div
          ref={(element) => { resultRefs.current[slot.id] = element; }}
          onScroll={() => handleResultScroll(slot.id)}
          style={{ overflow: "auto", flex: 1, padding: 8 }}
        >
          {option ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 6px" }}>{option.label} · {option.detail}</p> : null}
          {option ? (
            <p className="muted-text" style={{ fontSize: 11, margin: "0 0 10px" }}>
              {modelState.appliedSummary ?? summarizeLaunchSettings(settings)}
            </p>
          ) : null}
          {modelState.loadSeconds > 0 ? (
            <p className="muted-text" style={{ fontSize: 11, margin: "0 0 8px" }}>
              Load {number(modelState.loadSeconds)}s
              {modelState.totalSeconds > 0 ? ` · Total ${number(modelState.totalSeconds)}s` : ""}
            </p>
          ) : null}
          {runtimeWarning ? (
            <p style={{ fontSize: 11, margin: "0 0 8px", color: "var(--warning, #e4be75)" }}>
              {runtimeWarning}
            </p>
          ) : null}
          {speculativeSummary ? (
            <p className="muted-text" style={{ fontSize: 11, margin: "0 0 8px" }}>
              {speculativeSummary}
            </p>
          ) : null}
          {modelState.runtimeNote ? (
            <p className="muted-text" style={{ fontSize: 11, margin: "0 0 10px" }}>
              {modelState.runtimeNote}
            </p>
          ) : null}
          <ReasoningPanel text={modelState.reasoning} streaming={!modelState.reasoningDone} />
          {modelState.error ? (
            <p style={{ color: "#f87171" }}>{modelState.error}</p>
          ) : modelState.text ? (
            <div className="markdown-content">
              <RichMarkdown>{modelState.text}</RichMarkdown>
            </div>
          ) : modelState.loading ? (
            <p className="muted-text" style={{ fontSize: 13 }}>{modelState.loadingMessage ?? "Loading model..."}</p>
          ) : busy ? (
            <p className="muted-text" style={{ fontSize: 13 }}>{waitingLabel}</p>
          ) : null}
        </div>
      </Panel>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12, overflowY: "auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 4px", flexWrap: "wrap" }}>
        <button className="secondary-button" type="button" onClick={onBack} style={{ fontSize: 12 }}>
          Back to Chat
        </button>
        <h3 style={{ margin: 0, fontSize: 16, color: "#c8d0da" }}>Compare Models</h3>
        <small style={{ color: "#7a8594", fontSize: 11 }}>
          Queue 2-4 models. Each slot loads exclusively, runs, and unloads before the next slot starts.
        </small>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: gridColumns(slots.length), gap: 12 }}>
        {slots.map((slot) => renderModelCard(slot))}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 4px", minHeight: 28 }}>
        <button className="secondary-button" type="button" onClick={addSlot} disabled={busy || slots.length >= 4}>
          Add model
        </button>
        <span className="muted-text" style={{ fontSize: 11 }}>{slots.length}/4 queued</span>
        {duplicateSelected ? (
          <span className="muted-text" style={{ fontSize: 11 }}>
            Same model selected in multiple slots; useful for runtime-profile A/B tests.
          </span>
        ) : null}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !busy) void handleCompare();
          }}
          placeholder="Enter a prompt to compare..."
          className="text-input"
          style={{ flex: 1 }}
          disabled={busy}
        />
        {busy ? (
          <button className="secondary-button" type="button" onClick={handleCancel}>Cancel</button>
        ) : (
          <button
            className="primary-button"
            type="button"
            onClick={() => void handleCompare()}
            disabled={!prompt.trim() || !allSelected}
          >
            Compare
          </button>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: gridColumns(slots.length), gap: 12, flex: 1, minHeight: 420, overflow: "hidden" }}>
        {slots.map((slot, index) => renderResultPanel(slot, index))}
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
