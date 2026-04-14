import { useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import { getCachePreview, resolveApiBase } from "../../api";
import { ModelLaunchModal } from "../../components/ModelLaunchModal";
import { Panel } from "../../components/Panel";
import { ReasoningPanel } from "../../components/ReasoningPanel";
import { emptyPreview } from "../../defaults";
import type { LaunchPreferences, PreviewMetrics, SystemStats } from "../../types";
import type { ChatModelOption } from "../../types/chat";
import {
  detectBitsPerWeight,
  estimateArchFromParams,
  estimateParamsBFromDisk,
  number,
  sizeLabel,
} from "../../utils";

type CompareTarget = "a" | "b";

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
  error?: string;
}

interface CompareViewProps {
  modelOptions: ChatModelOption[];
  onBack: () => void;
  launchSettings: LaunchPreferences;
  availableMemoryGb: number;
  totalMemoryGb: number;
  availableCacheStrategies?: SystemStats["availableCacheStrategies"];
  dflashInfo?: SystemStats["dflash"];
  onInstallPackage?: (strategyId: string) => void;
  installingPackage?: string | null;
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
});

function cloneLaunchSettings(settings: LaunchPreferences): LaunchPreferences {
  return { ...settings };
}

function formatTokenSetting(value: number) {
  if (value >= 1024) return `${Math.round(value / 1024)}K`;
  return String(value);
}

function summarizeLaunchSettings(settings: LaunchPreferences) {
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

function useLaunchPreview(option: ChatModelOption | null, settings: LaunchPreferences) {
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

function buildComparePayload(option: ChatModelOption, settings: LaunchPreferences) {
  return {
    modelRef: option.modelRef,
    modelName: option.model,
    canonicalRepo: option.canonicalRepo ?? undefined,
    source: option.source,
    path: option.path ?? undefined,
    backend: option.backend,
    launch: settings,
  };
}

export function CompareView({
  modelOptions,
  onBack,
  launchSettings,
  availableMemoryGb,
  totalMemoryGb,
  availableCacheStrategies,
  dflashInfo,
  onInstallPackage,
  installingPackage,
}: CompareViewProps) {
  const [modelKeyA, setModelKeyA] = useState("");
  const [modelKeyB, setModelKeyB] = useState("");
  const [settingsA, setSettingsA] = useState<LaunchPreferences>(() => cloneLaunchSettings(launchSettings));
  const [settingsB, setSettingsB] = useState<LaunchPreferences>(() => cloneLaunchSettings(launchSettings));
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [modelA, setModelA] = useState<CompareModelState>(emptyModelState());
  const [modelB, setModelB] = useState<CompareModelState>(emptyModelState());
  const [pickerTarget, setPickerTarget] = useState<CompareTarget | null>(null);
  const [pickerSearch, setPickerSearch] = useState("");
  const [pickerDraftKey, setPickerDraftKey] = useState("");
  const [pickerDraftSettings, setPickerDraftSettings] = useState<LaunchPreferences>(() => cloneLaunchSettings(launchSettings));
  const abortRef = useRef<AbortController | null>(null);

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

  const selectedA = textModelOptions.find((option) => option.key === modelKeyA) ?? null;
  const selectedB = textModelOptions.find((option) => option.key === modelKeyB) ?? null;
  const sameModel = selectedA != null && selectedB != null && selectedA.key === selectedB.key;
  const pickerDraftOption =
    textModelOptions.find((option) => option.key === pickerDraftKey)
    ?? (pickerTarget ? textModelOptions[0] ?? null : null);
  const pickerDraftPreview = useLaunchPreview(pickerDraftOption, pickerDraftSettings);
  const installPackage = onInstallPackage ?? (() => {});

  async function handleCompare() {
    if (!prompt.trim() || !selectedA || !selectedB) return;

    setBusy(true);
    setModelA(emptyModelState());
    setModelB(emptyModelState());

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const apiBase = await resolveApiBase();
      const response = await fetch(`${apiBase}/api/chat/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim(),
          modelA: buildComparePayload(selectedA, settingsA),
          modelB: buildComparePayload(selectedB, settingsB),
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        setModelA((prev) => ({ ...prev, error: err?.detail ?? "Request failed" }));
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
            const event = JSON.parse(line.slice(6));
            if (event.model === "a") {
              if (event.reasoning) {
                setModelA((prev) => ({
                  ...prev,
                  reasoning: prev.reasoning + event.reasoning,
                  reasoningDone: false,
                }));
              }
              if (event.reasoningDone) {
                setModelA((prev) => ({ ...prev, reasoningDone: true }));
              }
              if (event.loading) {
                setModelA((prev) => ({ ...prev, loading: true, loadingMessage: event.message }));
              }
              if (event.loaded) {
                setModelA((prev) => ({
                  ...prev,
                  loading: false,
                  appliedSummary: event.appliedSummary ?? prev.appliedSummary,
                  runtimeNote: event.runtimeNote ?? prev.runtimeNote,
                }));
              }
              if (event.token) {
                setModelA((prev) => ({ ...prev, loading: false, text: prev.text + event.token }));
              }
              if (event.done) {
                setModelA((prev) => ({
                  ...prev,
                  done: true,
                  loading: false,
                  reasoningDone: true,
                  tokS: event.tokS ?? 0,
                  promptTokens: event.promptTokens ?? 0,
                  completionTokens: event.completionTokens ?? 0,
                  responseSeconds: event.responseSeconds ?? 0,
                  appliedSummary: event.appliedSummary ?? prev.appliedSummary,
                  runtimeNote: event.runtimeNote ?? prev.runtimeNote,
                }));
              }
              if (event.error) {
                setModelA((prev) => ({ ...prev, error: event.error, done: true, loading: false, reasoningDone: true }));
              }
            } else if (event.model === "b") {
              if (event.reasoning) {
                setModelB((prev) => ({
                  ...prev,
                  reasoning: prev.reasoning + event.reasoning,
                  reasoningDone: false,
                }));
              }
              if (event.reasoningDone) {
                setModelB((prev) => ({ ...prev, reasoningDone: true }));
              }
              if (event.loading) {
                setModelB((prev) => ({ ...prev, loading: true, loadingMessage: event.message }));
              }
              if (event.loaded) {
                setModelB((prev) => ({
                  ...prev,
                  loading: false,
                  appliedSummary: event.appliedSummary ?? prev.appliedSummary,
                  runtimeNote: event.runtimeNote ?? prev.runtimeNote,
                }));
              }
              if (event.token) {
                setModelB((prev) => ({ ...prev, loading: false, text: prev.text + event.token }));
              }
              if (event.done) {
                setModelB((prev) => ({
                  ...prev,
                  done: true,
                  loading: false,
                  reasoningDone: true,
                  tokS: event.tokS ?? 0,
                  promptTokens: event.promptTokens ?? 0,
                  completionTokens: event.completionTokens ?? 0,
                  responseSeconds: event.responseSeconds ?? 0,
                  appliedSummary: event.appliedSummary ?? prev.appliedSummary,
                  runtimeNote: event.runtimeNote ?? prev.runtimeNote,
                }));
              }
              if (event.error) {
                setModelB((prev) => ({ ...prev, error: event.error, done: true, loading: false, reasoningDone: true }));
              }
            }
            if (event.allDone) {
              setBusy(false);
            }
          } catch {
            // Skip malformed chunks.
          }
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setModelA((prev) => ({ ...prev, error: String(err) }));
    } finally {
      setBusy(false);
    }
  }

  function handleCancel() {
    abortRef.current?.abort();
    setBusy(false);
  }

  function openPicker(target: CompareTarget) {
    setPickerSearch("");
    setPickerDraftKey(target === "a" ? modelKeyA : modelKeyB);
    setPickerDraftSettings(cloneLaunchSettings(target === "a" ? settingsA : settingsB));
    setPickerTarget(target);
  }

  function renderModelCard(label: string, option: ChatModelOption | null, settings: LaunchPreferences, target: CompareTarget) {
    return (
      <div>
        <span className="eyebrow">{label}</span>
        <div className="model-selected-card" style={{ minHeight: 92 }}>
          <div className="model-selected-info">
            <strong>{option?.label ?? "Select a model"}</strong>
            <div className="model-selected-meta">
              {option?.format ? <span className="badge muted">{option.format}</span> : null}
              {option?.quantization ? <span className="badge muted">{option.quantization}</span> : null}
              {option?.sizeGb ? <span className="badge muted">{sizeLabel(option.sizeGb)}</span> : null}
              {option?.contextWindow ? <span className="badge muted">{option.contextWindow}</span> : null}
            </div>
            <small className="muted-text">{summarizeLaunchSettings(settings)}</small>
          </div>
          <button
            className="secondary-button"
            type="button"
            onClick={() => openPicker(target)}
            disabled={busy}
          >
            {option ? "Change" : "Select"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12, overflowY: "auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 4px" }}>
        <button className="secondary-button" type="button" onClick={onBack} style={{ fontSize: 12 }}>
          Back to Chat
        </button>
        <h3 style={{ margin: 0, fontSize: 16, color: "#c8d0da" }}>Compare Models</h3>
        <small style={{ color: "#7a8594", fontSize: 11 }}>
          Models run sequentially, so each side keeps its own runtime profile.
        </small>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {renderModelCard("Model A", selectedA, settingsA, "a")}
        {renderModelCard("Model B", selectedB, settingsB, "b")}
      </div>

      {sameModel ? (
        <p style={{ fontSize: 11, color: "#7a8594", margin: 0, padding: "0 4px" }}>
          Same model selected for both. Useful for A/B testing two runtime profiles on the same prompt.
        </p>
      ) : null}

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
            disabled={!prompt.trim() || !selectedA || !selectedB}
          >
            Compare
          </button>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, flex: 1, overflow: "hidden" }}>
        <Panel title="Model A" subtitle={modelA.done ? `${number(modelA.tokS)} tok/s | ${number(modelA.responseSeconds)}s` : modelA.loading ? "Loading..." : modelA.text ? "Generating..." : ""}>
          <div style={{ overflow: "auto", flex: 1, padding: 8 }}>
            {selectedA ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 6px" }}>{selectedA.label} · {selectedA.detail}</p> : null}
            {selectedA ? (
              <p className="muted-text" style={{ fontSize: 11, margin: "0 0 10px" }}>
                {modelA.appliedSummary ?? summarizeLaunchSettings(settingsA)}
              </p>
            ) : null}
            {modelA.runtimeNote ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 10px" }}>{modelA.runtimeNote}</p> : null}
            <ReasoningPanel text={modelA.reasoning} streaming={!modelA.reasoningDone} />
            {modelA.error ? (
              <p style={{ color: "#f87171" }}>{modelA.error}</p>
            ) : modelA.text ? (
              <div className="markdown-content">
                <Markdown>{modelA.text}</Markdown>
              </div>
            ) : modelA.loading ? (
              <p className="muted-text" style={{ fontSize: 13 }}>{modelA.loadingMessage ?? "Loading model..."}</p>
            ) : busy ? (
              <p className="muted-text" style={{ fontSize: 13 }}>Waiting...</p>
            ) : null}
          </div>
        </Panel>

        <Panel title="Model B" subtitle={modelB.done ? `${number(modelB.tokS)} tok/s | ${number(modelB.responseSeconds)}s` : modelB.loading ? "Loading..." : modelB.text ? "Generating..." : ""}>
          <div style={{ overflow: "auto", flex: 1, padding: 8 }}>
            {selectedB ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 6px" }}>{selectedB.label} · {selectedB.detail}</p> : null}
            {selectedB ? (
              <p className="muted-text" style={{ fontSize: 11, margin: "0 0 10px" }}>
                {modelB.appliedSummary ?? summarizeLaunchSettings(settingsB)}
              </p>
            ) : null}
            {modelB.runtimeNote ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 10px" }}>{modelB.runtimeNote}</p> : null}
            <ReasoningPanel text={modelB.reasoning} streaming={!modelB.reasoningDone} />
            {modelB.error ? (
              <p style={{ color: "#f87171" }}>{modelB.error}</p>
            ) : modelB.text ? (
              <div className="markdown-content">
                <Markdown>{modelB.text}</Markdown>
              </div>
            ) : modelB.loading ? (
              <p className="muted-text" style={{ fontSize: 13 }}>{modelB.loadingMessage ?? "Loading model..."}</p>
            ) : busy ? (
              <p className="muted-text" style={{ fontSize: 13 }}>Waiting for Model A to finish...</p>
            ) : null}
          </div>
        </Panel>
      </div>

      <ModelLaunchModal
        open={pickerTarget != null}
        title={pickerTarget === "a" ? "Select Model A" : "Select Model B"}
        confirmLabel={pickerTarget === "a" ? "Use for Model A" : "Use for Model B"}
        selectedKey={pickerDraftKey}
        collapseOnOpen={Boolean(pickerDraftKey)}
        search={pickerSearch}
        options={textModelOptions}
        settings={pickerDraftSettings}
        preview={pickerDraftPreview}
        availableMemoryGb={availableMemoryGb}
        totalMemoryGb={totalMemoryGb}
        availableCacheStrategies={availableCacheStrategies}
        dflashInfo={dflashInfo}
        installingPackage={installingPackage ?? null}
        onSelectedKeyChange={setPickerDraftKey}
        onSearchChange={setPickerSearch}
        onSettingChange={(key, value) => {
          setPickerDraftSettings((current) => ({ ...current, [key]: value }));
        }}
        onConfirm={(selectedKey) => {
          if (pickerTarget === "a") {
            setModelKeyA(selectedKey);
            setSettingsA(cloneLaunchSettings(pickerDraftSettings));
          }
          if (pickerTarget === "b") {
            setModelKeyB(selectedKey);
            setSettingsB(cloneLaunchSettings(pickerDraftSettings));
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
