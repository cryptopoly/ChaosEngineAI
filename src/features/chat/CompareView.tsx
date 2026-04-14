import { useState, useRef } from "react";
import Markdown from "react-markdown";
import { Panel } from "../../components/Panel";
import { RuntimeControls } from "../../components/RuntimeControls";
import { resolveApiBase } from "../../api";
import type { LaunchPreferences, LibraryItem, PreviewMetrics, SystemStats } from "../../types";
import { number } from "../../utils";

interface CompareModelState {
  text: string;
  done: boolean;
  loading: boolean;
  loadingMessage?: string;
  tokS: number;
  promptTokens: number;
  completionTokens: number;
  responseSeconds: number;
  error?: string;
}

interface CompareViewProps {
  library: LibraryItem[];
  onBack: () => void;
  launchSettings: LaunchPreferences;
  onLaunchSettingChange: <K extends keyof LaunchPreferences>(key: K, value: LaunchPreferences[K]) => void;
  preview: PreviewMetrics;
  availableMemoryGb: number;
  totalMemoryGb: number;
  availableCacheStrategies?: SystemStats["availableCacheStrategies"];
  dflashInfo?: SystemStats["dflash"];
  onInstallPackage?: (strategyId: string) => void;
  installingPackage?: string | null;
}

const emptyModelState = (): CompareModelState => ({
  text: "",
  done: false,
  loading: false,
  tokS: 0,
  promptTokens: 0,
  completionTokens: 0,
  responseSeconds: 0,
});

export function CompareView({
  library,
  onBack,
  launchSettings,
  onLaunchSettingChange,
  preview,
  availableMemoryGb,
  totalMemoryGb,
  availableCacheStrategies,
  dflashInfo,
  onInstallPackage,
  installingPackage,
}: CompareViewProps) {
  const [modelRefA, setModelRefA] = useState("");
  const [modelRefB, setModelRefB] = useState("");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [modelA, setModelA] = useState<CompareModelState>(emptyModelState());
  const [modelB, setModelB] = useState<CompareModelState>(emptyModelState());
  const [showSettings, setShowSettings] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Filter to text-generation models only (exclude image/vision-only models).
  // Image models (diffusers) typically have no backend assigned by the library
  // discovery, so `b !== ""` is the primary gate.  The name/format checks are
  // an additional safety net for edge cases.
  const textModels = library.filter((item) => {
    const b = (item.backend ?? "").toLowerCase();
    const f = (item.format ?? "").toLowerCase();
    const n = item.name.toLowerCase();
    return b !== "" && !f.includes("diffuser")
      && !n.includes("stable-diffusion") && !n.includes("flux") && !n.includes("sana");
  });
  const libraryOptions = textModels.map((item) => ({
    value: item.path || item.name,
    label: item.name,
    detail: `${item.format} · ${item.quantization ?? "BF16"} · ${item.sizeGb?.toFixed(1) ?? "?"}GB`,
    backend: item.backend ?? "auto",
    sizeGb: item.sizeGb,
  }));

  const selectedA = libraryOptions.find((o) => o.value === modelRefA);
  const selectedB = libraryOptions.find((o) => o.value === modelRefB);
  const sameModel = modelRefA !== "" && modelRefA === modelRefB;
  // For strategy compatibility, use the most restrictive backend.
  // If both are selected and different, pick the GGUF/llama.cpp one since
  // it has narrower strategy support than MLX. If same or only one selected,
  // use that backend.
  const effectiveBackend = (() => {
    const a = selectedA?.backend;
    const b = selectedB?.backend;
    if (!a && !b) return "auto";
    if (!a) return b!;
    if (!b) return a;
    if (a === b) return a;
    // Mixed backends: pick the more restrictive one for validation
    const isGguf = (be: string) => be.includes("gguf") || be.includes("llama");
    if (isGguf(a) || isGguf(b)) return "gguf";
    return a;
  })();

  async function handleCompare() {
    if (!prompt.trim() || !modelRefA || !modelRefB) return;

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
          modelRefA,
          modelRefB,
          temperature: launchSettings.temperature,
          maxTokens: launchSettings.maxTokens,
          cacheStrategy: launchSettings.cacheStrategy,
          cacheBits: launchSettings.cacheBits,
          fp16Layers: launchSettings.fp16Layers,
          fusedAttention: launchSettings.fusedAttention,
          fitModelInMemory: launchSettings.fitModelInMemory,
          contextTokens: launchSettings.contextTokens,
          speculativeDecoding: launchSettings.speculativeDecoding,
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
              if (event.loading) {
                setModelA((prev) => ({ ...prev, loading: true, loadingMessage: event.message }));
              }
              if (event.token) {
                setModelA((prev) => ({ ...prev, loading: false, text: prev.text + event.token }));
              }
              if (event.done) {
                setModelA((prev) => ({
                  ...prev,
                  done: true,
                  loading: false,
                  tokS: event.tokS ?? 0,
                  promptTokens: event.promptTokens ?? 0,
                  completionTokens: event.completionTokens ?? 0,
                  responseSeconds: event.responseSeconds ?? 0,
                }));
              }
              if (event.error) {
                setModelA((prev) => ({ ...prev, error: event.error, done: true, loading: false }));
              }
            } else if (event.model === "b") {
              if (event.loading) {
                setModelB((prev) => ({ ...prev, loading: true, loadingMessage: event.message }));
              }
              if (event.token) {
                setModelB((prev) => ({ ...prev, loading: false, text: prev.text + event.token }));
              }
              if (event.done) {
                setModelB((prev) => ({
                  ...prev,
                  done: true,
                  loading: false,
                  tokS: event.tokS ?? 0,
                  promptTokens: event.promptTokens ?? 0,
                  completionTokens: event.completionTokens ?? 0,
                  responseSeconds: event.responseSeconds ?? 0,
                }));
              }
              if (event.error) {
                setModelB((prev) => ({ ...prev, error: event.error, done: true, loading: false }));
              }
            }
            if (event.allDone) {
              setBusy(false);
            }
          } catch {
            // skip malformed
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

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12, overflowY: "auto" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 4px" }}>
        <button className="secondary-button" type="button" onClick={onBack} style={{ fontSize: 12 }}>
          Back to Chat
        </button>
        <h3 style={{ margin: 0, fontSize: 16, color: "#c8d0da" }}>Compare Models</h3>
        <small style={{ color: "#7a8594", fontSize: 11 }}>Models run sequentially to conserve memory</small>
      </div>

      {/* Model selectors */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <span className="eyebrow">Model A</span>
          <select
            value={modelRefA}
            onChange={(e) => setModelRefA(e.target.value)}
            className="select-input"
            style={{ width: "100%" }}
            disabled={busy}
          >
            <option value="">Select model...</option>
            {libraryOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label} ({opt.detail})</option>
            ))}
          </select>
        </div>
        <div>
          <span className="eyebrow">Model B</span>
          <select
            value={modelRefB}
            onChange={(e) => setModelRefB(e.target.value)}
            className="select-input"
            style={{ width: "100%" }}
            disabled={busy}
          >
            <option value="">Select model...</option>
            {libraryOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label} ({opt.detail})</option>
            ))}
          </select>
        </div>
      </div>

      {sameModel ? (
        <p style={{ fontSize: 11, color: "#7a8594", margin: 0, padding: "0 4px" }}>
          Same model selected for both — useful for comparing runs or testing different settings.
        </p>
      ) : null}
      {selectedA && selectedB && selectedA.backend !== selectedB.backend && selectedA.backend !== "auto" && selectedB.backend !== "auto" ? (
        <p style={{ fontSize: 11, color: "#d4a053", margin: 0, padding: "0 4px" }}>
          Mixed backends ({selectedA.backend} + {selectedB.backend}) — some cache strategies may not apply to both models.
        </p>
      ) : null}

      {/* Settings toggle + RuntimeControls */}
      <div>
        <button
          type="button"
          className="secondary-button"
          style={{ fontSize: 11, padding: "3px 8px" }}
          onClick={() => setShowSettings(!showSettings)}
        >
          {showSettings ? "Hide settings" : "Launch settings"}
        </button>
      </div>
      {showSettings ? (
        <div style={{ border: "1px solid var(--border)", borderRadius: 8, padding: 12 }}>
          <RuntimeControls
            settings={launchSettings}
            onChange={onLaunchSettingChange}
            preview={preview}
            availableMemoryGb={availableMemoryGb}
            totalMemoryGb={totalMemoryGb}
            availableCacheStrategies={availableCacheStrategies}
            dflashInfo={dflashInfo}
            selectedBackend={effectiveBackend}
            onInstallPackage={onInstallPackage}
            installingPackage={installingPackage}
            compact
            showPreview={false}
          />
        </div>
      ) : null}

      {/* Prompt + action */}
      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !busy) void handleCompare(); }}
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
            disabled={!prompt.trim() || !modelRefA || !modelRefB}
          >
            Compare
          </button>
        )}
      </div>

      {/* Side-by-side results */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, flex: 1, overflow: "hidden" }}>
        <Panel title="Model A" subtitle={modelA.done ? `${number(modelA.tokS)} tok/s | ${number(modelA.responseSeconds)}s` : modelA.loading ? "Loading..." : modelA.text ? "Generating..." : ""}>
          <div style={{ overflow: "auto", flex: 1, padding: 8 }}>
            {selectedA ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 8px" }}>{selectedA.label} · {selectedA.detail}</p> : null}
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
            {selectedB ? <p className="muted-text" style={{ fontSize: 11, margin: "0 0 8px" }}>{selectedB.label} · {selectedB.detail}</p> : null}
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
    </div>
  );
}
