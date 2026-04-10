import { useState, useRef } from "react";
import Markdown from "react-markdown";
import { Panel } from "../../components/Panel";
import type { LibraryItem } from "../../types";
import { number } from "../../utils";

interface CompareModelState {
  text: string;
  done: boolean;
  tokS: number;
  promptTokens: number;
  completionTokens: number;
  responseSeconds: number;
  error?: string;
}

interface CompareViewProps {
  library: LibraryItem[];
  onBack: () => void;
}

const emptyModelState = (): CompareModelState => ({
  text: "",
  done: false,
  tokS: 0,
  promptTokens: 0,
  completionTokens: 0,
  responseSeconds: 0,
});

export function CompareView({ library, onBack }: CompareViewProps) {
  const [modelRefA, setModelRefA] = useState("");
  const [modelRefB, setModelRefB] = useState("");
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [modelA, setModelA] = useState<CompareModelState>(emptyModelState());
  const [modelB, setModelB] = useState<CompareModelState>(emptyModelState());
  const abortRef = useRef<AbortController | null>(null);

  const libraryOptions = library.map((item) => ({
    value: item.path || item.name,
    label: `${item.name} (${item.format}, ${item.sizeGb?.toFixed(1) ?? "?"}GB)`,
  }));

  async function handleCompare() {
    if (!prompt.trim() || !modelRefA || !modelRefB) return;

    setBusy(true);
    setModelA(emptyModelState());
    setModelB(emptyModelState());

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch("/api/chat/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim(),
          modelRefA,
          modelRefB,
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
              if (event.token) {
                setModelA((prev) => ({ ...prev, text: prev.text + event.token }));
              }
              if (event.done) {
                setModelA((prev) => ({
                  ...prev,
                  done: true,
                  tokS: event.tokS ?? 0,
                  promptTokens: event.promptTokens ?? 0,
                  completionTokens: event.completionTokens ?? 0,
                  responseSeconds: event.responseSeconds ?? 0,
                }));
              }
              if (event.error) {
                setModelA((prev) => ({ ...prev, error: event.error, done: true }));
              }
            } else if (event.model === "b") {
              if (event.token) {
                setModelB((prev) => ({ ...prev, text: prev.text + event.token }));
              }
              if (event.done) {
                setModelB((prev) => ({
                  ...prev,
                  done: true,
                  tokS: event.tokS ?? 0,
                  promptTokens: event.promptTokens ?? 0,
                  completionTokens: event.completionTokens ?? 0,
                  responseSeconds: event.responseSeconds ?? 0,
                }));
              }
              if (event.error) {
                setModelB((prev) => ({ ...prev, error: event.error, done: true }));
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
    <div style={{ display: "flex", flexDirection: "column", height: "100%", gap: 12 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "0 4px" }}>
        <button className="secondary-button" type="button" onClick={onBack} style={{ fontSize: 12 }}>
          Back to Chat
        </button>
        <h3 style={{ margin: 0, fontSize: 16, color: "#c8d0da" }}>Compare Models</h3>
      </div>

      {/* Model selectors + prompt */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <label style={{ fontSize: 11, color: "#7a8594", marginBottom: 4, display: "block" }}>Model A</label>
          <select
            value={modelRefA}
            onChange={(e) => setModelRefA(e.target.value)}
            className="select-input"
            style={{ width: "100%" }}
          >
            <option value="">Select model...</option>
            {libraryOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label style={{ fontSize: 11, color: "#7a8594", marginBottom: 4, display: "block" }}>Model B</label>
          <select
            value={modelRefB}
            onChange={(e) => setModelRefB(e.target.value)}
            className="select-input"
            style={{ width: "100%" }}
          >
            <option value="">Select model...</option>
            {libraryOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>
      </div>

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
        <Panel title="Model A" subtitle={modelA.done ? `${number(modelA.tokS)} tok/s | ${number(modelA.responseSeconds)}s` : busy ? "Generating..." : ""}>
          <div style={{ overflow: "auto", flex: 1, padding: 8 }}>
            {modelA.error ? (
              <p style={{ color: "#f87171" }}>{modelA.error}</p>
            ) : modelA.text ? (
              <div className="markdown-content">
                <Markdown>{modelA.text}</Markdown>
              </div>
            ) : (
              <p className="muted-text" style={{ fontSize: 13 }}>Waiting for response...</p>
            )}
          </div>
        </Panel>

        <Panel title="Model B" subtitle={modelB.done ? `${number(modelB.tokS)} tok/s | ${number(modelB.responseSeconds)}s` : modelA.done && busy ? "Generating..." : ""}>
          <div style={{ overflow: "auto", flex: 1, padding: 8 }}>
            {modelB.error ? (
              <p style={{ color: "#f87171" }}>{modelB.error}</p>
            ) : modelB.text ? (
              <div className="markdown-content">
                <Markdown>{modelB.text}</Markdown>
              </div>
            ) : (
              <p className="muted-text" style={{ fontSize: 13 }}>Waiting for response...</p>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );
}
