import { useEffect, useState } from "react";
import { Panel } from "../../components/Panel";

interface Adapter {
  id: string;
  name: string;
  base_model: string;
  created: string;
  size_mb: number;
  config: Record<string, unknown>;
}

interface FineTuningTabProps {
  backendOnline: boolean;
}

export function FineTuningTab({ backendOnline }: FineTuningTabProps) {
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!backendOnline) return;
    fetch("/api/adapters")
      .then((r) => r.json())
      .then((data) => {
        setAdapters(data.adapters ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [backendOnline]);

  useEffect(() => {
    if (!backendOnline) return;
    fetch("/api/finetuning/status")
      .then((r) => r.json())
      .then((data) => setTrainingStatus(data.status ?? null))
      .catch(() => {});
  }, [backendOnline]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Panel title="LoRA Adapters" subtitle={`${adapters.length} adapter${adapters.length !== 1 ? "s" : ""} found`}>
        {loading ? (
          <p className="muted-text">Loading adapters...</p>
        ) : adapters.length === 0 ? (
          <div style={{ padding: 24, textAlign: "center" }}>
            <p className="muted-text" style={{ marginBottom: 8 }}>No LoRA adapters found.</p>
            <p style={{ color: "#5a6574", fontSize: 12 }}>
              Place adapter directories (with adapter_config.json) in your model directories,
              or use the fine-tuning feature below to create new adapters.
            </p>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 12, padding: 8 }}>
            {adapters.map((adapter) => (
              <div
                key={adapter.id}
                style={{
                  border: "1px solid #27303a",
                  borderRadius: 8,
                  padding: 12,
                  background: "#1a1f26",
                }}
              >
                <div style={{ fontWeight: 600, color: "#c8d0da", marginBottom: 4 }}>{adapter.name}</div>
                <div style={{ fontSize: 12, color: "#7a8594" }}>Base: {adapter.base_model}</div>
                <div style={{ fontSize: 12, color: "#7a8594" }}>Size: {adapter.size_mb.toFixed(1)} MB</div>
                <div style={{ fontSize: 11, color: "#5a6574", marginTop: 4 }}>Created: {adapter.created}</div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel title="Fine-Tuning" subtitle={trainingStatus ?? "Ready"}>
        <div style={{ padding: 24, textAlign: "center" }}>
          <p className="muted-text" style={{ marginBottom: 12 }}>
            Fine-tune models with QLoRA on Apple Silicon (MLX) or via llama.cpp.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 16, background: "#1a1f26", maxWidth: 240 }}>
              <div style={{ fontWeight: 600, color: "#8fb4ff", marginBottom: 4 }}>1. Upload Dataset</div>
              <p style={{ fontSize: 12, color: "#7a8594" }}>JSONL format with "prompt" and "completion" fields</p>
            </div>
            <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 16, background: "#1a1f26", maxWidth: 240 }}>
              <div style={{ fontWeight: 600, color: "#8fb4ff", marginBottom: 4 }}>2. Configure</div>
              <p style={{ fontSize: 12, color: "#7a8594" }}>Set learning rate, epochs, LoRA rank, and batch size</p>
            </div>
            <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 16, background: "#1a1f26", maxWidth: 240 }}>
              <div style={{ fontWeight: 600, color: "#8fb4ff", marginBottom: 4 }}>3. Train</div>
              <p style={{ fontSize: 12, color: "#7a8594" }}>Monitor loss and progress in real-time</p>
            </div>
          </div>
          <button
            className="primary-button"
            type="button"
            style={{ marginTop: 16 }}
            disabled={!backendOnline}
            onClick={() => {/* TODO: open training config modal */}}
          >
            Start Fine-Tuning
          </button>
        </div>
      </Panel>
    </div>
  );
}
