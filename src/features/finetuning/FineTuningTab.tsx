import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchJson } from "../../api";
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
  const { t } = useTranslation("common");
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!backendOnline) return;
    fetchJson<{ adapters?: Adapter[] }>("/api/adapters")
      .then((data) => {
        setAdapters(data.adapters ?? []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [backendOnline]);

  useEffect(() => {
    if (!backendOnline) return;
    fetchJson<{ status?: string | null }>("/api/finetuning/status")
      .then((data) => setTrainingStatus(data.status ?? null))
      .catch(() => {});
  }, [backendOnline]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <Panel
        title={t("panels.loraAdapters", { defaultValue: "LoRA Adapters" })}
        subtitle={t("panels.loraAdaptersFound", {
          count: adapters.length,
          defaultValue: "{{count}} adapters found",
        })}
      >
        {loading ? (
          <p className="muted-text">{t("fineTuning.loadingAdapters", { defaultValue: "Loading adapters..." })}</p>
        ) : adapters.length === 0 ? (
          <div style={{ padding: 24, textAlign: "center" }}>
            <p className="muted-text" style={{ marginBottom: 8 }}>{t("fineTuning.noAdapters", { defaultValue: "No LoRA adapters found." })}</p>
            <p style={{ color: "#5a6574", fontSize: 12 }}>
              {t("fineTuning.noAdaptersHint", { defaultValue: "Place adapter directories (with adapter_config.json) in your model directories, or use the fine-tuning feature below to create new adapters." })}
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
                <div style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.adapterBase", { value: adapter.base_model, defaultValue: "Base: {value}" })}</div>
                <div style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.adapterSize", { value: adapter.size_mb.toFixed(1), defaultValue: "Size: {value} MB" })}</div>
                <div style={{ fontSize: 11, color: "#5a6574", marginTop: 4 }}>{t("fineTuning.adapterCreated", { value: adapter.created, defaultValue: "Created: {value}" })}</div>
              </div>
            ))}
          </div>
        )}
      </Panel>

      <Panel
        title={t("tabs.fineTuning")}
        subtitle={trainingStatus ?? t("status.ready")}
      >
        <div style={{ padding: 24, textAlign: "center" }}>
          <p className="muted-text" style={{ marginBottom: 12 }}>
            {t("fineTuning.intro", { defaultValue: "Fine-tune models with QLoRA on Apple Silicon (MLX) or via llama.cpp." })}
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 16, background: "#1a1f26", maxWidth: 240 }}>
              <div style={{ fontWeight: 600, color: "#8fb4ff", marginBottom: 4 }}>{t("fineTuning.step1Title", { defaultValue: "1. Upload Dataset" })}</div>
              <p style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.step1Body", { defaultValue: "JSONL format with \"prompt\" and \"completion\" fields" })}</p>
            </div>
            <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 16, background: "#1a1f26", maxWidth: 240 }}>
              <div style={{ fontWeight: 600, color: "#8fb4ff", marginBottom: 4 }}>{t("fineTuning.step2Title", { defaultValue: "2. Configure" })}</div>
              <p style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.step2Body", { defaultValue: "Set learning rate, epochs, LoRA rank, and batch size" })}</p>
            </div>
            <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 16, background: "#1a1f26", maxWidth: 240 }}>
              <div style={{ fontWeight: 600, color: "#8fb4ff", marginBottom: 4 }}>{t("fineTuning.step3Title", { defaultValue: "3. Train" })}</div>
              <p style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.step3Body", { defaultValue: "Monitor loss and progress in real-time" })}</p>
            </div>
          </div>
          <button
            className="primary-button"
            type="button"
            style={{ marginTop: 16 }}
            disabled={!backendOnline}
            onClick={() => {/* TODO: open training config modal */}}
          >
            {t("fineTuning.startButton", { defaultValue: "Start Fine-Tuning" })}
          </button>
        </div>
      </Panel>
    </div>
  );
}
