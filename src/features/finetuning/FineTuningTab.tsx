import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchJson } from "../../api";
import { Panel } from "../../components/Panel";

/** One row of GET /api/adapters (backend_service/helpers/finetuning.py `_read_adapter_meta`). */
export interface Adapter {
  name: string;
  path: string;
  base_model: string;
  created?: string;
  size_bytes: number;
  config: Record<string, unknown>;
}

interface FineTuningTabProps {
  backendOnline: boolean;
}

export function AdapterCard({ adapter }: { adapter: Adapter }) {
  const { t } = useTranslation("common");
  return (
    <div
      style={{
        border: "1px solid #27303a",
        borderRadius: 8,
        padding: 12,
        background: "#1a1f26",
        overflowWrap: "anywhere",
      }}
    >
      <div style={{ fontWeight: 600, color: "#c8d0da", marginBottom: 4 }}>{adapter.name}</div>
      <div style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.adapterBase", { value: adapter.base_model, defaultValue: "Base: {value}" })}</div>
      <div style={{ fontSize: 12, color: "#7a8594" }}>{t("fineTuning.adapterSize", { value: ((adapter.size_bytes ?? 0) / 1048576).toFixed(1), defaultValue: "Size: {value} MB" })}</div>
      {adapter.created ? (
        <div style={{ fontSize: 11, color: "#5a6574", marginTop: 4 }}>{t("fineTuning.adapterCreated", { value: adapter.created, defaultValue: "Created: {value}" })}</div>
      ) : null}
    </div>
  );
}

export function FineTuningTab({ backendOnline }: FineTuningTabProps) {
  const { t } = useTranslation("common");
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!backendOnline) return;
    setLoadError(null);
    fetchJson<{ adapters?: Adapter[] }>("/api/adapters")
      .then((data) => {
        setAdapters(data.adapters ?? []);
        setLoading(false);
      })
      .catch((error: unknown) => {
        setLoadError(error instanceof Error ? error.message : String(error));
        setLoading(false);
      });
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
        subtitle={loading || loadError ? undefined : t("panels.loraAdaptersFound", {
          count: adapters.length,
          defaultValue: "{{count}} adapters found",
        })}
      >
        {loading && !backendOnline ? (
          <p className="muted-text">{t("fineTuning.offline", { defaultValue: "The backend is offline. Adapters are listed when it is back." })}</p>
        ) : loading ? (
          <p className="muted-text">{t("fineTuning.loadingAdapters", { defaultValue: "Loading adapters..." })}</p>
        ) : loadError ? (
          <p className="muted-text" role="alert">{t("fineTuning.loadError", { message: loadError, defaultValue: "Could not load the adapter list: {message}" })}</p>
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
              <AdapterCard key={adapter.path} adapter={adapter} />
            ))}
          </div>
        )}
      </Panel>

      <Panel
        title={t("tabs.fineTuning")}
        subtitle={trainingStatus && trainingStatus !== "idle" ? trainingStatus : t("fineTuning.notAvailable", { defaultValue: "Not available in this version" })}
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
          {/* No training backend exists yet (POST /api/finetuning/start answers 501), so the button stays off. */}
          <button className="primary-button" type="button" style={{ marginTop: 16 }} disabled aria-describedby="finetuning-not-available">
            {t("fineTuning.startButton", { defaultValue: "Start Fine-Tuning" })}
          </button>
          <p id="finetuning-not-available" className="muted-text" style={{ marginTop: 12, fontSize: 12 }}>
            {t("fineTuning.notAvailableBody", {
              defaultValue: "Training is not built yet: this page only lists the adapters it finds on disk, and an adapter cannot be applied to a model yet. The steps above show what is planned.",
            })}
          </p>
        </div>
      </Panel>
    </div>
  );
}
