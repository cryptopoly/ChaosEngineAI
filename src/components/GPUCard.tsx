import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { fetchJson } from "../api";
import { number } from "../utils";

interface GPUMetrics {
  gpu_name: string;
  vram_total_gb: number;
  vram_used_gb: number;
  utilization_pct: number;
  temperature_c: number | null;
  power_w: number | null;
}

interface GPUCardProps {
  backendOnline: boolean;
  visible: boolean;
}

export function GPUCard({ backendOnline, visible }: GPUCardProps) {
  const { t } = useTranslation("common");
  const [metrics, setMetrics] = useState<GPUMetrics | null>(null);

  useEffect(() => {
    if (!backendOnline || !visible) return;

    function poll() {
      fetchJson<{ gpu: GPUMetrics | null }>("/api/metrics/gpu")
        .then((data) => setMetrics(data.gpu ?? null))
        .catch(() => {});
    }

    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [backendOnline, visible]);

  if (!metrics) return null;

  const usedPct = metrics.vram_total_gb > 0
    ? Math.round((metrics.vram_used_gb / metrics.vram_total_gb) * 100)
    : 0;

  const barColor = usedPct > 90 ? "#f87171" : usedPct > 70 ? "#f0b060" : "#8fcf9f";

  return (
    <div style={{ border: "1px solid #27303a", borderRadius: 8, padding: 12, background: "#1a1f26" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontWeight: 600, color: "#c8d0da", fontSize: 13 }}>{t("gpuCard.heading", { defaultValue: "GPU" })}</span>
        <span style={{ fontSize: 11, color: "#7a8594" }}>{metrics.gpu_name}</span>
      </div>

      {/* VRAM bar */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#7a8594", marginBottom: 3 }}>
          <span>{t("gpuCard.vramLabel", { defaultValue: "VRAM" })}</span>
          <span>{t("gpuCard.vramValue", {
            defaultValue: "{used} / {total} GB ({percent}%)",
            used: number(metrics.vram_used_gb),
            total: number(metrics.vram_total_gb),
            percent: usedPct,
          })}</span>
        </div>
        <div style={{ height: 6, borderRadius: 3, background: "#0f1215", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${usedPct}%`, background: barColor, borderRadius: 3, transition: "width 0.3s" }} />
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: "flex", gap: 16, fontSize: 12 }}>
        <div>
          <span style={{ color: "#5a6574" }}>{t("gpuCard.utilLabel", { defaultValue: "Util: " })}</span>
          <span style={{ color: "#c8d0da" }}>{metrics.utilization_pct}%</span>
        </div>
        {metrics.temperature_c !== null && (
          <div>
            <span style={{ color: "#5a6574" }}>{t("gpuCard.tempLabel", { defaultValue: "Temp: " })}</span>
            <span style={{ color: metrics.temperature_c > 85 ? "#f87171" : "#c8d0da" }}>{t("gpuCard.tempValue", { defaultValue: "{value}C", value: metrics.temperature_c })}</span>
          </div>
        )}
        {metrics.power_w !== null && (
          <div>
            <span style={{ color: "#5a6574" }}>{t("gpuCard.powerLabel", { defaultValue: "Power: " })}</span>
            <span style={{ color: "#c8d0da" }}>{t("gpuCard.powerValue", { defaultValue: "{value}W", value: number(metrics.power_w) })}</span>
          </div>
        )}
      </div>
    </div>
  );
}
