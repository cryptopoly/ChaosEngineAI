import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { MoeOffloadPreview } from "../types/observability";
import { SliderField } from "./SliderField";

// FU-550: a mixture-of-experts GGUF on a discrete NVIDIA card can keep the experts of its first N layers in system
// RAM (llama.cpp --n-cpu-moe). The slider runs from Automatic (-1) through 0 (every expert on the GPU) to every
// layer. The value applies only to the model it was set for (``moeCpuLayersFor``); the load sends nothing for
// another model, so a count chosen for one model never reaches a different one.
const AUTO = -1;

export function moeCpuLayersForLoad(
  settings: { moeCpuLayers?: number | null; moeCpuLayersFor?: string | null },
  modelRef: string,
): number | null {
  if (settings.moeCpuLayers === null || settings.moeCpuLayers === undefined) return null;
  return settings.moeCpuLayersFor === modelRef ? settings.moeCpuLayers : null;
}

export function MoeOffloadField(props: {
  offload: MoeOffloadPreview;
  modelRef: string;
  value: number | null;
  onChange: (layers: number | null) => void;
}) {
  const { t } = useTranslation("runtime");
  const [showInfo, setShowInfo] = useState(false);
  const { offload } = props;
  const total = offload.blocks;
  const current = props.value === null ? AUTO : Math.min(props.value, total);
  const half = Math.round(total / 2);
  const label = (value: number): string => {
    if (value === AUTO) {
      return offload.autoCpuLayers
        ? t("moeOffload.autoWithCount", { count: offload.autoCpuLayers, total, defaultValue: "Automatic (about {count} of {total} layers)" })
        : t("moeOffload.auto", { defaultValue: "Automatic" });
    }
    if (value === 0) return t("moeOffload.none", { defaultValue: "None, all on the GPU" });
    return t("moeOffload.layers", { count: value, total, defaultValue: "{count} of {total} layers" });
  };
  return (
    <div className="moe-offload-field">
      <SliderField
        label={t("moeOffload.label", { defaultValue: "Experts in system RAM" })}
        value={current}
        min={AUTO}
        max={total}
        step={1}
        ticks={[
          { value: AUTO, label: t("moeOffload.autoShort", { defaultValue: "Auto" }) },
          { value: 0, label: "0" },
          { value: half, label: String(half) },
          { value: total, label: String(total) },
        ]}
        formatValue={label}
        onChange={(value) => props.onChange(value === AUTO ? null : value)}
      />
      <small className="muted-text">
        {t("moeOffload.hint", {
          gpu: offload.gpuGb.toFixed(1),
          ram: offload.ramGb.toFixed(1),
          defaultValue:
            "A mixture-of-experts model uses a few experts per token, so their weights can wait in system RAM and run on the CPU while the rest stays on the GPU. More layers in RAM use less GPU memory and reply more slowly. Automatic now: about {gpu} GB on the GPU, {ram} GB in system RAM.",
        })}{" "}
        <button
          type="button"
          className="cache-strategy-info-btn"
          onClick={() => setShowInfo(!showInfo)}
          title={t("moeOffload.aboutTitle", { defaultValue: "About experts in system RAM" })}
        >
          i
        </button>
      </small>
      {showInfo ? (
        <div className="cache-strategy-info-panel" style={{ marginTop: 4 }}>
          <p>
            {t("moeOffload.body", {
              defaultValue:
                "Automatic keeps everything on the GPU when it fits. When it does not, and Fit in memory is on, the engine moves experts to system RAM one layer at a time until the model fits; with Fit in memory off, ChaosEngineAI picks the count from the GPU's free memory. Measured on an RTX 4090 with a 35B model that reads 3B parameters per token: 150 tokens per second with every expert on the GPU, about 85 with half the layers in RAM, about 52 with all of them in RAM.",
            })}
          </p>
        </div>
      ) : null}
    </div>
  );
}
