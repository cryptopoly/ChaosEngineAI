/**
 * The device and memory facts behind the Video Studio's run estimate, as one sentence for the memory
 * item's tooltip: which device the run is for (the one the backend reported, else a guess from the host,
 * tagged "detected"), how much memory it has, and how big the model and this run are.
 */
import type { VideoEffectiveDevice } from "../../utils";

type TFn = (key: string, options: Record<string, unknown>) => string;

export interface CapacityFacts {
  /** Device the backend reported ("mps", "cuda", "cpu", "mlx" ...), when it did. */
  device: string | null | undefined;
  activeEngine: string | null | undefined;
  /** What the safety helper settled on when the backend reported nothing. */
  effectiveDevice: VideoEffectiveDevice;
  /** The total memory the estimate used, and whether the backend really reported it. */
  deviceMemoryGb: number;
  memoryReported: boolean;
  modelFootprintGb: number;
  estimatedPeakGb: number;
}

export function formatCapacityGb(gb: number): string {
  return gb >= 10 ? `${gb.toFixed(0)} GB` : `${gb.toFixed(1)} GB`;
}

export function deviceLabel(t: TFn, facts: Pick<CapacityFacts, "device" | "activeEngine" | "effectiveDevice">): string {
  if (!facts.device) {
    if (facts.effectiveDevice === "cuda") return t("videoStudio.device.gpuDetected", { defaultValue: "GPU (detected)" });
    if (facts.effectiveDevice === "cpu") return t("videoStudio.device.cpuDetected", { defaultValue: "CPU (detected)" });
    return t("videoStudio.device.appleDetected", { defaultValue: "Apple Silicon (detected)" });
  }
  const reported = facts.device.toUpperCase();
  if (facts.activeEngine === "mlx-video") return t("videoStudio.device.appleMlx", { defaultValue: "Apple Silicon (MLX)" });
  if (reported.startsWith("CUDA")) return t("videoStudio.device.gpu", { defaultValue: "GPU" });
  if (reported === "MPS") return t("videoStudio.device.apple", { defaultValue: "Apple Silicon" });
  return reported;
}

export function capacityLine(t: TFn, facts: CapacityFacts): string {
  const memory = facts.memoryReported
    ? formatCapacityGb(facts.deviceMemoryGb)
    : t("videoStudio.memory.defaultFallback", {
        defaultValue: "~{value} (default — restart backend for real detection)",
        value: formatCapacityGb(facts.deviceMemoryGb),
      });
  return facts.modelFootprintGb > 0
    ? t("videoStudio.capacityLine.withModel", {
        defaultValue: "{device} · {memory} total · model ≈ {model}, this run peak ≈ {peak}",
        device: deviceLabel(t, facts),
        memory,
        model: formatCapacityGb(facts.modelFootprintGb),
        peak: formatCapacityGb(facts.estimatedPeakGb),
      })
    : t("videoStudio.capacityLine.short", {
        defaultValue: "{device} · {memory} total · this run peak ≈ {peak}",
        device: deviceLabel(t, facts),
        memory,
        peak: formatCapacityGb(facts.estimatedPeakGb),
      });
}
