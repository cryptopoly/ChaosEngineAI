import type { GenerationMetrics, PerfTelemetry } from "../types";

/**
 * Phase 3.5: cross-platform per-turn perf telemetry strip.
 *
 * Renders a compact row of substrate-side host metrics sampled at
 * the moment the turn finalised — CPU %, GPU %, available memory,
 * thermal state. Sits below the substrate routing badge to give
 * operators a thermal / load read alongside the runtime decision.
 *
 * All fields are optional: macOS today reads thermal via pmset,
 * Windows / Linux fall through to None. The strip omits any field
 * that's null so unsupported platforms still show a useful subset.
 */
export interface ChatPerfStripProps {
  metrics: GenerationMetrics;
}

interface PerfChip {
  key: string;
  label: string;
  title: string;
  tone: "default" | "warn" | "alert";
}

const THERMAL_TONE: Record<string, PerfChip["tone"]> = {
  nominal: "default",
  moderate: "warn",
  critical: "alert",
};

function buildPerfChips(telemetry: PerfTelemetry, tokS: number | null): PerfChip[] {
  const chips: PerfChip[] = [];

  if (tokS != null && tokS > 0) {
    chips.push({
      key: "toks",
      label: `${tokS.toFixed(1)} tok/s`,
      title: `Decode throughput for this turn (${tokS.toFixed(2)} tokens/sec)`,
      tone: tokS < 1 ? "alert" : tokS < 5 ? "warn" : "default",
    });
  }

  if (telemetry.cpuPercent != null) {
    chips.push({
      key: "cpu",
      label: `CPU ${telemetry.cpuPercent.toFixed(0)}%`,
      title: `CPU utilisation at turn finalisation (${telemetry.cpuPercent.toFixed(1)}%)`,
      tone: telemetry.cpuPercent > 90 ? "warn" : "default",
    });
  }

  if (telemetry.gpuPercent != null) {
    chips.push({
      key: "gpu",
      label: `GPU ${telemetry.gpuPercent.toFixed(0)}%`,
      title: `GPU / accelerator utilisation at turn finalisation (${telemetry.gpuPercent.toFixed(1)}%)`,
      tone: telemetry.gpuPercent > 90 ? "warn" : "default",
    });
  }

  if (telemetry.availableMemoryGb != null) {
    chips.push({
      key: "mem",
      label: `${telemetry.availableMemoryGb.toFixed(1)} GB free`,
      title: `Available RAM at turn finalisation (${telemetry.availableMemoryGb.toFixed(2)} GB)`,
      tone: telemetry.availableMemoryGb < 2 ? "alert" : telemetry.availableMemoryGb < 4 ? "warn" : "default",
    });
  }

  if (telemetry.thermalState) {
    chips.push({
      key: "thermal",
      label: `Thermal: ${telemetry.thermalState}`,
      title: `Host thermal state (${telemetry.thermalState}). Critical means active throttling.`,
      tone: THERMAL_TONE[telemetry.thermalState] ?? "default",
    });
  }

  return chips;
}

export function ChatPerfStrip({ metrics }: ChatPerfStripProps) {
  const telemetry = metrics.perfTelemetry;
  if (!telemetry) return null;
  const chips = buildPerfChips(telemetry, metrics.tokS ?? null);
  if (chips.length === 0) return null;
  return (
    <div className="chat-perf-strip" aria-label="Host telemetry for this turn">
      {chips.map((chip) => (
        <span
          key={chip.key}
          className={`perf-chip perf-chip--${chip.tone}`}
          title={chip.title}
        >
          {chip.label}
        </span>
      ))}
    </div>
  );
}

// Exported for unit testing.
export { buildPerfChips };
