import { useTranslation } from "react-i18next";
import type { GenerationMetrics, PerfTelemetry } from "../types";
import type { TFunction } from "i18next";

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

// Same fallback contract as SubstrateRoutingBadge — keeps the standalone
// buildPerfChips export usable from unit tests without an i18n provider.
function defaultTranslator(_key: string, options?: Record<string, unknown>): string {
  const fallback = (options?.defaultValue as string | undefined) ?? "";
  if (!options) return fallback;
  return fallback.replace(/\{(\w+)\}/g, (_match, name: string) => {
    const value = options[name];
    return value == null ? "" : String(value);
  });
}

function buildPerfChips(
  telemetry: PerfTelemetry,
  tokS: number | null,
  t: TFunction | typeof defaultTranslator = defaultTranslator,
): PerfChip[] {
  const chips: PerfChip[] = [];

  if (tokS != null && tokS > 0) {
    chips.push({
      key: "toks",
      label: t("chatPerfStrip.toksLabel", {
        defaultValue: "{value} tok/s",
        value: tokS.toFixed(1),
      }),
      title: t("chatPerfStrip.toksTitle", {
        defaultValue: "Decode throughput for this turn ({value} tokens/sec)",
        value: tokS.toFixed(2),
      }),
      tone: tokS < 1 ? "alert" : tokS < 5 ? "warn" : "default",
    });
  }

  if (telemetry.cpuPercent != null) {
    chips.push({
      key: "cpu",
      label: t("chatPerfStrip.cpuLabel", {
        defaultValue: "CPU {value}%",
        value: telemetry.cpuPercent.toFixed(0),
      }),
      title: t("chatPerfStrip.cpuTitle", {
        defaultValue: "CPU utilisation at turn finalisation ({value}%)",
        value: telemetry.cpuPercent.toFixed(1),
      }),
      tone: telemetry.cpuPercent > 90 ? "warn" : "default",
    });
  }

  if (telemetry.gpuPercent != null) {
    chips.push({
      key: "gpu",
      label: t("chatPerfStrip.gpuLabel", {
        defaultValue: "GPU {value}%",
        value: telemetry.gpuPercent.toFixed(0),
      }),
      title: t("chatPerfStrip.gpuTitle", {
        defaultValue: "GPU / accelerator utilisation at turn finalisation ({value}%)",
        value: telemetry.gpuPercent.toFixed(1),
      }),
      tone: telemetry.gpuPercent > 90 ? "warn" : "default",
    });
  }

  if (telemetry.availableMemoryGb != null) {
    chips.push({
      key: "mem",
      label: t("chatPerfStrip.memLabel", {
        defaultValue: "{value} GB free",
        value: telemetry.availableMemoryGb.toFixed(1),
      }),
      title: t("chatPerfStrip.memTitle", {
        defaultValue: "Available RAM at turn finalisation ({value} GB)",
        value: telemetry.availableMemoryGb.toFixed(2),
      }),
      tone: telemetry.availableMemoryGb < 2 ? "alert" : telemetry.availableMemoryGb < 4 ? "warn" : "default",
    });
  }

  if (telemetry.thermalState) {
    chips.push({
      key: "thermal",
      label: t("chatPerfStrip.thermalLabel", {
        defaultValue: "Thermal: {state}",
        state: telemetry.thermalState,
      }),
      title: t("chatPerfStrip.thermalTitle", {
        defaultValue: "Host thermal state ({state}). Critical means active throttling.",
        state: telemetry.thermalState,
      }),
      tone: THERMAL_TONE[telemetry.thermalState] ?? "default",
    });
  }

  return chips;
}

export function ChatPerfStrip({ metrics }: ChatPerfStripProps) {
  const { t } = useTranslation("common");
  const telemetry = metrics.perfTelemetry;
  if (!telemetry) return null;
  const chips = buildPerfChips(telemetry, metrics.tokS ?? null, t);
  if (chips.length === 0) return null;
  return (
    <div
      className="chat-perf-strip"
      aria-label={t("chatPerfStrip.ariaLabel", { defaultValue: "Host telemetry for this turn" })}
    >
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
