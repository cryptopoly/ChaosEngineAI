import type { GenerationMetrics } from "../types";

/**
 * Phase 3.4: Substrate routing inspector — concise per-turn badge
 * showing which engine + cache strategy + speculative-decode budget
 * served the response, plus DFLASH acceptance rate when available.
 *
 * The data already lands on each assistant message's `metrics` blob
 * via inference.py / mlx_worker.py. Rendering it inline (above the
 * collapsible Model Details fold-out) makes the substrate visible
 * by default — operators can tell at a glance whether the turn went
 * through MLX vs llama.cpp, ChaosEngine vs TurboQuant, and how well
 * speculative decoding is doing.
 *
 * No badge renders when metrics is missing entirely; partial metrics
 * still render the fields that are present so partial-fail turns
 * still surface useful detail.
 */
export interface SubstrateRoutingBadgeProps {
  metrics: GenerationMetrics;
}

interface Chip {
  key: string;
  label: string;
  title: string;
  tone: "default" | "accent" | "warn";
}

function buildChips(metrics: GenerationMetrics): Chip[] {
  const chips: Chip[] = [];

  // Engine — MLX / llama.cpp / vLLM / etc. The runtime ships its own
  // engineLabel; fall back to backend if missing.
  const engine = metrics.engineLabel || metrics.backend;
  if (engine) {
    chips.push({
      key: "engine",
      label: String(engine),
      title: `Inference runtime that served this turn (${engine})`,
      tone: "default",
    });
  }

  // Cache strategy + bits, e.g. "ChaosEngine bf16" or "TurboQuant 4-bit".
  const cacheLabel = metrics.cacheLabel
    || (metrics.cacheStrategy
      ? metrics.cacheBits
        ? `${metrics.cacheStrategy} ${metrics.cacheBits}-bit`
        : metrics.cacheStrategy
      : null);
  if (cacheLabel) {
    chips.push({
      key: "cache",
      label: String(cacheLabel),
      title: `KV cache strategy (${cacheLabel})`,
      tone: "default",
    });
  }

  // Speculative decoding state. When on, surface the tree budget so
  // users know how aggressively DDTree was drafting.
  if (metrics.speculativeDecoding) {
    const budget = metrics.treeBudget;
    chips.push({
      key: "spec",
      label: budget && budget > 0 ? `DDTree ${budget}` : "DDTree",
      title: budget
        ? `Tree-based speculative decoding active (budget ${budget} draft tokens per step)`
        : "Tree-based speculative decoding active",
      tone: "accent",
    });

    if (metrics.dflashAcceptanceRate != null && metrics.dflashAcceptanceRate > 0) {
      chips.push({
        key: "accept",
        label: `${metrics.dflashAcceptanceRate.toFixed(1)} avg accepted`,
        title: `Average draft tokens accepted per step (${metrics.dflashAcceptanceRate.toFixed(2)})`,
        tone: "accent",
      });
    }
  }

  if (metrics.runtimeNote) {
    chips.push({
      key: "note",
      label: metrics.runtimeNote.length > 48 ? `${metrics.runtimeNote.slice(0, 45)}…` : metrics.runtimeNote,
      title: metrics.runtimeNote,
      tone: "warn",
    });
  }

  return chips;
}

export function SubstrateRoutingBadge({ metrics }: SubstrateRoutingBadgeProps) {
  const chips = buildChips(metrics);
  if (chips.length === 0) return null;
  return (
    <div className="substrate-routing" aria-label="Substrate routing for this turn">
      {chips.map((chip) => (
        <span
          key={chip.key}
          className={`substrate-chip substrate-chip--${chip.tone}`}
          title={chip.title}
        >
          {chip.label}
        </span>
      ))}
    </div>
  );
}

// Exported for unit tests so the chip-building logic can be exercised
// without rendering React.
export { buildChips };
