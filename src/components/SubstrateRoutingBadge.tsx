import { useTranslation } from "react-i18next";
import type { GenerationMetrics } from "../types";
import type { TFunction } from "i18next";

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

// Fallback translator used by unit tests where the i18n provider is
// not mounted; mirrors react-i18next's contract of "use defaultValue
// when no key is registered" by simply interpolating {var} placeholders
// in the defaultValue itself.
function defaultTranslator(_key: string, options?: Record<string, unknown>): string {
  const fallback = (options?.defaultValue as string | undefined) ?? "";
  if (!options) return fallback;
  return fallback.replace(/\{(\w+)\}/g, (_match, name: string) => {
    const value = options[name];
    return value == null ? "" : String(value);
  });
}

function buildChips(metrics: GenerationMetrics, t: TFunction | typeof defaultTranslator = defaultTranslator): Chip[] {
  const chips: Chip[] = [];

  // Engine — MLX / llama.cpp / vLLM / etc. The runtime ships its own
  // engineLabel; fall back to backend if missing.
  const engine = metrics.engineLabel || metrics.backend;
  if (engine) {
    chips.push({
      key: "engine",
      label: String(engine),
      title: t("substrateRoutingBadge.engineTitle", {
        defaultValue: "Inference runtime that served this turn ({engine})",
        engine,
      }),
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
      title: t("substrateRoutingBadge.cacheTitle", {
        defaultValue: "KV cache strategy ({label})",
        label: cacheLabel,
      }),
      tone: "default",
    });
  }

  // Speculative decoding state. When on, surface the tree budget so
  // users know how aggressively DDTree was drafting.
  if (metrics.speculativeDecoding) {
    const budget = metrics.treeBudget;
    chips.push({
      key: "spec",
      label: budget && budget > 0
        ? t("substrateRoutingBadge.specLabelWithBudget", {
            defaultValue: "DDTree {budget}",
            budget,
          })
        : t("substrateRoutingBadge.specLabel", { defaultValue: "DDTree" }),
      title: budget
        ? t("substrateRoutingBadge.specTitleWithBudget", {
            defaultValue: "Tree-based speculative decoding active (budget {budget} draft tokens per step)",
            budget,
          })
        : t("substrateRoutingBadge.specTitle", {
            defaultValue: "Tree-based speculative decoding active",
          }),
      tone: "accent",
    });

    if (metrics.dflashAcceptanceRate != null && metrics.dflashAcceptanceRate > 0) {
      chips.push({
        key: "accept",
        label: t("substrateRoutingBadge.acceptLabel", {
          defaultValue: "{value} avg accepted",
          value: metrics.dflashAcceptanceRate.toFixed(1),
        }),
        title: t("substrateRoutingBadge.acceptTitle", {
          defaultValue: "Average draft tokens accepted per step ({value})",
          value: metrics.dflashAcceptanceRate.toFixed(2),
        }),
        tone: "accent",
      });
    }
  }

  if (metrics.runtimeNote) {
    chips.push({
      key: "note",
      label: metrics.runtimeNote.length > 48 ? `${metrics.runtimeNote.slice(0, 45)}…` : metrics.runtimeNote,
      title: metrics.runtimeNote,
      // Default tone for benign info ("Using python with MLX 0.31.x and
      // mlx-lm 0.31.y."); warn only when the note flags an actual fault
      // — DFLASH unavailable, cache strategy fell back, MTP head missing,
      // etc. Operators ignore the orange chip if every turn surfaces it,
      // which defeats its purpose for the rare real warnings.
      tone: runtimeNoteIsWarning(metrics.runtimeNote) ? "warn" : "default",
    });
  }

  return chips;
}

/**
 * Decide whether a runtime note describes a problem the user should
 * notice. The boring "which library versions ran" prefix is always
 * present and not actionable; the warn tone should fire only when a
 * substantive issue appears later in the same string.
 */
export function runtimeNoteIsWarning(note: string): boolean {
  const lowered = note.toLowerCase();
  const warningTokens = [
    "unavailable",
    "fell back",
    "fall back",
    "fallback",
    "failed",
    "error",
    " not applied",
    " not supported",
    "warning",
    "cannot ",
  ];
  return warningTokens.some((token) => lowered.includes(token));
}

export function SubstrateRoutingBadge({ metrics }: SubstrateRoutingBadgeProps) {
  const { t } = useTranslation("common");
  const chips = buildChips(metrics, t);
  if (chips.length === 0) return null;
  return (
    <div
      className="substrate-routing"
      aria-label={t("substrateRoutingBadge.ariaLabel", { defaultValue: "Substrate routing for this turn" })}
    >
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
