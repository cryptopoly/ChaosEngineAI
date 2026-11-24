import { useTranslation } from "react-i18next";
import type { MemoryBudget } from "../types/observability";

interface MemoryBarProps {
  /** What the bar is a share of: system RAM on a Mac, the card's VRAM on an NVIDIA host. */
  totalGb: number;
  /** Memory in use now, outside this launch. It includes a model that is loaded now, which loading replaces. */
  usedNowGb: number;
  weightsGb: number;
  cacheGb: number;
  budget?: MemoryBudget | null;
}

function fmt(value: number): string {
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function limitLabel(kind: MemoryBudget["kind"] | undefined, t: (key: string, options: Record<string, unknown>) => string): string {
  if (kind === "metal") return t("memoryBar.limitMetal", { defaultValue: "GPU limit" });
  if (kind === "vram") return t("memoryBar.limitVram", { defaultValue: "Card memory" });
  return t("memoryBar.limitRam", { defaultValue: "Usable RAM" });
}

/**
 * One bar for the whole question "will this launch fit": what is in use now, then this model's weights, then its
 * cache at the chosen window, with a tick at the ceiling the engine must stay under. Three numbers sit under it, so
 * the bar never has to be decoded.
 */
export function MemoryBar({ totalGb, usedNowGb, weightsGb, cacheGb, budget }: MemoryBarProps) {
  const { t } = useTranslation("common");
  if (!(totalGb > 0)) return null;
  const used = Math.max(0, Math.min(usedNowGb, totalGb));
  const needGb = weightsGb + cacheGb;
  const freeNowGb = Math.max(0, totalGb - used);
  const pct = (value: number) => Math.min(100, Math.max(0, (value / totalGb) * 100));
  const limit = budget && budget.limitGb > 0 ? Math.min(budget.limitGb, totalGb) : null;
  const overLimit = limit != null && needGb > limit;
  const overFree = needGb > freeNowGb;
  const label = limitLabel(budget?.kind, t as never);

  return (
    <div className="memory-bar">
      <div
        className="memory-bar__track"
        role="img"
        aria-label={t("memoryBar.aria", {
          defaultValue: "Memory: {used} GB in use now, {weights} GB model weights, {cache} GB cache, out of {total} GB",
          used: fmt(used), weights: fmt(weightsGb), cache: fmt(cacheGb), total: fmt(totalGb),
        })}
      >
        <span className="memory-bar__seg memory-bar__seg--used" style={{ width: `${pct(used)}%` }} />
        <span className="memory-bar__seg memory-bar__seg--weights" style={{ width: `${pct(Math.min(weightsGb, totalGb - used))}%` }} />
        <span className="memory-bar__seg memory-bar__seg--cache" style={{ width: `${pct(Math.min(cacheGb, Math.max(0, totalGb - used - weightsGb)))}%` }} />
        {limit != null ? <span className="memory-bar__limit" style={{ left: `${pct(limit)}%` }} title={`${label} ${fmt(limit)} GB`} /> : null}
      </div>
      <div className="memory-bar__legend">
        <span><i className="memory-bar__dot memory-bar__seg--used" />{t("memoryBar.usedNow", { defaultValue: "In use now {value} GB", value: fmt(used) })}</span>
        <span><i className="memory-bar__dot memory-bar__seg--weights" />{t("memoryBar.weights", { defaultValue: "Model {value} GB", value: fmt(weightsGb) })}</span>
        <span><i className="memory-bar__dot memory-bar__seg--cache" />{t("memoryBar.cache", { defaultValue: "Cache {value} GB", value: fmt(cacheGb) })}</span>
      </div>
      <p className="memory-bar__summary">
        {t("memoryBar.summary", {
          defaultValue: "Needs {need} GB. {free} GB free now. {label} {limit} GB.",
          need: fmt(needGb), free: fmt(freeNowGb), label, limit: limit != null ? fmt(limit) : fmt(totalGb),
        })}
      </p>
      {overLimit ? (
        <p className="perf-preview__advice" role="note">
          {t("memoryBar.overLimit", {
            defaultValue: "This launch needs more than the {label} allows, so it will not start as set.",
            label: label.toLowerCase(),
          })}
        </p>
      ) : overFree ? (
        <p className="perf-preview__advice" role="note">
          {t("memoryBar.overFree", {
            defaultValue: "More than the free memory right now. It can start, but other apps will be moved out of RAM and it may swap. Closing apps first helps.",
          })}
        </p>
      ) : null}
    </div>
  );
}
