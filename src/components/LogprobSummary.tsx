import { useState } from "react";
import type { TokenLogprob } from "../types";

/**
 * Phase 3.3: per-message logprob summary.
 *
 * Renders a collapsible block beneath the assistant bubble that
 * shows confidence stats + a hover-revealed list of any low-confidence
 * tokens with their top alternatives. We deliberately don't replace
 * the markdown body with hoverable token spans — that breaks
 * formatting + accessibility — instead we surface a compact summary
 * the user can drill into when something looks off.
 *
 * Visible only when message.tokenLogprobs is populated, which
 * requires `advancedLogprobs` to be enabled in settings.
 */
export interface LogprobSummaryProps {
  entries: TokenLogprob[];
}

interface SummaryStats {
  count: number;
  avgLogprob: number;
  lowConfidenceCount: number;
}

function computeStats(entries: TokenLogprob[]): SummaryStats {
  const valid = entries.filter((e) => typeof e.logprob === "number" && Number.isFinite(e.logprob));
  if (valid.length === 0) {
    return { count: entries.length, avgLogprob: 0, lowConfidenceCount: 0 };
  }
  const sum = valid.reduce((acc, e) => acc + (e.logprob as number), 0);
  // logprob < -3.0 ≈ probability < 5%. Flag those as low-confidence
  // so the user can see where the model was uncertain.
  const lowConfidenceCount = valid.filter((e) => (e.logprob as number) < -3.0).length;
  return {
    count: entries.length,
    avgLogprob: sum / valid.length,
    lowConfidenceCount,
  };
}

function lowConfidenceEntries(entries: TokenLogprob[]): TokenLogprob[] {
  return entries
    .filter((e) => typeof e.logprob === "number" && (e.logprob as number) < -3.0)
    .slice(0, 12);
}

export function LogprobSummary({ entries }: LogprobSummaryProps) {
  const [open, setOpen] = useState(false);
  if (!entries?.length) return null;
  const stats = computeStats(entries);
  const flagged = lowConfidenceEntries(entries);

  return (
    <details
      className="logprob-summary"
      open={open}
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="logprob-summary__head">
        <span>Token confidence</span>
        <small>
          {stats.count} tokens · avg logprob {stats.avgLogprob.toFixed(2)}
          {stats.lowConfidenceCount > 0 ? ` · ${stats.lowConfidenceCount} low confidence` : ""}
        </small>
      </summary>
      {flagged.length === 0 ? (
        <p className="logprob-summary__empty">No low-confidence tokens — model was steady throughout.</p>
      ) : (
        <div className="logprob-summary__list">
          <p className="logprob-summary__hint">
            Tokens emitted with probability under ~5%. Hover for the top
            alternatives the model considered.
          </p>
          <ul>
            {flagged.map((entry, idx) => (
              <li
                key={`${entry.token}-${idx}`}
                title={
                  entry.alternatives.length
                    ? entry.alternatives
                        .map((alt) => `${JSON.stringify(alt.token ?? "")} (${(alt.logprob ?? 0).toFixed(2)})`)
                        .join("\n")
                    : "No alternatives recorded."
                }
              >
                <code>{JSON.stringify(entry.token ?? "")}</code>
                <span className="logprob-summary__metric">
                  logprob {(entry.logprob ?? 0).toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </details>
  );
}

export { computeStats, lowConfidenceEntries };
