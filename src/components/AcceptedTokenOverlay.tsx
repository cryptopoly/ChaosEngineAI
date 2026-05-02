import { useState } from "react";
import type { GenerationMetrics } from "../types";

/**
 * Phase 3.1: DDTree accepted-span overlay.
 *
 * Renders a collapsible block that shows the assistant's response
 * with draft-accepted character ranges tinted (green) vs
 * verifier-decoded ranges (default). Substrate truth view —
 * doesn't replace the markdown body, sits alongside it so users
 * can see how aggressively DDTree's draft acceptance kicked in.
 *
 * Visible only when the message metrics carry accepted-span data,
 * which requires speculative decoding to have run on the turn.
 *
 * The text in `acceptedTokenText` is the per-token-decoded string
 * which can differ slightly from the markdown body (no formatting,
 * sometimes BPE artifacts) — that's OK; the overlay is for
 * substrate diagnostics, not display.
 */
export interface AcceptedTokenOverlayProps {
  metrics: GenerationMetrics;
}

interface SpanStats {
  totalChars: number;
  acceptedChars: number;
  acceptedRatio: number;
  spanCount: number;
}

export function computeSpanStats(
  spans: AcceptedTokenOverlayProps["metrics"]["acceptedSpans"],
): SpanStats {
  if (!spans || spans.length === 0) {
    return { totalChars: 0, acceptedChars: 0, acceptedRatio: 0, spanCount: 0 };
  }
  let total = 0;
  let accepted = 0;
  for (const span of spans) {
    total += span.length;
    if (span.accepted) accepted += span.length;
  }
  return {
    totalChars: total,
    acceptedChars: accepted,
    acceptedRatio: total > 0 ? accepted / total : 0,
    spanCount: spans.length,
  };
}

export function AcceptedTokenOverlay({ metrics }: AcceptedTokenOverlayProps) {
  const [open, setOpen] = useState(false);
  const spans = metrics.acceptedSpans;
  const text = metrics.acceptedTokenText;
  if (!spans?.length || !text) return null;
  const stats = computeSpanStats(spans);

  return (
    <details
      className="accepted-overlay"
      open={open}
      onToggle={(event) => setOpen((event.currentTarget as HTMLDetailsElement).open)}
    >
      <summary className="accepted-overlay__head">
        <span>DDTree acceptance overlay</span>
        <small>
          {(stats.acceptedRatio * 100).toFixed(1)}% of {stats.totalChars} chars
          accepted from draft · {stats.spanCount} runs
        </small>
      </summary>
      <p className="accepted-overlay__hint">
        Green ranges = tokens the verifier accepted from the draft model
        without re-decoding. Plain ranges = tokens the verifier produced
        directly. Higher acceptance means DDTree saved more compute.
      </p>
      <pre className="accepted-overlay__text">
        {spans.map((span, idx) => (
          <span
            key={`${span.start}-${idx}`}
            className={`accepted-overlay__span${span.accepted ? " accepted-overlay__span--accepted" : ""}`}
            title={span.accepted ? "Accepted from draft" : "Verifier-decoded"}
          >
            {text.slice(span.start, span.start + span.length)}
          </span>
        ))}
      </pre>
    </details>
  );
}
