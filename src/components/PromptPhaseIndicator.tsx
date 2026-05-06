import { useEffect, useState } from "react";
import type { ChatStreamPhase } from "../types";

interface PromptPhaseIndicatorProps {
  phase: ChatStreamPhase;
}

const PROMPT_EVAL_LABEL = "Processing prompt";
const GENERATING_LABEL = "Generating";

/**
 * Live phase indicator shown below an assistant placeholder while a
 * generation is in flight. Replaces the bare blinking cursor with an
 * explicit "Processing prompt..." or "Generating..." label plus an elapsed
 * counter, so the user knows the model is working through the prompt
 * before the first token arrives.
 *
 * Updates internally on a 250ms tick — the parent doesn't need to drive
 * re-renders for the timer.
 */
export function PromptPhaseIndicator({ phase }: PromptPhaseIndicatorProps) {
  const [elapsedMs, setElapsedMs] = useState(0);

  // Reset the counter whenever the phase flips so "Generating" starts at 0s
  // again rather than continuing from prompt-eval seconds.
  useEffect(() => {
    const startedAt = Date.now();
    setElapsedMs(0);
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAt);
    }, 250);
    return () => window.clearInterval(timer);
  }, [phase]);

  const seconds = Math.floor(elapsedMs / 1000);
  const tenths = Math.floor((elapsedMs % 1000) / 100);
  const formatted = `${seconds}.${tenths}s`;

  const label = phase === "prompt_eval" ? PROMPT_EVAL_LABEL : GENERATING_LABEL;
  const className = `prompt-phase-indicator prompt-phase-indicator--${phase}`;

  return (
    <div className={className} role="status" aria-live="polite">
      <span className="prompt-phase-indicator__spinner" aria-hidden="true" />
      <span className="prompt-phase-indicator__label">{label}...</span>
      <span className="prompt-phase-indicator__elapsed">{formatted}</span>
    </div>
  );
}
