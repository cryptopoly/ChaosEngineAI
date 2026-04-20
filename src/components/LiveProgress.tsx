import { useEffect, useState } from "react";
import type { GenerationProgressSnapshot } from "../types";

export interface LiveProgressPhase {
  id: string;
  label: string;
  estimatedSeconds: number;
}

interface LiveProgressProps {
  title: string;
  subtitle?: string;
  phases: LiveProgressPhase[];
  startedAt: number;
  accent?: "convert" | "benchmark" | "image";
  /**
   * Optional real-time progress snapshot from the backend tracker. When
   * present and ``active=true`` it overrides the time-based estimates: the
   * active phase is whichever phase ID matches ``snapshot.phase``, and the
   * fill percentage is driven by ``step / totalSteps`` during diffusion.
   *
   * Pass ``null`` (or omit) to keep the legacy pure-estimate behaviour —
   * benchmark and convert modals don't have a backend signal yet.
   */
  realProgress?: GenerationProgressSnapshot | null;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function LiveProgress({
  title,
  subtitle,
  phases,
  startedAt,
  accent = "benchmark",
  realProgress,
}: LiveProgressProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const elapsedSec = Math.max(0, (now - startedAt) / 1000);
  const totalEstimated = phases.reduce((acc, p) => acc + p.estimatedSeconds, 0);

  // Determine current phase index. When the backend has reported a real
  // phase ID, prefer that — it's accurate and the user sees the bar pin to
  // exactly the step they're on. Otherwise fall back to time-based estimates.
  const realPhase = realProgress?.active ? realProgress.phase : null;
  let activeIndex: number;
  if (realPhase) {
    const matched = phases.findIndex((phase) => phase.id === realPhase);
    activeIndex = matched >= 0 ? matched : 0;
  } else {
    let cumulative = 0;
    activeIndex = phases.length - 1;
    for (let i = 0; i < phases.length; i++) {
      if (elapsedSec < cumulative + phases[i].estimatedSeconds) {
        activeIndex = i;
        break;
      }
      cumulative += phases[i].estimatedSeconds;
    }
  }

  // Real diffusion progress (step counter from diffusers' callback_on_step_end)
  // is the strongest signal we have. Use it to compute fillPct directly so the
  // bar moves in lockstep with the model. Outside of the diffusion phase, fall
  // back to the time-based estimate so we still get *some* movement.
  const realStepFraction =
    realProgress?.active && realProgress.phase === "diffusing" && realProgress.totalSteps > 0
      ? Math.min(1, realProgress.step / realProgress.totalSteps)
      : null;

  let fillPct: number;
  let overrunning: boolean;
  if (realStepFraction !== null) {
    // Distribute fill across the phases proportionally to their estimates so
    // diffusion fills its own slice of the bar rather than stretching to 100%.
    const cumulativeBefore = phases.slice(0, activeIndex).reduce((acc, p) => acc + p.estimatedSeconds, 0);
    const phaseShare = phases[activeIndex]?.estimatedSeconds ?? 0;
    const filled = cumulativeBefore + phaseShare * realStepFraction;
    fillPct = Math.min(100, (filled / Math.max(1, totalEstimated)) * 100);
    overrunning = false;
  } else {
    overrunning = elapsedSec >= totalEstimated;
    fillPct = Math.min(100, (elapsedSec / Math.max(1, totalEstimated)) * 100);
  }

  // Cumulative time consumed by phases that come *before* the active one,
  // used by the legacy "X.Xs / ~Ys" label on the active phase. With a real
  // signal driving activeIndex this is meaningless, so we hide that label.
  const cumulativeBeforeActive = phases.slice(0, activeIndex).reduce((acc, p) => acc + p.estimatedSeconds, 0);
  const phaseElapsed = Math.max(0, elapsedSec - cumulativeBeforeActive);

  return (
    <div className={`live-progress live-progress--${accent}`}>
      <div className="live-progress__header">
        <div>
          <div className="live-progress__title">{title}</div>
          {subtitle ? <div className="live-progress__subtitle">{subtitle}</div> : null}
        </div>
        <div className="live-progress__elapsed">{formatElapsed(elapsedSec)}</div>
      </div>

      {accent === "convert" ? (
        <div className="live-progress__qgrid">
          {Array.from({ length: 48 }).map((_, i) => (
            <div
              key={i}
              className="live-progress__qgrid-cell"
              style={{ animationDelay: `${(i % 12) * 0.08 + Math.floor(i / 12) * 0.15}s` }}
            />
          ))}
        </div>
      ) : accent === "image" ? (
        <div className="live-progress__mosaic">
          {Array.from({ length: 40 }).map((_, i) => (
            <div
              key={i}
              className="live-progress__mosaic-cell"
              style={{ animationDelay: `${(i % 10) * 0.06 + Math.floor(i / 10) * 0.12}s` }}
            />
          ))}
        </div>
      ) : (
        <div className="live-progress__waveform">
          {[0, 1, 2].map((row) => (
            <div className="live-progress__wave-row" key={row}>
              {Array.from({ length: 28 }).map((_, i) => (
                <div
                  key={i}
                  className="live-progress__wave-dot"
                  style={{ animationDelay: `${(i * 0.06 + row * 0.18).toFixed(2)}s` }}
                />
              ))}
            </div>
          ))}
        </div>
      )}

      <div className="live-progress__bar">
        <div
          className={`live-progress__bar-fill${overrunning ? " live-progress__bar-fill--indeterminate" : ""}`}
          style={overrunning ? undefined : { width: `${fillPct}%` }}
        />
      </div>

      <div className="live-progress__phases">
        {phases.map((phase, i) => {
          const state = i < activeIndex ? "done" : i === activeIndex ? "active" : "pending";
          const showLiveStepLabel =
            state === "active"
            && realProgress?.active
            && realProgress.phase === phase.id
            && realProgress.totalSteps > 0;
          return (
            <div key={phase.id} className={`live-progress__phase live-progress__phase--${state}`}>
              <span className="live-progress__phase-dot" />
              <span className="live-progress__phase-label">{phase.label}</span>
              {state === "active" ? (
                showLiveStepLabel ? (
                  <span className="live-progress__phase-elapsed">
                    step {realProgress!.step} / {realProgress!.totalSteps}
                  </span>
                ) : realPhase ? (
                  <span className="live-progress__phase-elapsed">
                    {realProgress?.message || "in progress…"}
                  </span>
                ) : (
                  <span className="live-progress__phase-elapsed">
                    {overrunning ? "finalizing…" : `${phaseElapsed.toFixed(1)}s / ~${phase.estimatedSeconds}s`}
                  </span>
                )
              ) : null}
              {state === "done" ? <span className="live-progress__phase-check">✓</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
