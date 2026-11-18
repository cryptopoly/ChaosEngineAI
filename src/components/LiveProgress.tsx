import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
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
  accent?: "convert" | "benchmark" | "image" | "audio" | "storyboard";
  /**
   * Optional real-time progress snapshot from the backend tracker. When
   * present and ``active=true`` it overrides the time-based estimates: the
   * active phase is whichever phase ID matches ``snapshot.phase``, and the
   * fill percentage is driven by ``step / totalSteps`` during diffusion.
   *
   * Pass ``null`` (or omit) to keep the pure-estimate behaviour.
   */
  realProgress?: GenerationProgressSnapshot | null;
  /**
   * Benchmarks and conversion: show only what the job reports. The phase comes from
   * ``realProgress``; the bar has a value only while the phase counts items
   * (``totalSteps`` > 0) and is indeterminate otherwise. No time estimate is used,
   * so ``estimatedSeconds`` of the phases is ignored.
   */
  measuredOnly?: boolean;
  /** With ``measuredOnly``: the measured text next to the active phase ("12 of 100 questions"). */
  activeDetail?: string | null;
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

/**
 * Bar state for a job that reports its own progress (``measuredOnly``): a percentage only
 * while the current phase counts items, indeterminate for every phase that cannot.
 */
export function measuredBar(progress: GenerationProgressSnapshot | null | undefined): { fillPct: number; indeterminate: boolean } {
  if (!progress?.active || !(progress.totalSteps > 0)) return { fillPct: 0, indeterminate: true };
  return { fillPct: Math.max(0, Math.min(100, (progress.step / progress.totalSteps) * 100)), indeterminate: false };
}

/** Whole-number percent for the progress bar, or undefined while the length is unknown (the bar is indeterminate). */
export function progressValueNow(fillPct: number, overrunning: boolean): number | undefined {
  return overrunning ? undefined : Math.max(0, Math.min(100, Math.round(fillPct)));
}

export function LiveProgress({
  title,
  subtitle,
  phases,
  startedAt,
  accent = "benchmark",
  realProgress,
  measuredOnly = false,
  activeDetail = null,
}: LiveProgressProps) {
  const { t } = useTranslation("common");
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
  if (measuredOnly) {
    // Before the first poll answers, the job is on its first phase.
    activeIndex = Math.max(0, realPhase ? phases.findIndex((phase) => phase.id === realPhase) : 0);
  } else if (realPhase) {
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
    realProgress?.active && (realProgress.phase === "diffusing" || realProgress.phase === "compose") && realProgress.totalSteps > 0
      ? Math.min(1, realProgress.step / realProgress.totalSteps)
      : null;

  let fillPct: number;
  let overrunning: boolean;
  if (measuredOnly) {
    const bar = measuredBar(realProgress);
    fillPct = bar.fillPct;
    overrunning = bar.indeterminate;
  } else if (realStepFraction !== null) {
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

  const waitingText = realProgress?.queue?.waiting
    ? realProgress.queue.agentAhead
      ? t("liveProgress.waitingForAgent", { defaultValue: "Waiting for an agent job to finish" })
      : t("liveProgress.waitingInQueue", {
          defaultValue: "Waiting for another job to finish (position {position})",
          position: realProgress.queue.position,
        })
    : null;
  // Read out the phase, not the 250 ms clock: this text changes only when the phase or the queue changes.
  const announcement = waitingText ?? `${title}: ${phases[activeIndex]?.label ?? ""}`;

  return (
    <div className={`live-progress live-progress--${accent}`}>
      <p className="sr-only" role="status">{announcement}</p>
      <div className="live-progress__header">
        <div>
          <div className="live-progress__title">{title}</div>
          {subtitle ? <div className="live-progress__subtitle">{subtitle}</div> : null}
        </div>
        <div className="live-progress__elapsed" aria-hidden="true">{formatElapsed(elapsedSec)}</div>
      </div>

      {accent === "convert" ? (
        <div className="live-progress__qgrid" aria-hidden="true">
          {Array.from({ length: 48 }).map((_, i) => (
            <div
              key={i}
              className="live-progress__qgrid-cell"
              style={{ animationDelay: `${(i % 12) * 0.08 + Math.floor(i / 12) * 0.15}s` }}
            />
          ))}
        </div>
      ) : accent === "image" ? (
        <div className="live-progress__mosaic" aria-hidden="true">
          {Array.from({ length: 40 }).map((_, i) => (
            <div
              key={i}
              className="live-progress__mosaic-cell"
              style={{ animationDelay: `${(i % 10) * 0.06 + Math.floor(i / 10) * 0.12}s` }}
            />
          ))}
        </div>
      ) : (
        <div className="live-progress__waveform" aria-hidden="true">
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

      {waitingText ? <div className="live-progress__subtitle">{waitingText}</div> : null}

      <div
        className="live-progress__bar"
        role="progressbar"
        aria-label={title}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progressValueNow(fillPct, overrunning)}
      >
        <div
          className={`live-progress__bar-fill${overrunning ? " live-progress__bar-fill--indeterminate" : ""}`}
          style={overrunning ? undefined : { width: `${fillPct}%` }}
        />
      </div>

      {realProgress?.active && realProgress.thumbnail ? (
        <div className="live-progress__thumbnail">
          <img
            src={`data:image/png;base64,${realProgress.thumbnail}`}
            alt={t("liveProgress.thumbnailAlt", { defaultValue: "Live denoise preview" })}
          />
          <span className="live-progress__thumbnail-caption">
            {t("liveProgress.thumbnailCaption", { defaultValue: "Live preview · TAESD decode" })}
          </span>
        </div>
      ) : null}

      <div className="live-progress__phases">
        {phases.map((phase, i) => {
          const state = i < activeIndex ? "done" : i === activeIndex ? "active" : "pending";
          const showLiveStepLabel =
            state === "active"
            && realProgress?.active
            && realProgress.phase === phase.id
            && realProgress.totalSteps > 0;
          return (
            <div key={phase.id} className={`live-progress__phase live-progress__phase--${state}`} aria-current={state === "active" ? "step" : undefined}>
              <span className="live-progress__phase-dot" />
              <span className="live-progress__phase-label">{phase.label}</span>
              {state === "active" ? (
                measuredOnly ? (
                  <span className="live-progress__phase-elapsed">
                    {activeDetail || t("liveProgress.inProgress", { defaultValue: "in progress…" })}
                  </span>
                ) : showLiveStepLabel ? (
                  <span className="live-progress__phase-elapsed">
                    {t("liveProgress.stepCounter", {
                      defaultValue: "step {current} / {total}",
                      current: realProgress!.step,
                      total: realProgress!.totalSteps,
                    })}
                  </span>
                ) : realPhase ? (
                  <span className="live-progress__phase-elapsed">
                    {realProgress?.message || t("liveProgress.inProgress", { defaultValue: "in progress…" })}
                  </span>
                ) : (
                  <span className="live-progress__phase-elapsed">
                    {overrunning
                      ? t("liveProgress.finalizing", { defaultValue: "finalizing…" })
                      : t("liveProgress.phaseTiming", {
                          defaultValue: "{elapsed}s / ~{estimated}s",
                          elapsed: phaseElapsed.toFixed(1),
                          estimated: phase.estimatedSeconds,
                        })}
                  </span>
                )
              ) : null}
              {state === "done" ? <span className="live-progress__phase-check" aria-hidden="true">✓</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
