import { useEffect, useState } from "react";

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
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

export function LiveProgress({ title, subtitle, phases, startedAt, accent = "benchmark" }: LiveProgressProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 250);
    return () => window.clearInterval(id);
  }, []);

  const elapsedSec = Math.max(0, (now - startedAt) / 1000);
  const totalEstimated = phases.reduce((acc, p) => acc + p.estimatedSeconds, 0);

  // Determine current phase index from cumulative seconds
  let cumulative = 0;
  let activeIndex = phases.length - 1;
  for (let i = 0; i < phases.length; i++) {
    if (elapsedSec < cumulative + phases[i].estimatedSeconds) {
      activeIndex = i;
      break;
    }
    cumulative += phases[i].estimatedSeconds;
  }
  const phaseElapsed = Math.max(0, elapsedSec - cumulative);
  const overrunning = elapsedSec >= totalEstimated;
  const fillPct = Math.min(100, (elapsedSec / totalEstimated) * 100);

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
          return (
            <div key={phase.id} className={`live-progress__phase live-progress__phase--${state}`}>
              <span className="live-progress__phase-dot" />
              <span className="live-progress__phase-label">{phase.label}</span>
              {state === "active" ? (
                <span className="live-progress__phase-elapsed">
                  {overrunning ? "finalizing…" : `${phaseElapsed.toFixed(1)}s / ~${phase.estimatedSeconds}s`}
                </span>
              ) : null}
              {state === "done" ? <span className="live-progress__phase-check">✓</span> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
