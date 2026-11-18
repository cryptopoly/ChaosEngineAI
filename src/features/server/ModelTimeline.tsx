import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { buildTimeline, formatSeconds, type ModelEvent, type TimelinePhase } from "./modelEvents";

const WINDOWS: Array<{ id: string; seconds: number | null }> = [
  { id: "15m", seconds: 15 * 60 },
  { id: "1h", seconds: 60 * 60 },
  { id: "24h", seconds: 24 * 60 * 60 },
  { id: "all", seconds: null },
];

const PHASES: TimelinePhase[] = ["loading", "ready", "generating", "idle", "warm", "failed"];

export interface ModelTimelineProps {
  events: ModelEvent[];
  /** Seconds since the epoch; closes phases still open. */
  now: number;
  onSelectModel?: (model: string) => void;
}

/** One bar per model: loading, ready, generating, idle and warm phases, newest model first. */
export function ModelTimeline({ events, now, onSelectModel }: ModelTimelineProps) {
  const { t } = useTranslation("common");
  const [windowId, setWindowId] = useState("1h");
  const span = WINDOWS.find((w) => w.id === windowId)?.seconds ?? null;
  const rows = useMemo(() => buildTimeline(events, now), [events, now]);
  const visible = useMemo(() => {
    const from = span == null ? -Infinity : now - span;
    return rows.filter((row) => row.end >= from);
  }, [rows, span, now]);
  const axisStart = visible.length
    ? Math.max(span == null ? -Infinity : now - span, Math.min(...visible.map((row) => row.start)))
    : now;
  const axisEnd = Math.max(axisStart + 1, now, ...visible.map((row) => row.end));
  const total = axisEnd - axisStart;
  const phaseLabel = (phase: TimelinePhase) =>
    t(`serverActivity.phase.${phase}`, { defaultValue: phase });

  return (
    <section className="activity-timeline" aria-label={t("serverActivity.timeline.title", { defaultValue: "Model timeline" })}>
      <div className="activity-section-head">
        <h3>{t("serverActivity.timeline.title", { defaultValue: "Model timeline" })}</h3>
        <div className="segmented activity-window" role="group" aria-label={t("serverActivity.timeline.window", { defaultValue: "Time window" })}>
          {WINDOWS.map((w) => (
            <button
              key={w.id}
              type="button"
              className={`segment${w.id === windowId ? " active" : ""}`}
              aria-pressed={w.id === windowId}
              onClick={() => setWindowId(w.id)}
            >
              {t(`serverActivity.timeline.windows.${w.id}`, { defaultValue: w.id })}
            </button>
          ))}
        </div>
      </div>
      {visible.length === 0 ? (
        <p className="muted-text activity-empty">
          {t("serverActivity.timeline.empty", { defaultValue: "No model activity in this window. Load a model to see its timeline." })}
        </p>
      ) : (
        <div className="activity-timeline-rows">
          {visible.map((row) => {
            const segments = row.segments.filter((segment) => segment.end >= axisStart);
            const summary = segments
              .map((segment) => `${phaseLabel(segment.phase)} ${formatSeconds(segment.end - segment.start)}`)
              .join(", ");
            return (
              <div className="activity-timeline-row" key={row.model}>
                <button
                  type="button"
                  className="activity-timeline-label"
                  title={row.model}
                  onClick={() => onSelectModel?.(row.model)}
                >
                  <span className="activity-timeline-name">{row.model}</span>
                  <small className="muted-text">
                    {row.engine ?? ""}
                    {row.live ? ` · ${t("serverActivity.timeline.live", { defaultValue: "now" })}` : ""}
                  </small>
                </button>
                <div className="activity-timeline-track" role="img" aria-label={`${row.model}: ${summary}`}>
                  {segments.map((segment, index) => {
                    const start = Math.max(segment.start, axisStart);
                    const left = ((start - axisStart) / total) * 100;
                    const width = ((segment.end - start) / total) * 100;
                    return (
                      <span
                        key={`${segment.phase}-${segment.start}-${index}`}
                        className={`activity-segment phase-${segment.phase}`}
                        style={{ left: `${left}%`, width: `${width}%` }}
                        title={`${phaseLabel(segment.phase)}: ${formatSeconds(segment.end - segment.start)}`}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
      <ul className="activity-legend" aria-label={t("serverActivity.timeline.legend", { defaultValue: "Legend" })}>
        {PHASES.map((phase) => (
          <li key={phase}>
            <span className={`activity-swatch phase-${phase}`} aria-hidden="true" />
            {phaseLabel(phase)}
          </li>
        ))}
      </ul>
    </section>
  );
}
