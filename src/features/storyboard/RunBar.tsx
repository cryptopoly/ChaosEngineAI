/**
 * The Storyboard tab's run bar (D-65 4.6): sticky at the bottom of the tab. What needs a take, the estimate for
 * Make missing (time, peak memory, disk; Estimated until a run here measures it), the Make buttons with the count
 * and time on the primary one, a red state with an acknowledgement when it will not fit, and the batch progress
 * with Cancel while a batch runs.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardEstimate } from "../../api/storyboard";
import { RunEstimateStrip, formatDuration, memoryRisk } from "../../components/RunEstimateStrip";
import type { StoryboardBatchRecord } from "../../types";
import { BatchBar } from "./BatchBar";

export interface RunBarProps {
  missing: number;
  stills: number;
  stale: number;
  unlocked: number;
  estimate: StoryboardEstimate | null;
  batch: StoryboardBatchRecord | null;
  running: boolean;
  busy: boolean;
  onMakeMissing: () => void;
  onMakeAll: () => void;
  onCancel: () => void;
}

type TFn = (key: string, options: Record<string, unknown>) => string;

export function RunBar({ missing, stills, stale, unlocked, estimate, batch, running, busy, onMakeMissing, onMakeAll, onCancel }: RunBarProps) {
  const { t } = useTranslation("studio");
  const [ack, setAck] = useState(false);
  if (running && batch) {
    return (
      <div className="run-bar run-bar--running">
        <BatchBar batch={batch} onCancel={onCancel} />
      </div>
    );
  }
  const risk = memoryRisk(estimate?.peakGb ?? null, estimate?.totalGb ?? null, "safe");
  const blockedByRisk = risk === "danger" && !ack;
  const seconds = estimate?.seconds ?? null;
  const makeLabel =
    missing === 0
      ? t("runBar.makeNone", { defaultValue: "Make missing" })
      : seconds != null
        ? t("runBar.makeTimed", {
            defaultValue: "Make {n} missing · about {time}",
            n: missing,
            time: formatDuration(t as unknown as TFn, seconds),
          })
        : t("runBar.make", { defaultValue: "Make {n} missing", n: missing });
  const need =
    missing === 0
      ? t("board.estimate.none", { defaultValue: "Every unlocked shot has a take." })
      : stills > 0
        ? t("runBar.needStills", {
            defaultValue: "{n, plural, one {# shot needs work} other {# shots need work}}: {stills, plural, one {# has a still and needs its clip} other {# have a still and need their clip}}",
            n: missing,
            stills,
          })
        : t("board.estimate.need", {
            defaultValue: "{n, plural, one {# shot needs a new take} other {# shots need a new take}}",
            n: missing,
          });

  return (
    <div className={`run-bar${risk === "danger" ? " run-bar--danger" : ""}`} role="region" aria-label={t("runBar.label", { defaultValue: "Make shots" })}>
      <p className="run-bar__need">
        {need}
        {stale > 0 ? ` · ${t("runBar.stale", { defaultValue: "{n} stale", n: stale })}` : ""}
      </p>
      {estimate?.missing.length ? (
        <ul className="run-bar__missing">
          {estimate.missing.map((line) => (
            <li key={line} className="muted-text">{line}</li>
          ))}
        </ul>
      ) : null}
      <RunEstimateStrip
        compactUnknown
        seconds={missing > 0 ? seconds : null}
        peakGb={missing > 0 ? (estimate?.peakGb ?? null) : null}
        totalGb={missing > 0 ? (estimate?.totalGb ?? null) : null}
        diskMb={missing > 0 ? (estimate?.diskMb ?? null) : null}
        basis={missing > 0 && (seconds != null || estimate?.peakGb != null) ? (estimate?.basis ?? null) : null}
        risk={risk}
        ack={{ checked: ack, onChange: setAck }}
        action={
          <div className="board-estimate__actions">
            <button
              type="button"
              className="secondary-button"
              disabled={busy || unlocked === 0}
              title={t("board.makeAllTitle", { defaultValue: "Adds a new take to every unlocked shot. Nothing is overwritten." })}
              onClick={onMakeAll}
            >
              {t("board.makeAll", { defaultValue: "Make all unlocked" })}
            </button>
            <button type="button" className="primary-button" disabled={busy || missing === 0 || blockedByRisk} onClick={onMakeMissing}>
              {makeLabel}
            </button>
          </div>
        }
      />
    </div>
  );
}
