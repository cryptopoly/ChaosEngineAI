/**
 * The Storyboard tab's run bar (D-65 4.6): sticky at the bottom of the tab. What needs a take, the estimate for
 * Make missing (time, peak memory, disk; Estimated until a run here measures it), the Make buttons with the count
 * and time on the primary one, a red state with an acknowledgement when it will not fit, and the batch progress
 * with Cancel while a batch runs. Below them the two passes (S6): Make drafts (the fast still model at a smaller
 * size, clips optional) and Final pass on kept shots (the quality model at full size, then clips), each with its time.
 * One of Make missing and Make drafts is primary, as `primaryPass` in storyboardLogic decides.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardEstimate } from "../../api/storyboard";
import { RunEstimateStrip, formatDuration, memoryRisk } from "../../components/RunEstimateStrip";
import type { JobProgress, StoryboardBatchRecord } from "../../types";
import { BatchBar } from "./BatchBar";
import type { PrimaryPass } from "./storyboardLogic";

export interface RunBarProps {
  missing: number;
  stills: number;
  stale: number;
  unlocked: number;
  estimate: StoryboardEstimate | null;
  batch: StoryboardBatchRecord | null;
  batchProgress?: JobProgress | null;
  running: boolean;
  busy: boolean;
  onMakeMissing: () => void;
  onMakeAll: () => void;
  onCancel: () => void;
  /** Shots marked Keep, and how many of them the final pass would redo. */
  kept?: number;
  finalShots?: number;
  onMakeDrafts?: (withClips: boolean) => void;
  onFinalPass?: () => void;
  /** Which button is primary (`primaryPass`); left out, Make missing is. */
  primary?: PrimaryPass;
}

type TFn = (key: string, options: Record<string, unknown>) => string;

function PassesRow({ missing, kept = 0, finalShots = 0, estimate, busy, onMakeDrafts, onFinalPass, primary }: RunBarProps) {
  const { t } = useTranslation("studio");
  const [withClips, setWithClips] = useState(false);
  const draft = estimate?.passes?.draft;
  const final = estimate?.passes?.final;
  const tf = t as unknown as TFn;
  const draftSeconds = draft?.seconds != null ? draft.seconds + (withClips ? (draft.clipSeconds ?? 0) : 0) : null;
  const draftLabel =
    missing > 0 && draftSeconds != null
      ? t("board.passes.draftTimed", { defaultValue: "Make drafts · about {time}", time: formatDuration(tf, draftSeconds) })
      : t("board.passes.draft", { defaultValue: "Make drafts" });
  const finalSeconds = final && !final.assumedAllKept && finalShots > 0 ? final.seconds : null;
  const finalLabel =
    finalSeconds != null
      ? t("board.passes.finalTimed", {
          defaultValue: "Final pass on {n, plural, one {# kept shot} other {# kept shots}} · about {time}",
          n: finalShots,
          time: formatDuration(tf, finalSeconds),
        })
      : t("board.passes.final", {
          defaultValue: "Final pass on {n, plural, one {# kept shot} other {# kept shots}}",
          n: finalShots,
        });
  const draftTitle = draft?.imageModel
    ? t("board.passes.draftTitle", {
        defaultValue: "Quick stills with {model} at {width} x {height}. Mark the shots you like with Keep, then run the final pass.",
        model: draft.imageModel,
        width: draft.width,
        height: draft.height,
      })
    : undefined;
  const finalTitle = final?.imageModel
    ? t("board.passes.finalTitle", {
        defaultValue: "Makes each kept shot again with {model} at {width} x {height}, then its clip from the new still.",
        model: final.imageModel,
        width: final.width,
        height: final.height,
      })
    : undefined;
  return (
    <div className="run-bar__passes" role="group" aria-label={t("board.passes.label", { defaultValue: "Draft and final passes" })}>
      {/* Drafts only make what is missing: with every shot made they would be a dead control. */}
      {missing > 0 ? (
        <>
          <label className="run-bar__passes-clips">
            <input type="checkbox" checked={withClips} disabled={busy} onChange={(event) => setWithClips(event.target.checked)} />
            {t("board.passes.withClips", { defaultValue: "Drafts with clips" })}
          </label>
          <button type="button" className={primary === "drafts" ? "primary-button" : "secondary-button"} disabled={busy} title={draftTitle}
            onClick={() => onMakeDrafts?.(withClips)}>
            {draftLabel}
          </button>
        </>
      ) : null}
      <button type="button" className="secondary-button" disabled={busy || finalShots === 0} title={finalTitle}
        onClick={onFinalPass}>
        {finalLabel}
      </button>
      {kept === 0 ? (
        <span className="muted-text">{t("board.passes.keepHint", { defaultValue: "Mark shots with Keep for final to choose them for the final pass." })}</span>
      ) : null}
    </div>
  );
}

export function RunBar(props: RunBarProps) {
  const { missing, stills, stale, unlocked, estimate, batch, batchProgress, running, busy, onMakeMissing, onMakeAll, onCancel } = props;
  const { t } = useTranslation("studio");
  const [ack, setAck] = useState(false);
  if (running && batch) {
    return (
      <div className="run-bar run-bar--running">
        <BatchBar batch={batch} progress={batchProgress ?? null} onCancel={onCancel} />
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

  const makeAll = (
    <button
      type="button"
      className="secondary-button"
      disabled={busy || unlocked === 0}
      title={t("board.makeAllTitle", { defaultValue: "Adds a new take to every unlocked shot. Nothing is overwritten." })}
      onClick={onMakeAll}
    >
      {t("board.makeAll", { defaultValue: "Make all unlocked" })}
    </button>
  );

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
      {missing === 0 ? (
        // Nothing to make: no estimate to show and no disabled primary button, only the deliberate remake.
        <div className="run-estimate run-estimate--safe">
          <div className="run-estimate__row">
            <div className="run-estimate__items" />
            <div className="run-estimate__action">
              <div className="board-estimate__actions">{makeAll}</div>
            </div>
          </div>
        </div>
      ) : (
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
            {makeAll}
            <button
              type="button"
              className={props.primary === "drafts" ? "secondary-button" : "primary-button"}
              disabled={busy || missing === 0 || blockedByRisk}
              onClick={onMakeMissing}
            >
              {makeLabel}
            </button>
          </div>
        }
      />
      )}
      {props.onMakeDrafts ? <PassesRow {...props} /> : null}
    </div>
  );
}
