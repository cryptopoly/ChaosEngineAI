/**
 * The Storyboard tab's run bar (D-65 4.6): sticky at the bottom of the tab. What needs a take, the draft-or-final
 * choice in sizes and time (audit change 2), the estimate for the one primary action (time, peak memory, disk;
 * Estimated until a run here measures it), a red state with an acknowledgement when it will not fit, and the batch
 * progress with Cancel while a batch runs. There is one primary action at a time (audit change 11): Make missing,
 * Make drafts (`primaryPass` in storyboardLogic decides) or, when nothing is missing, the final pass on the chosen
 * shots. Every other way to make shots sits under "More ways to make".
 */
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardEstimate, StoryboardPassEstimate } from "../../api/storyboard";
import { RunEstimateStrip, formatDuration, memoryRisk } from "../../components/RunEstimateStrip";
import type { JobProgress, StoryboardBatchRecord, StoryboardPausedProgress, StoryboardQuality } from "../../types";
import { BatchBar } from "./BatchBar";
import { PausedBatchBar } from "./PausedBatchBar";
import { draftChoiceText, finalChoiceText, sizeText, startChoice } from "./qualityText";
import { modelsMissingReason, type PrimaryPass } from "./storyboardLogic";
import "./quality.css";

export interface RunBarProps {
  missing: number;
  stills: number;
  stale: number;
  unlocked: number;
  estimate: StoryboardEstimate | null;
  batch: StoryboardBatchRecord | null;
  batchProgress?: JobProgress | StoryboardPausedProgress | null;
  running: boolean;
  busy: boolean;
  onMakeMissing: () => void;
  onMakeAll: () => void;
  onCancel: () => void;
  /** Shots chosen for final, and how many of them the final pass would redo. */
  kept?: number;
  finalShots?: number;
  onMakeDrafts?: (withClips: boolean) => void;
  onFinalPass?: () => void;
  /** Which button is primary (`primaryPass`); left out, Make missing is. */
  primary?: PrimaryPass;
  /** Pause and resume (S-M2): a paused batch shows its own bar with Resume and Cancel. */
  onPause?: () => void;
  onResume?: () => void;
  pausing?: boolean;
  /** `guide.quality`: the film's size and the draft size, so the bar says what each choice makes. */
  quality?: StoryboardQuality | null;
}

export type TFn = (key: string, options: Record<string, unknown>) => string;

/**
 * The Make drafts tooltip: the draft model, its size and the time per draft. The run bar and the Overview's next-step
 * card both use it, so they say the same numbers (S-O). Undefined until the estimate names the draft model.
 */
export function draftPassTitle(t: TFn, draft: StoryboardPassEstimate | null | undefined): string | undefined {
  if (!draft?.imageModel) return undefined;
  const base = t("board.passes.draftTitleChoose", {
    defaultValue: "Quick stills with {model} at {width} x {height}. Choose the shots you like for final, then run the final pass.",
    model: draft.imageModel,
    width: draft.width,
    height: draft.height,
  });
  if (draft.stillSeconds == null || !draft.stills) return base;
  const per = t("board.passes.draftPer", {
    defaultValue: "About {time} per draft.",
    time: formatDuration(t, draft.stillSeconds / draft.stills),
  });
  return `${base} ${per}`;
}

export function RunBar(props: RunBarProps) {
  const { missing, stills, stale, unlocked, estimate, batch, batchProgress, running, busy, onMakeMissing, onMakeAll, onCancel } = props;
  const { kept = 0, finalShots = 0, quality } = props;
  const { t } = useTranslation("studio");
  const tf = t as unknown as TFn;
  const [ack, setAck] = useState(false);
  const [withClips, setWithClips] = useState(false);
  if (running && batch) {
    return (
      <div className="run-bar run-bar--running">
        <BatchBar batch={batch} progress={(batchProgress as JobProgress | null) ?? null} onCancel={onCancel} onPause={props.onPause} pausing={props.pausing} />
      </div>
    );
  }
  if (batch?.state === "paused" && props.onResume) {
    return (
      <div className="run-bar run-bar--running">
        <PausedBatchBar batch={batch} progress={batchProgress ?? null} busy={busy} onResume={props.onResume} onCancel={onCancel} />
      </div>
    );
  }
  const risk = memoryRisk(estimate?.peakGb ?? null, estimate?.totalGb ?? null, "safe");
  const blockedByRisk = risk === "danger" && !ack;
  // FU-474: the sentences listed above say which model is not installed; Make stays off until it is.
  const noModelReason = modelsMissingReason(estimate);
  const seconds = estimate?.seconds ?? null;
  const draft = estimate?.passes?.draft;
  const final = estimate?.passes?.final;

  const canDrafts = missing > 0 && Boolean(props.onMakeDrafts);
  const draftsPrimary = canDrafts && props.primary === "drafts";
  const makePrimary = missing > 0 && !draftsPrimary;
  const finalPrimary = missing === 0 && finalShots > 0 && Boolean(props.onFinalPass);

  const makeLabel =
    missing === 0
      ? t("runBar.makeNone", { defaultValue: "Make missing" })
      : seconds != null
        ? t("runBar.makeTimed", {
            defaultValue: "Make {n} missing · about {time}",
            n: missing,
            time: formatDuration(tf, seconds),
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

  const draftSeconds = draft?.seconds != null ? draft.seconds + (withClips ? (draft.clipSeconds ?? 0) : 0) : null;
  const draftLabel =
    missing > 0 && draftSeconds != null
      ? t("board.passes.draftTimed", { defaultValue: "Make drafts · about {time}", time: formatDuration(tf, draftSeconds) })
      : t("board.passes.draft", { defaultValue: "Make drafts" });
  const finalSeconds = final && !final.assumedAllKept && finalShots > 0 ? final.seconds : null;
  const finalLabel =
    finalSeconds != null
      ? t("board.passes.finalChosenTimed", {
          defaultValue: "Final pass on {n, plural, one {# chosen shot} other {# chosen shots}} · about {time}",
          n: finalShots,
          time: formatDuration(tf, finalSeconds),
        })
      : t("board.passes.finalChosen", {
          defaultValue: "Final pass on {n, plural, one {# chosen shot} other {# chosen shots}}",
          n: finalShots,
        });
  const finalTitle = final?.imageModel
    ? t("board.passes.finalChosenTitle", {
        defaultValue: "Makes each chosen shot again with {model} at {width} x {height}, then its clip from the new still.",
        model: final.imageModel,
        width: final.width,
        height: final.height,
      })
    : undefined;

  const makeAllBtn = (
    <button
      key="all"
      type="button"
      className="secondary-button"
      disabled={busy || unlocked === 0 || noModelReason !== null}
      title={noModelReason ?? t("board.makeAllTitle", { defaultValue: "Adds a new take to every unlocked shot. Nothing is overwritten." })}
      onClick={onMakeAll}
    >
      {t("board.makeAll", { defaultValue: "Make all unlocked" })}
    </button>
  );
  const makeMissingBtn = (primary: boolean) => (
    <button
      key="missing"
      type="button"
      className={primary ? "primary-button" : "secondary-button"}
      disabled={busy || missing === 0 || blockedByRisk || noModelReason !== null}
      title={noModelReason ?? undefined}
      onClick={onMakeMissing}
    >
      {makeLabel}
    </button>
  );
  const clipsToggle = (
    <label key="clips" className="run-bar__passes-clips">
      <input type="checkbox" checked={withClips} disabled={busy} onChange={(event) => setWithClips(event.target.checked)} />
      {t("board.passes.withClips", { defaultValue: "Drafts with clips" })}
    </label>
  );
  const draftsBtn = (primary: boolean) => (
    <button
      key="drafts"
      type="button"
      className={primary ? "primary-button" : "secondary-button"}
      disabled={busy || noModelReason !== null}
      title={noModelReason ?? draftPassTitle(tf, draft)}
      onClick={() => props.onMakeDrafts?.(withClips)}
    >
      {draftLabel}
    </button>
  );
  const finalBtn = (primary: boolean) => (
    <button
      key="final"
      type="button"
      className={primary ? "primary-button" : "secondary-button"}
      disabled={busy || finalShots === 0}
      title={finalTitle}
      onClick={props.onFinalPass}
    >
      {finalLabel}
    </button>
  );

  // The draft-or-final choice, in sizes and time (audit change 2).
  const choice = missing > 0 && canDrafts ? startChoice(quality, estimate, withClips) : null;
  const finalNote =
    finalShots > 0 && quality?.target
      ? t("runBar.choice.finalPass", {
          defaultValue: "The final pass makes the chosen shots again at {size}, then their clips.",
          size: sizeText(quality.target),
        })
      : null;

  const more: ReactNode[] = [];
  if (missing > 0 && !makePrimary) more.push(makeMissingBtn(false));
  if (canDrafts && !draftsPrimary) more.push(clipsToggle, draftsBtn(false));
  more.push(makeAllBtn);
  if (props.onFinalPass && !finalPrimary) more.push(finalBtn(false));
  if (props.onFinalPass && kept === 0) {
    more.push(
      <span key="hint" className="muted-text">
        {t("board.passes.chooseHint", { defaultValue: "Use Choose for final on a shot to put it in the final pass." })}
      </span>,
    );
  }
  // Make missing can be held by the memory check while another button is primary: the acknowledgement lives here then.
  if (missing > 0 && !makePrimary && risk === "danger") {
    more.push(
      <label key="ack" className="check-row run-estimate__ack">
        <input type="checkbox" checked={ack} onChange={(event) => setAck(event.target.checked)} />
        {t("estimate.dangerAck", { defaultValue: "Run anyway. I accept that the backend may crash." })}
      </label>,
    );
  }

  const primaryButton = makePrimary ? makeMissingBtn(true) : draftsPrimary ? (
    <>
      {clipsToggle}
      {draftsBtn(true)}
    </>
  ) : null;
  const stripRisk = makePrimary ? risk : "safe";

  return (
    <div className={`run-bar${stripRisk === "danger" ? " run-bar--danger" : ""}`} role="region" aria-label={t("runBar.label", { defaultValue: "Make shots" })}>
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
      {choice && (choice.draft || choice.final) ? (
        <ul className="run-bar__choice" aria-label={t("runBar.choice.label", { defaultValue: "Drafts or final" })}>
          {choice.draft ? <li className={draftsPrimary ? "is-primary" : undefined}>{draftChoiceText(tf, choice.draft)}</li> : null}
          {choice.final ? <li className={makePrimary ? "is-primary" : undefined}>{finalChoiceText(tf, choice.final)}</li> : null}
        </ul>
      ) : finalNote && missing === 0 ? (
        <p className="run-bar__choice">{finalNote}</p>
      ) : null}
      {missing > 0 ? (
        <RunEstimateStrip
          compactUnknown
          seconds={makePrimary ? seconds : draftSeconds}
          peakGb={makePrimary ? (estimate?.peakGb ?? null) : null}
          totalGb={makePrimary ? (estimate?.totalGb ?? null) : null}
          diskMb={makePrimary ? (estimate?.diskMb ?? null) : null}
          basis={makePrimary ? (seconds != null || estimate?.peakGb != null ? (estimate?.basis ?? null) : null) : (draft?.basis ?? null)}
          risk={stripRisk}
          ack={makePrimary ? { checked: ack, onChange: setAck } : undefined}
          action={<div className="board-estimate__actions">{primaryButton}</div>}
        />
      ) : finalPrimary ? (
        <div className="run-bar__primary">{finalBtn(true)}</div>
      ) : null}
      <details className="run-bar__more">
        <summary>{t("runBar.more", { defaultValue: "More ways to make" })}</summary>
        <div className="run-bar__more-body">{more}</div>
      </details>
    </div>
  );
}
