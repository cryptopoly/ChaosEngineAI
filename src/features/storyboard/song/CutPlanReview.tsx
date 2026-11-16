/**
 * A proposed cut plan for review (S-M2): the shots the backend cut from the song's sections and beats, grouped by
 * section, with the total, the longest shot the video model allows and the backend's note. Nothing is written until
 * "Use these N shots", which replaces the shots with the ordinary `apply_plan` op (one Undo); when shots with takes
 * would leave the board, a confirm says so first. Locked shots keep Replace off, as on the Plan tab.
 */
import { useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardCutPlan, StoryboardProject } from "../../../types";
import { ConfirmDialog } from "../ConfirmDialog";
import { lockedCount } from "../planLogic";
import { clock, cutPlanGroups, shotsWithTakes } from "./songLogic";

export interface CutPlanReviewProps {
  plan: StoryboardCutPlan;
  project: StoryboardProject;
  busy: boolean;
  onUse: () => void;
  onDiscard: () => void;
}

export function CutPlanReview({ plan, project, busy, onUse, onDiscard }: CutPlanReviewProps) {
  const { t } = useTranslation("studio");
  const [confirm, setConfirm] = useState(false);
  const locked = lockedCount(project);
  const withTakes = shotsWithTakes(project.shots);
  const groups = cutPlanGroups(plan);
  const n = plan.shots.length;

  return (
    <section className="cut-plan" aria-labelledby="cut-plan-title">
      <div className="plan-section-head">
        <div>
          <h4 id="cut-plan-title" tabIndex={-1}>
            {t("board.song.cut.title", { defaultValue: "Cut plan: {n, plural, one {# shot} other {# shots}} over {time}", n, time: clock(plan.totalSeconds) })}
          </h4>
          <p className="muted-text">
            {plan.clipSecondsSource === "model" && plan.videoModel
              ? t("board.song.cut.capModel", { defaultValue: "Shots are at most {n} s, the clip length of {model}.", n: plan.clipSeconds, model: plan.videoModel })
              : plan.clipSecondsSource === "request"
                ? t("board.song.cut.capRequest", { defaultValue: "Shots are at most {n} s, as asked.", n: plan.clipSeconds })
                : t("board.song.cut.capDefault", { defaultValue: "Shots are at most {n} s (no video model is set for the project).", n: plan.clipSeconds })}
          </p>
          {plan.note ? <p className="muted-text">{plan.note}</p> : null}
        </div>
      </div>
      <ol className="cut-plan__groups">
        {groups.map((group, gi) => (
          <li key={gi} className="cut-plan__group">
            <p className="cut-plan__section">
              <strong>{group.label || "?"}</strong>
              <span className="muted-text">{`${clock(group.start)} – ${clock(group.end)}`}</span>
            </p>
            <ol className="cut-plan__shots">
              {group.shots.map(({ number, shot }) => (
                <li key={number} className="cut-plan__shot">
                  <span className="plan-number" aria-hidden="true">{number}</span>
                  <span className="cut-plan__shot-title">{shot.title || t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })}</span>
                  <span className="muted-text">{t("board.plan.seconds", { defaultValue: "{n} s", n: Math.round(shot.seconds * 100) / 100 })}</span>
                  {shot.stillPrompt ? <span className="cut-plan__prompt muted-text">{shot.stillPrompt}</span> : null}
                </li>
              ))}
            </ol>
          </li>
        ))}
      </ol>
      <div className="plan-proposal__actions">
        <button
          type="button"
          className="primary-button"
          disabled={busy || locked > 0 || n === 0}
          title={locked > 0 ? t("board.plan.proposal.lockedHint", { defaultValue: "Unlock the locked shots to replace them." }) : undefined}
          onClick={() => (withTakes > 0 ? setConfirm(true) : onUse())}
        >
          {t("board.song.cut.use", { defaultValue: "{n, plural, one {Use this shot} other {Use these # shots}}", n })}
        </button>
        <button type="button" className="secondary-button" onClick={onDiscard}>
          {t("board.song.cut.discard", { defaultValue: "Discard" })}
        </button>
      </div>
      {locked > 0 ? (
        <p className="muted-text">{t("board.plan.proposal.lockedNote", { defaultValue: "{n, plural, one {# shot is locked, so Replace is off.} other {# shots are locked, so Replace is off.}}", n: locked })}</p>
      ) : project.shots.length > 0 ? (
        <p className="muted-text">{t("board.song.cut.replaceNote", { defaultValue: "These replace the {n, plural, one {# shot} other {# shots}} on the board; with Keep prompts, shot 1 keeps the prompts of today's shot 1 and so on. Undo brings the old shots back.", n: project.shots.length })}</p>
      ) : null}
      {confirm ? (
        <ConfirmDialog
          title={t("board.plan.proposal.confirmTitle", { defaultValue: "Replace the shots?" })}
          body={<p>{t("board.song.cut.confirmBody", { defaultValue: "{n, plural, one {# shot has takes and leaves} other {# shots have takes and leave}} the board. Undo brings them back.", n: withTakes })}</p>}
          confirmLabel={t("board.plan.proposal.replace", { defaultValue: "Replace the shots" })}
          busy={busy}
          onConfirm={() => {
            setConfirm(false);
            onUse();
          }}
          onCancel={() => setConfirm(false)}
        />
      ) : null}
    </section>
  );
}
