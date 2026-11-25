/**
 * The shots the chat model proposed (S8), before they touch the project. Each shows its number, title, length
 * and prompts; the total is compared with the length aimed for. "Use these shots" adds them in one step (one
 * Undo); on a project that has shots already it offers "Add after" or "Replace", and Replace asks first when
 * shots with pictures would leave the board (they stay in Undo).
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardProposal } from "../../api/storyboardPlan";
import type { StoryboardProject } from "../../types";
import { ConfirmDialog } from "./ConfirmDialog";
import { lengthVerdict, lockedCount, replaceLosesTakes, shotSeconds, totalSeconds } from "./planLogic";

export interface PlanProposalProps {
  proposal: StoryboardProposal;
  project: StoryboardProject;
  target: number | null;
  busy: boolean;
  onUse: (mode: "append" | "replace") => void;
  onAgain: () => void;
  onDiscard: () => void;
}

export function PlanProposal({ proposal, project, target, busy, onUse, onAgain, onDiscard }: PlanProposalProps) {
  const { t } = useTranslation("studio");
  const [confirm, setConfirm] = useState(false);
  const total = totalSeconds(proposal.shots);
  const verdict = lengthVerdict(total, target);
  const existing = project.shots.length;
  const withTakes = replaceLosesTakes(project);
  const locked = lockedCount(project);
  const heading = useRef<HTMLHeadingElement>(null);
  // A proposal lands below the brief, often off screen: bring it into view and put focus on its heading so a
  // screen reader reads it and Tab goes on to "Use these shots".
  useEffect(() => {
    const el = heading.current;
    if (!el) return;
    const still = typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    el.scrollIntoView?.({ block: "start", behavior: still ? "auto" : "smooth" });
    el.focus({ preventScroll: true });
  }, [proposal]);

  return (
    <section className="plan-proposal" aria-labelledby="plan-proposal-title">
      <div className="plan-section-head plan-proposal__head">
        <div>
          <span className="eyebrow">{t("board.plan.proposal.eyebrow", { defaultValue: "Proposed by {model} in {n} s", model: proposal.model, n: Math.round(proposal.seconds) })}</span>
          <h3 id="plan-proposal-title" ref={heading} tabIndex={-1}>
            {t("board.plan.proposal.title", { defaultValue: "{n, plural, one {# shot} other {# shots}} · {total} s", n: proposal.shots.length, total })}
          </h3>
          {proposal.note ? <p className="plan-proposal__note">{proposal.note}</p> : null}
        </div>
        {verdict !== "none" ? (
          <span className={`badge ${verdict === "match" ? "success" : "warning"}`}>
            {verdict === "match"
              ? t("board.plan.proposal.lengthOk", { defaultValue: "About {target} s, as asked", target })
              : verdict === "short"
                ? t("board.plan.proposal.lengthShort", { defaultValue: "Shorter than the {target} s asked", target })
                : t("board.plan.proposal.lengthLong", { defaultValue: "Longer than the {target} s asked", target })}
          </span>
        ) : null}
      </div>

      <ol className="plan-proposal__list">
        {proposal.shots.map((shot, index) => (
          <li key={index} className="plan-proposal__shot">
            <span className="plan-number" aria-hidden="true">{index + 1}</span>
            <div className="plan-proposal__body">
              <p className="plan-proposal__title">
                <strong>{shot.title || t("board.shot.untitled", { defaultValue: "Shot {n}", n: index + 1 })}</strong>
                <span className="muted-text"> · {t("board.plan.seconds", { defaultValue: "{n} s", n: shotSeconds(shot.seconds) })}</span>
              </p>
              <p className="plan-proposal__prompt">{shot.stillPrompt}</p>
              {shot.motionPrompt ? (
                <p className="plan-proposal__motion muted-text">
                  <span className="plan-proposal__motion-label">{t("board.plan.motion", { defaultValue: "Movement" })}</span> {shot.motionPrompt}
                </p>
              ) : null}
            </div>
          </li>
        ))}
      </ol>

      <div className="plan-proposal__actions">
        {existing === 0 ? (
          <button type="button" className="primary-button" disabled={busy} onClick={() => onUse("append")}>
            {t("board.plan.proposal.use", { defaultValue: "Use these {n} shots", n: proposal.shots.length })}
          </button>
        ) : (
          <>
            <button type="button" className="primary-button" disabled={busy} onClick={() => onUse("append")}>
              {t("board.plan.proposal.append", { defaultValue: "Add after the {n, plural, one {# shot} other {# shots}}", n: existing })}
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={busy || locked > 0}
              title={locked > 0 ? t("board.plan.proposal.lockedHint", { defaultValue: "Unlock the locked shots to replace them." }) : undefined}
              onClick={() => (withTakes > 0 ? setConfirm(true) : onUse("replace"))}
            >
              {t("board.plan.proposal.replace", { defaultValue: "Replace the shots" })}
            </button>
          </>
        )}
        <button type="button" className="secondary-button" disabled={busy} onClick={onAgain}>
          {t("board.plan.proposal.again", { defaultValue: "Try again" })}
        </button>
        <button type="button" className="secondary-button plan-quiet-button" onClick={onDiscard}>
          {t("board.plan.proposal.discard", { defaultValue: "Discard" })}
        </button>
      </div>
      {locked > 0 && existing > 0 ? (
        <p className="muted-text">{t("board.plan.proposal.lockedNote", { defaultValue: "{n, plural, one {# shot is locked, so Replace is off.} other {# shots are locked, so Replace is off.}}", n: locked })}</p>
      ) : null}

      {confirm ? (
        <ConfirmDialog
          title={t("board.plan.proposal.confirmTitle", { defaultValue: "Replace the shots?" })}
          body={
            <p>
              {t("board.plan.proposal.confirmBody", {
                defaultValue: "{n, plural, one {# shot has pictures or clips.} other {# shots have pictures or clips.}} They leave the board; Undo brings them back with their takes.",
                n: withTakes,
              })}
            </p>
          }
          confirmLabel={t("board.plan.proposal.replace", { defaultValue: "Replace the shots" })}
          danger
          busy={busy}
          onConfirm={() => {
            setConfirm(false);
            onUse("replace");
          }}
          onCancel={() => setConfirm(false)}
        />
      ) : null}
    </section>
  );
}
