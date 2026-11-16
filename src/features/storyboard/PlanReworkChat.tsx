/**
 * The body of a shot's "Rework with the model" chat (S8): the turns, each suggestion with the fields it would change
 * and "Use this", the typing line, and the ask box. The state comes from `useRework`, so the Plan tab's shot row and
 * the freestyle board's card chat (S-M2) show the same conversation the same way.
 */
import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { SendIcon } from "../../components/icons/ActionIcons";
import type { StoryboardShot } from "../../types";
import { changedFields, type ShotField } from "./planLogic";
import type { ReworkState } from "./useRework";

export interface PlanReworkChatProps {
  id: string;
  shot: StoryboardShot;
  number: number;
  rework: ReworkState;
  busy: boolean;
  /** Escape in the ask box. */
  onClose?: () => void;
  /** Focus the ask box when it appears (default true). */
  autoFocus?: boolean;
}

export function PlanReworkChat({ id, shot, number, rework, busy, onClose, autoFocus = true }: PlanReworkChatProps) {
  const { t } = useTranslation("studio");
  const askField = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (autoFocus) askField.current?.focus();
  }, [autoFocus]);
  const { turns, ask, setAsk, pending, error, elapsed, send, use } = rework;

  const fieldLabel: Record<ShotField, string> = {
    title: t("board.plan.field.title", { defaultValue: "Title" }),
    stillPrompt: t("board.plan.field.still", { defaultValue: "Picture" }),
    motionPrompt: t("board.plan.field.motion", { defaultValue: "Movement" }),
    seconds: t("board.plan.field.seconds", { defaultValue: "Length" }),
  };

  return (
    <div className="plan-rework" id={id} role="region" aria-label={t("board.plan.rework.label", { defaultValue: "Rework shot {n}", n: number })}>
      {turns.length === 0 ? (
        <p className="muted-text">{t("board.plan.rework.intro", { defaultValue: "Say what should change, for example \"closer, warmer light\" or \"show the cup, not the pot\". The model rewrites this shot and says why." })}</p>
      ) : (
        <ol className="plan-chat__list" aria-live="polite">
          {turns.map((turn, index) => (
            <li key={index} className={`plan-chat__line plan-chat__line--${turn.role}`}>
              {turn.suggestion ? (
                <div className="plan-suggestion">
                  <p className="plan-suggestion__why">{turn.text}</p>
                  {changedFields(shot, turn.suggestion.shot).length === 0 && !turn.applied ? (
                    <p className="muted-text">{t("board.plan.rework.same", { defaultValue: "This matches the shot as it is now." })}</p>
                  ) : null}
                  <dl className="plan-suggestion__fields">
                    {(turn.applied ? (["stillPrompt"] as ShotField[]) : changedFields(shot, turn.suggestion.shot)).map((key) => (
                      <div key={key}>
                        <dt>{fieldLabel[key]}</dt>
                        <dd>{key === "seconds" ? t("board.plan.seconds", { defaultValue: "{n} s", n: turn.suggestion!.shot.seconds }) : turn.suggestion!.shot[key]}</dd>
                      </div>
                    ))}
                  </dl>
                  {turn.applied ? (
                    <p className="plan-suggestion__applied">{t("board.plan.rework.applied", { defaultValue: "Used. Undo puts the old prompt back." })}</p>
                  ) : changedFields(shot, turn.suggestion.shot).length > 0 ? (
                    <div className="plan-suggestion__actions">
                      <button type="button" className="primary-button" disabled={busy} onClick={() => void use(index, turn.suggestion!)}>
                        {t("board.plan.rework.use", { defaultValue: "Use this" })}
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p>{turn.text}</p>
              )}
            </li>
          ))}
          {pending ? (
            <li className="plan-chat__line plan-chat__line--assistant plan-chat__line--pending" role="status">
              <span className="plan-typing" aria-hidden="true"><span /><span /><span /></span>
              {t("board.plan.rework.thinking", { defaultValue: "Rewriting… {n} s", n: elapsed })}
            </li>
          ) : null}
        </ol>
      )}
      {error ? <p className="plan-error" role="alert">{error}</p> : null}
      <form
        className="plan-chat__form"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <textarea
          ref={askField}
          className="text-area"
          rows={1}
          maxLength={2000}
          value={ask}
          placeholder={t("board.plan.rework.placeholder", { defaultValue: "What should change?" })}
          aria-label={t("board.plan.rework.ask", { defaultValue: "What should change in shot {n}?", n: number })}
          onChange={(event) => setAsk(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              void send();
            }
            if (event.key === "Escape" && onClose) {
              event.stopPropagation();
              onClose();
            }
          }}
        />
        <button type="submit" className="primary-button icon-button plan-chat__send" disabled={!ask.trim() || pending} aria-label={t("board.plan.send", { defaultValue: "Send" })} title={t("board.plan.send", { defaultValue: "Send" })}>
          <SendIcon aria-hidden="true" />
        </button>
      </form>
    </div>
  );
}
