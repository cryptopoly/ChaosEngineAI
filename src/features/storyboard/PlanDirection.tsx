/**
 * The direction chat on the Plan tab (S8): one conversation about the whole film, kept with the project. The
 * model answers in words and may propose changes across several shots (edit, add, remove, or a whole
 * re-plan). The changes are listed in plain words and applied together as one step, so one Undo takes them
 * all back. The chat itself is not part of Undo.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { directStoryboardPlan, type StoryboardDirection } from "../../api/storyboardPlan";
import { SendIcon } from "../../components/icons/ActionIcons";
import type { StoryboardProject } from "../../types";
import { changeCount, changeRows, type ChangeRow } from "./planLogic";
import { failureOf, useElapsed } from "./usePlan";
import type { UseStoryboardResult } from "./useStoryboard";

export interface PlanDirectionProps {
  story: UseStoryboardResult;
  project: StoryboardProject;
  canAsk: boolean;
  onModelMissing: () => void;
}

export function PlanDirection({ story, project, canAsk, onModelMissing }: PlanDirectionProps) {
  const { t } = useTranslation("studio");
  const [ask, setAsk] = useState("");
  const [pending, setPending] = useState<string | null>(null);
  const [answer, setAnswer] = useState<StoryboardDirection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const elapsed = useElapsed(pending !== null);
  const list = useRef<HTMLOListElement>(null);
  const chat = project.planChat ?? [];

  useEffect(() => {
    const el = list.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chat.length, pending, answer]);

  async function send() {
    const message = ask.trim();
    if (!message || pending !== null) return;
    if (!canAsk) {
      onModelMissing();
      return;
    }
    setPending(message);
    setAsk("");
    setError(null);
    setAnswer(null);
    try {
      const reply = await directStoryboardPlan(project.id, message);
      await story.reload(); // the chat lines landed on the server
      setAnswer(changeCount(reply.changes) > 0 ? reply : null);
    } catch (err) {
      const failure = failureOf(err);
      if (failure.code === "no-model") onModelMissing();
      setError(failure.message);
      setAsk(message);
    } finally {
      setPending(null);
    }
  }

  async function apply() {
    if (!answer) return;
    if (await story.op("plan_changes", answer.changes)) setAnswer(null);
  }

  const fieldName = (field: string): string =>
    (field === "title"
      ? t("board.plan.field.title", { defaultValue: "Title" })
      : field === "stillPrompt"
        ? t("board.plan.field.still", { defaultValue: "Picture" })
        : field === "motionPrompt"
          ? t("board.plan.field.motion", { defaultValue: "Movement" })
          : t("board.plan.field.seconds", { defaultValue: "Length" })
    ).toLowerCase();
  const shotName = (number: number, title: string): string =>
    title
      ? t("board.plan.direct.shotTitled", { defaultValue: "Shot {n} ({title})", n: number, title })
      : t("board.shot.untitled", { defaultValue: "Shot {n}", n: number });
  const rowText = (row: ChangeRow): string => {
    if (row.kind === "edit") {
      return t("board.plan.direct.edit", { defaultValue: "{shot}: new {fields}", shot: shotName(row.number, row.title), fields: row.fields.map(fieldName).join(", ") });
    }
    if (row.kind === "remove") return t("board.plan.direct.remove", { defaultValue: "Remove {shot}", shot: shotName(row.number, row.title) });
    return row.after === 0
      ? t("board.plan.direct.addFirst", { defaultValue: "New first shot: {title}", title: row.title || "…" })
      : t("board.plan.direct.add", { defaultValue: "New shot after shot {n}: {title}", n: row.after, title: row.title || "…" });
  };
  const rows = answer ? changeRows(answer.changes, project.shots) : [];

  return (
    <section className="plan-direction" aria-labelledby="plan-direction-title">
      <div className="plan-direction__head">
        <div>
          <h3 id="plan-direction-title">{t("board.plan.direct.title", { defaultValue: "Direction" })}</h3>
          <p className="muted-text">{t("board.plan.direct.subtitle", { defaultValue: "Talk about the whole film. The model can change several shots at once; you apply them in one step." })}</p>
        </div>
        {chat.length > 0 ? (
          <button type="button" className="secondary-button plan-quiet-button" disabled={story.busy || pending !== null} onClick={() => void story.op("clear_plan_chat", {})}>
            {t("board.plan.direct.clear", { defaultValue: "Clear" })}
          </button>
        ) : null}
      </div>

      <ol ref={list} className="plan-chat__list plan-direction__list" aria-live="polite" aria-label={t("board.plan.direct.log", { defaultValue: "Direction chat" })}>
        {chat.length === 0 && pending === null ? (
          <li className="plan-direction__empty muted-text">
            {t("board.plan.direct.empty", { defaultValue: "Try \"make it feel like early morning\", \"end on the shop sign\" or \"cut it to 30 seconds\"." })}
          </li>
        ) : null}
        {chat.map((message) => (
          <li key={message.id} className={`plan-chat__line plan-chat__line--${message.role}`}>
            <p>{message.text}</p>
          </li>
        ))}
        {pending !== null ? (
          <>
            <li className="plan-chat__line plan-chat__line--user">
              <p>{pending}</p>
            </li>
            <li className="plan-chat__line plan-chat__line--assistant plan-chat__line--pending" role="status">
              <span className="plan-typing" aria-hidden="true"><span /><span /><span /></span>
              {t("board.plan.direct.thinking", { defaultValue: "Thinking… {n} s", n: elapsed })}
            </li>
          </>
        ) : null}
      </ol>

      {answer && rows.length > 0 ? (
        <div className="plan-changes" role="group" aria-label={t("board.plan.direct.changesLabel", { defaultValue: "Proposed changes" })}>
          <p className="plan-changes__title">
            {t("board.plan.direct.changesTitle", { defaultValue: "{n, plural, one {# change proposed} other {# changes proposed}}", n: rows.length })}
          </p>
          <ul className="plan-changes__list">
            {rows.map((row, index) => (
              <li key={index} className={`plan-changes__row plan-changes__row--${row.kind}`}>{rowText(row)}</li>
            ))}
          </ul>
          <div className="plan-changes__actions">
            <button type="button" className="primary-button" disabled={story.busy} onClick={() => void apply()}>
              {t("board.plan.direct.apply", { defaultValue: "{n, plural, one {Apply the change} other {Apply # changes}}", n: rows.length })}
            </button>
            <button type="button" className="secondary-button plan-quiet-button" onClick={() => setAnswer(null)}>
              {t("board.plan.direct.dismiss", { defaultValue: "Dismiss" })}
            </button>
          </div>
        </div>
      ) : null}

      {error ? <p className="plan-error" role="alert">{error}</p> : null}
      <form
        className="plan-chat__form"
        onSubmit={(event) => {
          event.preventDefault();
          void send();
        }}
      >
        <textarea
          className="text-area"
          rows={2}
          maxLength={2000}
          value={ask}
          placeholder={t("board.plan.direct.placeholder", { defaultValue: "Tell the model what to change across the film" })}
          aria-label={t("board.plan.direct.ask", { defaultValue: "Direction for the whole film" })}
          onChange={(event) => setAsk(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              void send();
            }
          }}
        />
        <button type="submit" className="primary-button icon-button plan-chat__send" disabled={!ask.trim() || pending !== null} aria-label={t("board.plan.send", { defaultValue: "Send" })} title={t("board.plan.send", { defaultValue: "Send" })}>
          <SendIcon aria-hidden="true" />
        </button>
      </form>
    </section>
  );
}
