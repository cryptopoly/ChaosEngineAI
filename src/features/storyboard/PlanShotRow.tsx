/**
 * One shot on the Plan tab (S8): its title, length and prompts, editable in place (each saves when it loses
 * focus, as an ordinary edit), and "Rework with the model": a short chat about this shot. The model answers
 * with a new prompt and why; "Use this" applies only the fields it changed, as one edit (Undo works), and the
 * chat can go on from there.
 */
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { StoryboardShot } from "../../types";
import { takeStyle } from "./planLogic";
import { PlanReworkChat } from "./PlanReworkChat";
import { roundSeconds, shotSeconds } from "./planLogic";
import type { UseStoryboardResult } from "./useStoryboard";
import { useCardTabStops } from "./useCardTabStops";
import { useRework } from "./useRework";
import { useUnsavedText } from "./useUnsavedText";

export interface PlanShotRowProps {
  story: UseStoryboardResult;
  projectId: string;
  shot: StoryboardShot;
  number: number;
  canAsk: boolean;
  /** The row that is the list's one Tab stop (PlanShotList); the others' fields leave the Tab order. Default true. */
  tabbable?: boolean;
  onModelMissing: () => void;
}

export function PlanShotRow({ story, projectId, shot, number, canAsk, tabbable = true, onModelMissing }: PlanShotRowProps) {
  const { t } = useTranslation("studio");
  const rowRef = useRef<HTMLLIElement>(null);
  useCardTabStops(rowRef, tabbable);
  const [fields, setFields] = useState({ title: shot.title, stillPrompt: shot.stillPrompt, motionPrompt: shot.motionPrompt, seconds: shotSeconds(shot.seconds) });
  const saved = useRef({ title: shot.title, stillPrompt: shot.stillPrompt, motionPrompt: shot.motionPrompt, seconds: roundSeconds(shot.seconds) });
  useEffect(() => {
    // Take what the project now says (an Undo, a rework, the direction chat), but keep a field being typed in.
    const next = { title: shot.title, stillPrompt: shot.stillPrompt, motionPrompt: shot.motionPrompt, seconds: roundSeconds(shot.seconds) };
    const before = saved.current; // read now: the updater below runs later, after `saved` has moved on
    setFields((current) => ({
      title: current.title === before.title ? next.title : current.title,
      stillPrompt: current.stillPrompt === before.stillPrompt ? next.stillPrompt : current.stillPrompt,
      motionPrompt: current.motionPrompt === before.motionPrompt ? next.motionPrompt : current.motionPrompt,
      seconds: Number(current.seconds) === before.seconds ? shotSeconds(next.seconds) : current.seconds,
    }));
    saved.current = next;
  }, [shot.title, shot.stillPrompt, shot.motionPrompt, shot.seconds]);

  const [open, setOpen] = useState(false);
  const rework = useRework(story, projectId, shot, canAsk, onModelMissing);

  // Each field keeps its text when the save is refused (S-O): the latest shows, with "Put it back" under it.
  const textField = (key: "title" | "stillPrompt" | "motionPrompt") => ({
    text: fields[key],
    dirty: fields[key] !== saved.current[key],
    send: (text: string) => story.op("edit_shot", { shotId: shot.id, [key]: text }),
    restore: (text: string) => setFields((current) => ({ ...current, [key]: text })),
  });
  const unsaved = {
    title: useUnsavedText(`plan:${shot.id}:title`, textField("title")),
    stillPrompt: useUnsavedText(`plan:${shot.id}:stillPrompt`, textField("stillPrompt")),
    motionPrompt: useUnsavedText(`plan:${shot.id}:motionPrompt`, textField("motionPrompt")),
  };
  const unsavedSeconds = useUnsavedText(`plan:${shot.id}:seconds`, {
    text: fields.seconds,
    dirty: Number(fields.seconds) !== saved.current.seconds,
    send: (text) => story.op("edit_shot", { shotId: shot.id, seconds: Math.round(Number(text) * 10) / 10 }),
    restore: (text) => setFields((current) => ({ ...current, seconds: text })),
  });

  function save(key: "title" | "stillPrompt" | "motionPrompt") {
    const value = fields[key];
    if (value === saved.current[key]) return;
    saved.current = { ...saved.current, [key]: value };
    unsaved[key].save(value);
  }

  function saveSeconds() {
    const value = Math.round(Number(fields.seconds) * 10) / 10;
    if (!Number.isFinite(value) || value < 0.5 || value > 60) {
      setFields((current) => ({ ...current, seconds: shotSeconds(saved.current.seconds) }));
      return;
    }
    if (value === saved.current.seconds) return;
    saved.current = { ...saved.current, seconds: value };
    unsavedSeconds.save(String(value));
  }
  const style = takeStyle(shot);
  const rowId = `plan-shot-${shot.id}`;
  const lastSuggestion = rework.waiting;

  return (
    <li
      ref={rowRef}
      className={`plan-shot${open ? " is-open" : ""}${shot.locked ? " is-locked" : ""}`}
      data-plan-shot-id={shot.id}
      tabIndex={tabbable ? 0 : -1}
      // The number is drawn aria-hidden and the title is a field: name the row "Shot 3 · Title" for a screen reader.
      aria-label={shot.title ? `${t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })} · ${shot.title}` : t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })}
    >
      <span className="plan-number" aria-hidden="true">{number}</span>
      <div className="plan-shot__main">
        <div className="plan-shot__top">
          <input
            className="text-input plan-shot__title"
            value={fields.title}
            maxLength={200}
            disabled={shot.locked}
            placeholder={t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })}
            aria-label={t("board.plan.field.titleOf", { defaultValue: "Title of shot {n}", n: number })}
            onChange={(event) => setFields({ ...fields, title: event.target.value })}
            onBlur={() => save("title")}
            onKeyDown={(event) => event.key === "Enter" && save("title")}
          />
          <label className="plan-shot__seconds">
            <input
              className="text-input"
              type="number"
              inputMode="decimal"
              min={0.5}
              max={60}
              step={0.5}
              value={fields.seconds}
              disabled={shot.locked}
              aria-label={t("board.plan.field.secondsOf", { defaultValue: "Length of shot {n} in seconds", n: number })}
              onChange={(event) => setFields({ ...fields, seconds: event.target.value })}
              onBlur={saveSeconds}
              onKeyDown={(event) => event.key === "Enter" && saveSeconds()}
            />
            <span aria-hidden="true">s</span>
          </label>
          {shot.locked ? <span className="badge muted">{t("board.plan.locked", { defaultValue: "Locked" })}</span> : null}
          {style ? (
            <span className="badge subtle" title={style === "pictures" ? t("board.style.usedPicturesHint", { defaultValue: "The still model took the style pictures as references." }) : t("board.style.usedWordsHint", { defaultValue: "The still model cannot take reference pictures, so the style went into the prompt as words." })}>
              {style === "pictures" ? t("board.style.usedPictures", { defaultValue: "Style as pictures" }) : t("board.style.usedWords", { defaultValue: "Style as words" })}
            </span>
          ) : null}
        </div>
        {unsaved.title.notice}
        {unsavedSeconds.notice}
        <textarea
          className="text-area plan-shot__prompt"
          rows={3}
          maxLength={4000}
          value={fields.stillPrompt}
          disabled={shot.locked}
          placeholder={t("board.plan.field.stillPlaceholder", { defaultValue: "What the picture shows: subject, place, light, framing" })}
          aria-label={t("board.plan.field.stillOf", { defaultValue: "Picture prompt of shot {n}", n: number })}
          onChange={(event) => setFields({ ...fields, stillPrompt: event.target.value })}
          onBlur={() => save("stillPrompt")}
        />
        {unsaved.stillPrompt.notice}
        <textarea
          className="text-area plan-shot__motion"
          rows={1}
          maxLength={4000}
          value={fields.motionPrompt}
          disabled={shot.locked}
          placeholder={t("board.plan.field.motionPlaceholder", { defaultValue: "What moves, and how the camera moves" })}
          aria-label={t("board.plan.field.motionOf", { defaultValue: "Movement prompt of shot {n}", n: number })}
          onChange={(event) => setFields({ ...fields, motionPrompt: event.target.value })}
          onBlur={() => save("motionPrompt")}
        />
        {unsaved.motionPrompt.notice}
        <div className="plan-shot__actions">
          <button
            type="button"
            className="secondary-button plan-shot__rework"
            aria-expanded={open}
            aria-controls={`${rowId}-chat`}
            disabled={shot.locked}
            onClick={() => setOpen(!open)}
          >
            {open ? t("board.plan.rework.close", { defaultValue: "Close" }) : t("board.plan.rework.open", { defaultValue: "Rework with the model" })}
          </button>
          {!open && lastSuggestion ? <span className="muted-text">{t("board.plan.rework.waiting", { defaultValue: "A suggestion is waiting" })}</span> : null}
        </div>

        {open ? (
          <PlanReworkChat id={`${rowId}-chat`} shot={shot} number={number} rework={rework} busy={story.busy} onClose={() => setOpen(false)} />
        ) : null}
      </div>
    </li>
  );
}
