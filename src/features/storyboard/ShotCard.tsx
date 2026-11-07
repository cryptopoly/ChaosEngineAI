/** One shot on the Board (UX spec D.5): thumbnail, status, takes, lock, redo, transition, ordering. */
import { useEffect, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import {
  ChevronDownIcon,
  ChevronUpIcon,
  GripIcon,
  LockIcon,
  PlayIcon,
  RedoIcon,
  UnlockIcon,
} from "../../components/icons/ActionIcons";
import type { StoryboardShot, StoryboardShotStatus, StoryboardTransition } from "../../types";
import { ActionMenu } from "./ActionMenu";
import { ProjectImage, ProjectVideo } from "./ProjectMedia";
import { StaleBadge, TransitionPicker } from "./ShotParts";
import { TakesStrip } from "./TakesStrip";
import { activeTakeOf, shotCardClassName, thumbnailOf } from "./storyboardLogic";

export const SHOT_DRAG_TYPE = "application/x-chaos-shot";

export interface ShotCardProps {
  projectId: string;
  shot: StoryboardShot;
  index: number;
  count: number;
  /** The status to show: the batch's live state wins over the stored one. */
  status: StoryboardShotStatus;
  queuePosition?: number;
  /** Zero-based index of the shot this one continues from when it is stale. */
  staleAfter: number | null;
  /** The user chose Keep for the stale badge. */
  kept: boolean;
  /** Why the last batch failed this shot, if it did. */
  failure: string | null;
  /** Roving tabindex: only one card in the list is a tab stop. */
  tabbable: boolean;
  dragOver: boolean;
  disabled: boolean;
  onRename: (title: string) => void;
  onPickTake: (takeId: string) => void;
  onToggleLock: () => void;
  onToggleRedo: () => void;
  onTransition: (next: StoryboardTransition) => void;
  onMove: (step: -1 | 1) => void;
  onEdit: () => void;
  onKeep: () => void;
  onRerun: () => void;
  onRetry: () => void;
  onRemove: () => void;
  onDragStart: (shotId: string) => void;
  onDragEnd: () => void;
  onDropOn: (dragId: string, overId: string) => void;
  onDragOverCard: (shotId: string | null) => void;
}

export function ShotCard(props: ShotCardProps) {
  const {
    projectId, shot, index, count, status, queuePosition, staleAfter, kept, failure, tabbable, dragOver, disabled,
  } = props;
  const { t } = useTranslation("studio");
  const [title, setTitle] = useState(shot.title);
  const [playing, setPlaying] = useState(false);
  const take = activeTakeOf(shot);
  const thumb = thumbnailOf(shot);
  const number = index + 1;
  const showStale = status === "stale" && !kept;

  useEffect(() => setTitle(shot.title), [shot.title]);
  useEffect(() => setPlaying(false), [shot.activeTake]);

  const statusText = (() => {
    switch (status) {
      case "queued":
        return t("board.shot.status.queued", { defaultValue: "Queued · {position}", position: queuePosition ?? 1 });
      case "running":
        return t("board.shot.status.running", { defaultValue: "Making…" });
      case "done":
        return take?.measuredSeconds
          ? t("board.shot.status.done", { defaultValue: "Done · {seconds} s", seconds: Math.round(take.measuredSeconds * 100) / 100 })
          : t("board.shot.status.doneNoLength", { defaultValue: "Done" });
      case "failed":
        return t("board.shot.status.failed", { defaultValue: "Failed" });
      case "stale":
        return staleAfter != null
          ? t("board.shot.status.stale", { defaultValue: "Stale · after shot {n}", n: staleAfter + 1 })
          : t("board.shot.status.staleNoShot", { defaultValue: "Stale" });
      case "missing":
        return t("board.shot.status.missing", { defaultValue: "Take file missing" });
      default:
        return t("board.shot.status.idle", { defaultValue: "Not made yet" });
    }
  })();

  const statusId = `shot-status-${shot.id}`;
  const titleId = `shot-title-${shot.id}`;

  const onDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    const dragId = event.dataTransfer.getData(SHOT_DRAG_TYPE);
    props.onDragOverCard(null);
    if (dragId) props.onDropOn(dragId, shot.id);
  };

  return (
    <article
      className={`${shotCardClassName(status, shot.locked)}${dragOver ? " shot-card--drop" : ""}${shot.redo ? " shot-card--redo" : ""}`}
      data-shot-id={shot.id}
      aria-labelledby={titleId}
      aria-describedby={statusId}
      tabIndex={tabbable ? 0 : -1}
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes(SHOT_DRAG_TYPE)) return;
        event.preventDefault();
        props.onDragOverCard(shot.id);
      }}
      onDragLeave={() => props.onDragOverCard(null)}
      onDrop={onDrop}
    >
      <div className="shot-card__head">
        <span
          className="shot-card__handle"
          draggable={!disabled}
          title={t("board.shot.drag", { defaultValue: "Drag to reorder" })}
          onDragStart={(event) => {
            event.dataTransfer.setData(SHOT_DRAG_TYPE, shot.id);
            event.dataTransfer.effectAllowed = "move";
            const card = event.currentTarget.closest("article");
            if (card) event.dataTransfer.setDragImage(card, 16, 16);
            props.onDragStart(shot.id);
          }}
          onDragEnd={props.onDragEnd}
        >
          <GripIcon />
        </span>
        <span className="shot-card__number" aria-hidden="true">{number}</span>
        <input
          id={titleId}
          className="shot-card__title"
          type="text"
          value={title}
          maxLength={200}
          placeholder={t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })}
          aria-label={t("board.shot.titleLabel", { defaultValue: "Shot {n} title", n: number })}
          disabled={disabled}
          onChange={(event) => setTitle(event.target.value)}
          onBlur={() => {
            if (title !== shot.title) props.onRename(title.trim());
          }}
          onKeyDown={(event) => {
            event.stopPropagation();
            if (event.key === "Enter") event.currentTarget.blur();
            if (event.key === "Escape") {
              setTitle(shot.title);
              event.currentTarget.blur();
            }
          }}
        />
        <div className="shot-card__order">
          <button
            type="button"
            className="secondary-button icon-button"
            disabled={disabled || index === 0}
            aria-label={t("board.shot.menu.up", { defaultValue: "Move up" })}
            title={t("board.shot.menu.up", { defaultValue: "Move up" })}
            onClick={() => props.onMove(-1)}
          >
            <ChevronUpIcon />
          </button>
          <button
            type="button"
            className="secondary-button icon-button"
            disabled={disabled || index === count - 1}
            aria-label={t("board.shot.menu.down", { defaultValue: "Move down" })}
            title={t("board.shot.menu.down", { defaultValue: "Move down" })}
            onClick={() => props.onMove(1)}
          >
            <ChevronDownIcon />
          </button>
        </div>
      </div>

      <div className="shot-card__body">
        <div className="shot-card__media">
          {playing && take?.clip ? (
            <ProjectVideo projectId={projectId} rel={take.clip} version={take.hashes.clip} className="shot-card__video" controls autoPlay />
          ) : thumb ? (
            <>
              {thumb.kind === "still" ? (
                <ProjectImage projectId={projectId} rel={thumb.path} version={take?.hashes.still} className="shot-card__thumb" />
              ) : (
                <ProjectVideo projectId={projectId} rel={thumb.path} version={take?.hashes.clip} className="shot-card__thumb" />
              )}
              {take?.clip ? (
                <button
                  type="button"
                  className="shot-card__play"
                  aria-label={t("board.shot.play", { defaultValue: "Play clip" })}
                  title={t("board.shot.play", { defaultValue: "Play clip" })}
                  onClick={() => setPlaying(true)}
                >
                  <PlayIcon width={22} height={22} />
                </button>
              ) : null}
            </>
          ) : (
            <span className="shot-card__thumb-empty" aria-hidden="true" />
          )}
        </div>
        <div className="shot-card__info">
          <p id={statusId} className="shot-card__status">{statusText}</p>
          {shot.seed != null ? (
            <p className="shot-card__meta">{t("board.shot.seed", { defaultValue: "seed {seed}", seed: shot.seed })}</p>
          ) : null}
          {shot.redo ? <span className="badge accent">{t("board.shot.redoMarked", { defaultValue: "Marked for redo" })}</span> : null}
          {shot.locked ? <span className="badge muted">{t("board.shot.status.locked", { defaultValue: "Locked" })}</span> : null}
        </div>
      </div>

      {status === "running" || status === "queued" ? (
        <div className={`shot-card__bar shot-card__bar--${status}`} role="progressbar" aria-label={statusText} />
      ) : null}

      {showStale ? (
        <div className="shot-card__stale">
          <StaleBadge
            reason={statusText}
            title={
              staleAfter != null
                ? t("board.shot.staleTitle", {
                    defaultValue: "Shot {n} got a new take; this shot starts from its last frame.",
                    n: staleAfter + 1,
                  })
                : t("board.shot.staleTitleGeneric", { defaultValue: "Something this shot depends on changed." })
            }
          />
          <button type="button" className="secondary-button" disabled={disabled} onClick={props.onRerun}>
            {t("board.shot.rerun", { defaultValue: "Re-run" })}
          </button>
          <button type="button" className="secondary-button" onClick={props.onKeep}>
            {t("board.shot.keep", { defaultValue: "Keep" })}
          </button>
        </div>
      ) : null}

      {status === "failed" ? (
        <div className="callout error shot-card__failed" role="alert">
          <p>{t("board.shot.failedNote", { defaultValue: "The shot did not render. Try again or edit the prompts." })}</p>
          {failure ? (
            <details className="debug-details">
              <summary>{t("error.technicalDetails", { defaultValue: "Technical details" })}</summary>
              <pre>{failure}</pre>
            </details>
          ) : null}
          <button type="button" className="secondary-button" disabled={disabled || shot.locked} onClick={props.onRetry}>
            {t("error.retry", { defaultValue: "Retry" })}
          </button>
        </div>
      ) : null}

      {status === "missing" ? (
        <div className="callout warning shot-card__missing">
          <p>{t("board.shot.missingNote", { defaultValue: "Take file missing. Relink or make a new take." })}</p>
          <button type="button" className="secondary-button" onClick={props.onEdit}>
            {t("board.shot.relink", { defaultValue: "Relink" })}
          </button>
        </div>
      ) : null}

      <TakesStrip
        projectId={projectId}
        takes={shot.takes}
        activeId={shot.activeTake}
        onPick={props.onPickTake}
        disabled={disabled}
      />

      <div className="shot-card__actions">
        <button
          type="button"
          className="secondary-button icon-button"
          aria-pressed={shot.locked}
          aria-label={shot.locked ? t("board.shot.unlock", { defaultValue: "Unlock this shot" }) : t("board.shot.lock", { defaultValue: "Lock this shot" })}
          title={
            shot.locked
              ? t("board.shot.unlock", { defaultValue: "Unlock this shot" })
              : t("board.shot.lockedNote", { defaultValue: "Locked shots are skipped by Make shots." })
          }
          disabled={disabled}
          onClick={props.onToggleLock}
        >
          {shot.locked ? <LockIcon /> : <UnlockIcon />}
        </button>
        <button
          type="button"
          className={`secondary-button${shot.redo ? " is-active" : ""}`}
          aria-pressed={shot.redo}
          title={t("board.shot.redoTitle", { defaultValue: "Mark for redo: the next Make shots adds a new take. Nothing is overwritten." })}
          disabled={disabled}
          onClick={props.onToggleRedo}
        >
          <RedoIcon /> {t("board.shot.redoLabel", { defaultValue: "Redo" })}
        </button>
        {index > 0 ? <TransitionPicker value={shot.transition} onChange={props.onTransition} disabled={disabled} /> : null}
        <ActionMenu
          label={t("board.shot.moreActions", { defaultValue: "More actions for shot {n}", n: number })}
          items={[
            { id: "edit", label: t("board.shot.menu.edit", { defaultValue: "Edit prompts" }), onSelect: props.onEdit },
            { id: "import", label: t("board.shot.menu.import", { defaultValue: "Import footage" }), onSelect: props.onEdit },
            { id: "up", label: t("board.shot.menu.up", { defaultValue: "Move up" }), onSelect: () => props.onMove(-1), disabled: index === 0 },
            { id: "down", label: t("board.shot.menu.down", { defaultValue: "Move down" }), onSelect: () => props.onMove(1), disabled: index === count - 1 },
            { id: "delete", label: t("board.shot.menu.delete", { defaultValue: "Delete" }), onSelect: props.onRemove, danger: true },
          ]}
        />
      </div>
    </article>
  );
}
