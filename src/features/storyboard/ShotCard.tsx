/** One shot on the Board (UX spec D.5): thumbnail, status, takes, lock, redo, transition, ordering. */
import { useEffect, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import {
  ChevronDownIcon,
  ChevronUpIcon,
  GripIcon,
  LockIcon,
  PlayIcon,
  UnlockIcon,
} from "../../components/icons/ActionIcons";
import type { StoryboardFailureAction, StoryboardShot, StoryboardShotStatus, StoryboardTransition } from "../../types";
import { ActionMenu } from "../../components/ActionMenu";
import { ProjectImage, ProjectVideo } from "./ProjectMedia";
import { syncChip, type ShotCardGuide } from "./cardGuide";
import { failureButton, failureSentence } from "./failureText";
import { QualityTag } from "./QualityTag";
import { takeQualityOf, type TFn } from "./qualityText";
import { StaleBadge, TransitionPicker } from "./ShotParts";
import { TakeCompare } from "./TakeCompare";
import { TakesStrip } from "./TakesStrip";
import { useCardTabStops } from "./useCardTabStops";
import type { SaveResult } from "./unsavedText";
import { useUnsavedText } from "./useUnsavedText";
import { shotSeconds } from "./planLogic";
import { canRedoClip, canRedoStill, clipOutdated } from "./remakeLogic";
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
  /** The raw error text of the last batch's failure of this shot, if it failed (shown behind Details). */
  failure: string | null;
  /** What the guide says about this shot: take quality, its failure, lip sync and cast (`shotCardGuide`). */
  guide?: ShotCardGuide;
  /** A failure's button other than Retry and Edit (make a draft, open models, install video tools). */
  onFailureAction?: (action: StoryboardFailureAction) => void;
  /** Roving tabindex: only one card in the list is a tab stop, and only its inner controls are in the Tab order. */
  tabbable: boolean;
  dragOver: boolean;
  disabled: boolean;
  /** Names of the characters and places this shot uses, shown as chips. */
  castNames?: string[];
  /** Saves the title; a promise of whether it landed lets the card keep refused text (S-O). */
  onRename: (title: string) => SaveResult;
  onPickTake: (takeId: string) => void;
  onToggleLock: () => void;
  onToggleRedo: () => void;
  /** Marks the shot for the final pass (S6), or takes it out. */
  onToggleKept?: () => void;
  onTransition: (next: StoryboardTransition) => void;
  onMove: (step: -1 | 1) => void;
  onEdit: () => void;
  onKeep: () => void;
  onRerun: () => void;
  onRetry: () => void;
  /** Makes only the clip of a still-only take, from its still. */
  onMakeClip?: () => void;
  /** Redo clip only (S11): a new take that keeps the active still and has a new clip (new seed). */
  onRedoClip?: () => void;
  /** Redo still only (S11): a new still; the old clip stays (marked stale) until it is remade. */
  onRedoStill?: () => void;
  /** Adds a cast entry whose picture is this shot's active still. */
  onUseAsReference?: () => void;
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
  const unsavedTitle = useUnsavedText(`card:${shot.id}:title`, {
    text: title,
    dirty: title !== shot.title,
    send: (text) => props.onRename(text),
    restore: setTitle,
  });
  const tf = t as unknown as TFn;
  const [playing, setPlaying] = useState(false);
  const [comparing, setComparing] = useState(false);
  const cardRef = useRef<HTMLElement>(null);
  useCardTabStops(cardRef, tabbable);
  const take = activeTakeOf(shot);
  const thumb = thumbnailOf(shot);
  const number = index + 1;
  const guide = props.guide;
  const takeQuality = take ? takeQualityOf(take, guide?.takes?.[take.id]) : null;
  const sync = syncChip(tf, guide?.sync);
  const failButton = failureButton(tf, guide?.failure, {
    draftSize: guide?.draftSize,
    canHandle: () => Boolean(props.onFailureAction),
  });
  const rawFailure = failure || guide?.failure?.error || null;
  const showStale = status === "stale" && !kept;
  const outdated = status === "stale" && clipOutdated(shot);
  // Every way to make this shot again lives in the menu, next to each other (S6, S11): the card keeps one
  // pass action (Choose for final) so the row fits one line.
  const redoItems = [
    ...(shot.takes.length > 1
      ? [{ id: "compare", label: t("board.compare.open", { defaultValue: "Compare takes" }), onSelect: () => setComparing(true) }]
      : []),
    {
      id: "new-take",
      label: shot.redo
        ? t("board.shot.menu.unmarkNewTake", { defaultValue: "Cancel the new take" })
        : t("board.shot.menu.markNewTake", { defaultValue: "Make a new take next run" }),
      onSelect: props.onToggleRedo,
      disabled,
    },
    ...(props.onRedoClip && canRedoClip(shot)
      ? [{ id: "redo-clip", label: t("board.remake.clip", { defaultValue: "Redo clip only" }), onSelect: props.onRedoClip, disabled }]
      : []),
    ...(props.onRedoStill && canRedoStill(shot)
      ? [{ id: "redo-still", label: t("board.remake.still", { defaultValue: "Redo still only" }), onSelect: props.onRedoStill, disabled }]
      : []),
  ];

  useEffect(() => setTitle(shot.title), [shot.title]);
  useEffect(() => setPlaying(false), [shot.activeTake]);

  const statusText = (() => {
    switch (status) {
      case "queued":
        return t("board.shot.status.queued", { defaultValue: "Queued · {position}", position: queuePosition ?? 1 });
      case "running":
        return t("board.shot.status.running", { defaultValue: "Making…" });
      case "done":
        // A take with a still and no clip is not finished: the film leaves it out.
        if (take && !take.clip) return t("board.shot.status.stillOnly", { defaultValue: "Still only" });
        return take?.measuredSeconds
          ? t("board.shot.status.done", { defaultValue: "Done · {seconds} s", seconds: shotSeconds(take.measuredSeconds) })
          : t("board.shot.status.doneNoLength", { defaultValue: "Done" });
      case "failed":
        return t("board.shot.status.failed", { defaultValue: "Failed" });
      case "stale":
        if (clipOutdated(shot)) return t("board.remake.outdatedStatus", { defaultValue: "New still · clip not remade yet" });
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
      ref={cardRef}
      className={`${shotCardClassName(status, shot.locked)}${dragOver ? " shot-card--drop" : ""}${shot.redo ? " shot-card--redo" : ""}`}
      data-shot-id={shot.id}
      // The number is drawn aria-hidden and the title is an input: name the card "Shot 3 · Title" so a screen
      // reader says which shot it is, also while the title is empty.
      aria-label={
        shot.title
          ? `${t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })} · ${shot.title}`
          : t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })
      }
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
          title={title || undefined}
          maxLength={200}
          placeholder={t("board.shot.untitled", { defaultValue: "Shot {n}", n: number })}
          aria-label={t("board.shot.titleLabel", { defaultValue: "Shot {n} title", n: number })}
          disabled={disabled}
          onChange={(event) => setTitle(event.target.value)}
          onBlur={() => {
            if (title !== shot.title) unsavedTitle.save(title.trim());
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
      {unsavedTitle.notice}

      <div className="shot-card__body">
        <div className="shot-card__media">
          {/* Every draft picture or clip carries its size, also while the clip plays. */}
          <QualityTag quality={takeQuality} target={guide?.target} />
          {playing && take?.clip ? (
            <ProjectVideo projectId={projectId} rel={take.clip} version={take.hashes.clip} className="shot-card__video" controls autoPlay />
          ) : thumb ? (
            <>
              {thumb.kind === "still" ? (
                <ProjectImage projectId={projectId} rel={thumb.path} version={take?.hashes.still} className="shot-card__thumb" thumb={512} />
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
          <p className={`shot-card__prompt${shot.stillPrompt ? "" : " shot-card__prompt--empty"}`} title={shot.stillPrompt || undefined}>
            {shot.stillPrompt || t("board.shot.noPrompt", { defaultValue: "No prompt yet. Press Enter or Edit prompts." })}
          </p>
          {props.castNames && props.castNames.length > 0 ? (
            <ul className="shot-card__cast" aria-label={t("board.shot.castLabel", { defaultValue: "Cast in this shot" })}>
              {props.castNames.map((name, i) => (
                <li key={`${name}-${i}`} className="shot-card__chip">{name}</li>
              ))}
            </ul>
          ) : null}
          {sync || guide?.noCharacter ? (
            <ul className="shot-card__tags" aria-label={t("board.shot.tagsLabel", { defaultValue: "Lip sync and cast" })}>
              {sync ? (
                <li className={`shot-card__chip shot-card__chip--${sync.tone}`} title={sync.title}>{sync.text}</li>
              ) : null}
              {guide?.noCharacter ? (
                <li
                  className="shot-card__chip shot-card__chip--quiet"
                  title={t("board.chip.noCharacterTitle", { defaultValue: "No character is in this shot, so its picture is made without one. Open the shot to add one." })}
                >
                  {t("board.chip.noCharacter", { defaultValue: "No character" })}
                </li>
              ) : null}
            </ul>
          ) : null}
          {shot.seed != null ? (
            <p className="shot-card__meta">{t("board.shot.seed", { defaultValue: "seed {seed}", seed: shot.seed })}</p>
          ) : null}
          {shot.redo ? (
            <span className="badge accent" title={t("board.shot.newTakeTitle", { defaultValue: "The next Make adds a new take of this shot. Nothing is overwritten." })}>
              {t("board.shot.newTakeMarked", { defaultValue: "New take next run" })}
            </span>
          ) : null}
          {shot.locked ? <span className="badge muted">{t("board.shot.status.locked", { defaultValue: "Locked" })}</span> : null}
          {props.onMakeClip && take?.still && !take.clip && status === "done" ? (
            <button
              type="button"
              className="secondary-button shot-card__make-clip"
              disabled={disabled}
              title={t("board.shot.makeClipTitle", { defaultValue: "Make the clip of this shot from its still." })}
              onClick={props.onMakeClip}
            >
              {t("board.shot.makeClip", { defaultValue: "Make clip" })}
            </button>
          ) : null}
        </div>
      </div>

      {status === "running" || status === "queued" ? (
        <div className={`shot-card__bar shot-card__bar--${status}`} role="progressbar" aria-label={statusText} />
      ) : null}

      {showStale ? (
        <div className="shot-card__stale">
          <StaleBadge
            reason={outdated ? t("board.remake.outdatedBadge", { defaultValue: "Clip is behind" }) : statusText}
            title={
              outdated
                ? t("board.remake.outdatedTitle", { defaultValue: "This take has a new still; its clip was made from the old one. Redo the clip to match." })
                : staleAfter != null
                ? t("board.shot.staleTitle", {
                    defaultValue: "Shot {n} got a new take; this shot starts from its last frame.",
                    n: staleAfter + 1,
                  })
                : t("board.shot.staleTitleGeneric", { defaultValue: "Something this shot depends on changed." })
            }
          />
          {outdated && props.onRedoClip ? (
            <button type="button" className="secondary-button" disabled={disabled} onClick={props.onRedoClip}>
              {t("board.remake.outdatedAction", { defaultValue: "Redo clip" })}
            </button>
          ) : (
            <button type="button" className="secondary-button" disabled={disabled} onClick={props.onRerun}>
              {t("board.shot.rerun", { defaultValue: "Re-run" })}
            </button>
          )}
          {/* Only hides the warning: nothing is remade and the shot is not chosen for final. */}
          <button
            type="button"
            className="secondary-button"
            title={t("board.shot.hideStaleTitle", { defaultValue: "Hides this warning. The shot stays as it is and nothing is made again." })}
            onClick={props.onKeep}
          >
            {t("board.shot.hideStale", { defaultValue: "Hide this warning" })}
          </button>
        </div>
      ) : null}

      {status === "failed" ? (
        <div className="callout error shot-card__failed" role="alert">
          <p>{failureSentence(tf, guide?.failure)}</p>
          <div className="shot-card__failed-actions">
            <button
              type="button"
              className="secondary-button"
              title={failButton.title}
              disabled={disabled || (shot.locked && !["edit-shot", "open-models", "install-video-tools"].includes(failButton.action))}
              onClick={() => {
                if (failButton.action === "retry") props.onRetry();
                else if (failButton.action === "edit-shot") props.onEdit();
                else props.onFailureAction?.(failButton.action);
              }}
            >
              {failButton.label}
            </button>
          </div>
          {rawFailure ? (
            <details className="debug-details">
              <summary>{t("board.failure.details", { defaultValue: "Details" })}</summary>
              <pre>{rawFailure}</pre>
            </details>
          ) : null}
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
        onCompare={() => setComparing(true)}
        tabStops={tabbable}
        quality={guide?.takes}
        target={guide?.target}
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
        {/* Choose for final picks a take worth redoing at full quality: nothing to choose before the first take, or
            when the active take is already a final one (unless the shot is marked, so the mark can be taken off). */}
        {props.onToggleKept && (take || shot.kept) && !(takeQuality && !takeQuality.draft && !shot.kept) ? (
          <button
            type="button"
            className={`secondary-button${shot.kept ? " is-active" : ""}`}
            aria-pressed={Boolean(shot.kept)}
            title={t("board.shot.chooseForFinalTitle", {
              defaultValue: "The final pass makes the shots you choose again with the quality model, at the film's full size.",
            })}
            disabled={disabled}
            onClick={props.onToggleKept}
          >
            {t("board.shot.chooseForFinal", { defaultValue: "Choose for final" })}
          </button>
        ) : null}
        {index > 0 ? <TransitionPicker value={shot.transition} onChange={props.onTransition} disabled={disabled} /> : null}
        <ActionMenu
          label={t("board.shot.moreActions", { defaultValue: "More actions for shot {n}", n: number })}
          items={[
            { id: "edit", label: t("board.shot.menu.edit", { defaultValue: "Edit prompts" }), onSelect: props.onEdit },
            { id: "import", label: t("board.shot.menu.import", { defaultValue: "Import footage" }), onSelect: props.onEdit },
            ...redoItems,
            ...(props.onUseAsReference && take?.still
              ? [{ id: "reference", label: t("board.shot.menu.useAsReference", { defaultValue: "Use still as reference" }), onSelect: props.onUseAsReference }]
              : []),
            { id: "up", label: t("board.shot.menu.up", { defaultValue: "Move up" }), onSelect: () => props.onMove(-1), disabled: index === 0 },
            { id: "down", label: t("board.shot.menu.down", { defaultValue: "Move down" }), onSelect: () => props.onMove(1), disabled: index === count - 1 },
            { id: "delete", label: t("board.shot.menu.delete", { defaultValue: "Delete" }), onSelect: props.onRemove, danger: true },
          ]}
        />
      </div>
      {comparing ? (
        <TakeCompare
          projectId={projectId}
          shot={shot}
          number={number}
          disabled={disabled}
          onPick={(takeId) => props.onPickTake(takeId)}
          onClose={() => setComparing(false)}
          quality={guide?.takes}
          target={guide?.target}
        />
      ) : null}
    </article>
  );
}
