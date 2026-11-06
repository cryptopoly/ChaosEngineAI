/**
 * The shared shell of a library card (Voice transcripts and speech, Music
 * tracks and sounds): select box, title, kind badge and meta line, optional
 * preview text, tags, a player, an error line, then the card's own actions.
 * What differs per library (exports, tags, waveform) comes in as slots.
 */
import type { DragEventHandler, ReactNode } from "react";

export interface MediaLibraryCardProps {
  title: string;
  /** Kind label shown before the meta line, e.g. "Transcript". */
  badge?: string;
  /** Pre-joined meta line, e.g. "2 Nov 2026 · 0:30". */
  meta: string;
  preview?: string;
  selected: boolean;
  onSelect: (on: boolean) => void;
  /** Accessible name of the select box. */
  selectLabel: string;
  tags?: string[];
  player?: ReactNode;
  error?: string | null;
  actions: ReactNode;
  draggable?: boolean;
  onDragStart?: DragEventHandler<HTMLElement>;
}

export function MediaLibraryCard({
  title,
  badge,
  meta,
  preview,
  selected,
  onSelect,
  selectLabel,
  tags,
  player,
  error,
  actions,
  draggable,
  onDragStart,
}: MediaLibraryCardProps) {
  return (
    <article
      className={`media-library-card${selected ? " media-library-card--selected" : ""}`}
      draggable={draggable}
      onDragStart={onDragStart}
    >
      <div className="media-library-card__head">
        <input type="checkbox" checked={selected} onChange={(e) => onSelect(e.target.checked)} aria-label={selectLabel} />
        <div className="media-library-card__title">
          <strong>{title}</strong>
          <span className="muted-text media-library-card__meta">
            {badge ? <span className="badge subtle">{badge}</span> : null}
            {badge ? " " : null}
            {meta}
          </span>
        </div>
      </div>
      {preview ? <p className="media-library-card__preview">{preview}</p> : null}
      {tags && tags.length > 0 ? (
        <ul className="media-library-card__tags">
          {tags.map((tag) => (
            <li key={tag} className="badge subtle">
              {tag}
            </li>
          ))}
        </ul>
      ) : null}
      {player}
      {error ? (
        <p className="media-library-card__error" role="alert">
          {error}
        </p>
      ) : null}
      <div className="button-row">{actions}</div>
    </article>
  );
}
