/**
 * One picture card on the freestyle board (S-M2): a shot's still (with its place in the film) or a character, place or
 * style picture. The card is one focusable button (roving tab stop: only the current card is in the Tab order);
 * selection shows as a ring and `aria-pressed`. Pointer, pen and touch go to the board's drag handlers.
 */
import { memo, useRef, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { useTranslation } from "react-i18next";
import { useProjectFileUrl } from "../ProjectMedia";
import type { BoardCardModel, Pos } from "./boardLogic";

export interface BoardCardProps {
  projectId: string;
  card: BoardCardModel;
  pos: Pos;
  selected: boolean;
  current: boolean;
  dragging: boolean;
  onPointerDown: (event: ReactPointerEvent<HTMLDivElement>, id: string) => void;
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>, id: string) => void;
  onFocus: (id: string) => void;
}

function FilmIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4" />
    </svg>
  );
}

function PictureIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <circle cx="9" cy="10" r="2" />
      <path d="m21 16-5-5-9 9" />
    </svg>
  );
}

export const BoardCard = memo(function BoardCard({ projectId, card, pos, selected, current, dragging, onPointerDown, onKeyDown, onFocus }: BoardCardProps) {
  const { t } = useTranslation("studio");
  const ref = useRef<HTMLDivElement>(null);
  const url = useProjectFileUrl(projectId, card.imageIsClip ? null : card.image, card.image, { thumb: 256, target: ref });
  const kindLabel =
    card.kind === "shot"
      ? t("board.shot.untitled", { defaultValue: "Shot {n}", n: card.number ?? 0 })
      : card.kind === "character"
        ? t("board.freestyle.kindCharacter", { defaultValue: "Character" })
        : card.kind === "place"
          ? t("board.freestyle.kindPlace", { defaultValue: "Place" })
          : t("board.freestyle.kindStyle", { defaultValue: "Style" });
  const name = card.title ? `${kindLabel} · ${card.title}` : kindLabel;
  return (
    <div
      ref={ref}
      role="button"
      tabIndex={current ? 0 : -1}
      aria-pressed={selected}
      aria-label={name}
      data-card-id={card.id}
      className={`board-card board-card--${card.kind}${selected ? " is-selected" : ""}${dragging ? " is-dragging" : ""}`}
      style={{ transform: `translate(${pos.x}px, ${pos.y}px)` }}
      onPointerDown={(event) => onPointerDown(event, card.id)}
      onKeyDown={(event) => onKeyDown(event, card.id)}
      onFocus={() => onFocus(card.id)}
    >
      <div className="board-card__picture">
        {url ? <img src={url} alt="" draggable={false} /> : <span className="board-card__empty">{card.imageIsClip ? <FilmIcon /> : <PictureIcon />}</span>}
        {card.number != null ? <span className="board-card__number" aria-hidden="true">{card.number}</span> : null}
      </div>
      <div className="board-card__label">
        <span className="board-card__kind">{card.kind === "shot" ? null : kindLabel}</span>
        <span className="board-card__title">{card.title || (card.kind === "shot" ? kindLabel : "")}</span>
      </div>
    </div>
  );
});
