/**
 * Pointer and keyboard input on the freestyle board's cards (S-M2).
 *
 * Pointer, pen and touch: press a card to select it (Shift, Cmd or Ctrl, or the "Select several" switch, adds or
 * removes it) and drag to move every selected card; a press without a drag on a card of a larger selection selects
 * only that card. A press on the empty board clears the selection (unless adding) and pans.
 *
 * Keyboard (roving tab stop: Tab reaches the current card once): arrows move the selected cards by a step, Shift +
 * arrows by a larger step; on a card that is not selected, arrows move the focus to the nearest card that way.
 * Space adds or removes the focused card, Enter selects only it, Home and End go to the first and last card, Cmd or
 * Ctrl + A selects every card, Escape clears the selection.
 */
import { useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type RefObject } from "react";
import { arrowDelta, isArrow, nearestCard, type BoardCardModel, type Pos } from "./boardLogic";
import type { BoardPositions } from "./useBoardPositions";
import type { BoardViewControls } from "./useBoardView";

const DRAG_SLOP = 4;

export interface BoardSelection {
  selected: string[];
  setSelected: (ids: string[]) => void;
  current: string | null;
  setCurrent: (id: string | null) => void;
}

interface Drag {
  pointerId: number;
  id: string;
  sx: number;
  sy: number;
  from: Record<string, Pos>;
  moved: boolean;
  additive: boolean;
}

export function useBoardInput(
  viewport: RefObject<HTMLElement | null>,
  cards: BoardCardModel[],
  positions: BoardPositions,
  view: BoardViewControls,
  selection: BoardSelection,
  multi: boolean,
) {
  const drag = useRef<Drag | null>(null);
  const [dragging, setDragging] = useState(false);
  const { pos } = positions;
  const { selected, setSelected, setCurrent } = selection;

  const focusCard = (id: string) => {
    setCurrent(id);
    if (pos[id]) view.reveal(pos[id]);
    // At once, not on the next frame: a fast second key press must reach the card it moved to.
    viewport.current?.querySelector<HTMLElement>(`[data-card-id="${CSS.escape(id)}"]`)?.focus({ preventScroll: true });
  };

  const onCardPointerDown = (event: ReactPointerEvent<HTMLDivElement>, id: string) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.stopPropagation();
    const additive = multi || event.shiftKey || event.metaKey || event.ctrlKey;
    let next = selected;
    if (additive) next = selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id];
    else if (!selected.includes(id)) next = [id];
    setSelected(next);
    setCurrent(id);
    const moving = next.includes(id) ? next : [];
    drag.current = {
      pointerId: event.pointerId,
      id,
      sx: event.clientX,
      sy: event.clientY,
      from: Object.fromEntries(moving.filter((m) => pos[m]).map((m) => [m, pos[m]])),
      moved: false,
      additive,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLElement>) => {
    const d = drag.current;
    if (!d || d.pointerId !== event.pointerId) {
      view.onPanMove(event);
      return;
    }
    const dx = event.clientX - d.sx;
    const dy = event.clientY - d.sy;
    if (!d.moved && Math.hypot(dx, dy) < DRAG_SLOP) return;
    if (!d.moved) {
      d.moved = true;
      setDragging(true);
    }
    positions.moveTo(d.from, dx / view.view.zoom, dy / view.view.zoom);
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLElement>) => {
    const d = drag.current;
    if (!d || d.pointerId !== event.pointerId) {
      view.onPanEnd(event);
      return;
    }
    drag.current = null;
    setDragging(false);
    if (d.moved) positions.save();
    else if (!d.additive && selected.length > 1) setSelected([d.id]);
  };

  const onBackgroundPointerDown = (event: ReactPointerEvent<HTMLElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    if (!(multi || event.shiftKey || event.metaKey || event.ctrlKey) && selected.length > 0) setSelected([]);
    view.onPanStart(event);
  };

  const onCardKeyDown = (event: KeyboardEvent<HTMLDivElement>, id: string) => {
    const key = event.key;
    if (isArrow(key)) {
      event.preventDefault();
      if (selected.includes(id)) {
        const { dx, dy } = arrowDelta(key, event.shiftKey);
        const from = Object.fromEntries(selected.filter((s) => pos[s]).map((s) => [s, pos[s]]));
        positions.moveTo(from, dx, dy);
        positions.saveSoon();
        view.reveal({ x: pos[id].x + dx, y: pos[id].y + dy });
        return;
      }
      const next = nearestCard(id, key, pos);
      if (next) focusCard(next);
      return;
    }
    if (key === " " || key === "Spacebar") {
      event.preventDefault();
      setSelected(selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id]);
    } else if (key === "Enter") {
      event.preventDefault();
      setSelected([id]);
    } else if (key === "Escape") {
      if (selected.length > 0) {
        event.preventDefault();
        event.stopPropagation();
        setSelected([]);
      }
    } else if (key === "Home" || key === "End") {
      event.preventDefault();
      const target = key === "Home" ? cards[0] : cards[cards.length - 1];
      if (target) focusCard(target.id);
    } else if ((key === "a" || key === "A") && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      setSelected(cards.map((c) => c.id));
    }
  };

  return { dragging, dragId: dragging ? drag.current?.id ?? null : null, onCardPointerDown, onPointerMove, onPointerUp, onBackgroundPointerDown, onCardKeyDown };
}
