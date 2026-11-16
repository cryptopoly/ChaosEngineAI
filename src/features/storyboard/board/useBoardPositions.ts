/**
 * Card positions on the freestyle board (S-M2): the saved ones (`project.board.pos`), a grid place for the rest, and
 * moves in progress on top. A drag or a run of arrow presses moves the cards here at once and is saved with one
 * `place_cards` op when it ends (arrows: half a second after the last press), so each move is one Undo step and the
 * revision check holds. When the op lands or fails, the moved cards show what the project says.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { StoryboardProject } from "../../../types";
import type { UseStoryboardResult } from "../useStoryboard";
import { boardCards, clampPos, layout, type BoardCardModel, type Pos } from "./boardLogic";

export const KEY_SAVE_MS = 500;

export interface BoardPositions {
  cards: BoardCardModel[];
  pos: Record<string, Pos>;
  /** Move cards locally to `from[id] + (dx, dy)` (a drag passes where each card started). */
  moveTo: (from: Record<string, Pos>, dx: number, dy: number) => void;
  /** Save the cards moved so far now. */
  save: () => void;
  /** Save after KEY_SAVE_MS without another call (arrow keys). */
  saveSoon: () => void;
}

export function useBoardPositions(story: UseStoryboardResult, project: StoryboardProject): BoardPositions {
  const cards = useMemo(() => boardCards(project), [project]);
  const saved = project.board?.pos;
  const [drafts, setDrafts] = useState<Record<string, Pos>>({});
  const draftsRef = useRef(drafts);
  draftsRef.current = drafts;
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const pos = useMemo(() => {
    const placed = layout(cards, saved ?? {});
    for (const [id, p] of Object.entries(drafts)) if (placed[id]) placed[id] = p;
    return placed;
  }, [cards, saved, drafts]);

  const moveTo = useCallback((from: Record<string, Pos>, dx: number, dy: number) => {
    setDrafts((current) => {
      const next = { ...current };
      for (const [id, p] of Object.entries(from)) next[id] = clampPos({ x: p.x + dx, y: p.y + dy });
      return next;
    });
  }, []);

  const save = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    const moved = draftsRef.current;
    const ids = Object.keys(moved);
    if (ids.length === 0) return;
    const positions = Object.fromEntries(ids.map((id) => [id, moved[id]]));
    void story.op("place_cards", { positions }).finally(() => {
      // Drop only the drafts this save carried, unless the card moved again meanwhile.
      setDrafts((current) => {
        const next = { ...current };
        for (const id of ids) if (next[id] === moved[id]) delete next[id];
        return next;
      });
    });
  }, [story]);

  const saveSoon = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(save, KEY_SAVE_MS);
  }, [save]);

  // Leaving the tab with an arrow move waiting saves it rather than dropping it.
  const saveRef = useRef(save);
  saveRef.current = save;
  useEffect(() => () => {
    if (timer.current) saveRef.current();
  }, []);

  return { cards, pos, moveTo, save, saveSoon };
}
