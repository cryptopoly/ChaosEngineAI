/**
 * Pure rules of the freestyle board (S-M2), kept here so they are tested without a screen:
 *
 * - Cards: one per shot (its active take's still) and one per character, place or style (its first picture).
 * - Layout: a card with a saved position sits there; one without is laid out in a grid, shots first in film order,
 *   references in a row below them.
 * - "Order these": reading order, left to right then top to bottom (cards whose tops are within half a card of each
 *   other are one row). With two or more shots selected, only those shots change places, in the slots they already
 *   hold in the film; otherwise every shot is ordered.
 * - Keyboard: arrows move the selected cards by STEP board pixels (shift: BIG_STEP); with nothing selected under the
 *   focus, arrows move the focus to the nearest card that way.
 * - View: screen = board * zoom + offset; zoom stays within ZOOM_MIN and ZOOM_MAX and zooms about a point.
 */
import type { StoryboardBoardPos, StoryboardProject, StoryboardReferenceKind } from "../../../types";
import { thumbnailOf } from "../storyboardLogic";

export const CARD_W = 176;
export const CARD_H = 140;
export const GAP = 24;
export const COLUMNS = 6;
export const STEP = 8;
export const BIG_STEP = 64;
export const ZOOM_MIN = 0.25;
export const ZOOM_MAX = 2.5;

export type Pos = StoryboardBoardPos;

export interface BoardCardModel {
  id: string;
  kind: "shot" | StoryboardReferenceKind;
  /** 1-based place in the film, shots only. */
  number: number | null;
  title: string;
  /** Project-relative picture to show, or null. */
  image: string | null;
  /** The picture is a clip (a shot with no still). */
  imageIsClip: boolean;
  locked: boolean;
}

export function boardCards(project: StoryboardProject): BoardCardModel[] {
  const shots: BoardCardModel[] = project.shots.map((shot, index) => {
    const thumb = thumbnailOf(shot);
    return {
      id: shot.id,
      kind: "shot",
      number: index + 1,
      title: shot.title,
      image: thumb?.path ?? null,
      imageIsClip: thumb?.kind === "clip",
      locked: shot.locked,
    };
  });
  const refs = project.references;
  const groups: [StoryboardReferenceKind, typeof refs.characters][] = [
    ["character", refs.characters],
    ["place", refs.places],
    ["style", refs.styles ?? []],
  ];
  const others = groups.flatMap(([kind, list]) =>
    list.map((ref) => ({ id: ref.id, kind, number: null, title: ref.name, image: ref.images[0]?.path ?? null, imageIsClip: false, locked: false })),
  );
  return [...shots, ...others];
}

function slot(index: number, rowOffset = 0): Pos {
  return { x: (index % COLUMNS) * (CARD_W + GAP), y: (Math.floor(index / COLUMNS) + rowOffset) * (CARD_H + GAP) };
}

/** Every card's position: the saved one, or a grid place (shots in film order, then references a row below). */
export function layout(cards: BoardCardModel[], saved: Record<string, Pos>): Record<string, Pos> {
  const out: Record<string, Pos> = {};
  const shots = cards.filter((c) => c.kind === "shot");
  const refs = cards.filter((c) => c.kind !== "shot");
  const shotRows = Math.ceil(shots.length / COLUMNS);
  shots.forEach((card, i) => (out[card.id] = saved[card.id] ?? slot(i)));
  refs.forEach((card, i) => (out[card.id] = saved[card.id] ?? slot(i, shotRows + (shotRows > 0 ? 0.5 : 0))));
  return out;
}

/** Ids in reading order: rows top to bottom (tops within half a card are one row), each row left to right. */
export function readingOrder(ids: string[], pos: Record<string, Pos>): string[] {
  const items = ids.filter((id) => pos[id]).map((id) => ({ id, ...pos[id] }));
  items.sort((a, b) => a.y - b.y || a.x - b.x);
  const rows: (typeof items)[] = [];
  for (const item of items) {
    const row = rows[rows.length - 1];
    if (row && Math.abs(item.y - row[0].y) < CARD_H / 2) row.push(item);
    else rows.push([item]);
  }
  return rows.flatMap((row) => row.sort((a, b) => a.x - b.x || a.y - b.y).map((item) => item.id));
}

/**
 * The film's new shot order. With two or more of `selected` being shots, those shots take the slots they hold now in
 * reading order and every other shot stays put; otherwise all shots go in reading order.
 */
export function orderShots(shotIds: string[], selected: string[], pos: Record<string, Pos>): string[] {
  const chosen = selected.filter((id) => shotIds.includes(id));
  if (chosen.length < 2) return readingOrder(shotIds, pos);
  const slots = shotIds.map((id, i) => (chosen.includes(id) ? i : -1)).filter((i) => i >= 0);
  const ordered = readingOrder(chosen, pos);
  const next = [...shotIds];
  slots.forEach((slotIndex, i) => (next[slotIndex] = ordered[i]));
  return next;
}

export function sameOrder(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((id, i) => id === b[i]);
}

export type ArrowKey = "ArrowLeft" | "ArrowRight" | "ArrowUp" | "ArrowDown";

export function isArrow(key: string): key is ArrowKey {
  return key === "ArrowLeft" || key === "ArrowRight" || key === "ArrowUp" || key === "ArrowDown";
}

/** The board-pixel offset of one arrow press. */
export function arrowDelta(key: ArrowKey, big: boolean): { dx: number; dy: number } {
  const step = big ? BIG_STEP : STEP;
  if (key === "ArrowLeft") return { dx: -step, dy: 0 };
  if (key === "ArrowRight") return { dx: step, dy: 0 };
  if (key === "ArrowUp") return { dx: 0, dy: -step };
  return { dx: 0, dy: step };
}

export const EXTENT = 100000;

export function clampPos(p: Pos): Pos {
  const clamp = (v: number) => Math.max(-EXTENT, Math.min(EXTENT, Math.round(v * 10) / 10));
  return { x: clamp(p.x), y: clamp(p.y) };
}

/** The nearest card from `from` in the arrow's direction (centre to centre), or null. */
export function nearestCard(from: string, key: ArrowKey, pos: Record<string, Pos>): string | null {
  const origin = pos[from];
  if (!origin) return null;
  let best: string | null = null;
  let bestScore = Infinity;
  for (const [id, p] of Object.entries(pos)) {
    if (id === from) continue;
    const dx = p.x - origin.x;
    const dy = p.y - origin.y;
    const along = key === "ArrowRight" ? dx : key === "ArrowLeft" ? -dx : key === "ArrowDown" ? dy : -dy;
    const across = key === "ArrowRight" || key === "ArrowLeft" ? Math.abs(dy) : Math.abs(dx);
    if (along <= 0) continue;
    const score = along + across * 2;
    if (score < bestScore) {
      bestScore = score;
      best = id;
    }
  }
  return best;
}

export interface View {
  x: number;
  y: number;
  zoom: number;
}

export function clampZoom(zoom: number): number {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom));
}

/** The view zoomed by `factor` keeping the screen point (cx, cy) over the same board point. */
export function zoomAbout(view: View, factor: number, cx: number, cy: number): View {
  const zoom = clampZoom(view.zoom * factor);
  const k = zoom / view.zoom;
  return { zoom, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k };
}

/** Two fingers on the board: where they were and where they are (midpoint in viewport pixels, finger distance). */
export interface PinchState {
  mid: Pos;
  distance: number;
}

/**
 * One step of a two-finger gesture: zoom by the change in finger distance and pan with the midpoint, so the board
 * point that was under the fingers stays under them. Moving both fingers together pans without zooming (S-P: the
 * board zoomed about the midpoint only, so a two-finger pan did not pan and drifted the zoom).
 */
export function pinchStep(view: View, prev: PinchState, next: PinchState): View {
  if (prev.distance <= 0 || next.distance <= 0) return view;
  const zoom = clampZoom(view.zoom * (next.distance / prev.distance));
  const bx = (prev.mid.x - view.x) / view.zoom;
  const by = (prev.mid.y - view.y) / view.zoom;
  return { zoom, x: next.mid.x - bx * zoom, y: next.mid.y - by * zoom };
}

/** A view that shows every card in a `width` x `height` viewport with a margin; zoom at most 1. */
export function fitView(pos: Pos[], width: number, height: number, margin = 24): View {
  if (pos.length === 0 || width <= 0 || height <= 0) return { x: margin, y: margin, zoom: 1 };
  const minX = Math.min(...pos.map((p) => p.x));
  const minY = Math.min(...pos.map((p) => p.y));
  const maxX = Math.max(...pos.map((p) => p.x + CARD_W));
  const maxY = Math.max(...pos.map((p) => p.y + CARD_H));
  const zoom = clampZoom(Math.min(1, (width - margin * 2) / (maxX - minX), (height - margin * 2) / (maxY - minY)));
  return { zoom, x: (width - (maxX - minX) * zoom) / 2 - minX * zoom, y: margin - minY * zoom };
}

/** Screen point to board point. */
export function toBoard(view: View, sx: number, sy: number): Pos {
  return { x: (sx - view.x) / view.zoom, y: (sy - view.y) / view.zoom };
}

/** The file name without folder or extension, as a title for a dropped picture. */
export function titleFromFile(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "";
  return base.replace(/\.[A-Za-z0-9]{1,5}$/, "").replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 200);
}
