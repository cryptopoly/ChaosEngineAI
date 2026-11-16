import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../../types";
import {
  BIG_STEP,
  CARD_H,
  CARD_W,
  COLUMNS,
  GAP,
  STEP,
  ZOOM_MAX,
  arrowDelta,
  boardCards,
  clampPos,
  fitView,
  layout,
  nearestCard,
  orderShots,
  readingOrder,
  sameOrder,
  titleFromFile,
  toBoard,
  zoomAbout,
} from "../boardLogic";

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return { id, title: id.toUpperCase(), stillPrompt: "", motionPrompt: "", takes: [], activeTake: null, locked: false, status: "idle", ...extra } as unknown as StoryboardShot;
}

function project(shots: StoryboardShot[], board?: StoryboardProject["board"]): StoryboardProject {
  return {
    shots,
    references: { characters: [{ id: "c1", name: "Mara", description: "", images: [{ path: "media/refs/c1.png", hash: null }] }], places: [], styles: [] },
    board,
  } as unknown as StoryboardProject;
}

describe("board cards and layout", () => {
  it("makes one card per shot (film number, still) and per reference (first picture)", () => {
    const withTake = shot("b", { takes: [{ id: "t1", still: "media/b/t1/still.png", clip: null }] as never, activeTake: "t1" });
    const cards = boardCards(project([shot("a"), withTake]));
    expect(cards.map((c) => [c.id, c.kind, c.number, c.image])).toEqual([
      ["a", "shot", 1, null],
      ["b", "shot", 2, "media/b/t1/still.png"],
      ["c1", "character", null, "media/refs/c1.png"],
    ]);
  });

  it("keeps saved positions and lays out the rest: shots in a grid, references a row below", () => {
    const shots = Array.from({ length: COLUMNS + 1 }, (_, i) => shot(`s${i}`));
    const cards = boardCards(project(shots));
    const pos = layout(cards, { s1: { x: 900, y: -50 } });
    expect(pos.s0).toEqual({ x: 0, y: 0 });
    expect(pos.s1).toEqual({ x: 900, y: -50 });
    expect(pos.s2).toEqual({ x: 2 * (CARD_W + GAP), y: 0 });
    expect(pos[`s${COLUMNS}`]).toEqual({ x: 0, y: CARD_H + GAP });
    expect(pos.c1.y).toBeGreaterThan(pos[`s${COLUMNS}`].y);
  });
});

describe("Order these", () => {
  const pos = {
    a: { x: 400, y: 0 },
    b: { x: 0, y: 10 }, // same row as a (within half a card), to its left
    c: { x: 0, y: 300 }, // next row
    d: { x: 200, y: 290 },
  };

  it("reads left to right, then top to bottom", () => {
    expect(readingOrder(["a", "b", "c", "d"], pos)).toEqual(["b", "a", "c", "d"]);
  });

  it("orders every shot when fewer than two are selected", () => {
    expect(orderShots(["a", "b", "c", "d"], [], pos)).toEqual(["b", "a", "c", "d"]);
    expect(orderShots(["a", "b", "c", "d"], ["c"], pos)).toEqual(["b", "a", "c", "d"]);
  });

  it("with a selection, only the selected shots swap, inside the slots they hold", () => {
    // d and a are selected: they hold slots 0 and 3; by place a (row 1) comes before d (row 2).
    expect(orderShots(["d", "b", "c", "a"], ["a", "d", "c1"], pos)).toEqual(["a", "b", "c", "d"]);
    expect(sameOrder(["a", "b"], ["a", "b"])).toBe(true);
    expect(sameOrder(["a", "b"], ["b", "a"])).toBe(false);
  });
});

describe("keyboard", () => {
  it("moves by a step, or a bigger step with Shift", () => {
    expect(arrowDelta("ArrowRight", false)).toEqual({ dx: STEP, dy: 0 });
    expect(arrowDelta("ArrowUp", true)).toEqual({ dx: 0, dy: -BIG_STEP });
  });

  it("finds the nearest card that way, preferring the same row", () => {
    const pos = { a: { x: 0, y: 0 }, b: { x: 200, y: 0 }, c: { x: 150, y: 200 }, d: { x: -300, y: 0 } };
    expect(nearestCard("a", "ArrowRight", pos)).toBe("b");
    expect(nearestCard("a", "ArrowDown", pos)).toBe("c");
    expect(nearestCard("a", "ArrowLeft", pos)).toBe("d");
    expect(nearestCard("a", "ArrowUp", pos)).toBeNull();
  });

  it("keeps a card on the board", () => {
    expect(clampPos({ x: 1e9, y: -1e9 })).toEqual({ x: 100000, y: -100000 });
  });
});

describe("view", () => {
  it("zooms about a point, which stays over the same board point, within the limits", () => {
    const view = { x: 10, y: 20, zoom: 1 };
    const before = toBoard(view, 300, 200);
    const after = zoomAbout(view, 2, 300, 200);
    expect(toBoard(after, 300, 200)).toEqual(before);
    expect(zoomAbout(view, 100, 0, 0).zoom).toBe(ZOOM_MAX);
  });

  it("fits every card, never above 100 %", () => {
    const fit = fitView([{ x: 0, y: 0 }, { x: 3000, y: 1000 }], 800, 600);
    expect(fit.zoom).toBeLessThan(1);
    expect(fitView([{ x: 0, y: 0 }], 800, 600).zoom).toBe(1);
  });

  it("names a dropped picture after its file", () => {
    expect(titleFromFile("/Users/me/Pictures/night_drive-01.png")).toBe("night drive 01");
    expect(titleFromFile("C:\\pics\\Sunset.webp")).toBe("Sunset");
  });
});
