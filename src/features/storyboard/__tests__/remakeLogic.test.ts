import { describe, expect, it } from "vitest";
import type { StoryboardShot, StoryboardTake } from "../../../types";
import {
  canRedoClip,
  canRedoStill,
  clipOutdated,
  compareRows,
  compareSummary,
  compareView,
  defaultComparePair,
  syncTime,
  takeFacts,
} from "../remakeLogic";

const H1 = "a".repeat(64);
const H2 = "b".repeat(64);
const C1 = "c".repeat(64);
const C2 = "d".repeat(64);

function take(id: string, extra: Partial<StoryboardTake> = {}): StoryboardTake {
  return {
    id, still: "media/s.png", clip: "media/c.mp4", hashes: { still: H1, clip: C1 }, settings: {}, measuredSeconds: 2,
    createdAt: 0, origin: "batch", basis: null, ...extra,
  };
}

function shot(extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id: "s1", order: 0, title: "", status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 2, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

const withTakes = (takes: StoryboardTake[], active = takes[0]?.id, extra: Partial<StoryboardShot> = {}) =>
  shot({ takes, activeTake: active ?? null, ...extra });

describe("redo one half of a take", () => {
  it("sees a clip made from an older still, and never on takes that do not record it", () => {
    expect(clipOutdated(withTakes([take("t1", { settings: { clipStill: H1 } })]))).toBe(false);
    expect(clipOutdated(withTakes([take("t1", { hashes: { still: H2, clip: C1 }, settings: { clipStill: H1 } })]))).toBe(true);
    expect(clipOutdated(withTakes([take("t1")]))).toBe(false);
    expect(clipOutdated(withTakes([take("t1", { clip: null, settings: { clipStill: H2 } })]))).toBe(false);
  });

  it("offers Redo clip only with a still to keep (or a shot that needs none), Redo still only on still shots", () => {
    expect(canRedoClip(withTakes([take("t1")]))).toBe(true);
    expect(canRedoClip(withTakes([take("t1", { still: null })]))).toBe(false);
    expect(canRedoClip(withTakes([take("t1", { still: null })], "t1", { startFrom: "text" }))).toBe(true);
    expect(canRedoClip(withTakes([take("t1")], "t1", { startFrom: "upload" }))).toBe(false);
    expect(canRedoClip(shot())).toBe(false);
    expect(canRedoStill(withTakes([take("t1")]))).toBe(true);
    expect(canRedoStill(withTakes([take("t1")], "t1", { startFrom: "previous-shot" }))).toBe(false);
    expect(canRedoStill(shot())).toBe(false);
  });
});

describe("take compare", () => {
  it("reads the still and clip settings apart, and an older take's single seed as the clip's", () => {
    const facts = takeFacts(take("t1", {
      renderPass: "draft",
      settings: { stillModel: "Iris", stillSeed: 11, clipModel: "LTX", clipSeed: 22, width: 854, height: 480, steps: 4 },
    }));
    expect(facts).toMatchObject({ pass: "draft", stillModel: "Iris", stillSeed: "11", clipModel: "LTX", clipSeed: "22", size: "854 x 480", length: "2.0" });
    const old = takeFacts(take("t2", { settings: { model: "LTX", seed: 5 } }));
    expect(old).toMatchObject({ pass: "final", stillSeed: null, clipSeed: "5", clipModel: "LTX" });
    const stillOnly = takeFacts(take("t3", { clip: null, settings: { model: "Iris", seed: 9 } }));
    expect(stillOnly).toMatchObject({ stillModel: "Iris", stillSeed: "9", clipSeed: null, clipModel: null });
  });

  it("marks the rows that differ and leaves out facts neither take records", () => {
    const a = take("a", { settings: { clipSeed: 1, clipModel: "LTX" } });
    const b = take("b", { renderPass: "draft", settings: { clipSeed: 2, clipModel: "LTX" } });
    const rows = compareRows(a, b);
    expect(rows.map((r) => r.key)).toEqual(["pass", "clipModel", "clipSeed", "length"]);
    expect(rows.filter((r) => r.differs).map((r) => r.key)).toEqual(["pass", "clipSeed"]);
  });

  it("names how two takes relate", () => {
    expect(compareSummary(take("a"), take("b", { hashes: { still: H1, clip: C2 } }))).toBe("sameStill");
    expect(compareSummary(take("a"), take("b", { hashes: { still: H2, clip: C1 } }))).toBe("sameClip");
    expect(compareSummary(take("a"), take("b", { hashes: { still: H2, clip: C2 } }))).toBe("different");
    expect(compareSummary(take("a"), take("b"))).toBe("identical");
  });

  it("opens on the active take and the newest other one", () => {
    const takes = [take("t1", { createdAt: 1 }), take("t2", { createdAt: 3 }), take("t3", { createdAt: 2 })];
    expect(defaultComparePair(withTakes(takes, "t1"))).toEqual(["t1", "t2"]);
    expect(defaultComparePair(withTakes(takes, "t2"))).toEqual(["t2", "t3"]);
    expect(defaultComparePair(withTakes([take("t1")]))).toBeNull();
  });

  it("shows clips when both have one, else stills", () => {
    expect(compareView(take("a"), take("b"))).toBe("clip");
    expect(compareView(take("a"), take("b", { clip: null }))).toBe("still");
    expect(compareView(take("a", { still: null }), take("b", { clip: null }))).toBe("clip");
  });

  it("keeps the second clip with the first, within a little drift, and never past its end", () => {
    expect(syncTime(1.0, 1.05, 4)).toBeNull();
    expect(syncTime(1.0, 1.5, 4)).toBe(1.0);
    expect(syncTime(3.5, 1.0, 2)).toBe(2);
    expect(syncTime(1.0, 0, Number.NaN)).toBe(1.0);
  });
});
