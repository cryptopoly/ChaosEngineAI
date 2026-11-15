import { describe, expect, it } from "vitest";
import type { StoryboardShot, StoryboardTake } from "../../../types";
import { draftShotIds, finalPassShotIds, isDraftTake, keptShotIds, stillModelPick } from "../storyboardLogic";

function take(id: string, extra: Partial<StoryboardTake> = {}): StoryboardTake {
  return { id, still: "media/s.png", clip: "media/c.mp4", hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "batch", basis: null, ...extra };
}

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: "", status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

const withTake = (id: string, t: StoryboardTake, extra: Partial<StoryboardShot> = {}) =>
  shot(id, { takes: [t], activeTake: t.id, ...extra });

describe("draft and final passes", () => {
  it("reads a take without a pass (an older project) as final", () => {
    expect(isDraftTake(take("t1"))).toBe(false);
    expect(isDraftTake(take("t1", { renderPass: "final" }))).toBe(false);
    expect(isDraftTake(take("t1", { renderPass: "draft" }))).toBe(true);
    expect(isDraftTake(null)).toBe(false);
  });

  it("lists the shots whose active take is a draft, for the export warning", () => {
    const shots = [
      withTake("a", take("t1", { renderPass: "draft" })),
      withTake("b", take("t2")),
      shot("c"),
    ];
    expect(draftShotIds(shots)).toEqual(["a"]);
  });

  it("mirrors needs_final: kept, unlocked, not an upload, and without a finished final take", () => {
    const shots = [
      withTake("draft-kept", take("t1", { renderPass: "draft" }), { kept: true }),
      withTake("final-kept", take("t2", { renderPass: "final" }), { kept: true }),
      withTake("final-still-only", take("t3", { renderPass: "final", clip: null }), { kept: true }),
      withTake("redo", take("t4"), { kept: true, redo: true }),
      withTake("not-kept", take("t5", { renderPass: "draft" })),
      withTake("locked", take("t6", { renderPass: "draft" }), { kept: true, locked: true }),
      shot("upload", { kept: true, startFrom: "upload" }),
      shot("empty", { kept: true, status: "idle" }),
    ];
    expect(finalPassShotIds(shots)).toEqual(["draft-kept", "final-still-only", "redo", "empty"]);
    expect(keptShotIds(shots)).toEqual(["draft-kept", "final-kept", "final-still-only", "redo", "locked", "upload", "empty"]);
  });
});

describe("still model pick", () => {
  const models = [
    { id: "qwen", name: "Qwen-Image 2.1", fit: "fits", territoryRestricted: false, secondsPerStill: 450 },
    { id: "iris", name: "Iris-3B Turbo", fit: "fits", territoryRestricted: false, secondsPerStill: 15.2 },
  ];

  it("names what Automatic picks and the time per still of the chosen model", () => {
    expect(stillModelPick(models, null, "iris")).toEqual({ id: "iris", name: "Iris-3B Turbo", seconds: 15.2, automatic: true });
    expect(stillModelPick(models, "qwen", "iris")).toEqual({ id: "qwen", name: "Qwen-Image 2.1", seconds: 450, automatic: false });
  });

  it("keeps a saved model that is no longer installed, without a time", () => {
    expect(stillModelPick(models, "gone", "iris")).toEqual({ id: "gone", name: "gone", seconds: null, automatic: false });
    expect(stillModelPick([], null, null)).toEqual({ id: null, name: null, seconds: null, automatic: true });
  });
});
