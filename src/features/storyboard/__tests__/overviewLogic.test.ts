import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../types";
import { clipCount, recentActivity, stillOnlyCount } from "../overviewLogic";

function shot(id: string, take?: { still?: string; clip?: string }): StoryboardShot {
  const takes = take ? [{ id: `${id}-t`, still: take.still ?? null, clip: take.clip ?? null, hashes: {}, settings: {}, measuredSeconds: 4, createdAt: 0, origin: "upload" as const, basis: null }] : [];
  return {
    id, order: 0, title: "", status: take ? "done" : "idle", stillPrompt: "", motionPrompt: "", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes, activeTake: take ? `${id}-t` : null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
  };
}

function project(shots: StoryboardShot[], extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return { shots, audio: [], references: { characters: [], places: [] }, ...extra } as unknown as StoryboardProject;
}

describe("Recent activity", () => {
  it("lists the last finished batches newest first with what they made", () => {
    const batch = (id: string, state: string, at: number, results: { shotId: string; state: string }[]) =>
      ({ id, state, createdAt: at - 10, finishedAt: state === "running" ? null : at, results });
    const p = {
      batches: [
        batch("b1", "done", 100, [{ shotId: "a", state: "done" }, { shotId: "a", state: "done" }, { shotId: "b", state: "done" }]),
        batch("b2", "done", 200, [{ shotId: "a", state: "failed" }, { shotId: "b", state: "done" }]),
        batch("b3", "interrupted", 300, []),
        batch("b4", "running", 400, []),
        batch("b5", "paused", 500, []),
      ],
    } as unknown as StoryboardProject;
    expect(recentActivity(p).map((l) => [l.id, l.state, l.shots, l.failed])).toEqual([
      ["b3", "interrupted", 0, 0],
      ["b2", "done", 2, 1],
      ["b1", "done", 2, 0],
    ]);
  });
});

describe("Overview counts", () => {
  it("counts shots with a clip and shots with a still but no clip", () => {
    const p = project([shot("a", { still: "s.png" }), shot("b", { still: "s.png", clip: "c.mp4" }), shot("c")]);
    expect(stillOnlyCount(p)).toBe(1);
    expect(clipCount(p)).toBe(1);
  });
});