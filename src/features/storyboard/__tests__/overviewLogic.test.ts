import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../types";
import { checklist, checklistComplete, nextStep, preflight, recentActivity, stillOnlyCount } from "../overviewLogic";

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

const none = { previewCurrent: false, exported: false };

describe("Before you export", () => {
  const exportSettings = { loudnessLufs: -14, captionsSrt: true, captionsBurn: false, aiLabel: { embedTag: false }, upscale: "none" };

  it("lists format, upscale, clips, stale, loudness, captions, AI tag, an old preview and a cloned voice", () => {
    const p = {
      ...project([shot("a", { clip: "c.mp4" }), { ...shot("b"), status: "stale" as const }]),
      captions: [{}, {}],
      export: exportSettings,
    } as unknown as StoryboardProject;
    const lines = preflight(p, { leftOut: 1, previewStale: true, clonedVoice: true, format: "1920 × 1080 · 24 fps", upscaledTo: "3840 × 2160" });
    expect(lines.map((l) => [l.id, l.ok, l.n, l.detail ?? null])).toEqual([
      ["format", true, 0, "1920 × 1080 · 24 fps"],
      ["upscale", true, 0, "3840 × 2160"],
      ["clips", false, 1, null],
      ["stale", false, 1, null],
      ["loudness", true, -14, "-14"],
      ["captions", true, 2, "srt"],
      ["aiTag", true, 1, null], // forced on by the cloned voice
      ["preview", false, 0, null],
      ["cloned", true, 0, null],
    ]);
  });

  it("leaves out what does not apply", () => {
    const p = { ...project([shot("a", { clip: "c.mp4" })]), captions: [], export: { ...exportSettings, loudnessLufs: null } } as unknown as StoryboardProject;
    const clean = preflight(p, { leftOut: 0, previewStale: false, clonedVoice: false });
    expect(clean.map((l) => l.id)).toEqual(["clips", "loudness", "captions", "aiTag"]);
    expect(clean[0].ok).toBe(true);
    expect(clean.find((l) => l.id === "loudness")?.detail).toBe("");
    expect(clean.find((l) => l.id === "aiTag")?.n).toBe(0);
  });
});

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
      ],
    } as unknown as StoryboardProject;
    expect(recentActivity(p).map((l) => [l.id, l.state, l.shots, l.failed])).toEqual([
      ["b3", "interrupted", 0, 0],
      ["b2", "done", 2, 1],
      ["b1", "done", 2, 0],
    ]);
  });
});

describe("Overview next step", () => {
  it("walks write, make, clips, sound, preview, export, done and never asks for a cast", () => {
    expect(nextStep(project([]), none).id).toBe("writeShots");
    expect(nextStep(project([shot("a"), shot("b")]), none)).toEqual({ id: "makeMissing", tab: "storyboard", count: 2, stills: 0 });
    // A still without a clip is missing its clip: Make missing counts it (and makes only the clip).
    const stillOnly = project([shot("a", { still: "s.png" }), shot("b")]);
    expect(stillOnlyCount(stillOnly)).toBe(1);
    expect(nextStep(stillOnly, none)).toEqual({ id: "makeMissing", tab: "storyboard", count: 2, stills: 1 });
    // A locked still-only shot is not made by a batch; the step points at it.
    const locked = project([{ ...shot("a", { still: "s.png" }), locked: true }]);
    expect(nextStep(locked, none).id).toBe("makeClips");
    const made = project([shot("a", { still: "s.png", clip: "c.mp4" })]);
    expect(nextStep(made, none).id).toBe("addSound");
    expect(nextStep({ ...made, audio: [{}] } as unknown as StoryboardProject, none).id).toBe("renderPreview");
    expect(nextStep(made, { previewCurrent: true, exported: false }).id).toBe("export");
    expect(nextStep(made, { previewCurrent: true, exported: true }).id).toBe("done");
  });

  it("ticks the checklist as work happens; cast and sound are optional, so a film without them completes", () => {
    const made = project([shot("a", { clip: "c.mp4" })]);
    const rows = checklist(made, { previewCurrent: true, exported: true });
    expect(rows.find((r) => r.id === "cast")).toMatchObject({ done: false, optional: true });
    expect(rows.find((r) => r.id === "takes")?.done).toBe(true);
    expect(checklistComplete(rows)).toBe(true);
    expect(checklistComplete(checklist(made, none))).toBe(false);
    expect(checklist(project([shot("a", { still: "s.png" })]), none).find((r) => r.id === "takes")?.done).toBe(false);
  });
});
