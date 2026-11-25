import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../types";
import { clipCount, preflight, recentActivity, stillOnlyCount } from "../overviewLogic";

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

describe("Before you export: captions", () => {
  it("counts the captions the film shows when given, not every caption of the project", () => {
    const p = { ...project([shot("a", { clip: "c.mp4" })]), captions: Array.from({ length: 84 }, () => ({})), export: { loudnessLufs: null, captionsBurn: true, aiLabel: { embedTag: false } } } as unknown as StoryboardProject;
    const line = preflight(p, { leftOut: 0, previewStale: false, clonedVoice: false, captions: 6 }).find((l) => l.id === "captions");
    expect(line).toMatchObject({ n: 6, detail: "burn" });
  });
});

describe("Before you export: drafts", () => {
  it("warns when an active take is a draft, after the stale line", () => {
    const draft = shot("a", { clip: "c.mp4" });
    draft.takes[0].renderPass = "draft";
    const p = { ...project([draft, shot("b", { clip: "d.mp4" })]), captions: [], export: { loudnessLufs: null, aiLabel: { embedTag: false } } } as unknown as StoryboardProject;
    const lines = preflight(p, { leftOut: 0, previewStale: false, clonedVoice: false });
    expect(lines.find((l) => l.id === "drafts")).toEqual({ id: "drafts", ok: false, n: 1, total: 2, tab: "storyboard" });
    expect(lines.map((l) => l.id).indexOf("drafts")).toBe(1);
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