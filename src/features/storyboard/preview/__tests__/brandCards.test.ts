import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardShot, StoryboardTimeline } from "../../../../types";
import { brandOnFilm, cardLook, cardSeconds, cardText, contrast } from "../brandCards";
import { buildRoughCut, captionAt, cutMarkers, logoVisibleAt, mediaFiles, pieceIndexAt, visibleAt } from "../roughCutLogic";

function shot(id: string): StoryboardShot {
  return {
    id, order: 0, title: id, status: "done", stillPrompt: "", motionPrompt: "", referenceImages: [], startFrom: "generate",
    endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null, speed: 1,
    takes: [{ id: "t", still: null, clip: `media/${id}/clip.mp4`, hashes: {}, settings: {}, measuredSeconds: 3, createdAt: 0, origin: "batch", basis: null }],
    activeTake: "t", transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
  };
}

function project(extra: Partial<StoryboardProject> = {}, exportExtra: Partial<StoryboardProject["export"]> = {}): StoryboardProject {
  return {
    schemaVersion: 1, revision: 3, id: "p", name: "Tea Shop", createdAt: 0,
    defaults: { fps: 24 } as StoryboardProject["defaults"],
    references: { characters: [], places: [], styles: [] },
    shots: [shot("a"), shot("b")],
    audio: [
      { id: "m", kind: "music", source: { path: "media/audio/m.wav", hash: null, measuredSeconds: 10, clonedVoice: false, label: "" },
        anchor: null, start: 1, trimIn: 0, trimOut: null, gainDb: 0, fadeIn: 0, fadeOut: 0, duck: false, loop: false, status: "ok" },
    ],
    export: { aiLabel: { embedTag: true }, loudnessLufs: -16, captionsSrt: false, captionsBurn: false, stems: false, upscale: "none", ...exportExtra },
    captions: [{ id: "c", clipId: null, start: 0.5, end: 1.5, text: "Hello" }],
    batches: [], undoStack: [], redoStack: [],
    ...extra,
  } as StoryboardProject;
}

const timeline: StoryboardTimeline = {
  total: 6,
  shots: [
    { id: "a", start: 0, end: 3, duration: 3, source_in: 0, source_out: 3, transition_type: "cut", transition_seconds: 0, measured: true, ready: true },
    { id: "b", start: 3, end: 6, duration: 3, source_in: 0, source_out: 3, transition_type: "cut", transition_seconds: 0, measured: true, ready: true },
  ],
  audio: [{ id: "m", start: 1, duration: 5, trim_in: 0, end: 6 }],
  problems: [],
};

const logo = { path: "media/brand/logo.png", hash: null };
const brand = (over: Partial<NonNullable<StoryboardProject["brand"]>> = {}) => ({
  logo: null, logoCorner: "bottom-right" as const, captions: { font: null, colour: "#ffffff", position: "bottom" as const }, styleUse: "auto" as const, ...over,
});

describe("cardLook (the same choices as the export)", () => {
  it("is white on black with no style and no brand", () => {
    expect(cardLook(project())).toEqual({ background: "#000000", colour: "#FFFFFF", font: null });
  });
  it("takes the first style palette and keeps the text readable", () => {
    const styles = [{ id: "s0", name: "", description: "", images: [] }, { id: "s1", name: "", description: "", images: [], palette: ["#f5e6c8", "#123456"] }];
    const p = project({ references: { characters: [], places: [], styles } });
    expect(cardLook(p)).toMatchObject({ background: "#F5E6C8", colour: "#000000" });
    const branded = project({ references: { characters: [], places: [], styles }, brand: brand({ captions: { font: "Didot", colour: "#5a2d0c", position: "top" } }) });
    expect(cardLook(branded)).toEqual({ background: "#F5E6C8", colour: "#5A2D0C", font: "Didot" });
    expect(contrast("#000000", "#FFFFFF")).toBeCloseTo(21, 5);
  });
});

describe("card text and length", () => {
  it("defaults to the project name and lasts whole frames", () => {
    const p = project({}, { titleText: " ", endText: "Visit us", cardSeconds: 2.01 });
    expect([cardText(p, "title"), cardText(p, "end")]).toEqual(["Tea Shop", "Visit us"]);
    expect(cardSeconds(p)).toBe(48 / 24);
    expect(cardSeconds(project({ defaults: { fps: 30 } as StoryboardProject["defaults"] }, { cardSeconds: 1.5 }))).toBe(1.5);
  });
  it("draws the logo only when it is on, but always on the cards", () => {
    const off = brandOnFilm(project({ brand: brand({ logo }) }, { showLogo: false }));
    expect([off.logo, off.cardLogo]).toEqual([null, logo.path]);
    expect(brandOnFilm(project({ brand: brand({ logo }) })).logo).toBe(logo.path);
  });
});

describe("the rough cut with cards", () => {
  const p = project({ brand: brand({ logo, logoCorner: "top-left" }) }, { titleCard: true, endCard: true, cardSeconds: 2 });
  const cut = buildRoughCut(p, timeline);

  it("puts the title card first and the end card last, and moves everything else behind the title", () => {
    expect(cut.total).toBe(10);
    expect(cut.pieces.map((piece) => [piece.shotId, piece.start, piece.end])).toEqual([
      ["card:title", 0, 2], ["a", 2, 5], ["b", 5, 8], ["card:end", 8, 10],
    ]);
    expect(cut.sounds.find((s) => s.id === "m")?.start).toBe(3);
    expect(cut.sounds.find((s) => s.id === "native:a")?.start).toBe(2);
    expect(captionAt(cut, 2.6)).toBe("Hello");
    expect(captionAt(cut, 1)).toBeNull();
    expect(visibleAt(cut, 1)[0].piece.media).toEqual({ kind: "card", which: "title", text: "Tea Shop" });
    expect(cut.pieces[pieceIndexAt(cut, 9)].shotId).toBe("card:end");
    expect(cut.missing).toBe(0);
  });

  it("marks the cards on the scrubber and loads the logo", () => {
    expect(cutMarkers(cut).map((m) => m.state)).toEqual(["card", "clip", "clip", "card"]);
    expect(mediaFiles(cut).stills).toEqual([logo.path]);
  });

  it("shows the corner logo over the shots only", () => {
    expect([1, 2, 6, 8, 9].map((t) => logoVisibleAt(cut, t))).toEqual([false, true, true, true, false]);
    const plain = buildRoughCut(project({ brand: brand({ logo }) }), timeline);
    expect([0, 6].map((t) => logoVisibleAt(plain, t))).toEqual([true, true]);
  });

  it("leaves the brand out for the Mixer, which plays the timeline as drawn", () => {
    const mixer = buildRoughCut(p, timeline, { brand: false });
    expect(mixer.total).toBe(6);
    expect(mixer.pieces[0].start).toBe(0);
    expect(logoVisibleAt(mixer, 3)).toBe(false);
  });
});
