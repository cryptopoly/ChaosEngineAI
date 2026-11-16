import { describe, expect, it } from "vitest";
import type { StoryboardProject, StoryboardShot } from "../../../types";
import {
  DEFAULT_BRAND,
  brandOf,
  changeCount,
  changeRows,
  changedFields,
  inkOn,
  isHexColour,
  lengthLabel,
  lengthVerdict,
  lockedCount,
  planFailure,
  planOf,
  projectPalette,
  replaceLosesTakes,
  reworkEdit,
  styleWithRoom,
  takeStyle,
  totalSeconds,
} from "../planLogic";

function shot(id: string, extra: Partial<StoryboardShot> = {}): StoryboardShot {
  return {
    id, order: 0, title: id, status: "idle", stillPrompt: "a cup", motionPrompt: "steam", referenceImages: [],
    startFrom: "generate", endImage: null, seconds: 4, seed: null, locked: false, trimIn: 0, trimOut: null,
    speed: 1, takes: [], activeTake: null, transition: { type: "cut", seconds: 0 }, nativeSoundDb: 0, redo: false,
    ...extra,
  };
}

const ref = (id: string, images: number, palette: string[] = []) => ({
  id, name: id, description: "", palette,
  images: Array.from({ length: images }, (_, i) => ({ path: `media/refs/${id}-${i}.png`, hash: null })),
});

function project(extra: Partial<StoryboardProject> = {}): StoryboardProject {
  return {
    schemaVersion: 3, revision: 1, id: "0123456789abcdef", name: "Tea", createdAt: 0,
    defaults: { width: 1280, height: 720, fps: 24, imageModel: null, videoModel: null, voice: null, exportPreset: "16:9", resolution: "720p" },
    references: { characters: [], places: [] }, shots: [], audio: [],
    export: { aiLabel: { embedTag: true }, loudnessLufs: -16, captionsSrt: false, captionsBurn: false, stems: false, upscale: "none" },
    captions: [], batches: [], undoStack: [], redoStack: [], ...extra,
  } as StoryboardProject;
}

describe("plan logic (S8)", () => {
  it("fills the plan and brand of a project saved before them", () => {
    expect(planOf(project())).toEqual({ brief: "", filmSeconds: null });
    expect(brandOf(project())).toEqual(DEFAULT_BRAND);
    const partial = brandOf(project({ brand: { ...DEFAULT_BRAND, captions: { font: "Avenir", colour: "#ffcc00", position: "top" } } }));
    expect(partial.captions.font).toBe("Avenir");
  });

  it("adds up the seconds and compares them with the aim", () => {
    expect(totalSeconds([{ seconds: 4 }, { seconds: 6.25 }, { seconds: Number.NaN }])).toBe(10.3);
    expect(lengthVerdict(58, 60)).toBe("match");
    expect(lengthVerdict(40, 60)).toBe("short");
    expect(lengthVerdict(80, 60)).toBe("long");
    expect(lengthVerdict(80, null)).toBe("none");
    expect(lengthLabel(90)).toEqual({ minutes: 1, seconds: 30 });
  });

  it("lists the direction chat's changes in words and drops unknown shots", () => {
    const shots = [shot("a", { title: "Pot" }), shot("b", { title: "Cup" })];
    const changes = {
      edits: [{ shotId: "a", stillPrompt: "warmer", seconds: 3 }, { shotId: "zz", title: "gone" }],
      add: [{ after: "b", title: "Sign", stillPrompt: "x", motionPrompt: "", seconds: 4 }, { after: null, title: "Dawn", stillPrompt: "y", motionPrompt: "", seconds: 4 }],
      remove: ["b", "zz"],
    };
    expect(changeCount(changes)).toBe(6);
    expect(changeCount(null)).toBe(0);
    expect(changeRows(changes, shots)).toEqual([
      { kind: "edit", number: 1, title: "Pot", fields: ["stillPrompt", "seconds"] },
      { kind: "remove", number: 2, title: "Cup" },
      { kind: "add", after: 2, title: "Sign" },
      { kind: "add", after: 0, title: "Dawn" },
    ]);
  });

  it("applies only what a rework changed", () => {
    const current = shot("a", { title: "Pot", stillPrompt: "a pot", motionPrompt: "steam", seconds: 4 });
    const suggestion = { title: "Pot", stillPrompt: "a pot, golden light", motionPrompt: "", seconds: 4 };
    expect(changedFields(current, suggestion)).toEqual(["stillPrompt"]);
    expect(reworkEdit(current, suggestion)).toEqual({ shotId: "a", stillPrompt: "a pot, golden light" });
    expect(reworkEdit(current, { ...suggestion, seconds: 6 })).toEqual({ shotId: "a", stillPrompt: "a pot, golden light", seconds: 6 });
  });

  it("knows what a replace would take off the board", () => {
    const withTake = shot("a", { takes: [{ id: "t", still: "media/a/t/still.png", clip: null, hashes: {}, settings: {}, measuredSeconds: null, createdAt: 0, origin: "batch", basis: null }] });
    const p = project({ shots: [withTake, shot("b", { locked: true }), shot("c")] });
    expect(replaceLosesTakes(p)).toBe(1);
    expect(lockedCount(p)).toBe(1);
  });

  it("reads the error code of a plan call", () => {
    expect(planFailure("no-model")).toBe("noModel");
    expect(planFailure("busy")).toBe("busy");
    expect(planFailure(null)).toBe("other");
  });
});

describe("style logic (S15)", () => {
  it("puts a brief picture in the first style with room, else asks for a new style", () => {
    expect(styleWithRoom(project())).toBeNull();
    const p = project({ references: { characters: [], places: [], styles: [ref("full", 4), ref("room", 1)] } });
    expect(styleWithRoom(p)?.id).toBe("room");
  });

  it("merges the palettes of every style once each", () => {
    const p = project({ references: { characters: [], places: [], styles: [ref("a", 1, ["#112233", "#ffffff"]), ref("b", 1, ["#ffffff", "#aa0000"])] } });
    expect(projectPalette(p)).toEqual(["#112233", "#ffffff", "#aa0000"]);
  });

  it("picks readable ink and checks colours", () => {
    expect(inkOn("#ffffff")).toBe("#000000");
    expect(inkOn("#102030")).toBe("#ffffff");
    expect(isHexColour("#a1b2c3")).toBe(true);
    expect(isHexColour("#A1B2C3")).toBe(false);
    expect(isHexColour("red")).toBe(false);
  });

  it("reads which way the style went into the active take", () => {
    const t = { id: "t", still: null, clip: null, hashes: {}, measuredSeconds: null, createdAt: 0, origin: "batch" as const, basis: null };
    expect(takeStyle(shot("a"))).toBeNull();
    expect(takeStyle(shot("a", { takes: [{ ...t, settings: { style: "words" } }], activeTake: "t" }))).toBe("words");
    expect(takeStyle(shot("a", { takes: [{ ...t, settings: { style: "pictures" } }], activeTake: "t" }))).toBe("pictures");
  });
});
