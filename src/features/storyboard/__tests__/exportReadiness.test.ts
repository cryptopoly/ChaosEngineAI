import { describe, expect, it } from "vitest";
import type { StoryboardGuideAction, StoryboardReadiness, StoryboardReadinessItem } from "../../../types";
import {
  exportAllowed,
  exportAnyway,
  fixForAction,
  localLines,
  readinessGroups,
  stopReason,
} from "../exportReadiness";

function item(code: StoryboardReadinessItem["code"], action: StoryboardGuideAction | null, shotIds: string[] = [], message = `${code} sentence`): StoryboardReadinessItem {
  return { code, message, shotIds, action, data: {} };
}

function readiness(parts: Partial<StoryboardReadiness> = {}): StoryboardReadiness {
  return { ready: true, stops: [], worthFixing: [], forInfo: [], ...parts };
}

describe("fixForAction", () => {
  it("sends the make, remake and choose actions to the Storyboard", () => {
    for (const action of ["open-storyboard", "retry-failed", "make-drafts", "make-missing", "make-final", "choose-final", "make-clips", "remake-stale", "watch-batch", "resume-batch"] as StoryboardGuideAction[]) {
      expect(fixForAction(action, ["s1", "s2"])).toEqual({ kind: "storyboard", tab: "storyboard", shotId: null });
    }
  });

  it("sends planning and song actions to Plan, the cast to Cast and the sound to the Mixer", () => {
    expect(fixForAction("plan-shots")).toEqual({ kind: "plan", tab: "plan", shotId: null });
    for (const action of ["attach-song", "cut-plan", "assign-vocals", "make-lipsync", "open-coverage"] as StoryboardGuideAction[]) {
      expect(fixForAction(action)).toEqual({ kind: "song", tab: "plan", shotId: null });
    }
    expect(fixForAction("open-cast")).toEqual({ kind: "cast", tab: "cast", shotId: null });
    expect(fixForAction("open-mixer")).toEqual({ kind: "mixer", tab: "mixer", shotId: null });
  });

  it("opens the one shot a length or lock fix is about, and the Storyboard when it is about several", () => {
    for (const action of ["lengthen", "shorten", "unlock", "real-speed"] as StoryboardGuideAction[]) {
      expect(fixForAction(action, ["s4"])).toEqual({ kind: "shot", tab: "storyboard", shotId: "s4" });
      expect(fixForAction(action, ["s4", "s5"])).toEqual({ kind: "shot", tab: "storyboard", shotId: null });
      expect(fixForAction(action, [])).toEqual({ kind: "shot", tab: "storyboard", shotId: null });
    }
  });

  it("gives no button for no action, or for the check and export actions (this tab)", () => {
    expect(fixForAction(null)).toBeNull();
    expect(fixForAction("open-check")).toBeNull();
    expect(fixForAction("export")).toBeNull();
  });
});

describe("readinessGroups", () => {
  it("puts each backend item in its own group, in the order stops, worth fixing, for information", () => {
    const groups = readinessGroups(
      readiness({
        ready: false,
        stops: [item("no-clips", "make-clips")],
        worthFixing: [item("drafts", "choose-final", ["s1", "s2"]), item("song-length", "open-mixer")],
        forInfo: [item("format", null)],
      }),
      [],
    );
    expect(groups.map((g) => g.id)).toEqual(["stops", "worthFixing", "forInfo"]);
    expect(groups[0].lines).toHaveLength(1);
    expect(groups[1].lines.map((l) => (l.source === "guide" ? l.code : ""))).toEqual(["drafts", "song-length"]);
    expect(groups[2].lines).toHaveLength(1);
  });

  it("carries the sentence, the shots and the mapped fix of each line", () => {
    const [, worth] = readinessGroups(readiness({ worthFixing: [item("lipsync-gap", "lengthen", ["s4"], "Shot 4 is too short. It is not lip-synced.")] }), []);
    expect(worth.lines[0]).toEqual({
      source: "guide",
      code: "lipsync-gap",
      message: "Shot 4 is too short. It is not lip-synced.",
      shotIds: ["s4"],
      fix: { kind: "shot", tab: "storyboard", shotId: "s4" },
    });
  });

  it("adds the app's own lines to the information group only", () => {
    const local = localLines(
      { export: { aiLabel: { embedTag: false }, loudnessLufs: -14, captionsSrt: true, captionsBurn: false, stems: false, upscale: "none" } },
      { upscaledTo: null, method: "lanczos", previewStale: true, clonedVoice: false, captions: 3 },
    );
    const groups = readinessGroups(readiness({ forInfo: [item("format", null)] }), local);
    expect(groups[2].lines.map((l) => (l.source === "guide" ? l.code : l.local.id))).toEqual(["format", "loudness", "captions", "aiTag", "preview"]);
    expect(groups[0].lines).toHaveLength(0);
    expect(groups[1].lines).toHaveLength(0);
  });

  it("shows only the local lines when the guide is not there (an older payload)", () => {
    const groups = readinessGroups(null, [{ id: "preview" }]);
    expect(groups[0].lines).toHaveLength(0);
    expect(groups[1].lines).toHaveLength(0);
    expect(groups[2].lines).toHaveLength(1);
  });
});

describe("localLines", () => {
  const base = { export: { aiLabel: { embedTag: true }, loudnessLufs: null, captionsSrt: true, captionsBurn: true, stems: false, upscale: "1440p" as const } };

  it("names the upscale size and method, the caption mode and the AI tag", () => {
    const lines = localLines(base, { upscaledTo: "2560 × 1440", method: "fast", previewStale: false, clonedVoice: false, captions: 2 });
    expect(lines).toEqual([
      { id: "upscale", size: "2560 × 1440", method: "fast" },
      { id: "loudness", lufs: null },
      { id: "captions", count: 2, how: "srt+burn" },
      { id: "aiTag", on: true, cloned: false },
    ]);
  });

  it("forces the AI tag line on for a cloned voice and adds the preview line only when it is stale", () => {
    const off = { export: { ...base.export, aiLabel: { embedTag: false }, captionsSrt: false, captionsBurn: false } };
    const lines = localLines(off, { upscaledTo: null, method: "lanczos", previewStale: true, clonedVoice: true, captions: 0 });
    expect(lines.find((l) => l.id === "aiTag")).toEqual({ id: "aiTag", on: true, cloned: true });
    expect(lines.find((l) => l.id === "captions")).toEqual({ id: "captions", count: 0, how: "none" });
    expect(lines[lines.length - 1]).toEqual({ id: "preview" });
  });
});

describe("the Export button rules", () => {
  it("is off only when the backend says something stops the export", () => {
    expect(exportAllowed(readiness({ ready: false, stops: [item("no-clips", "make-clips")] }), true)).toBe(false);
    expect(exportAllowed(readiness({ ready: true, worthFixing: [item("drafts", "choose-final")] }), false)).toBe(true);
  });

  it("falls back to whether a shot has a clip when there is no guide", () => {
    expect(exportAllowed(null, true)).toBe(true);
    expect(exportAllowed(null, false)).toBe(false);
  });

  it("says Export anyway only when it is ready with things worth fixing", () => {
    expect(exportAnyway(null)).toBe(false);
    expect(exportAnyway(readiness())).toBe(false);
    expect(exportAnyway(readiness({ worthFixing: [item("drafts", "choose-final")] }))).toBe(true);
    expect(exportAnyway(readiness({ ready: false, stops: [item("no-shots", "plan-shots")], worthFixing: [item("drafts", null)] }))).toBe(false);
  });

  it("gives the first stop's sentence as the reason", () => {
    expect(stopReason(readiness({ ready: false, stops: [item("no-shots", "plan-shots", [], "The project has no shots yet.")] }))).toBe("The project has no shots yet.");
    expect(stopReason(readiness())).toBeNull();
    expect(stopReason(null)).toBeNull();
  });
});
