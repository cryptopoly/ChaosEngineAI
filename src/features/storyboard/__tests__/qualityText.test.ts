import { describe, expect, it } from "vitest";
import type { StoryboardEstimate } from "../../../api/storyboard";
import type { StoryboardQuality, StoryboardTake, StoryboardTakeQuality } from "../../../types";
import { draftChoiceText, draftTag, finalChoiceText, startChoice, takeQualityOf, takeSize, type TFn } from "../qualityText";

const t: TFn = (key, options) => {
  // Mirror the app's helper well enough for the sentences under test: ICU placeholders only.
  void key;
  return String(options.defaultValue).replace(/\{(\w+)\}/g, (_m, name: string) => String(options[name] ?? ""));
};

function take(extra: Partial<StoryboardTake> = {}): StoryboardTake {
  return { id: "t1", still: "a.png", clip: null, hashes: {}, settings: {}, measuredSeconds: null, createdAt: 0, origin: "batch", basis: null, ...extra };
}

const draftQuality = (extra: Partial<StoryboardTakeQuality> = {}): StoryboardTakeQuality => ({
  pass: "draft", draft: true, still: { width: 640, height: 384 }, clip: null, stillStale: false, clipStale: false, ...extra,
});

describe("what a take was made at", () => {
  it("reads the half sizes the batch recorded", () => {
    const q = takeQualityOf(take({ clip: "c.mp4", renderPass: "draft", settings: { stillWidth: 640, stillHeight: 384, clipWidth: 640, clipHeight: 352, clipFps: 24 } }));
    expect(q.draft).toBe(true);
    expect(q.still).toEqual({ width: 640, height: 384 });
    expect(q.clip).toEqual({ width: 640, height: 352, fps: 24 });
    expect(takeSize(q)).toEqual({ width: 640, height: 352 });
  });

  it("uses the plain width and height only for the half made last (as the backend does)", () => {
    expect(takeQualityOf(take({ renderPass: "draft", settings: { width: 640, height: 384 } })).still).toEqual({ width: 640, height: 384 });
    // A take with a clip: the plain keys belong to the clip, not the still.
    const withClip = takeQualityOf(take({ clip: "c.mp4", settings: { width: 1280, height: 720, fps: 24 } }));
    expect(withClip.still).toBeNull();
    expect(withClip.clip).toEqual({ width: 1280, height: 720, fps: 24 });
  });

  it("is final without a pass, and has no size for an upload", () => {
    const q = takeQualityOf(take({ origin: "upload" }));
    expect(q.draft).toBe(false);
    expect(takeSize(q)).toBeNull();
  });

  it("prefers the guide's label", () => {
    const fromGuide = draftQuality();
    expect(takeQualityOf(take(), fromGuide)).toBe(fromGuide);
  });
});

describe("the draft tag", () => {
  it("names the size and explains it, with the film's size", () => {
    const tag = draftTag(t, draftQuality(), { width: 1280, height: 720 });
    expect(tag?.label).toBe("Draft 640 x 384");
    expect(tag?.title).toContain("made at 640 x 384");
    expect(tag?.title).toContain("The film is 1280 x 720");
    expect(tag?.title).toContain("choose this shot for final");
  });

  it("still explains a draft whose size was not recorded", () => {
    const tag = draftTag(t, draftQuality({ still: null }));
    expect(tag?.label).toBe("Draft");
    expect(tag?.title).toContain("smaller than the film");
    expect(draftTag(t, draftQuality(), null)?.title).toContain("made at 640 x 384, smaller than the film");
  });

  it("gives a final take no tag", () => {
    expect(draftTag(t, draftQuality({ pass: "final", draft: false }))).toBeNull();
    expect(draftTag(t, null)).toBeNull();
  });
});

describe("the draft-or-final choice in sizes", () => {
  const quality: StoryboardQuality = {
    target: { width: 1280, height: 720, fps: 24 }, draftSize: { width: 640, height: 384 }, draftShots: [], staleShots: [], takes: {},
  };
  const estimate = {
    what: "both", shots: 3, stills: 3, clips: 3, seconds: 2280, peakGb: 21, totalGb: 24, diskMb: 10, basis: "estimated", missing: [],
    imageModel: "A", videoModel: "B",
    passes: {
      draft: { shots: 3, stills: 3, clips: 3, seconds: 240, stillSeconds: 240, clipSeconds: 600, basis: "estimated", imageModel: "A", videoModel: "B", width: 640, height: 384 },
      final: { shots: 3, stills: 3, clips: 3, seconds: 2280, stillSeconds: 900, clipSeconds: 1380, basis: "estimated", imageModel: "A", videoModel: "B", width: 1280, height: 720 },
    },
  } as StoryboardEstimate;

  it("states both options in sizes and time", () => {
    const choice = startChoice(quality, estimate);
    expect(draftChoiceText(t, choice!.draft!)).toBe(
      "Drafts first: 640 x 384 · about 4 min · half size, to choose the shots you like. You then make those full size.",
    );
    expect(finalChoiceText(t, choice!.final!)).toBe("Straight to final: 1280 x 720 · about 38 min · peak 21 GB of 24 GB");
  });

  it("adds the clips to the draft time when asked", () => {
    expect(startChoice(quality, estimate, true)?.draft?.seconds).toBe(840);
  });

  it("works from the guide alone, and says nothing without either", () => {
    const sizes = startChoice(quality, null);
    expect(sizes?.draft).toEqual({ size: { width: 640, height: 384 }, seconds: null });
    expect(draftChoiceText(t, sizes!.draft!)).toContain("Drafts first: 640 x 384 · half size");
    expect(startChoice(null, null)).toBeNull();
  });
});
