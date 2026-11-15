import i18next from "i18next";
import { describe, expect, it } from "vitest";
import { busyEngineLines } from "../busyEngine";
import type { ActiveGeneration } from "../../types";

const t = i18next.getFixedT("en", "common");
const run = (extra: Partial<ActiveGeneration>): ActiveGeneration => ({
  kind: "image", runLabel: null, phase: "diffusing", message: "", step: 0, totalSteps: 0, elapsedSeconds: 0, ...extra,
});

describe("busyEngineLines", () => {
  it("is empty when nothing runs", () => {
    expect(busyEngineLines(t, undefined)).toEqual([]);
    expect(busyEngineLines(t, [])).toEqual([]);
  });

  it("gives the run label and the denoise step", () => {
    expect(busyEngineLines(t, [run({ runLabel: "Qwen-Image 2.1", step: 27, totalSteps: 40 })])).toEqual(["Image engine busy: Qwen-Image 2.1 · step 27 of 40"]);
  });

  it("shows no step outside the denoise phase and falls back to the status message", () => {
    expect(busyEngineLines(t, [run({ phase: "loading", message: "Loading model", step: 3, totalSteps: 40 })])).toEqual(["Image engine busy: Loading model"]);
  });

  it("names video and music runs and works with no label at all", () => {
    expect(busyEngineLines(t, [run({ kind: "video", runLabel: "LTX-2.3" }), run({ kind: "music" })])).toEqual(["Video engine busy: LTX-2.3", "Music engine busy"]);
  });

  it("never shows a step past the total", () => {
    expect(busyEngineLines(t, [run({ runLabel: "x", step: 9, totalSteps: 8 })])).toEqual(["Image engine busy: x · step 8 of 8"]);
  });
});
