import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { MusicEstimateResponse, MusicLibraryItem, MusicModelVariant } from "../../../types";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { RunEstimateStrip } from "../../../components/RunEstimateStrip";
import { stripFromEstimate } from "../musicLogic";
import {
  buildSoundPayload,
  clampSoundSteps,
  defaultSoundForm,
  effectiveSoundSteps,
  soundDefaultSteps,
  soundFormFromItem,
  soundStepsOptions,
  validateSoundForm,
  againSoundForm,
  varySoundForm,
  type SoundForm,
} from "../soundsLogic";

const small = { defaultSettings: { seconds: 5, steps: 8 }, stepsOptions: { min: 1, max: 32, presets: [4, 8, 16] } } as unknown as MusicModelVariant;
const full = { defaultSettings: { seconds: 10, steps: 100 }, stepsOptions: { min: 10, max: 150, presets: [25, 50, 100] } } as unknown as MusicModelVariant;
const withCaption = (form: SoundForm): SoundForm => ({ ...form, caption: "rain" });

describe("sound steps in the form", () => {
  it("starts on the row's default and does not send it", () => {
    const form = defaultSoundForm(full);
    expect(form.steps).toBeNull();
    expect(effectiveSoundSteps(form, full)).toBe(100);
    expect(effectiveSoundSteps(defaultSoundForm(small), small)).toBe(8);
    expect(buildSoundPayload(withCaption(form), "m")).not.toHaveProperty("steps");
  });

  it("sends a chosen number", () => {
    expect(buildSoundPayload({ ...withCaption(defaultSoundForm(full)), steps: 50 }, "m").steps).toBe(50);
  });

  it("reads the allowed range and the default from the catalog row", () => {
    expect(soundStepsOptions(full)).toEqual({ min: 10, max: 150, presets: [25, 50, 100] });
    expect(soundStepsOptions({} as MusicModelVariant)).toBeNull();
    expect(soundDefaultSteps(small)).toBe(8);
    expect(clampSoundSteps(50, full)).toBe(50);
    expect(clampSoundSteps(8, full)).toBeNull();
    expect(clampSoundSteps(100, small)).toBeNull();
  });

  it("validates against each row's own range", () => {
    const form = (steps: number | null) => ({ ...withCaption(defaultSoundForm(full)), steps });
    expect(validateSoundForm(form(10), full)).toEqual([]);
    expect(validateSoundForm(form(150), full)).toEqual([]);
    expect(validateSoundForm(form(9), full)).toEqual(["stepsRange"]);
    expect(validateSoundForm(form(151), full)).toEqual(["stepsRange"]);
    expect(validateSoundForm(form(10.5), full)).toEqual(["stepsRange"]);
    expect(validateSoundForm(form(33), small)).toEqual(["stepsRange"]);
    expect(validateSoundForm(form(1), small)).toEqual([]);
    expect(validateSoundForm(form(null), small)).toEqual([]);
  });

  it("Make another like this, Again and Vary reuse the steps", () => {
    const item = { id: "1", kind: "sound", title: "t", caption: "rain", lyrics: "", tags: [], createdAt: 0, updatedAt: 0, audio: ".flac", durationS: 4, seed: 9, steps: 50 } as MusicLibraryItem;
    const form = soundFormFromItem(item, full);
    expect(form.steps).toBe(50);
    expect(againSoundForm(form)).toMatchObject({ steps: 50, seed: null });
    expect(varySoundForm(form, 9)).toMatchObject({ steps: 50, seed: 10 });
    // The row's default stays unset; a step count the row does not accept falls back to the default.
    expect(soundFormFromItem({ ...item, steps: 100 }, full).steps).toBeNull();
    expect(soundFormFromItem({ ...item, steps: 100 }, small).steps).toBeNull();
    expect(soundFormFromItem({ ...item, steps: undefined }, full).steps).toBeNull();
  });
});

function response(over: Partial<MusicEstimateResponse["estimate"]> = {}): MusicEstimateResponse {
  return {
    estimate: { model: "m", seconds: 110, coldStartSeconds: 0, warm: true, peakGb: 22.8, idleGb: 7, writesMb: 1, modelDiskGb: 5.7, basis: "measured", ...over },
    fit: null,
    totalGb: 64,
    queue: { running: null, waiting: [] },
  };
}

describe("estimate label at a non-default step count", () => {
  it("keeps Measured at the measured steps", () => {
    const strip = stripFromEstimate(response());
    expect(strip.basis).toBe("measured");
    expect(strip.scaledFromMeasured).toBeUndefined();
  });

  it("says it is estimated from the measured run when the backend scaled it", () => {
    const strip = stripFromEstimate(response({ basis: "estimated", basisNote: "Estimated from the measured run at 100 steps.", steps: 50 }));
    expect(strip).toMatchObject({ basis: "estimated", scaledFromMeasured: true, seconds: 110 });
    const html = renderToStaticMarkup(
      <RunEstimateStrip seconds={strip.seconds} peakGb={strip.peakGb} totalGb={strip.totalGb} diskMb={strip.diskMb} basis={strip.basis} scaledFromMeasured={strip.scaledFromMeasured} risk={strip.risk} queue={null} action={null} />,
    );
    expect(html).toContain("Estimated from the measured run");
    expect(html).not.toContain(">Measured<");
    const measured = renderToStaticMarkup(<RunEstimateStrip seconds={208} peakGb={22.8} totalGb={64} diskMb={1} basis="measured" risk="safe" queue={null} action={null} />);
    expect(measured).toContain(">Measured<");
  });
});
