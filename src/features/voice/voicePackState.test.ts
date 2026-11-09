import { describe, expect, it } from "vitest";
import { chineseAddonPending, quickVoiceSummary, voicePackMode } from "./voicePackState";

describe("quickVoiceSummary", () => {
  it("counts the quick voices and languages this machine can offer, never the ones that cannot run here", () => {
    const voices = [{ language: "en-US" }, { language: "en-US" }, { language: "zh" }, { language: "ja", neverAvailable: true }];
    expect(quickVoiceSummary(voices)).toEqual({ count: 3, languages: 2 });
  });
});

const healthy = { installed: true, updateAvailable: false, vadAvailable: true, ttsModelInstalled: true, jobPhase: "done" };

describe("voicePackMode", () => {
  it("leads with install when the pack is missing", () => {
    expect(voicePackMode({ ...healthy, installed: false })).toBe("install");
  });
  it("leads with update when the pack lacks features", () => {
    expect(voicePackMode({ ...healthy, updateAvailable: true })).toBe("update");
    expect(voicePackMode({ ...healthy, vadAvailable: false })).toBe("update");
  });
  it("leads with repair after a failed run", () => {
    expect(voicePackMode({ ...healthy, jobPhase: "error" })).toBe("repair");
  });
  it("is healthy without Kokoro, an optional add-on (D-75)", () => {
    expect(voicePackMode({ ...healthy, ttsModelInstalled: false })).toBe("healthy");
  });
  it("is healthy otherwise, including right after a successful install", () => {
    expect(voicePackMode(healthy)).toBe("healthy");
    expect(voicePackMode({ ...healthy, jobPhase: "idle" })).toBe("healthy");
  });
});

describe("chineseAddonPending", () => {
  it("is true only for an uninstalled chinese extra", () => {
    expect(chineseAddonPending([{ id: "chinese", installed: false }])).toBe(true);
    expect(chineseAddonPending([{ id: "chinese", installed: true }])).toBe(false);
    expect(chineseAddonPending([])).toBe(false);
  });
});
