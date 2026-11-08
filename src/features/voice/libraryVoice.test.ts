import { describe, expect, it } from "vitest";
import { clearLibraryFilter, kindFromVoiceId, nameFromVoiceId, peekLibraryFilter, requestLibraryFilter, speechVoice } from "./libraryVoice";

describe("speechVoice", () => {
  it("uses the stored name and kind", () => {
    expect(speechVoice({ voice: "qwen3:Ryan", voiceName: "Ryan", voiceKind: "preset" })).toEqual({ name: "Ryan", kind: "preset" });
    expect(speechVoice({ voice: "clone:1_abcdef12", voiceName: "Ada", voiceKind: "clone", clonedVoice: true })).toEqual({ name: "Ada", kind: "clone" });
  });

  it("works the name and kind out of the id for items saved before they were stored", () => {
    expect(speechVoice({ voice: "af_heart" })).toEqual({ name: "Heart", kind: "builtin" });
    expect(speechVoice({ voice: "qwen3:Uncle_Fu" })).toEqual({ name: "Uncle Fu", kind: "preset" });
    expect(speechVoice({ voice: "clone:1_abcdef12", clonedVoice: true })).toEqual({ name: "", kind: "clone" });
    expect(speechVoice({ voice: "design" })).toEqual({ name: "", kind: "design" });
    expect(speechVoice({ voice: null })).toBeNull();
    expect(speechVoice({})).toBeNull();
  });

  it("reads ids", () => {
    expect(kindFromVoiceId("zf_xiaobei")).toBe("builtin");
    expect(nameFromVoiceId("zf_xiaobei")).toBe("Xiaobei");
    expect(nameFromVoiceId("custom")).toBe("Custom");
  });
});

describe("library filter request", () => {
  it("is read until cleared", () => {
    expect(peekLibraryFilter()).toBe("");
    requestLibraryFilter("speech");
    expect(peekLibraryFilter()).toBe("speech");
    expect(peekLibraryFilter()).toBe("speech");
    clearLibraryFilter();
    expect(peekLibraryFilter()).toBe("");
  });
});
