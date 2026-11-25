import { describe, expect, it } from "vitest";
import { sttFindsLanguageItself, sttLanguageMode, sttLanguageModeText } from "./voiceLanguages";

describe("speech to text language rule", () => {
  it("never takes a choice for Parakeet, Granite and Nemotron Transcribe", () => {
    for (const key of ["parakeet-tdt-v3", "granite-speech-5.0-2b", "nemotron-3.5-transcribe-30b"]) {
      expect(sttFindsLanguageItself(key)).toBe(true);
      expect(sttLanguageMode(key, "de")).toBe("auto");
    }
  });

  it("uses the choice for the other models, and finds the language when Detect is chosen", () => {
    expect(sttFindsLanguageItself("whisper-large-v3-turbo")).toBe(false);
    expect(sttFindsLanguageItself(undefined)).toBe(false);
    expect(sttLanguageMode("qwen3-asr-1.7b", "de")).toBe("choice");
    expect(sttLanguageMode("qwen3-asr-1.7b", "")).toBe("auto");
    expect(sttLanguageMode(undefined, "fr")).toBe("choice");
  });

  it("says it in plain words", () => {
    expect(sttLanguageModeText("auto")).toBe("finds the language itself");
    expect(sttLanguageModeText("choice")).toBe("uses your choice");
  });
});
