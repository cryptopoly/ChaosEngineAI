import { describe, expect, it } from "vitest";
import { VoiceApiError } from "../../api/voice";
import { VOICE_ERROR_ACTIONS, VOICE_ERROR_CODES } from "../../types/voice";
import type { VoiceErrorInfo } from "../../types";
import { describeVoiceError, describeVoiceErrorInfo, voiceErrorActionLabel, voiceErrorSentence } from "./voiceErrorText";

const info = (over: Partial<VoiceErrorInfo>): VoiceErrorInfo => ({
  code: "gpu_memory",
  message: "x",
  action: "smaller-model",
  extra: null,
  ...over,
});

describe("voiceErrorSentence", () => {
  it("gives every code its own plain sentence", () => {
    const sentences = VOICE_ERROR_CODES.map((code) => voiceErrorSentence(code));
    expect(new Set(sentences).size).toBe(VOICE_ERROR_CODES.length);
    for (const sentence of sentences) {
      expect(sentence.endsWith(".")).toBe(true);
      expect(sentence).not.toContain("studio:voice"); // never a bare translation key
    }
  });

  it("reads the way the backend's English does", () => {
    expect(voiceErrorSentence("gpu_memory")).toBe("This model needs more GPU memory than is free.");
    expect(voiceErrorSentence("model_missing")).toBe("The model for this is not downloaded yet.");
  });

  it("falls back to the generic sentence for a code it does not know", () => {
    expect(voiceErrorSentence("from-the-future" as never)).toBe(voiceErrorSentence("unknown"));
  });
});

describe("voiceErrorActionLabel", () => {
  it("labels every action but none", () => {
    for (const action of VOICE_ERROR_ACTIONS) {
      const label = voiceErrorActionLabel(action);
      if (action === "none") expect(label).toBeNull();
      else expect(label && label.length).toBeGreaterThan(0);
    }
  });

  it("uses short button text", () => {
    expect(voiceErrorActionLabel("retry")).toBe("Try again");
    expect(voiceErrorActionLabel("install-ffmpeg")).toBe("Install ffmpeg");
    expect(voiceErrorActionLabel("add-hf-token")).toBe("Add a Hugging Face token");
  });
});

describe("describeVoiceErrorInfo", () => {
  it("keeps the raw text for Details and the action for the button", () => {
    const view = describeVoiceErrorInfo(info({}), "OutOfMemoryError: CUDA out of memory.");
    expect(view).toEqual({
      code: "gpu_memory",
      sentence: "This model needs more GPU memory than is free.",
      action: "smaller-model",
      actionLabel: "Choose a smaller model",
      extra: null,
      details: "OutOfMemoryError: CUDA out of memory.",
    });
  });

  it("has no button for a sentence that stands alone and no Details when the raw text adds nothing", () => {
    const view = describeVoiceErrorInfo(info({ code: "voice_not_found", action: "none" }), "That voice no longer exists.");
    expect(view.actionLabel).toBeNull();
    expect(view.details).toBe("");
  });

  it("carries the add-on to tick", () => {
    const view = describeVoiceErrorInfo(info({ code: "addon_missing", action: "install-addon", extra: "speech-gpu" }), "raw");
    expect(view.extra).toBe("speech-gpu");
    expect(view.actionLabel).toBe("Install the add-on");
  });

  it("shows the raw text as the sentence, with no button, when the response had no code", () => {
    const view = describeVoiceErrorInfo(null, "Unknown voice 'zz'.");
    expect(view).toMatchObject({ code: "unknown", sentence: "Unknown voice 'zz'.", action: "none", actionLabel: null, details: "" });
    expect(describeVoiceErrorInfo(null, "").sentence).toBe(voiceErrorSentence("unknown"));
  });
});

describe("describeVoiceError", () => {
  it("reads a VoiceApiError", () => {
    const error = new VoiceApiError("The voice runtime is not installed. Install it from Voice > Models.", 503, info({
      code: "runtime_missing",
      action: "install-voice-runtime",
    }));
    const view = describeVoiceError(error);
    expect(view.sentence).toBe("The voice runtime is not installed yet.");
    expect(view.actionLabel).toBe("Install the voice runtime");
    expect(view.details).toContain("Install it from Voice > Models.");
  });

  it("treats any other Error or a string as raw text", () => {
    expect(describeVoiceError(new Error("Request cancelled")).sentence).toBe("Request cancelled");
    expect(describeVoiceError("plain").sentence).toBe("plain");
    expect(describeVoiceError(undefined).sentence).toBe(voiceErrorSentence("unknown"));
  });
});
