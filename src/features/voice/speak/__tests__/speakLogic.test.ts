import { describe, expect, it } from "vitest";
import type { VoiceEntry, VoiceModelRow } from "../../../../types";
import {
  canGenerate,
  chooseModel,
  cloneRequestText,
  cloneTextTooLong,
  defaultVoice,
  installedCloneModels,
  matchesFilter,
  modelNote,
  pickerVoices,
  styleState,
} from "../speakLogic";

const caps = (over: Partial<VoiceEntry["capabilities"]> = {}) => ({ speak: true, style: false, clone: false, design: false, ...over });

function row(over: Partial<VoiceModelRow>): VoiceModelRow {
  return {
    key: "m",
    name: "M",
    summary: "",
    repo: "r",
    sizeGb: 1,
    languages: "",
    default: false,
    kind: "custom",
    capabilities: caps(),
    supported: true,
    installed: false,
    download: { state: "idle", error: null },
    ...over,
  };
}

const small = row({ key: "custom-0.6b", name: "0.6B CustomVoice", sizeGb: 2.5 });
const big = row({ key: "custom-1.7b", name: "1.7B CustomVoice", sizeGb: 3.08, capabilities: caps({ style: true }) });
const design = row({ key: "design-1.7b", kind: "design", sizeGb: 3.08, capabilities: caps({ style: true, design: true }) });

function voice(over: Partial<VoiceEntry>): VoiceEntry {
  return { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", model: "custom-0.6b", models: ["custom-0.6b", "custom-1.7b"], capabilities: caps({ style: true }), available: true, ...over };
}

describe("style box gating", () => {
  it("is off for built-in and cloned voices, with the reason", () => {
    expect(styleState(voice({ kind: "builtin", id: "af_heart" }), [])).toMatchObject({ enabled: false, reason: "builtin" });
    expect(styleState(voice({ kind: "clone", id: "clone:1" }), [])).toMatchObject({ enabled: false, reason: "clone" });
  });

  it("is off for a preset until a style model is installed, and offers that model", () => {
    const state = styleState(voice({}), [{ ...small, installed: true }, big]);
    expect(state.enabled).toBe(false);
    expect(state.reason).toBe("smallModel");
    expect(state.unlockWith?.key).toBe("custom-1.7b");
    expect(styleState(voice({}), [{ ...small, installed: true }, { ...big, installed: true }]).enabled).toBe(true);
  });

  it("is on for a design voice", () => {
    expect(styleState(voice({ kind: "design", id: "design", models: ["design-1.7b"] }), [design]).enabled).toBe(true);
  });
});

describe("which model runs", () => {
  it("takes the small installed model for plain speech and the style model for style", () => {
    const models = [{ ...small, installed: true }, { ...big, installed: true }];
    expect(chooseModel(voice({}), models, false)?.key).toBe("custom-0.6b");
    expect(chooseModel(voice({}), models, true)?.key).toBe("custom-1.7b");
  });

  it("prefers an installed model, else the smallest, and has none for built-in voices", () => {
    expect(chooseModel(voice({}), [small, { ...big, installed: true }], false)?.key).toBe("custom-1.7b");
    expect(chooseModel(voice({}), [small, big], false)?.key).toBe("custom-0.6b");
    expect(chooseModel(voice({ kind: "builtin" }), [small], false)).toBeNull();
  });

  it("asks for a download only when a non-built-in voice's model is missing", () => {
    expect(modelNote(voice({}), [small, big], false).needsDownload).toBe(true);
    expect(modelNote(voice({}), [{ ...small, installed: true }], false).needsDownload).toBe(false);
    expect(modelNote(voice({ kind: "builtin" }), [], false).needsDownload).toBe(false);
    const busy = { ...small, download: { state: "downloading" as const, error: null } };
    expect(modelNote(voice({}), [busy], false).downloading).toBe(true);
  });
});

describe("generate gate", () => {
  const ready = { model: small, needsDownload: false, downloading: false };
  it("needs text and a downloaded model", () => {
    expect(canGenerate(voice({}), "hi", "", ready)).toBe(true);
    expect(canGenerate(voice({}), "  ", "", ready)).toBe(false);
    expect(canGenerate(voice({}), "hi", "", { ...ready, needsDownload: true })).toBe(false);
    expect(canGenerate(null, "hi", "", ready)).toBe(false);
  });
  it("needs the description for a design voice", () => {
    const d = voice({ kind: "design", id: "design" });
    expect(canGenerate(d, "hi", "", ready)).toBe(false);
    expect(canGenerate(d, "hi", "a calm woman", ready)).toBe(true);
  });
});

describe("text helpers and lists", () => {
  it("turns [pause] into a paragraph break for one-request voices and flags long text", () => {
    expect(cloneRequestText("one [pause] two")).toBe("one\n\ntwo");
    expect(cloneTextTooLong("x".repeat(20_001))).toBe(true);
    expect(cloneTextTooLong("x".repeat(20_000))).toBe(false);
  });

  it("filters and defaults", () => {
    const voices = [
      voice({ id: "af_heart", kind: "builtin" }),
      voice({ id: "jf_alpha", kind: "builtin", available: false }),
      voice({ id: "clone:1", kind: "clone" }),
      voice({ id: "qwen3:Ryan" }),
    ];
    expect(pickerVoices(voices).map((v) => v.id)).toEqual(["af_heart", "clone:1", "qwen3:Ryan"]);
    expect(voices.filter((v) => matchesFilter(v, "mine")).map((v) => v.id)).toEqual(["clone:1"]);
    expect(voices.filter((v) => matchesFilter(v, "builtin")).map((v) => v.id)).toEqual(["af_heart", "jf_alpha", "qwen3:Ryan"]);
    expect(defaultVoice(voices, "qwen3:Ryan")).toBe("qwen3:Ryan");
    expect(defaultVoice(voices, "gone")).toBe("af_heart");
  });
});

describe("cloning model choice", () => {
  const clone: VoiceEntry = {
    id: "clone:1", name: "Me", kind: "clone", language: "auto", model: "b06", models: ["b06", "b17"], capabilities: caps({ clone: true }), available: true,
  };
  const b06 = row({ key: "b06", name: "0.6B", kind: "base", sizeGb: 2.3, installed: true });
  const b17 = row({ key: "b17", name: "1.7B", kind: "base", sizeGb: 3.8, installed: true });

  it("runs the installed model when only the 1.7B is there, and sends its key", () => {
    const only17 = [{ ...b06, installed: false }, b17];
    expect(modelNote(clone, only17, false).model?.key).toBe("b17");
    expect(installedCloneModels(clone, only17).map((m) => m.key)).toEqual(["b17"]);
  });

  it("takes the smaller model by default and the picked one when both are installed", () => {
    expect(modelNote(clone, [b06, b17], false).model?.key).toBe("b06");
    expect(modelNote(clone, [b06, b17], false, "b17").model?.key).toBe("b17");
    expect(installedCloneModels(clone, [b06, b17])).toHaveLength(2);
  });

  it("ignores a pick that is not installed, and a pick for a non-clone voice", () => {
    expect(modelNote(clone, [b06, { ...b17, installed: false }], false, "b17").model?.key).toBe("b06");
    expect(installedCloneModels(null, [b06, b17])).toEqual([]);
  });
});
