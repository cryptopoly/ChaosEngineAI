import { describe, expect, it } from "vitest";
import type { VoiceEntry, VoiceModelRow } from "../../../../types";
import {
  canGenerate,
  chooseModel,
  cloneRequestText,
  cloneTextTooLong,
  costLabel,
  defaultVoice,
  ffmpegBlocked,
  groupVoices,
  installedCloneModels,
  quickShortlist,
  matchesFilter,
  modelNote,
  pickerVoices,
  recommendedNaturalModel,
  showInstallCard,
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
  it("runs the installed style model whether or not there is style text (no swap, no reload)", () => {
    const models = [{ ...small, installed: true }, { ...big, installed: true }];
    expect(chooseModel(voice({}), models)?.key).toBe("custom-1.7b");
  });

  it("runs the 0.6B model only when it is the only one installed", () => {
    expect(chooseModel(voice({}), [{ ...small, installed: true }, big])?.key).toBe("custom-0.6b");
    expect(chooseModel(voice({}), [small, { ...big, installed: true }])?.key).toBe("custom-1.7b");
  });

  it("names the style model when nothing is installed, and none for built-in voices", () => {
    expect(chooseModel(voice({}), [small, big])?.key).toBe("custom-1.7b");
    expect(chooseModel(voice({ kind: "builtin" }), [small])).toBeNull();
  });

  it("asks for a download only when a non-built-in voice's model is missing", () => {
    expect(modelNote(voice({}), [small, big]).needsDownload).toBe(true);
    expect(modelNote(voice({}), [{ ...small, installed: true }]).needsDownload).toBe(false);
    expect(modelNote(voice({ kind: "builtin" }), []).needsDownload).toBe(false);
    const busy = { ...big, download: { state: "downloading" as const, error: null } };
    expect(modelNote(voice({}), [small, busy]).downloading).toBe(true);
  });
});

describe("default voice", () => {
  const heart = voice({ id: "af_heart", kind: "builtin", language: "en-US", grade: "A" });
  const nicole = voice({ id: "af_nicole", kind: "builtin", language: "en-US", grade: "B-" });
  const emma = voice({ id: "bf_emma", kind: "builtin", language: "en-GB", grade: "B-" });
  const siwis = voice({ id: "ff_siwis", kind: "builtin", language: "fr", grade: "B-" });
  const ryan = voice({ id: "qwen3:Ryan", language: "en" });
  const serena = voice({ id: "qwen3:Serena", language: "zh" });
  const clone = voice({ id: "clone:1", kind: "clone", language: "auto" });

  it("1. starts on the voice last used in Speak when it is still listed", () => {
    expect(defaultVoice([nicole, heart, ryan, clone], "clone:1", "en")).toBe("clone:1");
    expect(defaultVoice([nicole, heart, ryan], "clone:gone", "en")).toBe("qwen3:Ryan");
  });

  it("2. else the best installed natural voice for the UI language", () => {
    expect(defaultVoice([heart, serena, ryan], "", "en-GB")).toBe("qwen3:Ryan");
    expect(defaultVoice([heart, serena, ryan], "", "zh-CN")).toBe("qwen3:Serena");
    expect(defaultVoice([heart, serena, ryan], "", "de")).toBe("qwen3:Serena"); // no German preset: the first one that runs
  });

  it("3. else the top-graded quick voice for the UI language, not the chat voice", () => {
    const notReady = { ...ryan, available: false };
    expect(defaultVoice([nicole, emma, heart, siwis, notReady], "", "en")).toBe("af_heart");
    expect(defaultVoice([nicole, emma, heart, siwis, notReady], "", "fr")).toBe("ff_siwis");
    expect(defaultVoice([nicole, emma, siwis], "", "de")).toBe("af_nicole"); // no voice in the language: the best graded
  });

  it("4. else the first usable voice", () => {
    expect(defaultVoice([clone], "", "en")).toBe("clone:1");
    expect(defaultVoice([], "", "en")).toBe("");
  });

  it("offers the install card only while no natural model is downloaded on a machine that runs one", () => {
    expect(showInstallCard([small, big, design], true)).toBe(true);
    expect(recommendedNaturalModel([small, big])?.key).toBe("custom-1.7b");
    expect(showInstallCard([{ ...small, installed: true }, big], true)).toBe(false);
    expect(showInstallCard([small, big], false)).toBe(false);
    expect(showInstallCard([], true)).toBe(false);
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
    expect(defaultVoice(voices, "qwen3:Ryan", "en")).toBe("qwen3:Ryan");
    expect(defaultVoice(voices, "jf_alpha", "en")).toBe("qwen3:Ryan"); // not usable here: the rule decides
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
    expect(modelNote(clone, only17).model?.key).toBe("b17");
    expect(installedCloneModels(clone, only17).map((m) => m.key)).toEqual(["b17"]);
  });

  it("takes the smaller model by default and the picked one when both are installed", () => {
    expect(modelNote(clone, [b06, b17]).model?.key).toBe("b06");
    expect(modelNote(clone, [b06, b17], "b17").model?.key).toBe("b17");
    expect(installedCloneModels(clone, [b06, b17])).toHaveLength(2);
  });

  it("ignores a pick that is not installed, and a pick for a non-clone voice", () => {
    expect(modelNote(clone, [b06, { ...b17, installed: false }], "b17").model?.key).toBe("b06");
    expect(installedCloneModels(null, [b06, b17])).toEqual([]);
  });
});

describe("picker groups and the quick shortlist", () => {
  // The 20 American voices with the model author's grades, in the backend's order.
  const AMERICAN: Array<[string, string, string]> = [
    ["af_heart", "female", "A"], ["af_alloy", "female", "C"], ["af_aoede", "female", "C+"], ["af_bella", "female", "A-"],
    ["af_jessica", "female", "D"], ["af_kore", "female", "C+"], ["af_nicole", "female", "B-"], ["af_nova", "female", "C"],
    ["af_river", "female", "D"], ["af_sarah", "female", "C+"], ["af_sky", "female", "C-"],
    ["am_adam", "male", "F+"], ["am_echo", "male", "D"], ["am_eric", "male", "D"], ["am_fenrir", "male", "C+"],
    ["am_liam", "male", "D"], ["am_michael", "male", "C+"], ["am_onyx", "male", "D"], ["am_puck", "male", "C+"], ["am_santa", "male", "D-"],
  ];
  const american = AMERICAN.map(([id, gender, grade]) =>
    voice({ id, kind: "builtin", gender: gender as "female" | "male", grade, language: "en-US", languageLabel: "American English" }),
  );

  it("shortlists the best graded, three of each gender, best first", () => {
    expect(quickShortlist(american).map((v) => v.id)).toEqual(["af_heart", "af_bella", "af_nicole", "am_fenrir", "am_michael", "am_puck"]);
  });

  it("shows the first six of a language with no voice good enough or none graded", () => {
    const mandarin = Array.from({ length: 8 }, (_, i) => voice({ id: `z${i}`, kind: "builtin", grade: "D", language: "zh" }));
    expect(quickShortlist(mandarin).map((v) => v.id)).toEqual(["z0", "z1", "z2", "z3", "z4", "z5"]);
    const spanish = [voice({ id: "ef_dora", kind: "builtin", grade: null }), voice({ id: "em_alex", kind: "builtin", grade: null })];
    expect(quickShortlist(spanish)).toHaveLength(2);
  });

  it("groups Recommended (natural, top quick, newest clone, design), Quick, Cloned, Designed and More", () => {
    const ryan = voice({ id: "qwen3:Ryan", language: "en" });
    const aiden = voice({ id: "qwen3:Aiden", language: "en" });
    const sohee = voice({ id: "qwen3:Sohee", language: "ko" });
    const clone = voice({ id: "clone:1", kind: "clone" });
    const designVoice = voice({ id: "design", kind: "design" });
    const mandarin = voice({ id: "zf_xiaobei", kind: "builtin", available: false, addon: "chinese", language: "zh" });
    const neverHere = voice({ id: "x_gone", kind: "builtin", available: false, addon: null });
    const groups = groupVoices([...american, ryan, aiden, sohee, clone, designVoice, mandarin, neverHere], "en", true);
    expect(groups.recommended.map((v) => v.id)).toEqual(["qwen3:Ryan", "af_heart", "clone:1", "design"]);
    expect(groups.quick).toHaveLength(21); // the Mandarin voice stays so its add-on card can show
    expect(groups.quick.some((v) => v.id === "x_gone")).toBe(false);
    expect(groups.more.map((v) => v.id)).toEqual(["qwen3:Aiden", "qwen3:Sohee"]);
    expect(groups.clones.map((v) => v.id)).toEqual(["clone:1"]);
    expect(groups.design?.id).toBe("design");
  });

  it("recommends no natural voice or design card where the models cannot run, and none until a model is installed", () => {
    const ryan = voice({ id: "qwen3:Ryan", language: "en" });
    expect(groupVoices([american[0], ryan, voice({ id: "design", kind: "design" })], "en", false).recommended.map((v) => v.id)).toEqual(["af_heart"]);
    const notYet = { ...ryan, available: false };
    const groups = groupVoices([american[0], notYet], "en", true);
    expect(groups.recommended.map((v) => v.id)).toEqual(["af_heart"]);
    expect(groups.more.map((v) => v.id)).toEqual(["qwen3:Ryan"]);
  });
});

describe("cost label", () => {
  const measured = row({
    key: "custom-1.7b",
    sizeGb: 3.08,
    estimates: { basis: "measured", measuredOn: "Apple M4 Max, 64 GB, 2026-11-08", coldStartSeconds: 2.4, secondsPer10sAudio: 2.45, peakGb: 3.5, idleGb: 3.48 },
  });

  it("gives the measured memory, speed, cold start and machine", () => {
    expect(costLabel(measured)).toEqual({
      kind: "measured",
      downloadGb: 3.08,
      peakGb: 3.5,
      secondsPer10s: 2.5,
      coldStartS: 2.4,
      measuredOn: "Apple M4 Max, 64 GB, 2026-11-08",
    });
  });

  it("says unmeasured instead of guessing when there is no measured run", () => {
    expect(costLabel(row({ sizeGb: 2.5 }))).toEqual({ kind: "unmeasured", downloadGb: 2.5 });
    expect(costLabel(row({ sizeGb: 2.5, estimates: null }))).toEqual({ kind: "unmeasured", downloadGb: 2.5 });
    expect(costLabel(row({ sizeGb: 2.5, estimates: { basis: "estimate", peakGb: 3, secondsPer10sAudio: 2 } }))).toEqual({ kind: "unmeasured", downloadGb: 2.5 });
  });

  it("names Kokoro for a quick voice (no model row)", () => {
    expect(costLabel(null)).toEqual({ kind: "kokoro" });
  });
});

describe("ffmpegBlocked", () => {
  const quick = { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", model: "kokoro", models: ["kokoro"], capabilities: caps(), available: true } as VoiceEntry;
  const preset = { ...quick, id: "qwen3:Ryan", kind: "preset", models: ["c17"] } as VoiceEntry;

  it("never blocks when ffmpeg is there", () => {
    expect(ffmpegBlocked(preset, 1.5, 3)).toBe(false);
    expect(ffmpegBlocked(null, 1.5, 3)).toBe(false);
  });

  it("blocks pitch on every voice and speed on all but quick voices when ffmpeg is missing", () => {
    const q = { ...quick, needsFfmpeg: true };
    const p = { ...preset, needsFfmpeg: true };
    expect(ffmpegBlocked(q, 1.5, 0)).toBe(false);
    expect(ffmpegBlocked(q, 1, 2)).toBe(true);
    expect(ffmpegBlocked(p, 1, 0)).toBe(false);
    expect(ffmpegBlocked(p, 1.2, 0)).toBe(true);
    expect(ffmpegBlocked(p, 1, -1)).toBe(true);
  });
});
