import { describe, expect, it } from "vitest";
import { gbText, installPlan, setupPresets, setupTasks, type SetupInput, type SetupModel } from "./voiceSetupLogic";

function model(key: string, overrides: Partial<SetupModel> = {}): SetupModel {
  return {
    key,
    name: key,
    kind: "custom",
    lanes: ["cuda"],
    capabilities: { speak: true, style: true, clone: false, design: false },
    installed: false,
    sizeGb: 4.52,
    unsupportedFacts: null,
    fit: null,
    download: { state: "idle", error: null },
    ...overrides,
  };
}

const custom = model("qwen3-tts-custom-1.7b-cuda", { name: "Qwen3-TTS 1.7B CustomVoice" });
const base = model("qwen3-tts-base-0.6b-cuda", { kind: "base", sizeGb: 2.52, capabilities: { speak: false, style: false, clone: true, design: false } });

function pc(overrides: Partial<SetupInput> = {}): SetupInput {
  return {
    lane: "portable",
    installed: false,
    needsUpdate: false,
    extras: [
      { id: "kokoro", installed: false },
      { id: "chinese", installed: false },
      { id: "speech-gpu", installed: false },
      { id: "stt-gpu", installed: false },
    ],
    ttsModelInstalled: false,
    models: [custom, base],
    modelsLoaded: true,
    ...overrides,
  };
}

const byTask = (input: SetupInput) => Object.fromEntries(setupTasks(input).map((t) => [t.task, t]));

describe("voice setup tasks", () => {
  it("lists what a fresh NVIDIA PC lacks for each task, with sizes from the code and the catalog", () => {
    const tasks = byTask(pc());
    expect(tasks.quick).toEqual({ task: "quick", state: "missing", parts: [{ id: "runtime", gb: 2.3 }, { id: "kokoro", gb: 0.33 }] });
    expect(tasks.natural).toMatchObject({
      state: "missing",
      parts: [{ id: "runtime" }, { id: "speech-gpu", gb: 5.7 }, { id: "natural-model", gb: 4.52, modelKey: custom.key }],
    });
    expect(tasks.clone).toMatchObject({ state: "missing", parts: [{ id: "runtime" }, { id: "speech-gpu" }, { id: "clone-model", modelKey: base.key, gb: 2.52 }] });
    expect(tasks.stt).toMatchObject({ state: "missing", parts: [{ id: "runtime" }, { id: "stt-gpu", gb: 2.9 }] });
    expect(tasks.calls).toEqual({ task: "calls", state: "missing", parts: [{ id: "runtime", gb: 2.3 }] });
  });

  it("is ready once the parts and the models are on disk", () => {
    const extras = pc().extras.map((e) => ({ ...e, installed: e.id !== "chinese" }));
    const tasks = setupTasks(pc({ installed: true, extras, ttsModelInstalled: true, models: [{ ...custom, installed: true }, { ...base, installed: true }] }));
    expect(tasks.every((t) => t.state === "ready")).toBe(true);
  });

  it("says a CPU-only PC cannot run natural voices, cloning or GPU speech to text", () => {
    const tasks = byTask(pc({ extras: [{ id: "kokoro", installed: true }], installed: true, ttsModelInstalled: true }));
    expect(tasks.quick.state).toBe("ready");
    expect(tasks.natural).toEqual({ task: "natural", state: "unavailable", reason: { kind: "needs-gpu" } });
    expect(tasks.clone).toEqual({ task: "clone", state: "unavailable", reason: { kind: "needs-gpu" } });
    expect(tasks.stt).toEqual({ task: "stt", state: "unavailable", reason: { kind: "needs-gpu" } });
  });

  it("gives the GPU's numbers when every natural model is too big", () => {
    const small = { kind: "memory" as const, need: 6, have: 4 };
    const tasks = byTask(pc({ models: [{ ...custom, unsupportedFacts: small }] }));
    expect(tasks.natural).toEqual({ task: "natural", state: "unavailable", reason: small });
    // A measured fit of "wont-fit" counts as too big before the add-on can report facts.
    const fit = { status: "wont-fit", needGb: 7, usableGb: 5, headroomPct: null, reason: "", basis: "measured" };
    expect(byTask(pc({ models: [{ ...custom, fit }] })).natural).toEqual({ task: "natural", state: "unavailable", reason: { kind: "memory", need: 7, have: 5 } });
  });

  it("needs only the runtime on a Mac for speech to text, and no add-on for natural voices", () => {
    const mac = pc({ lane: "mlx", extras: [{ id: "kokoro", installed: true }], installed: true, ttsModelInstalled: true, models: [{ ...custom, lanes: ["mlx"] }] });
    const tasks = byTask(mac);
    expect(tasks.stt.state).toBe("ready");
    expect(tasks.natural).toMatchObject({ state: "missing", parts: [{ id: "natural-model" }] });
    expect(tasks.clone).toEqual({ task: "clone", state: "unavailable", reason: { kind: "no-model" } });
  });

  it("counts a model as missing with no size until the list arrives, and an old runtime as needing its update", () => {
    const tasks = byTask(pc({ models: [], modelsLoaded: false }));
    expect(tasks.natural).toMatchObject({ state: "missing", parts: [{ id: "runtime" }, { id: "speech-gpu" }, { id: "natural-model", gb: null }] });
    const old = byTask(pc({ installed: true, needsUpdate: true, extras: [{ id: "kokoro", installed: true }], ttsModelInstalled: true }));
    expect(old.calls).toEqual({ task: "calls", state: "missing", parts: [{ id: "update", gb: null }] });
  });

  it("marks a model download in progress", () => {
    const downloading = { ...custom, download: { state: "downloading" as const, error: null, progress: 0.4 } };
    const extras = pc().extras.map((e) => ({ ...e, installed: true }));
    const tasks = byTask(pc({ installed: true, extras, ttsModelInstalled: true, models: [downloading, { ...base, installed: true }] }));
    expect(tasks.natural).toMatchObject({ state: "missing", parts: [{ id: "natural-model", downloading: 0.4 }] });
    // Already downloading: not installed again.
    expect(installPlan(setupTasks(pc({ installed: true, extras, ttsModelInstalled: true, models: [downloading] })))).toMatchObject({ runJob: false, modelKey: null });
  });
});

describe("install what is missing", () => {
  it("installs every add-on once and the natural voice model, leaving the cloning model to Clone a voice", () => {
    const plan = installPlan(setupTasks(pc()));
    expect(plan).toEqual({ runJob: true, addons: ["kokoro", "speech-gpu", "stt-gpu"], modelKey: custom.key, gb: 15.8, partial: false });
  });

  it("only downloads the model when the parts are in", () => {
    const extras = pc().extras.map((e) => ({ ...e, installed: true }));
    expect(installPlan(setupTasks(pc({ installed: true, extras, ttsModelInstalled: true })))).toEqual({
      runJob: false,
      addons: [],
      modelKey: custom.key,
      gb: 4.5,
      partial: false,
    });
  });

  it("runs the update with an unknown size", () => {
    const plan = installPlan(setupTasks(pc({ lane: "mlx", installed: true, needsUpdate: true, extras: [{ id: "kokoro", installed: true }], ttsModelInstalled: true, models: [] })));
    expect(plan).toEqual({ runJob: true, addons: [], modelKey: null, gb: 0, partial: true });
  });
});

describe("first install presets (D0141)", () => {
  it("offers Basic at about 2.6 GB and Full for this GPU at about 16 GB, Full preselected", () => {
    const presets = setupPresets(pc());
    expect(presets.basic).toEqual({ addons: ["kokoro"], modelKey: null, gb: 2.6, partial: false });
    expect(presets.full).toEqual({ addons: ["kokoro", "speech-gpu", "stt-gpu"], modelKey: custom.key, gb: 15.8, partial: false });
    expect(presets.preferred).toBe("full");
  });

  it("starts on Basic when the GPU is too small, and has no Full on a CPU-only PC", () => {
    const tooSmall = setupPresets(pc({ models: [{ ...custom, unsupportedFacts: { kind: "memory", need: 6, have: 4 } }] }));
    expect(tooSmall.preferred).toBe("basic");
    expect(tooSmall.full?.modelKey).toBeNull();
    const cpu = setupPresets(pc({ extras: [{ id: "kokoro", installed: false }] }));
    expect(cpu.full).toBeNull();
    expect(cpu.preferred).toBe("basic");
  });

  it("leaves the model out of Full until the model list names it, and has no Full on a Mac without one", () => {
    const early = setupPresets(pc({ models: [], modelsLoaded: false }));
    expect(early.full).toEqual({ addons: ["kokoro", "speech-gpu", "stt-gpu"], modelKey: null, gb: 11.2, partial: true });
    const mac = setupPresets(pc({ lane: "mlx", extras: [{ id: "kokoro", installed: false }], models: [], modelsLoaded: false }));
    expect(mac.full).toBeNull();
  });

  it("formats sizes", () => {
    expect(gbText(15.8)).toBe("15.8");
    expect(gbText(0.04)).toBe("0.04");
  });
});
