import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceModelRow, VoiceRuntime } from "../../../../types";
import { VoiceProvider } from "../../VoiceContext";
import { VoiceRuntimePanel } from "../../VoicePackPanel";
import { runtimePartsSummary } from "../../voicePackState";
import type { VoiceCatalog } from "../../speak/useVoiceCatalog";
import { DictationModelsPanel } from "../DictationModelsPanel";
import { SpeechModelsPanel } from "../SpeechModelsPanel";
import { voiceRowToEntry } from "../../../../components/models/adapters/voice";
import {
  collapsedFamilyRows,
  familyHeadingTracker,
  runtimeRemoveConfirmText,
  speechDeleteConfirmText,
  sttDeleteConfirmText,
  studioTargetFor,
} from "../voiceModelsLogic";

const caps = { speak: true, style: false, clone: false, design: false };
const fit = { status: "fits", needGb: 3.5, usableGb: 51.8, headroomPct: 93, reason: "Peak memory", basis: "measured" };

function model(key: string, name: string, over: Partial<VoiceModelRow> = {}): VoiceModelRow {
  return {
    key, name, summary: `${name} summary.`, repo: `mlx-community/${key}`, sizeGb: 2.5, languages: "English", default: false, kind: "custom",
    family: "qwen3-tts", capabilities: caps, supported: true, installed: false, fit, licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid" },
    estimates: { basis: "measured", peakGb: 3.5, secondsPer10sAudio: 2.45, measuredOn: "Apple M4 Max" }, download: { state: "idle", error: null },
    ...over,
  };
}

const models = [
  model("qwen3-tts-custom-1.7b", "Qwen3-TTS 1.7B CustomVoice", { installed: true, capabilities: { ...caps, style: true } }),
  model("qwen3-tts-0.6b", "Qwen3-TTS 0.6B Base", { kind: "base", capabilities: { ...caps, clone: true } }),
  model("chatterbox-turbo", "Chatterbox-Turbo", { kind: "turbo", family: "chatterbox-turbo", installed: true, capabilities: { ...caps, clone: true, tags: true, presets: true } }),
  model("orpheus-3b", "Orpheus 3B", { kind: "orpheus", family: "orpheus", capabilities: { ...caps, tags: true, presets: true } }),
];
const voices: VoiceEntry[] = [
  { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", model: "qwen3-tts-custom-1.7b", models: ["qwen3-tts-custom-1.7b"], capabilities: caps, available: true },
];
const catalog: VoiceCatalog = {
  voices, models, modelsAvailable: true, cloneAllowed: true, loaded: true, error: null, refresh: async () => {}, download: async () => {}, remove: async () => {},
  kokoro: { peakGb: 0.87, fit },
};

function runtime(over: Partial<VoiceRuntime> = {}): VoiceRuntime {
  return {
    platform: "Darwin", lane: "mlx", installed: true, sttAvailable: true, ttsAvailable: true, ttsModelInstalled: true, vadAvailable: true, speakersAvailable: true,
    updateAvailable: false, sttBackend: null, ttsBackend: null, cuda: false, nvidiaGpu: false,
    extras: [{ id: "chinese", label: "Mandarin Chinese voices", installed: false }], installedAt: "2026-11-09T09:00:00", workerPid: 123,
    parts: [{ id: "packages", sizeGb: 1.9 }, { id: "kokoro", sizeGb: 0.33 }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 }],
    inMemory: [{ lane: "tts", name: "Kokoro", peakGb: 0.87 }],
    install: { phase: "done", phaseLabel: "", phaseIndex: 0, phaseTotal: 5, percent: 100, message: "", error: null, log: [], running: false, extras: [] },
    sttModels: [
      { id: "mlx-community/parakeet", key: "parakeet", name: "Parakeet TDT 0.6B v3", languages: "25 European languages", sizeGb: 2.5, installed: true, default: true },
      { id: "mlx-community/whisper-turbo", key: "whisper-turbo", name: "Whisper Large v3 Turbo", languages: "99 languages", sizeGb: 0.8, installed: false, default: false },
    ],
    ttsVoices: [{ id: "af_heart", name: "Heart", language: "en-US" }, { id: "bf_emma", name: "Emma", language: "en-GB" }],
    settings: { sttModel: "", language: "", ttsVoice: "af_heart", speed: 1, autoSend: false, autoRead: false, vadSilenceMs: 700, bargeIn: true, idleUnloadMinutes: 10 },
    ...over,
  } as VoiceRuntime;
}

function wrap(node: React.ReactNode, rt = runtime()) {
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>{node}</VoiceProvider>,
  );
}

describe("Speech models on the shared Models table (D-73 M3b)", () => {
  const html = wrap(<SpeechModelsPanel catalog={catalog} runtime={runtime()} availability="all" onAvailabilityChange={() => {}} onOpenStudio={() => {}} />);

  it("starts with Kokoro, an add-on of the runtime, with Browse voices and no Delete", () => {
    const kokoro = html.slice(html.indexOf('data-model-id="kokoro"'), html.indexOf('data-model-id="qwen3-tts-custom-1.7b"'));
    expect(html.indexOf('data-model-id="kokoro"')).toBeLessThan(html.indexOf('data-model-id="qwen3-tts-custom-1.7b"'));
    expect(kokoro).toContain("2 quick voices in 2 languages");
    expect(kokoro).toContain("an add-on of the voice runtime");
    expect(kokoro).not.toContain("falls back to");
    expect(kokoro).toContain("Part of the runtime");
    expect(kokoro).toContain("Browse voices");
    expect(kokoro).not.toContain("Delete model");
  });

  it("shows Kokoro as not installed, with where to add it, when it was left out (D-75)", () => {
    const out = wrap(
      <SpeechModelsPanel catalog={catalog} runtime={runtime({ ttsModelInstalled: false })} availability="all" onAvailabilityChange={() => {}} onOpenStudio={() => {}} />,
    );
    const kokoro = out.slice(out.indexOf('data-model-id="kokoro"'), out.indexOf('data-model-id="qwen3-tts-custom-1.7b"'));
    expect(kokoro).toContain("Not installed");
    expect(kokoro).toContain("Tick it under Voice runtime &gt; Add-ons.");
    expect(kokoro).not.toContain("Browse voices");
  });

  it("groups the rows under family headings with their licence", () => {
    for (const family of ["Qwen3-TTS", "Chatterbox", "Orpheus"]) expect(html).toContain(`<strong>${family}</strong>`);
    expect(html.indexOf("<strong>Qwen3-TTS</strong>")).toBeLessThan(html.indexOf("<strong>Chatterbox</strong>"));
    expect(html.indexOf("<strong>Chatterbox</strong>")).toBeLessThan(html.indexOf("<strong>Orpheus</strong>"));
    expect(html).toContain("Natural voices, style instructions, voice design and cloning · Apache 2.0");
  });

  it("shows Makes, Size, Fit and Status columns, Delete on a downloaded row and Download on the others", () => {
    for (const column of ["Model", "Makes", "Size", "Fit", "Status"]) expect(html).toContain(column);
    const custom = html.slice(html.indexOf('data-model-id="qwen3-tts-custom-1.7b"'), html.indexOf('data-model-id="qwen3-tts-0.6b"'));
    expect(custom).toContain("Delete model");
    expect(custom).toContain("Use in Speak");
    expect(custom).toContain("Natural voices");
    expect(custom).toContain("Style");
    const base = html.slice(html.indexOf('data-model-id="qwen3-tts-0.6b"'), html.indexOf('data-model-id="chatterbox-turbo"'));
    expect(base).toContain("Download 2.50 GB");
    expect(base).not.toContain("Delete model");
  });

  it("says why a row cannot run on this GPU, from the numbers the backend sends, else that it is Apple-only", () => {
    const panel = (over: Partial<VoiceModelRow>) =>
      wrap(
        <SpeechModelsPanel
          catalog={{ ...catalog, models: [model("only-row", "Only row", { supported: false, ...over })] }}
          runtime={runtime()}
          availability="all"
          onAvailabilityChange={() => {}}
          onOpenStudio={() => {}}
        />,
      );
    expect(panel({ unsupportedFacts: { kind: "memory", need: 5.4, have: 4 } })).toContain("Needs about 5.4 GB of GPU memory; this GPU has 4 GB.");
    expect(panel({ unsupportedFacts: { kind: "compute", need: 7, have: 6.1 } })).toContain(
      "Needs an NVIDIA GPU of compute capability 7 or newer; this one is 6.1.",
    );
    expect(panel({})).toContain("Apple Silicon Macs only for now.");
  });

  it("offers a natural voice while none is downloaded (first-run callout)", () => {
    const none = wrap(
      <SpeechModelsPanel catalog={{ ...catalog, models: models.map((m) => ({ ...m, installed: false })) }} runtime={runtime()} availability="all" onAvailabilityChange={() => {}} />,
    );
    expect(none).toContain("Install a better voice");
    expect(html).not.toContain("Install a better voice");
  });
});

describe("Dictation models (D-73 M1)", () => {
  const html = wrap(<DictationModelsPanel runtime={runtime()} onChanged={() => {}} />);
  it("has the dictation language in its header and a row menu with Delete on a downloaded model", () => {
    expect(html).toContain("Dictation models");
    expect(html).toContain("Detect automatically");
    const parakeet = html.slice(html.indexOf('data-stt-model="parakeet"'), html.indexOf('data-stt-model="whisper-turbo"'));
    expect(parakeet).toContain("Delete model (2.5 GB)");
    expect(parakeet).toContain("More actions for Parakeet TDT 0.6B v3");
    const whisper = html.slice(html.indexOf('data-stt-model="whisper-turbo"'));
    expect(whisper).toContain("Download 0.8 GB");
    expect(whisper).not.toContain("Delete model");
    expect(html).toContain("fetched the first time you dictate");
    expect(html).not.toContain("Downloads on first use if you skip this");
  });
});

describe("Voice runtime panel (D-73 M4, M5)", () => {
  const pack = { job: runtime().install, error: null, setError: () => {}, start: async () => {}, cancel: () => undefined };
  it("names the runtime, lists its parts with sizes, what is in memory, the add-ons and Remove", () => {
    const html = wrap(<VoiceRuntimePanel runtime={runtime()} pack={pack} />);
    expect(html).toContain("Voice runtime");
    expect(html).not.toContain("Voice pack");
    expect(html).toContain("2.3 GB on disk: speech packages 1.9 GB, Kokoro quick voices 0.3 GB");
    expect(html).toContain("In memory now:");
    expect(html).toContain("Kokoro, 0.9 GB");
    expect(html).toContain(">Unload<");
    expect(html).toContain("minutes idle");
    expect(html).toMatch(/<input type="checkbox"\/>Mandarin Chinese quick voices/);
    expect(html).toContain("Installs when ticked; untick to remove.");
    expect(html).toContain("Remove voice runtime");
    expect(html).toContain("Downloaded dictation and speech models always stay");
  });

  it("before install, the add-ons are chosen with the install", () => {
    const html = wrap(<VoiceRuntimePanel runtime={runtime({ installed: false, parts: [], workerPid: null })} pack={pack} />);
    expect(html).toContain("Install voice runtime");
    expect(html).toContain("Ticked add-ons install with the runtime");
    expect(html).not.toContain("Remove voice runtime");
    expect(html).not.toContain("In memory now");
  });

  it("ticks an installed add-on", () => {
    const html = wrap(<VoiceRuntimePanel runtime={runtime({ extras: [{ id: "chinese", label: "x", installed: true }] })} pack={pack} />);
    expect(html).toMatch(/<input type="checkbox" checked=""\/>Mandarin Chinese quick voices/);
  });

  it("offers Kokoro as an add-on ticked by default at install (D-75)", () => {
    const extras = [
      { id: "kokoro", label: "Kokoro quick voices", installed: false },
      { id: "chinese", label: "Mandarin Chinese voices", installed: false },
    ];
    const before = wrap(<VoiceRuntimePanel runtime={runtime({ installed: false, parts: [], workerPid: null, extras, ttsModelInstalled: false })} pack={pack} />);
    expect(before).toMatch(/<input type="checkbox" checked=""\/>Kokoro quick voices \(about 330 MB\)/);
    expect(before).toMatch(/<input type="checkbox"\/>Mandarin Chinese quick voices/);
    expect(before).toContain("plus the Kokoro quick voices when ticked below");
    // Installed without Kokoro: unticked, the runtime is healthy, and the parts line says Kokoro is not installed.
    const without = wrap(
      <VoiceRuntimePanel
        runtime={runtime({ extras, ttsModelInstalled: false, parts: [{ id: "packages", sizeGb: 1.9 }, { id: "kokoro", sizeGb: 0 }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 }] })}
        pack={pack}
      />,
    );
    expect(without).toMatch(/<input type="checkbox"\/>Kokoro quick voices/);
    expect(without).toContain("the Kokoro quick voices are not installed");
    expect(without).toContain("Installed, checked OK");
    expect(without).not.toContain("Repair voice runtime");
  });
});

describe("Voice Models rules", () => {
  it("adds up the runtime's parts", () => {
    expect(runtimePartsSummary(runtime().parts ?? [])).toEqual({ total: "2.3", packages: "1.9", kokoro: "0.3", small: "0.04" });
    expect(runtimePartsSummary([])).toBeNull();
  });

  it("names the model, its size and what stays in the delete confirm", () => {
    const text = speechDeleteConfirmText(
      { key: "k", name: "Chatterbox-Turbo", remove: ["a"], kept: [{ repo: "mlx-community/S3TokenizerV2", usedBy: ["Chatterbox Multilingual V3"] }], watermark: true, sizeGb: 0.71 },
      1.2,
    );
    expect(text).toContain("Delete Chatterbox-Turbo (0.71 GB)");
    expect(text).toContain("Cloned voices and Library items stay");
    expect(text).toContain("Kept: mlx-community/S3TokenizerV2, still used by Chatterbox Multilingual V3.");
    expect(text).toContain("The watermark component goes too");
    expect(speechDeleteConfirmText({ key: "k", name: "X", remove: [], kept: [], watermark: false, sizeGb: 0 }, 1.2)).toContain("(1.20 GB)");
    expect(sttDeleteConfirmText("Parakeet", 2.5)).toContain("Delete Parakeet (2.50 GB)");
    expect(runtimeRemoveConfirmText(2.3)).toContain("Remove the voice runtime (2.3 GB)?");
    expect(runtimeRemoveConfirmText(2.3)).toContain("Downloaded dictation and speech models, cloned voices and the Library stay.");
  });

  it("opens a family heading once, on its first row, however often it is asked", () => {
    const first = familyHeadingTracker();
    const rows = [{ id: "a", familyId: "qwen3-tts" }, { id: "b", familyId: "qwen3-tts" }, { id: "c", familyId: "orpheus" }];
    expect(rows.map(first)).toEqual([true, false, true]);
    expect(rows.map(first)).toEqual([true, false, true]);
  });

  it("collapses a family of three or more to its downloaded, downloading or just-cancelled rows, else its first row", () => {
    const five = ["a", "b", "c", "d", "e"].map((k) => voiceRowToEntry(model(k, k)));
    expect([...collapsedFamilyRows(five).get("qwen3-tts")!.shown]).toEqual(["a"]);
    const some = ["a", "b", "c", "d", "e"].map((k) =>
      voiceRowToEntry(model(k, k, k === "c" ? { installed: true } : k === "e" ? { download: { state: "downloading", error: null } } : k === "d" ? { download: { state: "cancelled", error: null } } : {})),
    );
    const fold = collapsedFamilyRows(some).get("qwen3-tts")!;
    expect(fold.total).toBe(5);
    expect([...fold.shown]).toEqual(["c", "d", "e"]);
    expect(collapsedFamilyRows(five.slice(0, 2)).size).toBe(0);
  });

  it("lands Use in Speak on a preset of the model, the design voice, or Voices for a model that only clones", () => {
    const preset = (id: string, language: string, key: string): VoiceEntry => ({ id, name: id, kind: "preset", language, model: key, models: [key], capabilities: caps, available: true });
    const custom = models[0];
    expect(studioTargetFor(custom, [preset("qwen3:Serena", "zh", custom.key), preset("qwen3:Ryan", "en", custom.key)], "en-GB")).toEqual({ mode: "speak", voice: "qwen3:Ryan" });
    expect(studioTargetFor(custom, [preset("qwen3:Serena", "zh", custom.key), preset("qwen3:Ryan", "en", custom.key)], "zh-CN")).toEqual({ mode: "speak", voice: "qwen3:Serena" });
    expect(studioTargetFor({ ...custom, kind: "design" }, [], "en")).toEqual({ mode: "speak", voice: "design" });
    expect(studioTargetFor({ ...custom, kind: "voxcpm", capabilities: { ...caps, design: true, clone: true } }, [], "en")).toEqual({ mode: "speak", voice: "design" });
    expect(studioTargetFor(models[1], [], "en")).toEqual({ mode: "voices" });
    expect(studioTargetFor(models[2], [preset("chatterbox-turbo", "en", "chatterbox-turbo")], "en")).toEqual({ mode: "speak", voice: "chatterbox-turbo" });
  });
});
