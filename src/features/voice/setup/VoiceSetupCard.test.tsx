import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceModelRow, VoiceRuntime } from "../../../types";
import { VoiceProvider } from "../VoiceContext";
import { VoiceRuntimePanel } from "../VoicePackPanel";
import { VoiceSetupCard, type SetupCatalog } from "./VoiceSetupCard";
import type { VoicePackJob } from "./voicePackJob";

const custom: VoiceModelRow = {
  key: "qwen3-tts-custom-1.7b-cuda",
  name: "Qwen3-TTS 1.7B CustomVoice",
  summary: "",
  repo: "Qwen/x",
  sizeGb: 4.52,
  languages: "10 languages",
  default: false,
  kind: "custom",
  lanes: ["cuda"],
  capabilities: { speak: true, style: true, clone: false, design: false },
  supported: false,
  installed: false,
  download: { state: "idle", error: null },
};

function runtime(over: Partial<VoiceRuntime> = {}): VoiceRuntime {
  return {
    platform: "Windows",
    lane: "portable",
    installed: false,
    sttAvailable: false,
    ttsAvailable: false,
    ttsModelInstalled: false,
    vadAvailable: false,
    speakersAvailable: false,
    updateAvailable: false,
    sttBackend: null,
    ttsBackend: null,
    cuda: false,
    nvidiaGpu: true,
    extras: [
      { id: "kokoro", label: "Kokoro", installed: false },
      { id: "chinese", label: "Mandarin", installed: false },
      { id: "speech-gpu", label: "Speech GPU", installed: false },
      { id: "stt-gpu", label: "STT GPU", installed: false },
    ],
    workerPid: null,
    parts: [],
    install: { phase: "idle", phaseLabel: "", phaseIndex: 0, phaseTotal: 5, percent: 0, message: "", error: null, log: [], running: false, extras: [] },
    sttModels: [],
    ttsVoices: [],
    settings: { sttModel: "", language: "", ttsVoice: "af_heart", speed: 1, autoSend: false, autoRead: false, vadSilenceMs: 700, bargeIn: true, idleUnloadMinutes: 10 },
    ...over,
  } as VoiceRuntime;
}

const catalog: SetupCatalog = { models: [custom], loaded: true, refresh: async () => {} };

function pack(rt: VoiceRuntime, over: Partial<VoicePackJob> = {}): VoicePackJob {
  return { job: rt.install, error: null, setError: () => {}, start: async () => true, cancel: () => {}, ...over };
}

function wrap(node: React.ReactNode, rt: VoiceRuntime) {
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Windows", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>{node}</VoiceProvider>,
  );
}

describe("Voice setup card", () => {
  it("before install: every task with what it lacks, Full for this GPU preselected, and one Install", () => {
    const rt = runtime();
    const html = wrap(<VoiceSetupCard runtime={rt} pack={pack(rt)} catalog={catalog} onRuntimeChanged={() => {}} onOpenChat={() => {}} />, rt);
    for (const task of ["Quick speech", "Natural voices and style", "Clone or design a voice", "Speech to text on GPU", "Voice calls"]) expect(html).toContain(task);
    expect(html).toContain("Missing: voice runtime 2.3 GB + GPU speech part 5.7 GB + Qwen3-TTS 1.7B CustomVoice 4.5 GB");
    expect(html).toContain("Missing: voice runtime 2.3 GB. Happens in Chat");
    expect(html).toContain(">Open Chat<");
    expect(html).toMatch(/<input type="radio" name="voice-setup-preset" checked=""\/>Full for this GPU: natural voices, cloning and speech to text on the GPU, about 15.8 GB/);
    expect(html).toMatch(/<input type="radio" name="voice-setup-preset"\/>Basic: quick voices, about 2.6 GB/);
    expect(html).toContain("Install, about 15.8 GB");
    expect(html).toContain("<summary>Choose parts</summary>");
    expect(html).toContain("Optional parts");
    expect(html).toMatch(/<input type="checkbox" checked=""\/>Natural voices, cloning and design on this GPU \(about 5.7 GB\)/);
    expect(html).toMatch(/<input type="checkbox"\/>Mandarin Chinese quick voices \(8 voices, about 60 MB\)/);
    expect(html).toContain("Natural voice model: Qwen3-TTS 1.7B CustomVoice (4.5 GB, downloads after the install)");
  });

  it("on a CPU-only PC: says why, offers Basic only", () => {
    const rt = runtime({ nvidiaGpu: false, extras: [{ id: "kokoro", label: "Kokoro", installed: false }] });
    const html = wrap(<VoiceSetupCard runtime={rt} pack={pack(rt)} catalog={catalog} onRuntimeChanged={() => {}} />, rt);
    expect(html).toContain("Not on this machine: needs an NVIDIA GPU that this computer does not have");
    expect(html).not.toContain("Full for this GPU");
    expect(html).toContain("Install, about 2.6 GB");
  });

  it("installed: Install what is missing with its size; the cloning model is picked on Clone a voice", () => {
    const rt = runtime({
      installed: true,
      vadAvailable: true,
      ttsModelInstalled: true,
      extras: runtime().extras.map((e) => ({ ...e, installed: e.id === "kokoro" })),
    });
    const base = { ...custom, key: "qwen3-tts-base-0.6b-cuda", name: "Qwen3-TTS 0.6B Base", kind: "base" as const, sizeGb: 2.52, capabilities: { speak: false, style: false, clone: true, design: false } };
    const html = wrap(
      <VoiceSetupCard runtime={rt} pack={pack(rt)} catalog={{ ...catalog, models: [custom, base] }} onRuntimeChanged={() => {}} onOpenClone={() => {}} />,
      rt,
    );
    expect(html).toContain("Install what is missing, 13.1 GB");
    expect(html).toContain("Missing: GPU speech part 5.7 GB + a cloning model, such as Qwen3-TTS 0.6B Base 2.5 GB");
    expect(html).not.toContain("voice-setup-preset");
    // With the parts in, only the cloning model is left: its button opens Clone a voice.
    const ready = runtime({ ...rt, extras: rt.extras.map((e) => ({ ...e, installed: true })) });
    const after = wrap(
      <VoiceSetupCard runtime={ready} pack={pack(ready)} catalog={{ ...catalog, models: [{ ...custom, installed: true }, base] }} onRuntimeChanged={() => {}} onOpenClone={() => {}} />,
      ready,
    );
    expect(after).toContain(">Choose a cloning model<");
    expect(after).not.toContain("Install what is missing");
  });

  it("shows a failed install as one sentence with the raw text under Details", () => {
    const rt = runtime();
    const html = wrap(
      <VoiceSetupCard runtime={rt} pack={pack(rt, { error: "The install could not start.", errorDetails: "Traceback: boom" })} catalog={catalog} onRuntimeChanged={() => {}} />,
      rt,
    );
    expect(html).toContain("<p>The install could not start.</p>");
    expect(html).toContain("<summary>Details</summary>");
    expect(html).toContain("Traceback: boom");
  });
});

describe("Voice setup on Models", () => {
  it("names it Voice setup and keeps memory, Unload and Remove once installed", () => {
    const rt = runtime({
      installed: true,
      vadAvailable: true,
      ttsModelInstalled: true,
      workerPid: 12,
      inMemory: [{ lane: "tts", name: "Kokoro", peakGb: 0.87 }],
      parts: [{ id: "packages", sizeGb: 1.9 }, { id: "kokoro", sizeGb: 0.33 }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 }],
      extras: [{ id: "kokoro", label: "Kokoro", installed: true }],
    });
    const html = wrap(<VoiceRuntimePanel runtime={rt} pack={pack(rt)} />, rt);
    expect(html).toContain("Voice setup");
    expect(html).toContain("2.3 GB on disk: speech packages 1.9 GB, Kokoro quick voices 0.3 GB");
    expect(html).toContain("Kokoro, 0.9 GB");
    expect(html).toContain("A part installs when ticked; untick it to remove it.");
    expect(html).toContain("Remove voice runtime");
  });
});
