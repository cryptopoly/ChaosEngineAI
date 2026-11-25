import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceModelRow, VoiceRuntime } from "../../../../types";
import { VoiceProvider } from "../../VoiceContext";
import { VoiceRuntimePanel } from "../../VoicePackPanel";
import { runtimePartsSummary } from "../../voicePackState";
import type { VoiceCatalog } from "../../speak/useVoiceCatalog";
import { DictationModelsPanel, SttDownloadProgress } from "../DictationModelsPanel";
import { MacOnlyModels, SpeechModelsPanel } from "../SpeechModelsPanel";
import { voiceRowToEntry } from "../../../../components/models/adapters/voice";
import { addonSizeText, runtimeInstallGb } from "../../runtimeSize";
import {
  collapsedFamilyRows,
  familyHeadingTracker,
  gpuDefaultPending,
  licenceLabel,
  makesSentence,
  pickSizeText,
  recommendedSpeechPick,
  recommendedSttModel,
  speechGpuMemoryText,
  sttGpuMemoryText,
  macOnlyOnThisPc,
  splitForLane,
  sttFitText,
  speechGpuAddonMissing,
  unsupportedModelText,
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
    expect(kokoro).toContain("Tick it under Voice setup &gt; Choose parts.");
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

  it("on a PC without the speech GPU add-on, a GPU row says to tick the add-on and an Apple row stays Apple-only", () => {
    const pc = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] });
    // The rows only: the recommended card above them says the same about the add-on (it falls back to Kokoro here).
    const panel = (over: Partial<VoiceModelRow>, rt = pc) => {
      const out = wrap(
        <SpeechModelsPanel
          catalog={{ ...catalog, modelsAvailable: false, models: [model("only-row", "Only row", { supported: false, ...over })] }}
          runtime={rt}
          availability="all"
          onAvailabilityChange={() => {}}
          onOpenStudio={() => {}}
        />,
      );
      return out.slice(out.indexOf(">All models<"));
    };
    const addon = "Needs the speech GPU part. Tick it under Voice setup &gt; Choose parts.";
    expect(panel({ lanes: ["cuda"] })).toContain(addon);
    expect(panel({ lanes: ["mlx"] })).not.toContain(addon);
    expect(panel({ lanes: ["mlx"] })).toContain("Apple Silicon Macs only for now.");
    // With the add-on in, a GPU row that still cannot run has the GPU's numbers; no add-on line.
    const withAddon = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: true }] });
    expect(panel({ lanes: ["cuda"], unsupportedFacts: { kind: "memory", need: 5.4, have: 4 } }, withAddon)).not.toContain(addon);
    expect(unsupportedModelText({ lanes: ["cuda"] }, runtime())).toBe("Apple Silicon Macs only for now.");
    expect(speechGpuAddonMissing(pc)).toBe(true);
    expect(speechGpuAddonMissing(withAddon)).toBe(false);
    expect(speechGpuAddonMissing(runtime())).toBe(false);
  });

  it("the fit badge and tooltip name the missing speech GPU part, and keep Not for this machine for a real hardware limit", () => {
    const unknown = { status: "unknown", needGb: null, usableGb: 22.5, headroomPct: null, reason: "This model does not run on this machine.", basis: "estimated" };
    const pc = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] });
    const rows = (over: Partial<VoiceModelRow>, rt = pc) => {
      const out = wrap(
        <SpeechModelsPanel
          catalog={{ ...catalog, modelsAvailable: false, models: [model("only-row", "Only row", { supported: false, fit: unknown, ...over })] }}
          runtime={rt}
          availability="all"
          onAvailabilityChange={() => {}}
          onOpenStudio={() => {}}
        />,
      );
      return out.slice(out.indexOf(">All models<"));
    };
    const part = rows({ lanes: ["cuda"] });
    expect(part).toContain("Needs GPU part");
    expect(part).not.toContain("Not for this machine");
    expect(part).not.toContain("does not run on this machine");
    const small = rows({ lanes: ["cuda"], unsupportedFacts: { kind: "memory", need: 5.4, have: 4 } });
    expect(small).toContain("Not for this machine");
    expect(small).not.toContain("Needs GPU part");
    expect(small).toContain("Needs about 5.4 GB of GPU memory; this GPU has 4 GB.");
    expect(rows({ lanes: ["mlx"] })).toContain("Not for this machine");
  });

  it("opens with the one model recommended for this machine: what it makes, its size, about how much GPU memory, then All models", () => {
    const pickHtml = html.slice(html.indexOf("data-speech-pick"), html.indexOf(">All models<"));
    expect(html).toContain("Recommended for this machine");
    expect(html.indexOf("data-speech-pick")).toBeLessThan(html.indexOf(">All models<"));
    expect(html.indexOf(">All models<")).toBeLessThan(html.indexOf('data-model-id="kokoro"'));
    expect(pickHtml).toContain("<strong>Qwen3-TTS 1.7B CustomVoice</strong>: Natural voices, style");
    expect(pickHtml).toContain("2.5 GB download, about 3.5 GB GPU memory.");
    // Downloaded: a badge and Use in Speak, no Download button.
    expect(pickHtml).toContain("Downloaded");
    expect(pickHtml).toContain("Use in Speak");
    expect(pickHtml).not.toContain("Download 2.50 GB");
    // The table below still lists every model.
    expect(html).toContain('data-model-id="qwen3-tts-custom-1.7b"');
  });

  it("offers Download on the recommended model while none is downloaded, replacing the old install card", () => {
    const none = wrap(
      <SpeechModelsPanel catalog={{ ...catalog, models: models.map((m) => ({ ...m, installed: false })) }} runtime={runtime()} availability="all" onAvailabilityChange={() => {}} />,
    );
    const pickHtml = none.slice(none.indexOf("data-speech-pick"), none.indexOf(">All models<"));
    expect(pickHtml).toContain("Download 2.50 GB");
    expect(none).not.toContain("Install a better voice");
  });

  it("leaves out a memory figure the catalogue does not have, and falls back to Kokoro when no natural model runs here", () => {
    const unmeasured = wrap(
      <SpeechModelsPanel
        catalog={{ ...catalog, models: models.map((m) => ({ ...m, estimates: null })) }}
        runtime={runtime()}
        availability="all"
        onAvailabilityChange={() => {}}
      />,
    );
    const pickHtml = unmeasured.slice(unmeasured.indexOf("data-speech-pick"), unmeasured.indexOf(">All models<"));
    expect(pickHtml).toContain("2.5 GB download.");
    expect(pickHtml).not.toContain("GPU memory");
    const pc = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] });
    const kokoro = wrap(
      <SpeechModelsPanel
        catalog={{ ...catalog, modelsAvailable: false, models: models.map((m) => ({ ...m, supported: false, lanes: ["cuda"] })) }}
        runtime={pc}
        availability="all"
        onAvailabilityChange={() => {}}
        onOpenStudio={() => {}}
      />,
    );
    const kokoroPick = kokoro.slice(kokoro.indexOf("data-speech-pick"), kokoro.indexOf(">All models<"));
    expect(kokoroPick).toContain("<strong>Kokoro</strong>: 2 quick voices in 2 languages, fast on any machine");
    expect(kokoroPick).toContain("about 0.9 GB GPU memory.");
    expect(kokoroPick).toContain("The natural voices need the speech GPU part.");
    expect(kokoroPick).toContain("Browse voices");
  });

  it("shows a failed download as a plain sentence with the raw text under Details", () => {
    const failed = wrap(
      <SpeechModelsPanel
        catalog={{
          ...catalog,
          models: [model("only-row", "Only row", { kind: "custom", capabilities: { ...caps, style: true }, download: { state: "failed", error: "401 Client Error", errorInfo: { code: "download_blocked", message: "x", action: "add-hf-token", extra: null } } })],
        }}
        runtime={runtime()}
        availability="all"
        onAvailabilityChange={() => {}}
      />,
    );
    expect(failed).toContain("The download was refused; the model may need a Hugging Face sign-in.");
    expect(failed).toContain("<summary>Details</summary>");
    expect(failed).toContain("401 Client Error");
  });
});

describe("Dictation models (D-73 M1)", () => {
  const html = wrap(<DictationModelsPanel runtime={runtime()} onChanged={() => {}} />);
  it("is called Speech to text, has no language select (Transcribe owns it) and a row menu with Delete on a downloaded model", () => {
    expect(html).toContain("<h2>Speech to text</h2>");
    expect(html).not.toContain("Dictation models");
    expect(html).not.toContain("<select");
    expect(html).not.toContain("Detect automatically");
    const parakeet = html.slice(html.indexOf('data-stt-model="parakeet"'), html.indexOf('data-stt-model="whisper-turbo"'));
    expect(parakeet).toContain("Delete model (2.5 GB)");
    expect(parakeet).toContain("More actions for Parakeet TDT 0.6B v3");
    const whisper = html.slice(html.indexOf('data-stt-model="whisper-turbo"'));
    expect(whisper).toContain("Download 0.8 GB");
    expect(whisper).not.toContain("Delete model");
    expect(html).toContain("downloads before the first file you transcribe or dictate");
    expect(html).toContain("Use as default");
    expect(html).not.toContain("Use for dictation");
  });

  it("does not offer a GPU row this GPU cannot run, and says which GPU row becomes the default once downloaded", () => {
    const gpu = (key: string, name: string, over: Partial<VoiceRuntime["sttModels"][number]> = {}) => ({
      id: `org/${key}`, key, name, languages: "30 languages", sizeGb: 4.09, installed: false, default: false, backend: "transformers-asr", licence: "Apache-2.0", ...over,
    });
    const whisper = { id: "org/whisper", key: "whisper-large-v3-turbo", name: "Whisper Large v3 Turbo", languages: "99 languages", sizeGb: 1.6, installed: false, default: true, backend: "faster-whisper" };
    const big = gpu("big", "Big 30B", { sizeGb: 67.5, gpuFacts: { kind: "memory", need: 72, have: 24 } });
    const html = wrap(<DictationModelsPanel runtime={runtime({ lane: "portable", sttModels: [gpu("qwen3-asr-1.7b", "Qwen3-ASR 1.7B"), whisper, big] })} onChanged={() => {}} />);
    const card = html.slice(html.indexOf('data-stt-model="big"'));
    expect(card).toContain("Needs about 72 GB of GPU memory; this GPU has 24 GB.");
    expect(card).not.toContain("Download 67.5 GB");
    expect(card).toMatch(/<input[^>]*disabled=""[^>]*name="voice-stt-model"|<input[^>]*name="voice-stt-model"[^>]*disabled=""/);
    expect(html).toContain("Download 4.09 GB");
    expect(html).toContain("Qwen3-ASR 1.7B becomes the default speech to text model once it is downloaded, unless you pick another.");
    expect(gpuDefaultPending([gpu("q", "Q", { installed: true }), whisper])).toBeUndefined();
    expect(gpuDefaultPending([big, whisper])).toBeUndefined();
    expect(gpuDefaultPending([whisper])).toBeUndefined();
  });

  it("shows a GPU row's licence and note, and why this GPU cannot run it", () => {
    const gpu = wrap(
      <DictationModelsPanel
        runtime={runtime({
          sttModels: [
            {
              id: "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B", key: "nemotron-3.5-transcribe-30b", name: "Nemotron 3.5 Transcribe 30B-A4B",
              languages: "40 languages", sizeGb: 67.5, installed: false, default: false, backend: "transformers-asr", licence: "OpenMDW-1.1",
              note: "Needs a GPU with about 72 GB of memory. Untested here.", minVramGb: 72, gpuFacts: { kind: "memory", need: 72, have: 24 },
            },
          ],
        })}
        onChanged={() => {}}
      />,
    );
    // The licence by a readable name, linking to the model's page (D-102 D, P2).
    expect(gpu).toContain("40 languages · 67.5 GB · ");
    expect(gpu).toContain("OpenMDW 1.1 (open model licence)</button>");
    expect(gpu).not.toContain(">OpenMDW-1.1<");
    expect(gpu).toContain("Untested here.");
    expect(gpu).toContain("Needs about 72 GB of GPU memory; this GPU has 24 GB.");
    expect(gpu).not.toContain("Fits ·"); // no fit chip on a row this GPU cannot run
  });

  it("gives every card a line on what it is for, a fit chip saying measured or estimate, and badges the default (D-102 C, D)", () => {
    const qwen = {
      id: "Qwen/Qwen3-ASR-1.7B-hf", key: "qwen3-asr-1.7b", name: "Qwen3-ASR 1.7B", languages: "30 languages", sizeGb: 4.09, installed: true,
      default: true, backend: "transformers-asr", licence: "Apache-2.0", minVramGb: 5, vramBasis: "measured" as const, gpuFacts: null,
    };
    const whisper = { id: "org/whisper", key: "whisper-large-v3-turbo", name: "Whisper Large v3 Turbo", languages: "99 languages", sizeGb: 1.6, installed: true, default: false, backend: "faster-whisper", licence: "MIT" };
    const html = wrap(<DictationModelsPanel runtime={runtime({ lane: "portable", sttModels: [qwen, whisper] })} onChanged={() => {}} />);
    const card = html.slice(html.indexOf('data-stt-model="qwen3-asr-1.7b"'), html.indexOf('data-stt-model="whisper-large-v3-turbo"'));
    expect(card).toContain("The pick for an NVIDIA GPU");
    expect(card).toContain(">Default<"); // with Downloaded beside it, not instead of it
    expect(card).toContain(">Downloaded<");
    expect(card).toContain("Fits · 5 GB GPU memory");
    expect(card).toContain("Measured on an RTX 4090");
    const w = html.slice(html.indexOf('data-stt-model="whisper-large-v3-turbo"'));
    expect(w).toContain("does not translate");
    expect(w).toContain(">MIT</button>");
    expect(w).not.toContain("Fits ·");
    // Before the download: the GPU row is "Recommended for this GPU" and Whisper is the default.
    const pending = wrap(
      <DictationModelsPanel runtime={runtime({ lane: "portable", sttModels: [{ ...qwen, installed: false, default: false }, { ...whisper, default: true }] })} onChanged={() => {}} />,
    );
    expect(pending.slice(0, pending.indexOf('data-stt-model="whisper-large-v3-turbo"'))).toContain("Recommended for this GPU");
    expect(pending.slice(pending.indexOf('data-stt-model="whisper-large-v3-turbo"'))).toContain(">Default<");
    expect(sttFitText({ minVramGb: 72, vramBasis: "estimate", gpuFacts: null })?.text).toBe("Fits · about 72 GB GPU memory (estimate)");
    expect(licenceLabel("CC-BY-4.0")).toBe("Creative Commons BY 4.0 (credit the author)");
    expect(licenceLabel("Some-Other")).toBe("Some-Other");
    expect(licenceLabel(null)).toBeNull();
    // Each card says whether the model finds the language itself or uses the choice (here: Detect, so itself).
    expect(card).toContain("Language: finds the language itself");
  });

  it("opens with the one model recommended for this machine and lists the rest under All models", () => {
    const whisper = { id: "org/whisper", key: "whisper-large-v3-turbo", name: "Whisper Large v3 Turbo", languages: "99 languages", sizeGb: 1.6, installed: false, default: true, backend: "faster-whisper" };
    const qwen = {
      id: "Qwen/Qwen3-ASR-1.7B-hf", key: "qwen3-asr-1.7b", name: "Qwen3-ASR 1.7B", languages: "30 languages", sizeGb: 4.09, installed: false,
      default: false, backend: "transformers-asr", licence: "Apache-2.0", minVramGb: 5, vramBasis: "measured" as const, gpuFacts: null,
    };
    const html = wrap(<DictationModelsPanel runtime={runtime({ lane: "portable", sttModels: [whisper, qwen] })} onChanged={() => {}} />);
    const recommended = html.slice(html.indexOf("Recommended for this machine"), html.indexOf(">All models<"));
    expect(recommended).toContain('data-stt-model="qwen3-asr-1.7b"');
    expect(recommended).toContain("Download 4.09 GB");
    expect(recommended).toContain("Fits · 5 GB GPU memory");
    expect(recommended).not.toContain("whisper");
    const rest = html.slice(html.indexOf(">All models<"));
    expect(rest).toContain('data-stt-model="whisper-large-v3-turbo"');
    expect(rest).not.toContain('data-stt-model="qwen3-asr-1.7b"');
    // One model only: no "All models" heading to open.
    expect(wrap(<DictationModelsPanel runtime={runtime({ sttModels: [whisper] })} onChanged={() => {}} />)).not.toContain(">All models<");
  });

  it("picks the recommended speech to text model: the GPU row to come, else the default, else one that can run", () => {
    const row = (key: string, over: Partial<VoiceRuntime["sttModels"][number]> = {}) => ({
      id: `org/${key}`, key, name: key, sizeGb: 1, installed: false, default: false, backend: "faster-whisper", ...over,
    });
    const gpu = row("qwen3-asr-1.7b", { backend: "transformers-asr", minVramGb: 5 });
    const whisper = row("whisper-large-v3-turbo", { default: true });
    expect(recommendedSttModel([gpu, whisper])?.key).toBe("qwen3-asr-1.7b");
    // The GPU row cannot run here: Whisper, the default.
    expect(recommendedSttModel([{ ...gpu, gpuFacts: { kind: "memory", need: 72, have: 24 } }, whisper])?.key).toBe("whisper-large-v3-turbo");
    // Downloaded GPU row is the default.
    expect(recommendedSttModel([{ ...gpu, installed: true, default: true }, whisper])?.key).toBe("qwen3-asr-1.7b");
    expect(recommendedSttModel([])).toBeUndefined();
    expect(sttGpuMemoryText({ minVramGb: 5, gpuFacts: null })).toBe("about 5 GB GPU memory");
    expect(sttGpuMemoryText({ gpuFacts: null })).toBeNull();
    expect(sttGpuMemoryText({ minVramGb: 72, gpuFacts: { kind: "memory", need: 72, have: 24 } })).toBeNull();
  });

  it("offers the forced aligner as its own Word timings download, and Delete once it is there (D-101)", () => {
    const aligner = { key: "qwen3-forced-aligner-0.6b", id: "Qwen/Qwen3-ForcedAligner-0.6B-hf", name: "Qwen3-ForcedAligner 0.6B", sizeGb: 1.84, licence: "Apache-2.0", vramGb: 1.9, installed: false };
    const html = wrap(<DictationModelsPanel runtime={runtime({ lane: "portable", aligner })} onChanged={() => {}} />);
    const card = html.slice(html.indexOf('data-stt-aligner="qwen3-forced-aligner-0.6b"'));
    expect(card).toContain("Word timings");
    expect(card).toContain("Times each word");
    expect(card).toContain("Qwen3-ForcedAligner 0.6B · 1.84 GB · ");
    expect(card).toContain("Apache 2.0</button>");
    expect(card).toContain("Download 1.84 GB");
    const there = wrap(<DictationModelsPanel runtime={runtime({ lane: "portable", aligner: { ...aligner, installed: true } })} onChanged={() => {}} />);
    expect(there.slice(there.indexOf("data-stt-aligner"))).toContain("Delete model (1.84 GB)");
    expect(wrap(<DictationModelsPanel runtime={runtime()} onChanged={() => {}} />)).not.toContain("data-stt-aligner");
  });

  it("shows a running dictation download as a progress bar with the size and Cancel", () => {
    const html = renderToStaticMarkup(
      <SttDownloadProgress name="Qwen3-ASR 0.6B" sizeGb={1.58} onCancel={() => {}} status={{ repo: "r", state: "downloading", progress: 0.37, downloadedGb: 0.58, totalGb: 1.58, error: null }} />,
    );
    expect(html).toContain('role="progressbar"');
    expect(html).toContain('aria-valuenow="37"');
    expect(html).toContain("37% · 0.58 of 1.58 GB");
    expect(html).toContain(">Cancel<");
    expect(html).toContain('aria-label="Downloading Qwen3-ASR 0.6B"');
  });
});

describe("A PC shows only what a PC runs (D-102 A, P1)", () => {
  const pcModels = [
    model("qwen3-tts-custom-1.7b", "Qwen3-TTS 1.7B CustomVoice", { lanes: ["mlx"], supported: false }),
    model("chatterbox-turbo", "Chatterbox-Turbo", { kind: "turbo", family: "chatterbox-turbo", lanes: ["mlx"], supported: false }),
    model("qwen3-tts-custom-1.7b-cuda", "Qwen3-TTS 1.7B CustomVoice (GPU)", { lanes: ["cuda"], capabilities: { ...caps, style: true } }),
  ];
  const pcVoices: VoiceEntry[] = [
    { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", model: "qwen3-tts-custom-1.7b", models: ["qwen3-tts-custom-1.7b", "qwen3-tts-custom-1.7b-cuda"], capabilities: caps, available: true },
    { id: "chatterbox-turbo", name: "Turbo", kind: "preset", language: "en", model: "chatterbox-turbo", models: ["chatterbox-turbo"], capabilities: caps, available: false },
    { id: "clone:1", name: "Ada", kind: "clone", language: "auto", model: "chatterbox-turbo", models: ["chatterbox-turbo"], capabilities: caps, available: true },
  ];

  it("folds the Mac-only rows and their voices away on the portable lane, keeps cloned voices, drops (GPU)", () => {
    const split = splitForLane(pcModels, pcVoices, "portable");
    expect(split.models.map((m) => m.name)).toEqual(["Qwen3-TTS 1.7B CustomVoice"]);
    expect(split.macOnly.map((m) => m.key)).toEqual(["qwen3-tts-custom-1.7b", "chatterbox-turbo"]);
    expect(split.voices.map((v) => v.id)).toEqual(["qwen3:Ryan", "clone:1"]);
    expect(split.hiddenVoices).toBe(1);
    // A Mac, a downloaded Mac row, or an older backend without lanes: nothing is folded.
    expect(splitForLane(pcModels, pcVoices, "mlx").models).toHaveLength(3);
    expect(macOnlyOnThisPc({ lanes: ["mlx"], installed: true }, "portable")).toBe(false);
    expect(macOnlyOnThisPc({ installed: false }, "portable")).toBe(false);
  });

  it("lists them behind one closed line with their count", () => {
    const split = splitForLane(pcModels, pcVoices, "portable");
    const html = wrap(
      <SpeechModelsPanel
        catalog={{ ...catalog, models: split.models, voices: split.voices, macOnly: split.macOnly, hiddenVoices: split.hiddenVoices }}
        runtime={runtime({ lane: "portable" })}
        availability="all"
        onAvailabilityChange={() => {}}
      />,
    );
    expect(html).toContain('<details class="voice-models__mac-only"><summary>2 models for Apple Silicon Macs</summary>');
    expect(html).toContain("they and their 1 voices are left out");
    expect(html).not.toContain('data-model-id="chatterbox-turbo"');
    expect(html).not.toContain("Apple Silicon Macs only for now.");
    expect(wrap(<MacOnlyModels models={[]} />)).toBe("");
  });

  it("collapses a family to the recommended natural row when nothing is downloaded", () => {
    const rows = ["a", "b", "c"].map((k) => voiceRowToEntry(model(k, k)));
    expect([...collapsedFamilyRows(rows, "c").get("qwen3-tts")!.shown]).toEqual(["c"]);
  });
});

describe("Voice runtime panel (D-73 M4, M5)", () => {
  const pack = { job: runtime().install, error: null, setError: () => {}, start: async () => {}, cancel: () => undefined };
  it("names the runtime, lists its parts with sizes, what is in memory, the add-ons and Remove", () => {
    const html = wrap(<VoiceRuntimePanel runtime={runtime()} pack={pack} />);
    expect(html).toContain("Voice setup");
    expect(html).not.toContain("Voice pack");
    expect(html).toContain("2.3 GB on disk: speech packages 1.9 GB, Kokoro quick voices 0.3 GB");
    expect(html).toContain("In memory now:");
    expect(html).toContain("Kokoro, 0.9 GB");
    expect(html).toContain(">Unload<");
    expect(html).toContain("minutes idle");
    expect(html).toMatch(/<input type="checkbox"\/>Mandarin Chinese quick voices/);
    expect(html).toContain("A part installs when ticked; untick it to remove it.");
    expect(html).toContain("Remove voice runtime");
    expect(html).toContain("Downloaded speech-to-text and speech models always stay");
  });

  it("before install, the add-ons are chosen with the install", () => {
    const html = wrap(<VoiceRuntimePanel runtime={runtime({ installed: false, parts: [], workerPid: null })} pack={pack} />);
    expect(html).toContain("Install, about 2.3 GB");
    expect(html).toContain("Ticked parts install with Voice setup");
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
    // Kokoro is part of the Basic preset now, not a tick box (setup card).
    expect(before).toContain("Basic: quick voices, about 2.6 GB");
    expect(before).toMatch(/<input type="checkbox"\/>Mandarin Chinese quick voices/);
    // One size source (runtimeSize.ts): the base plus the ticked add-ons; Kokoro is ticked by default.
    expect(before).toContain("Install, about 2.6 GB");
    expect(runtimeInstallGb([])).toBe(2.3);
    expect(runtimeInstallGb(["kokoro", "speech-gpu"])).toBe(8.3);
    expect(addonSizeText("kokoro")).toBe("330 MB");
    expect(addonSizeText("speech-gpu")).toBe("5.7 GB");
    // Installed without Kokoro: unticked, the runtime is healthy, and the parts line says Kokoro is not installed.
    const without = wrap(
      <VoiceRuntimePanel
        runtime={runtime({ extras, ttsModelInstalled: false, parts: [{ id: "packages", sizeGb: 1.9 }, { id: "kokoro", sizeGb: 0 }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 }] })}
        pack={pack}
      />,
    );
    expect(without).toContain("Missing: Kokoro quick voices 0.3 GB");
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
    expect(runtimeRemoveConfirmText(2.3)).toContain("Downloaded speech to text and speech models, cloned voices and the Library stay.");
  });

  it("words the recommended card from the catalogue's own figures", () => {
    expect(pickSizeText(4.5)).toBe("4.5");
    expect(pickSizeText(5)).toBe("5");
    expect(pickSizeText(0.33)).toBe("0.33");
    expect(makesSentence(["Natural voices", "Style", "Sound tags"])).toBe("Natural voices, style, sound tags");
    expect(speechGpuMemoryText(4.66)).toBe("about 4.7 GB GPU memory");
    expect(speechGpuMemoryText(null)).toBeNull();
    expect(speechGpuMemoryText(0)).toBeNull();
    expect(recommendedSpeechPick(models)).toEqual({ kind: "model", row: models[0] });
    expect(recommendedSpeechPick(models.map((m) => ({ ...m, supported: false })))).toEqual({ kind: "kokoro" });
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

  it("with nothing downloaded, a collapsed family shows the first row that runs here, not an Apple-only row (a PC lists those first)", () => {
    const pc = [
      model("a", "a", { supported: false, lanes: ["mlx"] }),
      model("b", "b", { supported: false, lanes: ["mlx"] }),
      model("a-cuda", "a (GPU)", { supported: true, lanes: ["cuda"] }),
      model("b-cuda", "b (GPU)", { supported: true, lanes: ["cuda"] }),
    ].map((m) => voiceRowToEntry(m));
    expect([...collapsedFamilyRows(pc).get("qwen3-tts")!.shown]).toEqual(["a-cuda"]);
    // The add-on is not installed yet, so nothing is supported: the first GPU row, which the add-on would run.
    const noAddon = pc.map((entry) => voiceRowToEntry({ ...(entry.source as VoiceModelRow), supported: false }));
    expect([...collapsedFamilyRows(noAddon).get("qwen3-tts")!.shown]).toEqual(["a-cuda"]);
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
