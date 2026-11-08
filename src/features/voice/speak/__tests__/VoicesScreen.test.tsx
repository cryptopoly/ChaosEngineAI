import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceRuntime } from "../../../../types";
import { VoiceProvider } from "../../VoiceContext";
import type { VoiceCatalog } from "../useVoiceCatalog";
import { VoicesScreen } from "../VoicesScreen";

const caps = { speak: true, style: false, clone: false, design: false };
const voices: VoiceEntry[] = [
  { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", languageLabel: "American English", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true, grade: "A" },
  { id: "af_bella", name: "Bella", kind: "builtin", language: "en-US", languageLabel: "American English", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true, grade: "A-" },
  { id: "zf_xiaobei", name: "Xiaobei", kind: "builtin", language: "zh", languageLabel: "Chinese (Mandarin)", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: false, addon: "chinese" },
  { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", languageLabel: "English", gender: "male", model: "c17", models: ["c17"], capabilities: { ...caps, style: true }, available: true },
  { id: "design", name: "Describe a voice", kind: "design", language: "auto", model: "d17", models: ["d17"], capabilities: { ...caps, style: true, design: true }, available: true },
  { id: "clone:1", name: "Ada", kind: "clone", language: "auto", model: "b06", models: ["b06"], capabilities: { ...caps, clone: true }, available: true, consent: { speakerName: "Ada", basis: "own", consentedAt: "2026-11-08T10:00:00Z" } },
];
const catalog: VoiceCatalog = { voices, models: [], modelsAvailable: true, cloneAllowed: true, loaded: true, error: null, refresh: async () => {}, download: async () => {} };

function runtime(over: Partial<VoiceRuntime> = {}): VoiceRuntime {
  return {
    installed: true,
    extras: [{ id: "chinese", installed: false }],
    install: { phase: "idle", phaseLabel: "", phaseIndex: 0, phaseCount: 0, running: false, extras: [], log: [], error: null },
    settings: { sttModel: "", language: "", ttsVoice: "af_bella", speed: 1.2, autoSend: true, autoRead: false, vadSilenceMs: 900, bargeIn: true, idleUnloadMinutes: 10 },
    ...over,
  } as VoiceRuntime;
}

function render(rt = runtime()) {
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
      <VoicesScreen catalog={catalog} onUse={() => {}} onDesign={() => {}} onGetMore={() => {}} />
    </VoiceProvider>,
  );
}

/** The markup of the list row that names ``name``. */
function row(html: string, name: string): string {
  const rows = html.split('<li class="voice-list__row">').slice(1);
  return rows.find((r) => r.includes(`<strong>${name}</strong>`)) ?? "";
}

describe("VoicesScreen as the voice manager", () => {
  it("has the kind filters, a language select and the action cards, Mandarin included while it is missing", () => {
    const html = render();
    for (const text of ["All", "Recommended", "Quick", "Natural", "Cloned", "Designed", "All languages", "Clone a voice", "Design a voice", "Add Mandarin voices", "Get more voices", "Only clone your own voice"]) {
      expect(html).toContain(text);
    }
    expect(render(runtime({ extras: [{ id: "chinese", installed: true }] } as Partial<VoiceRuntime>))).not.toContain("Add Mandarin voices");
  });

  it("lists every voice that runs here, the design voice too, with a sample button on each", () => {
    const html = render();
    for (const name of ["Heart", "Bella", "Ryan", "Design a voice", "Ada"]) {
      expect(row(html, name)).toContain("Play a sample of");
      expect(row(html, name)).toContain("Use in Speak");
    }
    expect(row(html, "Xiaobei")).toBe("");
  });

  it("marks the chat voice and allows Use in Chat for quick voices only, with a plain note", () => {
    const html = render();
    expect(row(html, "Bella")).toContain("Chat voice");
    expect(row(html, "Bella")).toContain("In use in Chat");
    expect(row(html, "Heart")).toMatch(/<button type="button" class="secondary-button">Use in Chat<\/button>/);
    for (const name of ["Ryan", "Design a voice", "Ada"]) {
      expect(row(html, name)).toMatch(/<button type="button" class="secondary-button" disabled=""[^>]*>Use in Chat<\/button>/);
    }
    expect(html).toContain("Chat reads aloud with quick voices for now");
    expect(row(html, "Ada")).toContain("Delete the voice Ada");
    expect(row(html, "Heart")).not.toContain("Delete the voice");
  });

  it("holds every chat setting in Chat and calls, read from the same settings", () => {
    const html = render();
    expect(html).toContain("Chat and calls");
    expect(html).toContain("Chat voice: Bella.");
    for (const text of ["Reading speed", "(1.2×)", "Read every chat reply aloud", "Send the message as soon as dictation finishes", "pause that ends your turn", "(0.9 s)", "interrupt the model by speaking"]) {
      expect(html).toContain(text);
    }
  });
});
