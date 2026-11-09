import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceRuntime } from "../../../../types";
import { VoiceProvider } from "../../VoiceContext";
import type { VoiceCatalog } from "../useVoiceCatalog";
import { VoicesScreen } from "../VoicesScreen";

const caps = { speak: true, style: false, clone: false, design: false };
const both = { read: true, call: true };
const kokoroCost = { firstAudioS: 0.13, firstAudioColdS: 5.7, peakGb: 0.87, measuredOn: "Apple M4 Max, 64 GB, 2026-11-09", slowForCalls: false };
const voices: VoiceEntry[] = [
  { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", languageLabel: "American English", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true, grade: "A", chatUse: both, chatCost: kokoroCost },
  { id: "af_bella", name: "Bella", kind: "builtin", language: "en-US", languageLabel: "American English", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true, grade: "A-", chatUse: both, chatCost: kokoroCost },
  { id: "zf_xiaobei", name: "Xiaobei", kind: "builtin", language: "zh", languageLabel: "Chinese (Mandarin)", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: false, addon: "chinese" },
  { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", languageLabel: "English", gender: "male", model: "c17", models: ["c17"], capabilities: { ...caps, style: true }, available: true, chatUse: both, chatCost: { firstAudioS: 2.4, firstAudioColdS: 4.1, peakGb: 3.51, slowForCalls: true } },
  { id: "design", name: "Describe a voice", kind: "design", language: "auto", model: "d17", models: ["d17"], capabilities: { ...caps, style: true, design: true }, available: true, chatUse: { read: false, call: false } },
  { id: "clone:1", name: "Ada", kind: "clone", language: "auto", model: "b06", models: ["b06"], capabilities: { ...caps, clone: true }, available: true, chatUse: { read: true, call: false }, consent: { speakerName: "Ada", basis: "own", consentedAt: "2026-11-08T10:00:00Z" } },
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

  it("marks the chat and call voices and allows Use in Chat and Use for calls where the backend says so (D-72)", () => {
    const html = render();
    expect(row(html, "Bella")).toContain("Chat voice");
    expect(row(html, "Bella")).toContain("Call voice"); // no callVoice set: calls use the chat voice
    expect(row(html, "Bella")).toContain("In use in Chat");
    for (const name of ["Heart", "Ryan", "Ada"]) {
      expect(row(html, name)).toMatch(/<button type="button" class="secondary-button">Use in Chat<\/button>/);
    }
    expect(row(html, "Design a voice")).toMatch(/<button type="button" class="secondary-button" disabled=""[^>]*>Use in Chat<\/button>/);
    expect(row(html, "Design a voice")).not.toContain("Use for calls");
    expect(row(html, "Ryan")).toMatch(/<button type="button" class="secondary-button">Use for calls<\/button>/);
    expect(row(html, "Ada")).toMatch(/disabled="" title="Cloned voices are not used in voice calls.">Use for calls/);
    // The measured cost next to the choice, and the slow label when warm first audio is over 2 s.
    expect(row(html, "Ryan")).toContain("first audio 2.4 s (4.1 s cold), 3.51 GB");
    expect(row(html, "Ryan")).toContain("Slow for calls");
    expect(row(html, "Heart")).not.toContain("Slow for calls");
    expect(html).toContain("Chat and calls can use any voice that runs here.");
    expect(row(html, "Ada")).toContain("Delete the voice Ada");
    expect(row(html, "Heart")).not.toContain("Delete the voice");
  });

  it("holds every chat setting in Chat and calls, read from the same settings", () => {
    const html = render();
    expect(html).toContain("Chat and calls");
    expect(html).toContain("Chat voice (read-aloud)");
    expect(html).toMatch(/<option value="af_bella" selected="">Bella, 0.13 s<\/option>/);
    expect(html).toMatch(/<option value="" selected="">Same as the chat voice<\/option>/);
    expect(html).toContain("Ryan, 2.4 s (Slow for calls)");
    expect(html).not.toMatch(/aria-label="Call voice">[^]*value="clone:1"/); // a cloned voice is never offered for calls
    expect(html).not.toContain('value="design"');
    for (const text of ["Reading speed", "(1.2×)", "Read every chat reply aloud", "Send the message as soon as dictation finishes", "pause that ends your turn", "(0.9 s)", "interrupt the model by speaking"]) {
      expect(html).toContain(text);
    }
  });
});
