import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceRuntime } from "../../../../types";
import { VoiceProvider } from "../../VoiceContext";
import { VOICE_USE_SETTING } from "../speakLogic";
import type { VoiceCatalog } from "../useVoiceCatalog";
import { VoicesScreen } from "../VoicesScreen";

const caps = { speak: true, style: false, clone: false, design: false };
const both = { read: true, call: true };
const kokoroCost = { firstAudioS: 0.13, firstAudioColdS: 5.7, peakGb: 0.87, measuredOn: "Apple M4 Max, 64 GB, 2026-11-09", slowForCalls: false };
const quick = (id: string, name: string, grade: string, gender: "female" | "male" = "female"): VoiceEntry => ({
  id, name, kind: "builtin", language: "en-US", languageLabel: "American English", gender, model: "kokoro", models: ["kokoro"], capabilities: caps, available: true, grade, chatUse: both, chatCost: kokoroCost,
});
const voices: VoiceEntry[] = [
  // Ungraded-order first, as the backend lists them: the grouped list must not show them in list order.
  quick("af_alloy", "Alloy", "C"),
  quick("af_jessica", "Jessica", "D"),
  quick("af_heart", "Heart", "A"),
  quick("af_bella", "Bella", "A-"),
  quick("af_nicole", "Nicole", "B-"),
  quick("am_michael", "Michael", "C+", "male"),
  { id: "zf_xiaobei", name: "Xiaobei", kind: "builtin", language: "zh", languageLabel: "Chinese (Mandarin)", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: false, addon: "chinese" },
  { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", languageLabel: "English", gender: "male", about: "Dynamic", model: "c17", models: ["c17"], capabilities: { ...caps, style: true }, available: true, chatUse: both, chatCost: { firstAudioS: 2.4, firstAudioColdS: 4.1, peakGb: 3.51, slowForCalls: true } },
  { id: "design", name: "Describe a voice", kind: "design", language: "auto", model: "d17", models: ["d17"], capabilities: { ...caps, style: true, design: true }, available: true, chatUse: { read: false, call: false } },
  { id: "clone:1", name: "Ada", kind: "clone", language: "auto", model: "b06", models: ["b06"], capabilities: { ...caps, clone: true }, available: true, chatUse: { read: true, call: false }, consent: { speakerName: "Ada", basis: "own", consentedAt: "2026-11-08T10:00:00Z" } },
];
const catalog: VoiceCatalog = { voices, models: [], modelsAvailable: true, cloneAllowed: true, loaded: true, error: null, refresh: async () => {}, download: async () => {} };

function runtime(over: Partial<VoiceRuntime> = {}): VoiceRuntime {
  return {
    installed: true,
    extras: [{ id: "chinese", installed: false }],
    install: { phase: "idle", phaseLabel: "", phaseIndex: 0, phaseCount: 0, running: false, extras: [], log: [], error: null },
    settings: { sttModel: "", language: "", ttsVoice: "af_bella", callVoice: "", speed: 1.2, autoSend: true, autoRead: false, vadSilenceMs: 900, bargeIn: true, idleUnloadMinutes: 10 },
    ...over,
  } as VoiceRuntime;
}

function render(rt = runtime(), speakVoiceId = "qwen3:Ryan") {
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
      <VoicesScreen catalog={catalog} speakVoiceId={speakVoiceId} onUseInSpeak={() => {}} onDesign={() => {}} onGetMore={() => {}} />
    </VoiceProvider>,
  );
}

/** The markup of the list row that names ``name``. */
function row(html: string, name: string): string {
  const rows = html.split('<li class="voice-list__row"').slice(1);
  return rows.find((r) => r.includes(`<strong>${name}</strong>`)) ?? "";
}

function slot(html: string, use: string): string {
  const start = html.indexOf(`data-use="${use}"`);
  return start < 0 ? "" : html.slice(start, html.indexOf("</button>", start));
}

describe("VoicesScreen (D-73 S2)", () => {
  it("has the actions, kind filters and a language select; the cloning statement, agent switch and chat settings are gone", () => {
    const html = render();
    for (const text of ["New cloned voice", "Design a voice", "Get more voices", "All", "Natural", "Quick", "Cloned", "Designed", "All languages", "Add Mandarin voices"]) {
      expect(html).toContain(text);
    }
    for (const text of ["Only clone your own voice", "Allow agents", "Chat and calls", "Reading speed", "Recommended"]) {
      expect(html).not.toContain(text);
    }
    expect(render(runtime({ extras: [{ id: "chinese", installed: true }] } as Partial<VoiceRuntime>))).not.toContain("Add Mandarin voices");
  });

  it("shows the main voice for Speak, Chat read-aloud and voice calls, each with Change, and the fallback rule", () => {
    const html = render();
    expect(slot(html, "speak")).toContain("<strong>Ryan</strong>");
    expect(slot(html, "read")).toContain("<strong>Bella</strong>");
    expect(slot(html, "read")).toContain("First words after 0.13 s.");
    // No call voice set: calls use the chat voice.
    expect(slot(html, "call")).toContain("<strong>Bella</strong>");
    expect(slot(html, "call")).toContain("(same as Chat)");
    expect(slot(html, "call")).toContain("Cloned voices are not used in calls.");
    expect(html.match(/>Change</g)).toHaveLength(3);
    expect(html).toContain("Kokoro reads instead and Chat says so");
    const own = render(runtime({ settings: { ...runtime().settings, callVoice: "qwen3:Ryan" } }));
    expect(slot(own, "call")).toContain("<strong>Ryan</strong>");
    expect(slot(own, "call")).not.toContain("(same as Chat)");
  });

  it("groups the list Natural, Quick, Cloned, Designed, with the graded quick shortlist first", () => {
    const html = render();
    const order = ["voice-group-natural", "voice-group-quick", "voice-group-cloned", "voice-group-designed"].map((id) => html.indexOf(`id="${id}"`));
    expect(order.every((at) => at >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    // Heart (A) before Bella (A-) before Nicole (B-); Jessica (D) is not on the shortlist, behind the expander.
    const at = (id: string) => html.indexOf(`data-voice-row="${id}"`);
    expect(at("af_heart")).toBeLessThan(at("af_bella"));
    expect(at("af_bella")).toBeLessThan(at("af_nicole"));
    expect(row(html, "Jessica")).toBe("");
    expect(html).toContain("All 6 American English voices");
    expect(row(html, "Xiaobei")).toBe("");
  });

  it("gives every row one Use for menu, greyed with W6's reason where a use is not allowed, and marks the uses", () => {
    const html = render();
    for (const name of ["Heart", "Bella", "Ryan", "Ada"]) {
      expect(row(html, name)).toContain("Use for…");
      expect(row(html, name)).toContain("Play a sample of");
    }
    expect(row(html, "Bella")).toContain(">Chat<");
    expect(row(html, "Bella")).toContain(">Calls<");
    expect(row(html, "Ryan")).toContain(">Speak<");
    expect(row(html, "Ada")).toMatch(/disabled="" title="Cloned voices are not used in voice calls."/);
    expect(row(html, "Ryan")).toContain("(slow: 2.4 s)");
    // Describe a voice keeps a plain Use in Speak (it needs a description each time, so no Chat or calls).
    expect(row(html, "Design a voice")).toContain("Use in Speak");
    expect(row(html, "Design a voice")).not.toContain("Use for…");
    expect(row(html, "Ada")).toContain("Delete the voice Ada");
    expect(row(html, "Heart")).not.toContain("Delete the voice");
    expect(row(html, "Ryan")).toContain("first audio 2.4 s (4.1 s cold), 3.51 GB");
    expect(row(html, "Ryan")).toContain("Slow for calls");
  });

  it("writes the same settings keys as before", () => {
    expect(VOICE_USE_SETTING).toEqual({ speak: "speakVoice", read: "ttsVoice", call: "callVoice" });
  });
});
