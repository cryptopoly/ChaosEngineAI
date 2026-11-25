import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceModelRow, VoiceRuntime } from "../../../../types";
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

function render(rt = runtime(), speakVoiceId = "qwen3:Ryan", cat: VoiceCatalog = catalog) {
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
      <VoicesScreen catalog={cat} speakVoiceId={speakVoiceId} onUseInSpeak={() => {}} onDesign={() => {}} onGetMore={() => {}} />
    </VoiceProvider>,
  );
}

/** The markup of the list row that names ``name``. */
function row(html: string, name: string): string {
  const rows = html.split('<li class="voice-list__row"').slice(1);
  return rows.find((r) => r.includes(`<strong>${name}</strong>`)) ?? "";
}

describe("VoicesScreen as the voice catalogue (D-73 S2, D0142)", () => {
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

  it("no longer holds Chat's voice choices: no Main voices strip, no Change buttons, no Fallback wording (D0142)", () => {
    const html = render();
    for (const text of ["Main voices", "Change the voice for", ">Change<", "Voice calls", "Chat read-aloud", "Fallback", "fallback voice", "Use for…"]) {
      expect(html).not.toContain(text);
    }
    expect(html).not.toContain("data-use=");
    expect(html).toContain("Chat&#x27;s voices are chosen in Chat, under Voice.");
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

  it("gives every row one Use in Speak button, and marks where a voice is in use (Backup, not Fallback)", () => {
    const html = render();
    for (const name of ["Heart", "Bella", "Ryan", "Ada"]) {
      expect(row(html, name)).toContain("Play a sample of");
      expect(row(html, name)).toContain(`aria-label="Use ${name} in Speak"`);
    }
    // Ryan is the Speak voice: the button says so and is off.
    expect(row(html, "Ryan")).toMatch(/disabled="" aria-label="Use Ryan in Speak">In use in Speak/);
    expect(row(html, "Ryan")).toContain(">Speak<");
    expect(row(html, "Bella")).toContain(">Chat<");
    expect(row(html, "Bella")).toContain(">Calls<");
    expect(row(html, "Heart")).toContain(">Backup<");
    expect(row(html, "Heart")).not.toContain(">Fallback<");
    // Describe a voice keeps a plain Use in Speak.
    expect(row(html, "Design a voice")).toContain("Use in Speak");
    expect(row(html, "Ada")).toContain("Delete the voice Ada");
    expect(row(html, "Heart")).not.toContain("Delete the voice");
    expect(row(html, "Ryan")).toContain("first audio 2.4 s (4.1 s cold), 3.51 GB");
    expect(row(html, "Ryan")).toContain("Slow for calls");
  });

  it("writes the same settings keys as before, plus the fallback voice", () => {
    expect(VOICE_USE_SETTING).toEqual({ speak: "speakVoice", read: "ttsVoice", call: "callVoice", fallback: "fallbackVoice" });
  });

  it("greyed New cloned voice and Design a voice say why as text, with the fix button on a PC without the speech GPU add-on", () => {
    const pc = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] } as Partial<VoiceRuntime>);
    const html = render(pc, "", { ...catalog, modelsAvailable: false, cloneAllowed: false });
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>New cloned voice/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Design a voice/);
    expect(html.match(/Needs the speech GPU part./g)).toHaveLength(2);
    expect(html.match(/>Open Voice setup</g)).toHaveLength(2);
    // The reason is text on the page, not only a tooltip.
    expect(html).not.toContain('title="Needs the speech GPU');
  });

  it("a downloaded natural voice is off, with the reason, on a PC without the speech GPU part (B1324)", () => {
    const model = { key: "c17", name: "Qwen3-TTS 1.7B CustomVoice", installed: true, supported: true, lanes: ["cuda"], sizeGb: 4.5, download: { state: "idle" } } as unknown as VoiceModelRow;
    const cat = { ...catalog, models: [model] };
    const missing = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] } as Partial<VoiceRuntime>);
    const without = row(render(missing, "", cat), "Ryan");
    expect(without).toMatch(/<button[^>]*disabled=""[^>]*aria-label="Use Ryan in Speak"/);
    expect(without).toContain("Needs the speech GPU part.");
    const withPart = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: true }] } as Partial<VoiceRuntime>);
    const ok = row(render(withPart, "", cat), "Ryan");
    expect(ok).not.toMatch(/disabled=""[^>]*aria-label="Use Ryan in Speak"/);
    expect(ok).not.toContain("Needs the speech GPU part.");
  });

  it("a Mac-only model says so in words without a button; a window-only clone says so too", () => {
    const mac = render(runtime(), "", { ...catalog, modelsAvailable: false });
    expect(mac).toContain("Voice cloning runs on Apple Silicon Macs for now.");
    expect(mac).toContain("Designing a voice runs on Apple Silicon Macs for now.");
    expect(mac).not.toContain("Open Voice setup");
    const windowOnly = render(runtime(), "", { ...catalog, cloneAllowed: false });
    expect(windowOnly).toContain("Cloned voices can only be used from the ChaosEngineAI window on this computer.");
    expect(windowOnly).not.toContain("Designing a voice runs on");
    // Nothing greyed, nothing to explain.
    expect(render()).not.toContain("voice-voices__reason");
    // Before the catalogue loads, no reason is guessed.
    expect(render(runtime(), "", { ...catalog, loaded: false, modelsAvailable: false })).not.toContain("voice-voices__reason");
  });

  it("without Kokoro the Quick group says how to add Kokoro", () => {
    const noKokoro: VoiceCatalog = {
      ...catalog,
      voices: voices.map((v) => (v.kind === "builtin" ? { ...v, available: false, addon: null } : v)),
    };
    const html = renderToStaticMarkup(
      <VoiceProvider
        value={{
          runtime: runtime({ extras: [{ id: "kokoro", installed: false }, { id: "chinese", installed: false }] } as Partial<VoiceRuntime>),
          platform: "Darwin",
          refresh: () => {},
          saveSettings: async () => {},
          openVoiceModels: () => {},
        }}
      >
        <VoicesScreen catalog={noKokoro} speakVoiceId="qwen3:Ryan" onUseInSpeak={() => {}} onDesign={() => {}} onGetMore={() => {}} />
      </VoiceProvider>,
    );
    expect(html).toContain("The Kokoro quick voices are not installed");
    expect(html).toContain("Open Voice Models");
    expect(html).not.toContain("Add Mandarin voices");
    expect(row(html, "Heart")).toBe("");
  });

  it("offers no Download for a voice whose model cannot run here, and says why (a PC lists the Apple-only voices)", () => {
    const row = (key: string, over: Partial<VoiceModelRow>): VoiceModelRow => ({
      key, name: key, summary: "", repo: `org/${key}`, sizeGb: 3.54, languages: "English", default: false, kind: "custom", capabilities: caps,
      supported: true, installed: false, download: { state: "idle", error: null }, ...over,
    });
    const preset = (id: string, model: string): VoiceEntry => ({ id, name: id, kind: "preset", language: "en", model, models: [model], capabilities: caps, available: false, chatUse: both });
    const pc = runtime({ lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: true }] } as Partial<VoiceRuntime>);
    const html = renderToStaticMarkup(
      <VoiceProvider value={{ runtime: pc, platform: "Windows", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
        <VoicesScreen
          catalog={{ ...catalog, voices: [preset("Tara", "apple-only"), preset("Ryan", "gpu")], models: [row("apple-only", { supported: false, lanes: ["mlx"] }), row("gpu", { lanes: ["cuda"] })] }}
          speakVoiceId=""
          onUseInSpeak={() => {}}
          onDesign={() => {}}
          onGetMore={() => {}}
        />
      </VoiceProvider>,
    );
    const rows = html.split('<li class="voice-list__row"').slice(1);
    const tara = rows.find((r) => r.includes("<strong>Tara</strong>")) ?? "";
    const ryan = rows.find((r) => r.includes("<strong>Ryan</strong>")) ?? "";
    expect(tara).toContain("Apple Silicon Macs only for now.");
    expect(tara).not.toContain("Download 3.54 GB");
    expect(ryan).toContain("Download 3.54 GB");
  });
});
