import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import { Drawer } from "../../../components/Drawer";
import type { VoiceEntry, VoiceRuntime, VoiceSettings } from "../../../types";
import { ChatVoiceDrawer, type ChatVoiceUse } from "../ChatVoiceDrawer";
import { FallbackNote } from "../speak/FallbackNote";
import { VOICE_USE_SETTING } from "../speak/speakLogic";
import type { VoiceCatalog } from "../speak/useVoiceCatalog";
import { VoiceProvider } from "../VoiceContext";
import { VoicePopover, type VoiceRequirement } from "../VoicePopover";

const settings: VoiceSettings = {
  sttModel: "parakeet", language: "", ttsVoice: "qwen3:Ryan", callVoice: "", speakVoice: "", speed: 1.3, autoSend: true, autoRead: false,
  vadSilenceMs: 900, bargeIn: true, idleUnloadMinutes: 10,
};

function runtime(over: Partial<VoiceRuntime> = {}): VoiceRuntime {
  return {
    installed: true, sttAvailable: true, ttsAvailable: true, vadAvailable: true,
    sttModels: [{ id: "mlx-community/parakeet", key: "parakeet", name: "Parakeet TDT 0.6B v3", sizeGb: 2.5, installed: true, default: true }],
    ttsVoices: [{ id: "af_heart", name: "Heart", language: "en-US" }],
    extras: [], settings,
    ...over,
  } as VoiceRuntime;
}

const ready: VoiceRequirement[] = [
  { label: "a chat model loaded", ready: true },
  { label: "dictation", ready: true, fixOnModels: true },
  { label: "a voice", ready: true, fixOnModels: true },
  { label: "turn detection", ready: true, fixOnModels: true },
];

function render(requirements = ready, rt = runtime()) {
  return renderToStaticMarkup(
    <VoiceProvider value={{ runtime: rt, platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {}, openVoices: () => {} }}>
      <VoicePopover requirements={requirements} initialOpen />
    </VoiceProvider>,
  );
}

describe("Chat's Voice popover (D-73 S3)", () => {
  it("is a Voice button, closed by default", () => {
    const closed = renderToStaticMarkup(
      <VoiceProvider value={{ runtime: runtime(), platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
        <VoicePopover requirements={ready} />
      </VoiceProvider>,
    );
    expect(closed).toContain('aria-expanded="false"');
    expect(closed).toContain("Voice");
    expect(closed).not.toContain("Read-aloud");
  });

  it("names the voices in use and the dictation model, each with Change, and the fallback rule", () => {
    const html = render();
    expect(html).toContain("Voices in Chat");
    expect(html).toContain(">Read-aloud<");
    expect(html).toContain("<strong>Ryan</strong>");
    expect(html).toContain(">Calls<");
    expect(html).not.toContain("Chat read-aloud");
    expect(html).not.toContain("Voice calls");
    expect(html).toContain("(same as Chat)");
    expect(html).toContain("<strong>Parakeet TDT 0.6B v3</strong>");
    expect(html.match(/>Change</g)).toHaveLength(4);
    expect(html).toContain("Open Voice Studio");
    // Each Change opens the picker here and says which choice it changes.
    expect(html).toContain('aria-label="Change Read-aloud"');
    expect(html).toContain('aria-label="Change Calls"');
    expect(html).toContain('aria-label="Change Backup"');
    expect(html).toContain(">Dictate<");
    // The three voice choices come first, under their heading; dictation after them.
    expect(html.indexOf(">Read-aloud<")).toBeLessThan(html.indexOf(">Calls<"));
    expect(html.indexOf(">Calls<")).toBeLessThan(html.indexOf(">Backup<"));
    expect(html.indexOf(">Backup<")).toBeLessThan(html.indexOf(">Dictate<"));
    // The picker is a drawer that only exists once Change was pressed.
    expect(html).not.toContain("drawer-layer");
    expect(html).not.toContain("Kokoro reads instead");
  });

  it("names the fallback voice, or says none is set (D-75)", () => {
    const html = render();
    expect(html).toContain(">Backup<");
    expect(html).not.toContain(">Fallback<");
    expect(html).toContain("<strong>Heart</strong>");
    const none = render(ready, runtime({ settings: { ...settings, fallbackVoice: "" } }));
    expect(none).toContain("<strong>None set</strong>");
    // The note under it (shown once the voice list has loaded) names the voice or says none is set.
    const ryan = { id: "qwen3:Ryan", name: "Ryan" } as VoiceEntry;
    expect(renderToStaticMarkup(<FallbackNote voice={ryan} />)).toContain("the backup voice Ryan reads instead");
    expect(renderToStaticMarkup(<FallbackNote voice={null} />)).toContain("No backup voice is set");
  });

  it("holds the chat behaviour settings, read from the same keys", () => {
    const html = render();
    expect(html).toMatch(/<input type="checkbox"\/>Read every chat reply aloud/);
    expect(html).toMatch(/<input type="checkbox" checked=""\/>Send the message as soon as dictation finishes/);
    expect(html).toContain("Reading speed (1.3×)");
    expect(html).toContain("pause that ends your turn (0.9 s)");
    expect(html).toMatch(/<input type="checkbox" checked=""\/>Voice call: interrupt the model by speaking/);
  });

  it("lists what a call needs and offers one fix for what Voice Models installs", () => {
    expect(render()).not.toContain("Install what is missing");
    const missing = render(ready.map((req) => (req.label === "dictation" ? { ...req, ready: false } : req)));
    expect(missing).toContain("Missing");
    expect(missing).toContain("Install what is missing");
    // Only the chat model missing: nothing to install on Voice Models.
    const modelOnly = render(ready.map((req) => (req.label === "a chat model loaded" ? { ...req, ready: false } : req)));
    expect(modelOnly).not.toContain("Install what is missing");
    expect(missing).not.toContain("(Voice Models)");
  });

  it("does not break on a reading speed missing from an old settings file", () => {
    const html = render(ready, runtime({ settings: { ...settings, speed: undefined as unknown as number } }));
    expect(html).toContain("Reading speed (1.0×)");
  });

  it("asks for the runtime when it is not installed", () => {
    expect(render(ready, runtime({ installed: false }))).toContain("needs the voice runtime");
  });
});

describe("the voice picker that Change opens in Chat (D0142)", () => {
  const caps = { speak: true, style: false, clone: false, design: false };
  const heart = { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", languageLabel: "American English", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true, chatUse: { read: true, call: true } } as VoiceEntry;
  const ryan = { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", languageLabel: "English", model: "c17", models: ["c17"], capabilities: caps, available: true, chatUse: { read: true, call: true } } as VoiceEntry;
  const ada = { id: "clone:1", name: "Ada", kind: "clone", language: "auto", model: "b06", models: ["b06"], capabilities: { ...caps, clone: true }, available: true, chatUse: { read: true, call: false } } as VoiceEntry;
  const design = { id: "design", name: "Describe a voice", kind: "design", language: "auto", model: "d17", models: ["d17"], capabilities: { ...caps, design: true }, available: true, chatUse: { read: false, call: false } } as VoiceEntry;
  const catalog: VoiceCatalog = { voices: [heart, ryan, ada, design], models: [], modelsAvailable: true, cloneAllowed: true, loaded: true, error: null, refresh: async () => {}, download: async () => {} };

  function drawer(use: ChatVoiceUse, currentId: string) {
    return renderToStaticMarkup(
      <VoiceProvider value={{ runtime: runtime(), platform: "Darwin", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}>
        <ChatVoiceDrawer use={use} catalog={catalog} currentId={currentId} speed={1} onAssign={async () => {}} onClose={() => {}} onGetMore={() => {}} />
      </VoiceProvider>,
    );
  }

  it("offers the voices that can serve the use, in the same cards as Speak's picker", () => {
    const read = drawer("read", "qwen3:Ryan");
    expect(read).toContain("Choose the voice that reads Chat replies");
    expect(read).toContain('role="radiogroup"');
    expect(read).toContain("Ryan");
    expect(read).toContain("Ada");
    // Describe a voice needs a description each time: never a Chat voice.
    expect(read).not.toContain("Describe a voice");
    expect(read).not.toContain("No backup voice");
    // Cloned voices are not used in calls.
    const call = drawer("call", "af_heart");
    expect(call).toContain("Choose the voice for calls");
    expect(call).toContain("Heart");
    expect(call).not.toContain("Ada");
  });

  it("the backup picker adds No backup voice, pressed while none is set", () => {
    const backup = drawer("fallback", "");
    expect(backup).toContain("Choose the backup voice");
    expect(backup).toMatch(/aria-pressed="true"[^>]*>No backup voice/);
    expect(drawer("fallback", "af_heart")).toMatch(/aria-pressed="false"[^>]*>No backup voice/);
  });

  it("the three choices write the keys they always did", () => {
    expect(VOICE_USE_SETTING.read).toBe("ttsVoice");
    expect(VOICE_USE_SETTING.call).toBe("callVoice");
    expect(VOICE_USE_SETTING.fallback).toBe("fallbackVoice");
  });
});

describe("Drawer", () => {
  it("is a titled modal dialog with Close", () => {
    const html = renderToStaticMarkup(
      <Drawer title="Choose a voice" subtitle="Pick one." onClose={() => {}}>
        <p>body</p>
      </Drawer>,
    );
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toContain("Choose a voice");
    expect(html).toContain("Pick one.");
    expect(html).toContain(">Close<");
    expect(html).toContain("drawer-backdrop");
  });
});
