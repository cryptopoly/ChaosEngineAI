import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceRuntime } from "../../../../types";
import { VoiceProvider } from "../../VoiceContext";
import { VoicePicker } from "../VoicePicker";

const caps = { speak: true, style: false, clone: false, design: false };
const voices: VoiceEntry[] = [
  { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", languageLabel: "American English", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true },
  { id: "design", name: "Describe a voice", kind: "design", language: "auto", model: "d17", models: ["d17"], capabilities: { ...caps, design: true }, available: false },
];

function render(modelsAvailable: boolean, runtime: Partial<VoiceRuntime> = {}) {
  return renderToStaticMarkup(
    <VoiceProvider
      value={{ runtime: { installed: true, extras: [], settings: {}, ...runtime } as unknown as VoiceRuntime, platform: "Windows", refresh: () => {}, saveSettings: async () => {}, openVoiceModels: () => {} }}
    >
      <VoicePicker
        voices={voices}
        models={[]}
        selectedId="af_heart"
        onSelect={() => {}}
        modelsAvailable={modelsAvailable}
        isPlaying={() => false}
        onSample={() => {}}
        onNewClone={() => {}}
        onGetMore={() => {}}
      />
    </VoiceProvider>,
  );
}

describe("the picker's Clone a voice card (UX audit change 5)", () => {
  it("opens the Clone a voice screen when cloning can run here", () => {
    const html = render(true);
    expect(html).toContain("Clone a voice");
    expect(html).toContain(">Open Clone a voice<");
    expect(html).not.toContain("New cloned voice");
  });

  it("says why it is greyed, in words, with the fix button on a PC without the speech GPU add-on", () => {
    const html = render(false, { lane: "portable", extras: [{ id: "speech-gpu", label: "", installed: false }] } as Partial<VoiceRuntime>);
    expect(html).toContain("Clone a voice");
    expect(html).not.toContain(">Open Clone a voice<");
    expect(html).toContain("Needs the speech GPU add-on.");
    // One fix on the clone card, one beside the designed voice.
    expect(html.match(/>Open Voice setup</g)).toHaveLength(2);
  });

  it("gives the Mac-only note without a button elsewhere", () => {
    const html = render(false);
    expect(html).toContain("Voice cloning runs on Apple Silicon Macs for now.");
    expect(html).not.toContain("Open Voice setup");
  });
});
