import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceModelRow } from "../../../../types";
import { SpeakScreen } from "../SpeakScreen";
import type { VoiceCatalog } from "../useVoiceCatalog";

const caps = { speak: true, style: false, clone: false, design: false };
const voices: VoiceEntry[] = [
  { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", languageLabel: "English (US)", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps, available: true },
  { id: "qwen3:Ryan", name: "Ryan", kind: "preset", language: "en", languageLabel: "English", gender: "male", model: "c06", models: ["c06", "c17"], capabilities: { ...caps, style: true }, available: true },
  { id: "design", name: "Describe a voice", kind: "design", language: "auto", model: "d17", models: ["d17"], capabilities: { ...caps, style: true, design: true }, available: false },
  { id: "clone:1", name: "My voice", kind: "clone", language: "auto", model: "b06", models: ["b06"], capabilities: { ...caps, clone: true }, available: true, consent: { speakerName: "Dan", basis: "own", consentedAt: "2026-11-08T10:00:00Z" } },
];
const models: VoiceModelRow[] = [
  { key: "c06", name: "Qwen3-TTS 0.6B CustomVoice", summary: "", repo: "r", sizeGb: 2.5, languages: "", default: true, kind: "custom", capabilities: caps, supported: true, installed: true, download: { state: "idle", error: null } },
  { key: "c17", name: "Qwen3-TTS 1.7B CustomVoice", summary: "", repo: "r", sizeGb: 3.08, languages: "", default: false, kind: "custom", capabilities: { ...caps, style: true }, supported: true, installed: false, download: { state: "idle", error: null } },
];
const catalog: VoiceCatalog = { voices, models, modelsAvailable: true, cloneAllowed: true, loaded: true, error: null, refresh: async () => {}, download: async () => {} };

describe("SpeakScreen", () => {
  it("greys the style box and chips for a voice without style, with a note", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(html).toMatch(/<textarea[^>]*disabled=""[^>]*placeholder="Tone, pace, accent, in your own words/);
    expect(html).toContain("have no style control");
    expect(html).toContain(">Angry<");
    expect(html).toContain("Runs with Kokoro");
    expect(html).toContain("Design a voice from a description");
    expect(html).toContain("Advanced");
  });

  it("explains the small model and offers the style model download", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />);
    expect(html).toContain("The small model ignores style");
    expect(html).toContain("Download Qwen3-TTS 1.7B CustomVoice for style");
  });

  it("keeps the standing statement in the Cloned group, the same whichever voice is chosen", () => {
    const heart = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />);
    const clone = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="clone:1" onVoiceChange={() => {}} />);
    expect(heart.match(/Only clone your own voice/g)).toHaveLength(1);
    expect(clone.match(/Only clone your own voice/g)).toHaveLength(1);
    // In the picker, not between the picker and the style box.
    expect(clone.indexOf("Only clone your own voice")).toBeLessThan(clone.indexOf("voice-picker__group--designed"));
  });

  it("tells a designed voice what to describe and how to keep it", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="design" onVoiceChange={() => {}} />);
    expect(html).toContain("Describe the voice");
    expect(html).toContain("Say the age, gender, pitch, tone, accent and pace");
    expect(html).toContain("set a seed under Advanced");
  });

  it("shows the standing statement for a cloned voice", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="clone:1" onVoiceChange={() => {}} />);
    expect(html).toContain("Only clone your own voice");
    expect(html).toContain("own voice, confirmed 2026-11-08");
  });
});

describe("SpeakScreen cloning model and voice picker", () => {
  const base = (key: string, name: string, sizeGb: number, installed: boolean): VoiceModelRow => ({
    key, name, summary: "", repo: "r", sizeGb, languages: "", default: key === "b06", kind: "base", capabilities: { ...caps, clone: true }, supported: true, installed,
    download: { state: "idle", error: null },
  });
  const cloneCatalog = (installed06: boolean, installed17: boolean): VoiceCatalog => ({
    ...catalog,
    voices: voices.map((v) => (v.kind === "clone" ? { ...v, models: ["b06", "b17"] } : v)),
    models: [...models, base("b06", "Qwen3-TTS 0.6B", 2.3, installed06), base("b17", "Qwen3-TTS 1.7B", 3.8, installed17)],
  });

  it("offers a model choice only for a clone voice with two cloning models installed", () => {
    const both = renderToStaticMarkup(<SpeakScreen catalog={cloneCatalog(true, true)} voiceId="clone:1" onVoiceChange={() => {}} />);
    expect(both).toContain("Cloning model");
    const one = renderToStaticMarkup(<SpeakScreen catalog={cloneCatalog(false, true)} voiceId="clone:1" onVoiceChange={() => {}} />);
    expect(one).not.toContain("Cloning model");
    expect(one).toContain("Runs with Qwen3-TTS 1.7B");
    const notClone = renderToStaticMarkup(<SpeakScreen catalog={cloneCatalog(true, true)} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(notClone).not.toContain("Cloning model");
  });

  it("makes exactly one voice card a Tab stop", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />);
    const radios = html.match(/<button[^>]*role="radio"[^>]*>/g) ?? [];
    expect(radios.length).toBeGreaterThan(2);
    expect(radios.filter((r) => r.includes('tabindex="0"'))).toHaveLength(1);
    expect(radios.find((r) => r.includes('tabindex="0"'))).toContain('data-voice-id="qwen3:Ryan"');
  });

  it("links the greyed-out style box to its reason", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(html).toContain('aria-describedby="voice-speak-style-reason"');
    expect(html).toContain('id="voice-speak-style-reason"');
  });
});
