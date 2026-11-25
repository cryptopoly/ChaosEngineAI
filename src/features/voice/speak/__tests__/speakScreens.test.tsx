import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn(), isTauri: vi.fn(() => false) }));

import type { VoiceEntry, VoiceModelRow } from "../../../../types";
import { SpeakScreen } from "../SpeakScreen";
import { VoicePicker } from "../VoicePicker";
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
  it("hides the style box and chips for a voice without style: one sentence and one button instead", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(html).not.toContain("Tone, pace, accent, in your own words");
    expect(html).not.toContain(">Angry<");
    expect(html).toContain("Style needs a natural voice.");
    expect(html).toContain(">Get one, 3.1 GB<");
    // The install card would repeat that button.
    expect(html).not.toContain("Install a better voice");
    expect(html).toContain("Runs with Kokoro");
    expect(html).toContain("Advanced");
  });

  it("says what the voice can do in one line under the voice row", () => {
    const quick = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(quick).toContain("This voice can: speed. Languages: English (US).");
    expect(quick.indexOf("This voice can")).toBeGreaterThan(quick.indexOf("voice-row"));
    expect(quick.indexOf("This voice can")).toBeLessThan(quick.indexOf("voice-speak__style"));
    const natural = renderToStaticMarkup(
      <SpeakScreen catalog={{ ...catalog, models: models.map((m) => ({ ...m, installed: true })) }} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />,
    );
    expect(natural).toContain("This voice can: speed, style. Languages: English.");
  });

  it("moves Pitch under Advanced and disables it with the reason up front when ffmpeg is missing", () => {
    const ok = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(ok.indexOf("Pitch (0)")).toBeGreaterThan(ok.indexOf("Advanced"));
    expect(ok).not.toContain("needs ffmpeg");
    const noFfmpeg = { ...catalog, voices: voices.map((v) => ({ ...v, needsFfmpeg: true })) };
    const quick = renderToStaticMarkup(<SpeakScreen catalog={noFfmpeg} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(quick).toMatch(/Pitch \(0\)<\/span><input[^>]*disabled=""/);
    expect(quick).toContain("Pitch needs ffmpeg, which is not installed.");
    // Quick voices change speed themselves; the others need ffmpeg for it too.
    expect(quick).not.toContain("Speed for this voice needs ffmpeg");
    const natural = renderToStaticMarkup(<SpeakScreen catalog={noFfmpeg} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />);
    expect(natural).toContain("Speed for this voice needs ffmpeg");
    expect(natural).toMatch(/Speed \(1\.0×\)<\/span><input[^>]*disabled=""/);
  });

  it("shows one voice row with Change voice instead of the picker, so the form and Generate come first (D-73 S1)", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />);
    expect(html).not.toContain('role="radiogroup"');
    expect(html).toContain(">Change voice<");
    expect(html).toMatch(/<strong>Ryan<\/strong><span class="badge">Natural<\/span>/);
    expect(html).toContain("English, male");
    // Text, then the voice row, then the style box, then Generate.
    const order = ["Text to speak", "voice-row", "voice-speak__style", "Generate audio"].map((needle) => html.indexOf(needle));
    expect(order.every((at) => at >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
  });

  it("explains the small model and offers the style model download with its size, in place of the style box", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />);
    expect(html).toContain("The small model ignores style");
    expect(html).toContain("Download Qwen3-TTS 1.7B CustomVoice for style, 3.1 GB");
    expect(html).not.toContain("Tone, pace, accent, in your own words");
  });

  it("keeps the cloning statement off Speak: it lives in the New cloned voice form (D-73 S2)", () => {
    for (const id of ["af_heart", "clone:1"]) {
      expect(renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId={id} onVoiceChange={() => {}} />)).not.toContain("Only clone your own voice");
    }
  });

  it("tells a designed voice what to describe and how to keep it", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="design" onVoiceChange={() => {}} />);
    expect(html).toContain("Describe the voice");
    expect(html).toContain("Say the age, gender, pitch, tone, accent and pace");
    expect(html).toContain("set a seed under Advanced");
  });

  it("names a cloned voice's permission record on the voice row", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="clone:1" onVoiceChange={() => {}} />);
    expect(html).toContain("own voice, confirmed 2026-11-08");
    expect(html).toMatch(/<strong>My voice<\/strong><span class="badge">Cloned<\/span>/);
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
    // The cloning model sits with the voice it belongs to: right under the voice row, above the style box.
    expect(both.indexOf("Cloning model")).toBeGreaterThan(both.indexOf("voice-row"));
    expect(both.indexOf("Cloning model")).toBeLessThan(both.indexOf("voice-speak__style"));
  });

  it("makes exactly one voice card a Tab stop, and marks it to take focus when the drawer opens", () => {
    const html = renderToStaticMarkup(
      <VoicePicker voices={voices} models={models} selectedId="qwen3:Ryan" onSelect={() => {}} modelsAvailable isPlaying={() => false} onSample={() => {}} />,
    );
    const radios = html.match(/<button[^>]*role="radio"[^>]*>/g) ?? [];
    expect(radios.length).toBeGreaterThan(2);
    expect(radios.filter((r) => r.includes('tabindex="0"'))).toHaveLength(1);
    expect(radios.find((r) => r.includes('tabindex="0"'))).toContain('data-voice-id="qwen3:Ryan"');
    expect(radios.filter((r) => r.includes("data-autofocus"))).toHaveLength(1);
    expect(html).not.toContain("Only clone your own voice");
  });

  it("draws Turbo's tag chips, with its undocumented style tags labelled as such", () => {
    const turbo: VoiceModelRow = {
      key: "chatterbox-turbo", name: "Chatterbox-Turbo", summary: "", repo: "r", sizeGb: 0.71, languages: "English", default: false, kind: "turbo",
      family: "chatterbox-turbo", capabilities: { ...caps, clone: true, tags: true, presets: true },
      controls: { tags: ["laugh", "clear throat"], styleTags: ["whispering"] }, supported: true, installed: true, download: { state: "idle", error: null },
    };
    const turboVoice: VoiceEntry = { id: "chatterbox-turbo", name: "Turbo", kind: "preset", family: "chatterbox-turbo", language: "en", model: "chatterbox-turbo", models: ["chatterbox-turbo"], capabilities: caps, available: true };
    const html = renderToStaticMarkup(
      <SpeakScreen catalog={{ ...catalog, voices: [...voices, turboVoice], models: [...models, turbo] }} voiceId="chatterbox-turbo" onVoiceChange={() => {}} />,
    );
    expect(html).toContain("[Laugh]");
    expect(html).toContain("[Clear throat]");
    // The tag chips sit under the voice row with the other controls of that voice, not above it.
    expect(html.indexOf("[Laugh]")).toBeGreaterThan(html.indexOf("voice-row"));
    expect(html).toContain("[Whispering]");
    expect(html).toContain("its authors do not document them");
    expect(html).toContain("takes sound tags in the text");
    // Kokoro gets no chips.
    expect(renderToStaticMarkup(<SpeakScreen catalog={catalog} voiceId="af_heart" onVoiceChange={() => {}} />)).not.toContain("[Laugh]");
  });

  it("draws Chatterbox's language select, Expression and Guidance from the catalog", () => {
    const range = { min: 0, max: 1, step: 0.05, default: 0.5 };
    const mtl: VoiceModelRow = {
      key: "chatterbox-mtl", name: "Chatterbox Multilingual V3", summary: "", repo: "r", sizeGb: 2.71, languages: "23 languages", default: false, kind: "chatterbox",
      family: "chatterbox", capabilities: { ...caps, clone: true, expression: true, languageSelect: true },
      controls: { languages: ["de", "en"], expression: range, guidance: range }, supported: true, installed: true, download: { state: "idle", error: null },
    };
    const mtlVoice: VoiceEntry = { id: "chatterbox", name: "Chatterbox", kind: "preset", family: "chatterbox", language: "auto", model: "chatterbox-mtl", models: ["chatterbox-mtl"], capabilities: caps, available: true };
    const html = renderToStaticMarkup(
      <SpeakScreen catalog={{ ...catalog, voices: [...voices, mtlVoice], models: [...models, mtl] }} voiceId="chatterbox" onVoiceChange={() => {}} />,
    );
    expect(html).toContain("Expression (calm to dramatic)");
    expect(html).toContain("Follow the voice sample closely");
    expect(html).toContain(">German<");
    expect(html).toContain("use the Expression slider");
  });

  it("shows the style box, with its chips, only for a voice that can use it", () => {
    const styled = { ...catalog, models: models.map((m) => ({ ...m, installed: true })) };
    const html = renderToStaticMarkup(<SpeakScreen catalog={styled} voiceId="qwen3:Ryan" onVoiceChange={() => {}} />);
    expect(html).toMatch(/<textarea[^>]*placeholder="Tone, pace, accent, in your own words/);
    expect(html).not.toMatch(/<textarea[^>]*disabled=""[^>]*placeholder="Tone/);
    expect(html).toContain(">Angry<");
    expect(html).not.toContain("voice-speak-style-reason");
  });

  it("offers a natural voice to choose when the style model is already downloaded and a quick voice is picked", () => {
    const styled = { ...catalog, models: models.map((m) => ({ ...m, installed: true })) };
    const html = renderToStaticMarkup(<SpeakScreen catalog={styled} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(html).toContain("Style needs a natural voice.");
    expect(html).toContain(">Choose a natural voice<");
    expect(html).not.toContain("Get one");
  });

  it("gives a quick voice no button when natural voices cannot run here", () => {
    const html = renderToStaticMarkup(<SpeakScreen catalog={{ ...catalog, modelsAvailable: false, models: [] }} voiceId="af_heart" onVoiceChange={() => {}} />);
    expect(html).toContain("Quick voices have no style control");
    expect(html).not.toContain("Get one");
  });
});
