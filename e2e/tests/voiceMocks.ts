import type { Page } from "@playwright/test";

// Mocked voice routes shared by the voice specs: the voice pack looks installed, the model and voice lists are
// fixed, nothing is downloaded and no model runs.

const caps = (over: Record<string, boolean> = {}) => ({ speak: true, style: false, clone: false, design: false, ...over });

function model(key: string, name: string, kind: string, sizeGb: number, capabilities: Record<string, boolean>, installed: boolean) {
  return {
    key,
    name,
    summary: "Mock.",
    repo: `mock/${key}`,
    revision: "0",
    sizeGb,
    languages: "10 languages",
    default: key === "qwen3-tts-custom-0.6b",
    kind,
    capabilities: capabilities,
    supported: true,
    installed,
    licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/licence" },
    download: { state: "idle", error: null, phase: null },
  };
}

export function ttsModels(installed: string[]) {
  const rows: Array<[string, string, string, number, Record<string, boolean>]> = [
    ["qwen3-tts-0.6b", "Qwen3-TTS 0.6B Base", "base", 2.52, caps({ clone: true })],
    ["qwen3-tts-1.7b", "Qwen3-TTS 1.7B Base", "base", 3.8, caps({ clone: true })],
    ["qwen3-tts-custom-0.6b", "Qwen3-TTS 0.6B CustomVoice", "custom", 2.5, caps()],
    ["qwen3-tts-custom-1.7b", "Qwen3-TTS 1.7B CustomVoice", "custom", 3.08, caps({ style: true })],
    ["qwen3-tts-design-1.7b", "Qwen3-TTS 1.7B VoiceDesign", "design", 3.08, caps({ style: true, design: true })],
  ];
  return rows.map(([key, name, kind, gb, c]) => model(key, name, kind, gb, c, installed.includes(key)));
}

export const ADA = {
  id: "voice-1",
  name: "Ada",
  createdAt: 1_790_000_000,
  durationS: 12,
  referenceText: "",
  consent: { speakerName: "Ada Example", basis: "own", consentedAt: "2026-11-01T10:00:00Z", appVersion: "0.0.0" },
};

export function voiceList(clones: Array<typeof ADA>, installed: string[]) {
  const customReady = installed.some((k) => k.startsWith("qwen3-tts-custom"));
  const preset = (id: string, language: string, languageLabel: string, gender: string, about: string) => ({
    id: `qwen3:${id}`,
    name: id,
    kind: "preset",
    language,
    languageLabel,
    gender,
    about,
    model: "qwen3-tts-custom-0.6b",
    models: ["qwen3-tts-custom-0.6b", "qwen3-tts-custom-1.7b"],
    capabilities: caps({ style: true }),
    available: customReady,
    needs: customReady ? null : "Download Qwen3-TTS 0.6B CustomVoice in Voice Models.",
  });
  return [
    { id: "af_heart", name: "Heart", kind: "builtin", language: "en-US", languageLabel: "American English", gender: "female", model: "kokoro", models: ["kokoro"], capabilities: caps(), available: true, needs: null },
    { id: "am_adam", name: "Adam", kind: "builtin", language: "en-US", languageLabel: "American English", gender: "male", model: "kokoro", models: ["kokoro"], capabilities: caps(), available: true, needs: null },
    preset("Ryan", "en", "English", "male", "Dynamic, strong rhythm"),
    preset("Aiden", "en", "English", "male", "Sunny American"),
    {
      id: "design", name: "Describe a voice", kind: "design", language: "auto", languageLabel: "Any", gender: null,
      model: "qwen3-tts-design-1.7b", models: ["qwen3-tts-design-1.7b"], capabilities: caps({ style: true, design: true }),
      available: installed.includes("qwen3-tts-design-1.7b"), needs: null,
    },
    ...clones.map((c) => ({
      id: `clone:${c.id}`, name: c.name, kind: "clone", language: "auto", languageLabel: "Any", gender: null,
      model: "qwen3-tts-0.6b", models: ["qwen3-tts-0.6b", "qwen3-tts-1.7b"], capabilities: caps({ clone: true }), available: true, needs: null, consent: c.consent,
    })),
  ];
}

export interface VoiceMockOptions {
  /** Catalog keys that look downloaded. */
  installed?: string[];
  clones?: Array<typeof ADA>;
}

/** Patch the runtime to "pack installed" and mock the voice and model lists plus the clone routes. */
export async function mockVoiceApi(page: Page, options: VoiceMockOptions = {}): Promise<{ downloads: string[]; deleted: string[] }> {
  const installed = options.installed ?? ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b"];
  const clones = options.clones ?? [];
  const seen = { downloads: [] as string[], deleted: [] as string[] };
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    Object.assign(body.runtime ?? body, { installed: true });
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/voice/voices", (route) => route.fulfill({ json: { voices: voiceList(clones, installed) } }));
  await page.route("**/api/voice/models", (route) => route.fulfill({ json: { models: ttsModels(installed), available: true } }));
  await page.route("**/api/voice/clone/models", (route) =>
    route.fulfill({ json: { available: true, allowedHere: true, models: ttsModels(installed).filter((m) => m.kind === "base") } }),
  );
  await page.route("**/api/voice/clone/models/download", (route) => {
    seen.downloads.push((route.request().postDataJSON() as { key: string }).key);
    return route.fulfill({ json: { state: "downloading", error: null } });
  });
  await page.route("**/api/voice/clone/voices", (route) => route.fulfill({ json: { voices: clones } }));
  await page.route("**/api/voice/clone/voices/*", (route) => {
    if (route.request().method() === "DELETE") {
      seen.deleted.push(route.request().url().split("/").pop() ?? "");
      return route.fulfill({ status: 200, json: { deleted: true } });
    }
    return route.fallback();
  });
  return seen;
}

/** A one-second silent mono 16-bit WAV, base64, for mocked /api/voice/speak answers. */
export function silentWavBase64(seconds = 1): string {
  const rate = 16000;
  const data = Buffer.alloc(rate * seconds * 2);
  const head = Buffer.alloc(44);
  head.write("RIFF", 0);
  head.writeUInt32LE(36 + data.length, 4);
  head.write("WAVEfmt ", 8);
  head.writeUInt32LE(16, 16);
  head.writeUInt16LE(1, 20);
  head.writeUInt16LE(1, 22);
  head.writeUInt32LE(rate, 24);
  head.writeUInt32LE(rate * 2, 28);
  head.writeUInt16LE(2, 32);
  head.writeUInt16LE(16, 34);
  head.write("data", 36);
  head.writeUInt32LE(data.length, 40);
  return Buffer.concat([head, data]).toString("base64");
}

export function speakAnswer(over: Record<string, unknown> = {}) {
  return {
    audio: silentWavBase64(),
    durationS: 1,
    seconds: 0.1,
    chunks: 1,
    voice: "af_heart",
    kind: "builtin",
    model: "kokoro",
    modelName: "Kokoro",
    clonedVoice: false,
    libraryId: null,
    notes: [],
    ...over,
  };
}
