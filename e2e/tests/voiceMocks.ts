import type { Page } from "@playwright/test";

// Mocked voice routes shared by the voice specs: the voice pack looks installed, the model and voice lists are
// fixed, nothing is downloaded and no model runs.

const caps = (over: Record<string, boolean> = {}) => ({ speak: true, style: false, clone: false, design: false, ...over });

// The catalog's measured rows (Apple M4 Max, 64 GB, 2026-11-08): peak memory, seconds per 10 s of speech, cold start.
const MEASURED: Record<string, [number, number, number]> = {
  "qwen3-tts-0.6b": [4.3, 3.2, 3.0],
  "qwen3-tts-1.7b": [6.5, 3.6, 3.6],
  "qwen3-tts-custom-0.6b": [2.93, 2.65, 2.4],
  "qwen3-tts-custom-1.7b": [3.5, 2.45, 2.4],
  "qwen3-tts-design-1.7b": [3.51, 2.46, 2.3],
};

function model(key: string, name: string, kind: string, sizeGb: number, capabilities: Record<string, boolean>, installed: boolean) {
  const [peakGb, secondsPer10sAudio, coldStartSeconds] = MEASURED[key] ?? [0, 0, 0];
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
    estimates: MEASURED[key]
      ? { basis: "measured", measuredOn: "Apple M4 Max, 64 GB, 2026-11-08", coldStartSeconds, secondsPer10sAudio, peakGb, idleGb: peakGb - 0.1 }
      : null,
    supported: true,
    installed,
    licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/licence" },
    download: { state: "idle", error: null, phase: null },
  };
}

export function ttsModels(installed: string[], downloading: string[] = [], cancelled: string[] = []) {
  const rows: Array<[string, string, string, number, Record<string, boolean>]> = [
    ["qwen3-tts-0.6b", "Qwen3-TTS 0.6B Base", "base", 2.52, caps({ clone: true })],
    ["qwen3-tts-1.7b", "Qwen3-TTS 1.7B Base", "base", 3.8, caps({ clone: true })],
    ["qwen3-tts-custom-0.6b", "Qwen3-TTS 0.6B CustomVoice", "custom", 2.5, caps()],
    ["qwen3-tts-custom-1.7b", "Qwen3-TTS 1.7B CustomVoice", "custom", 3.08, caps({ style: true })],
    ["qwen3-tts-design-1.7b", "Qwen3-TTS 1.7B VoiceDesign", "design", 3.08, caps({ style: true, design: true })],
  ];
  return rows.map(([key, name, kind, gb, c]) => {
    const row = model(key, name, kind, gb, c, installed.includes(key));
    if (downloading.includes(key)) {
      row.download = { state: "downloading", error: null, phase: "weights", progress: 0.4, downloadedGb: 1, totalGb: gb } as typeof row.download;
    }
    if (cancelled.includes(key)) row.download = { state: "cancelled", error: null, phase: null } as typeof row.download;
    return row;
  });
}

export const ADA = {
  id: "voice-1",
  name: "Ada",
  createdAt: 1_790_000_000,
  durationS: 12,
  referenceText: "",
  consent: { speakerName: "Ada Example", basis: "own", consentedAt: "2026-11-01T10:00:00Z", appVersion: "0.0.0" },
};

const AMERICAN_VOICES: Array<[string, string]> = [
  ["af_heart", "A"], ["af_alloy", "C"], ["af_aoede", "C+"], ["af_bella", "A-"], ["af_jessica", "D"], ["af_kore", "C+"],
  ["af_nicole", "B-"], ["af_nova", "C"], ["af_river", "D"], ["af_sarah", "C+"], ["af_sky", "C-"],
  ["am_adam", "F+"], ["am_echo", "D"], ["am_eric", "D"], ["am_fenrir", "C+"], ["am_liam", "D"], ["am_michael", "C+"],
  ["am_onyx", "D"], ["am_puck", "C+"], ["am_santa", "D-"],
];

export function voiceList(clones: Array<typeof ADA>, installed: string[], needsFfmpeg = false) {
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
  const quick = (id: string, grade: string | null, language = "en-US", languageLabel = "American English", available = true) => ({
    id,
    name: id.slice(3).replace(/^./, (c) => c.toUpperCase()),
    kind: "builtin",
    language,
    languageLabel,
    gender: id[1] === "f" ? "female" : "male",
    model: "kokoro",
    models: ["kokoro"],
    capabilities: caps(),
    available,
    grade,
    addon: available ? null : "chinese",
    needs: available ? null : `${languageLabel} voices need an extra install.`,
  });
  const list = [
    // All 20 American voices with the model author's grades, so the shortlist and 375 px density are real.
    ...AMERICAN_VOICES.map(([id, grade]) => quick(id, grade)),
    quick("zf_xiaobei", "D", "zh", "Chinese (Mandarin)", false),
    quick("zm_yunxi", "D", "zh", "Chinese (Mandarin)", false),
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
  // Where each voice may speak in Chat and its measured chat cost, as the backend sends them (D-72).
  const chatUse = (kind: string) => ({ read: kind !== "design", call: kind === "builtin" || kind === "preset" });
  const chatCost = (kind: string) =>
    kind === "builtin"
      ? { firstAudioS: 0.13, firstAudioColdS: 5.7, peakGb: 0.87, measuredOn: "Apple M4 Max, 64 GB, 2026-11-09", slowForCalls: false }
      : kind === "preset"
        ? { firstAudioS: 1.3, firstAudioColdS: 4.1, peakGb: 3.51, measuredOn: "Apple M4 Max, 64 GB, 2026-11-09", slowForCalls: false }
        : null;
  return list.map((v) => ({ ...v, needsFfmpeg, chatUse: chatUse(v.kind), chatCost: chatCost(v.kind) }));
}

export interface VoiceMockOptions {
  /** Catalog keys that look downloaded. */
  installed?: string[];
  clones?: Array<typeof ADA>;
  /** The voice last used in Speak; kept in the mock so one test's pick never leaks into the next (the backend is shared). */
  speakVoice?: string;
  /** ffmpeg looks missing: every voice carries ``needsFfmpeg``. */
  needsFfmpeg?: boolean;
  /** The chat voice the runtime reports (``ttsVoice``); kept in the mock like ``speakVoice``. */
  ttsVoice?: string;
  /** The call voice (``callVoice``, "" = the chat voice); kept in the mock like ``ttsVoice``. */
  callVoice?: string;
  /** The fallback voice (``fallbackVoice``, "" = none, D-75); kept in the mock like ``ttsVoice``. */
  fallbackVoice?: string;
  /** The Kokoro quick voices look left out (D-75): the add-on unticked and the quick voices not available. */
  noKokoro?: boolean;
  /** Speech, speech-to-text and turn detection all report ready, so Read aloud and Call are enabled. */
  ready?: boolean;
  /** Catalog keys whose download is running; Cancel on one stops it (the mock records the key in ``cancels``). */
  downloading?: string[];
}

/** Patch the runtime to "pack installed" and mock the voice and model lists plus the clone routes. */
export async function mockVoiceApi(
  page: Page,
  options: VoiceMockOptions = {},
): Promise<{ downloads: string[]; cancels: string[]; deleted: string[]; speakVoices: string[]; ttsVoices: string[]; callVoices: string[]; fallbackVoices: string[] }> {
  const installed = options.installed ?? ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b"];
  const clones = options.clones ?? [];
  const downloading = [...(options.downloading ?? [])];
  const cancelled: string[] = [];
  const seen = { downloads: [] as string[], cancels: [] as string[], deleted: [] as string[], speakVoices: [] as string[], ttsVoices: [] as string[], callVoices: [] as string[], fallbackVoices: [] as string[] };
  let speakVoice = options.speakVoice ?? "";
  let ttsVoice = options.ttsVoice ?? "af_heart";
  let callVoice = options.callVoice ?? "";
  let fallbackVoice = options.fallbackVoice ?? "af_heart";
  let lastSettings: Record<string, unknown> = {};
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, { installed: true });
    if (options.ready) Object.assign(runtime, { ttsAvailable: true, sttAvailable: true, vadAvailable: true });
    runtime.settings = { ...(runtime.settings ?? {}), speakVoice, ttsVoice, callVoice, fallbackVoice };
    // The Kokoro add-on (D-75) follows the mocked voice list: installed unless ``noKokoro``. The throwaway backend has
    // no Kokoro files, so without this the runtime would say "not installed" while the voices say available.
    runtime.ttsModelInstalled = !options.noKokoro;
    runtime.extras = [
      { id: "kokoro", label: "Kokoro quick voices", installed: !options.noKokoro },
      ...(runtime.extras ?? []).filter((e: { id: string }) => e.id !== "kokoro"),
    ];
    lastSettings = runtime.settings;
    await route.fulfill({ response, json: body });
  });
  // The Speak voice and the chat voice are answered here; every other setting goes to the backend as before.
  await page.route("**/api/voice/settings", async (route) => {
    const patch = route.request().method() === "POST" ? (route.request().postDataJSON() as Record<string, unknown>) : {};
    if (typeof patch.speakVoice === "string") {
      speakVoice = patch.speakVoice;
      seen.speakVoices.push(speakVoice);
    } else if (typeof patch.ttsVoice === "string") {
      ttsVoice = patch.ttsVoice;
      seen.ttsVoices.push(ttsVoice);
    } else if (typeof patch.callVoice === "string") {
      callVoice = patch.callVoice;
      seen.callVoices.push(callVoice);
    } else if (typeof patch.fallbackVoice === "string") {
      fallbackVoice = patch.fallbackVoice;
      seen.fallbackVoices.push(fallbackVoice);
    } else {
      return route.fallback();
    }
    lastSettings = { ...lastSettings, speakVoice, ttsVoice, callVoice, fallbackVoice };
    return route.fulfill({ json: lastSettings });
  });
  await page.route("**/api/voice/voices", (route) => {
    const voices = voiceList(clones, installed, options.needsFfmpeg).map((v) =>
      options.noKokoro && v.kind === "builtin" ? { ...v, available: false, addon: null, needs: "The Kokoro quick voices are not installed." } : v,
    );
    return route.fulfill({ json: { voices } });
  });
  await page.route("**/api/voice/models", (route) => route.fulfill({ json: { models: ttsModels(installed, downloading, cancelled), available: true } }));
  await page.route("**/api/voice/clone/models", (route) =>
    route.fulfill({ json: { available: true, allowedHere: true, models: ttsModels(installed, downloading, cancelled).filter((m) => m.kind === "base") } }),
  );
  await page.route("**/api/voice/clone/models/download/cancel", (route) => {
    const key = (route.request().postDataJSON() as { key: string }).key;
    seen.cancels.push(key);
    const at = downloading.indexOf(key);
    if (at >= 0) downloading.splice(at, 1);
    cancelled.push(key);
    return route.fulfill({ json: { download: { state: "cancelled", error: null } } });
  });
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
