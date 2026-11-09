import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { ADA, mockVoiceApi, speakAnswer, ttsModels, voiceList } from "./voiceMocks";

// Wave-2 speech models on the Speak screen (D-70, FU-149): Chatterbox-Turbo's tag chips, Chatterbox V3's language,
// Expression and Guidance, the Orpheus preset voices, VoxCPM2 as a design and cloning model. Voice and model lists
// are mocked on top of the shared voice mocks; /api/voice/speak is mocked and its request body is checked.

const range = { min: 0, max: 1, step: 0.05, default: 0.5 };
const caps = (over: Record<string, boolean> = {}) => ({ speak: true, style: false, clone: false, design: false, ...over });
const measured = (peakGb: number, secondsPer10sAudio: number) => ({
  basis: "measured", measuredOn: "Apple M4 Max, 64 GB, 2026-11-09", coldStartSeconds: 2.4, secondsPer10sAudio, peakGb, idleGb: peakGb,
});

function wave2Models() {
  const base = { summary: "Mock.", revision: "0", supported: true, installed: true, download: { state: "idle", error: null, phase: null }, default: false };
  return [
    {
      ...base, key: "chatterbox-turbo", name: "Chatterbox-Turbo", repo: "mock/turbo", sizeGb: 0.71, languages: "English", kind: "turbo",
      family: "chatterbox-turbo", capabilities: caps({ clone: true, tags: true, presets: true }), minReferenceS: 5,
      controls: { tags: ["laugh", "sigh", "clear throat"], styleTags: ["whispering", "angry"] }, estimates: measured(1.72, 2.1),
      licence: { id: "mit", name: "MIT", url: "https://example.invalid/mit" },
    },
    {
      ...base, key: "voxcpm2", name: "VoxCPM2", repo: "mock/voxcpm2", sizeGb: 3.23, languages: "30 languages", kind: "voxcpm",
      family: "voxcpm2", capabilities: caps({ style: true, clone: true, design: true }), minReferenceS: 3, sampleRate: 48000,
      controls: {}, estimates: measured(3.8, 5.22), licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/a" },
    },
    {
      ...base, key: "chatterbox-mtl", name: "Chatterbox Multilingual V3", repo: "mock/mtl", sizeGb: 2.71, languages: "23 languages",
      kind: "chatterbox", family: "chatterbox", capabilities: caps({ clone: true, expression: true, presets: true, languageSelect: true }),
      controls: { languages: ["de", "en", "ja"], expression: range, guidance: range }, minReferenceS: 3, estimates: measured(3.67, 3.63),
      licence: { id: "mit", name: "MIT", url: "https://example.invalid/mit" },
    },
    {
      ...base, key: "orpheus-3b", name: "Orpheus 3B", repo: "mock/orpheus", sizeGb: 3.54, languages: "English", kind: "orpheus",
      family: "orpheus", capabilities: caps({ tags: true, presets: true }), controls: { tags: ["laugh", "yawn"] }, estimates: measured(4.19, 7.22),
      licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/a" },
    },
  ];
}

function wave2Voices(clones: Array<typeof ADA>) {
  const installed = ["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b", "qwen3-tts-design-1.7b"];
  const preset = (id: string, name: string, family: string, model: string, gender: string, about: string) => ({
    id, name, kind: "preset", group: "natural", family, language: "en", languageLabel: "English", gender, about, model, models: [model],
    capabilities: caps(), available: true, needs: null,
  });
  return [
    ...voiceList([], installed).map((v) =>
      v.kind === "design" ? { ...v, models: ["qwen3-tts-design-1.7b", "voxcpm2"] } : v,
    ),
    preset("chatterbox-turbo", "Turbo", "chatterbox-turbo", "chatterbox-turbo", "male", "Chatterbox-Turbo's own voice; reads sound tags"),
    preset("chatterbox", "Chatterbox", "chatterbox", "chatterbox-mtl", "female", "Chatterbox's default voice in 23 languages"),
    preset("orpheus:tara", "Tara", "orpheus", "orpheus-3b", "female", "Most natural of the eight (authors' rank 1)"),
    preset("orpheus:leo", "Leo", "orpheus", "orpheus-3b", "male", "Authors' rank 4"),
    ...clones.map((c) => ({
      id: `clone:${c.id}`, name: c.name, kind: "clone", group: "cloned", language: "auto", languageLabel: "Any", gender: null,
      model: "qwen3-tts-0.6b", models: ["qwen3-tts-0.6b", "chatterbox-turbo", "voxcpm2", "chatterbox-mtl"],
      capabilities: caps({ clone: true }), available: true, needs: null, consent: c.consent, durationS: c.durationS,
    })),
  ];
}

async function setup(page: Page, clones: Array<typeof ADA> = []): Promise<Array<Record<string, unknown>>> {
  await mockVoiceApi(page, { clones, installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b", "qwen3-tts-design-1.7b"], speakVoice: "af_heart" });
  // Registered after the shared mocks, so these answers win.
  const models = [...ttsModels(["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b", "qwen3-tts-design-1.7b"]), ...wave2Models()];
  await page.route("**/api/voice/voices", (route) => route.fulfill({ json: { voices: wave2Voices(clones) } }));
  await page.route("**/api/voice/models", (route) => route.fulfill({ json: { models, available: true } }));
  const bodies: Array<Record<string, unknown>> = [];
  await page.route("**/api/voice/speak", (route) => {
    bodies.push(route.request().postDataJSON() as Record<string, unknown>);
    return route.fulfill({ json: speakAnswer() });
  });
  await page.route("**/api/voice/library/speech", (route) =>
    route.fulfill({ json: { id: "1790000000_abcdef12", kind: "speech", title: "x", text: "x", createdAt: 1, updatedAt: 1, audio: ".wav", speakers: {}, segments: [] } }),
  );
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Speak" }).click();
  await expect(page.getByLabel("Text to speak")).toBeVisible();
  return bodies;
}

const pick = (page: Page, name: RegExp) => page.getByRole("radio", { name }).first().click();

test("Turbo: tag chips insert [laugh] at the cursor, its undocumented style tags say so, and the request carries no extra fields", async ({ page }) => {
  const bodies = await setup(page);
  await pick(page, /^Turbo/);
  const text = page.getByLabel("Text to speak");
  await text.fill("That is funny");
  await text.press("End");
  await page.getByRole("button", { name: "[Laugh]" }).click();
  await expect(text).toHaveValue("That is funny [laugh]");
  await expect(page.getByRole("group", { name: "Style tags (undocumented)" }).getByRole("button", { name: "[Whispering]" })).toBeVisible();
  await expect(page.getByText("its authors do not document them")).toBeVisible();
  await expect(page.getByRole("textbox", { name: "Style", exact: true })).toBeDisabled();
  await expect(page.getByText("takes sound tags in the text")).toBeVisible();
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect.poll(() => bodies.length).toBe(1);
  expect(bodies[0]).toMatchObject({ text: "That is funny [laugh]", voice: "chatterbox-turbo" });
  expect(bodies[0]).not.toHaveProperty("language");
  expect(bodies[0]).not.toHaveProperty("expression");
});

test("Chatterbox: the language select, Expression and Guidance go into the request", async ({ page }) => {
  const bodies = await setup(page);
  await pick(page, /^Chatterbox/);
  await expect(page.getByRole("button", { name: "[Laugh]" })).toHaveCount(0);
  await page.getByLabel("Speech language").selectOption("de");
  await page.getByText("Advanced").click();
  await expect(page.getByLabel("Follow the voice sample closely")).toBeVisible();
  await page.getByLabel("Text to speak").fill("Guten Morgen.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect.poll(() => bodies.length).toBe(1);
  expect(bodies[0]).toMatchObject({ voice: "chatterbox", language: "de", expression: 0.5, guidance: 0.5 });
});

test("Orpheus voices are natural voices; a designed voice can run on VoxCPM2", async ({ page }) => {
  const bodies = await setup(page);
  await expect(page.getByRole("radio", { name: /^Tara/ }).first()).toBeVisible();
  await pick(page, /^Leo/);
  await expect(page.getByRole("button", { name: "[Yawn]" })).toBeVisible();
  await pick(page, /Design a voice from a description/);
  await page.getByLabel("Design model").selectOption("voxcpm2");
  await page.getByRole("textbox", { name: "Describe the voice" }).fill("A calm older man.");
  await page.getByLabel("Text to speak").fill("Hello there.");
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect.poll(() => bodies.length).toBe(1);
  expect(bodies[0]).toMatchObject({ voice: "design", model: "voxcpm2", style: "A calm older man." });
});

test("a cloned voice can use Turbo or VoxCPM2; Turbo refuses a sample under 5 s, VoxCPM2 opens the style box", async ({ page }) => {
  const short = { ...ADA, durationS: 4 };
  const bodies = await setup(page, [short]);
  await pick(page, /^Ada\b/);
  const select = page.getByLabel("Cloning model");
  await expect(select.locator("option")).toHaveText(["Qwen3-TTS 0.6B Base", "Chatterbox-Turbo", "VoxCPM2", "Chatterbox Multilingual V3"]);
  await expect(page.getByText("Only clone your own voice").first()).toBeVisible();
  await select.selectOption("chatterbox-turbo");
  await expect(page.getByText("Chatterbox-Turbo needs a voice sample of at least 5 seconds")).toBeVisible();
  await page.getByLabel("Text to speak").fill("Hello.");
  await expect(page.getByRole("button", { name: "Generate audio" })).toBeDisabled();
  await select.selectOption("voxcpm2");
  await page.getByRole("textbox", { name: "Style", exact: true }).fill("cheerful");
  await expect(page.getByText("keeps a little less of the likeness")).toBeVisible();
  await page.getByRole("button", { name: "Generate audio" }).click();
  await expect.poll(() => bodies.length).toBe(1);
  expect(bodies[0]).toMatchObject({ voice: "clone:voice-1", model: "voxcpm2", style: "cheerful" });
});
