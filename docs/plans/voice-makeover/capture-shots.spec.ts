// Screenshot recipe for the voice makeover. To re-run: copy to e2e/tests/99-voice-shots.spec.ts, then from e2e/
// E2E_WEB_PORT=8961 E2E_API_PORT=8962 npx playwright test 99-voice-shots --workers=1 (own ports; never 8876).
// VOICE_SHOTS=shots writes the "before" set the review used; the default writes the finished screens to after/.
import { mkdirSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openApp, openTab } from "./helpers";
import { ADA, mockVoiceApi, ttsModels, voiceList } from "./voiceMocks";

// Screenshots of every Voice screen and state for docs/plans/voice-makeover. Not a test of behaviour.

const OUT = path.resolve(__dirname, "..", "..", "docs", "plans", "voice-makeover", process.env.VOICE_SHOTS ?? "after");
const DESKTOP = { width: 1280, height: 900 };
const PHONE = { width: 375, height: 812 };

async function shoot(page: Page, name: string, phone = false) {
  mkdirSync(OUT, { recursive: true });
  await page.waitForTimeout(500);
  // The app fills the viewport and scrolls inside its own boxes (zoom 0.9), so the capture grows the viewport
  // to the tallest scroll box instead of using Playwright's full-page mode.
  const size = page.viewportSize() ?? DESKTOP;
  const tallest = await page.evaluate(() => {
    let max = 0;
    for (const el of document.querySelectorAll<HTMLElement>("main, main *")) {
      const style = getComputedStyle(el);
      if (["auto", "scroll"].includes(style.overflowY) && el.scrollHeight > el.clientHeight + 1) {
        max = Math.max(max, el.scrollHeight + el.getBoundingClientRect().top / 0.9 + 40);
      }
    }
    return max;
  });
  const height = Math.min(Math.max(size.height, Math.ceil(tallest * 0.9)), 12000);
  if (height !== size.height) {
    await page.setViewportSize({ width: size.width, height });
    await page.waitForTimeout(300);
  }
  await page.screenshot({ path: path.join(OUT, `${phone ? "m" : "d"}-${name}.png`), fullPage: false });
  if (height !== size.height) await page.setViewportSize(size);
}

const pick = (page: Page, id: string) => page.locator(`.voice-card__main[data-voice-id="${id}"]`).first();

const range = { min: 0, max: 1, step: 0.05, default: 0.5 };
const caps = (over: Record<string, boolean> = {}) => ({ speak: true, style: false, clone: false, design: false, ...over });
const measured = (peakGb: number, secondsPer10sAudio: number) => ({
  basis: "measured", measuredOn: "Apple M4 Max, 64 GB, 2026-11-09", coldStartSeconds: 2.4, secondsPer10sAudio, peakGb, idleGb: peakGb,
});

function wave2Models(installed: string[]) {
  const base = (key: string) => ({ summary: "", revision: "0", supported: true, installed: installed.includes(key), download: { state: "idle", error: null, phase: null }, default: false });
  return [
    {
      ...base("chatterbox-turbo"), key: "chatterbox-turbo", name: "Chatterbox-Turbo", repo: "ResembleAI/chatterbox-turbo", sizeGb: 0.71, languages: "English", kind: "turbo",
      family: "chatterbox-turbo", capabilities: caps({ clone: true, tags: true, presets: true }), minReferenceS: 5,
      controls: { tags: ["laugh", "sigh", "clear throat", "cough"], styleTags: ["whispering", "angry", "happy"] }, estimates: measured(1.72, 2.1),
      licence: { id: "mit", name: "MIT", url: "https://example.invalid/mit" },
      summary: "Its own English voice that performs sound tags; clones from 5 s.",
    },
    {
      ...base("voxcpm2"), key: "voxcpm2", name: "VoxCPM2", repo: "openbmb/VoxCPM2", sizeGb: 3.23, languages: "30 languages", kind: "voxcpm",
      family: "voxcpm2", capabilities: caps({ style: true, clone: true, design: true }), minReferenceS: 3, sampleRate: 48000,
      controls: {}, estimates: measured(3.8, 5.22), licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/a" },
      summary: "Design a voice, or clone one, in 30 languages (found from the text); 48 kHz.",
    },
    {
      ...base("chatterbox-mtl"), key: "chatterbox-mtl", name: "Chatterbox Multilingual V3", repo: "ResembleAI/chatterbox", sizeGb: 2.71, languages: "23 languages",
      kind: "chatterbox", family: "chatterbox", capabilities: caps({ clone: true, expression: true, presets: true, languageSelect: true }),
      controls: { languages: ["de", "en", "fr", "ja"], expression: range, guidance: range }, minReferenceS: 3, estimates: measured(3.67, 3.63),
      licence: { id: "mit", name: "MIT", url: "https://example.invalid/mit" },
      summary: "A default voice in 23 languages with an Expression dial; clones.",
    },
    {
      ...base("orpheus-3b"), key: "orpheus-3b", name: "Orpheus 3B", repo: "canopylabs/orpheus-3b-0.1-ft", sizeGb: 3.54, languages: "English", kind: "orpheus",
      family: "orpheus", capabilities: caps({ tags: true, presets: true }), controls: { tags: ["laugh", "chuckle", "sigh", "yawn", "gasp"] }, estimates: measured(4.19, 7.22),
      licence: { id: "apache-2.0", name: "Apache 2.0", url: "https://example.invalid/a" },
      summary: "Eight expressive English voices with sound tags.",
    },
  ];
}

function wave2Voices(clones: Array<typeof ADA>, installed: string[]) {
  const preset = (id: string, name: string, family: string, model: string, gender: string, about: string) => ({
    id, name, kind: "preset", group: "natural", family, language: "en", languageLabel: "English", gender, about, model, models: [model],
    capabilities: caps(), available: installed.includes(model), needs: installed.includes(model) ? null : `Download ${model} in Voice Models.`,
  });
  return [
    ...voiceList([], installed).map((v) => (v.kind === "design" ? { ...v, models: ["qwen3-tts-design-1.7b", "voxcpm2"] } : v)),
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

const FULL = ["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b", "qwen3-tts-design-1.7b", "chatterbox-turbo", "orpheus-3b", "chatterbox-mtl"];

/** Pack installed and healthy, every list mocked. */
async function installedWorld(page: Page, opts: { installed?: string[]; clones?: Array<typeof ADA>; speakVoice?: string; ttsVoice?: string } = {}) {
  const installed = opts.installed ?? FULL;
  const clones = opts.clones ?? [ADA];
  await mockVoiceApi(page, { installed, clones, speakVoice: opts.speakVoice ?? "", ttsVoice: opts.ttsVoice ?? "af_heart" });
  // Settings answered here in full (the shared mock's copy never sees the runtime below, so it would answer with
  // only the two voice keys and the Chat and calls panel would read speed as undefined).
  let settings: Record<string, unknown> | null = null;
  await page.route("**/api/voice/settings", async (route) => {
    if (!settings) {
      const response = await route.fetch({ method: "GET", postData: undefined });
      settings = { ...(await response.json()), speakVoice: opts.speakVoice ?? "", ttsVoice: opts.ttsVoice ?? "af_heart" };
    }
    if (route.request().method() === "POST") Object.assign(settings, route.request().postDataJSON() as Record<string, unknown>);
    return route.fulfill({ json: settings });
  });
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, {
      installed: true, sttAvailable: true, ttsAvailable: true, ttsModelInstalled: true, vadAvailable: true, speakersAvailable: true,
      updateAvailable: false, workerPid: 48213, installedAt: "2026-11-09T09:12:00",
      parts: [{ id: "packages", sizeGb: 1.92 }, { id: "kokoro", sizeGb: 0.33 }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 }],
      inMemory: [{ lane: "tts", name: "Kokoro", peakGb: 0.87 }],
    });
    runtime.settings = { ...(runtime.settings ?? {}), ...(settings ?? {}), speakVoice: opts.speakVoice ?? "", ttsVoice: opts.ttsVoice ?? "af_heart" };
    if (Array.isArray(runtime.sttModels) && runtime.sttModels[0]) runtime.sttModels[0].installed = true;
    await route.fulfill({ response, json: body });
  });
  const models = [...ttsModels(installed), ...wave2Models(installed)];
  await page.route("**/api/voice/voices", (route) => route.fulfill({ json: { voices: wave2Voices(clones, installed) } }));
  const fit = (peakGb: number) => ({ status: peakGb > 4 ? "tight" : "fits", needGb: peakGb, usableGb: 51.8, headroomPct: 90, reason: "Peak memory while speaking", basis: "measured" });
  const withFit = models.map((m) => ({ ...m, fit: m.estimates ? fit(m.estimates.peakGb) : null }));
  await page.route("**/api/voice/models", (route) => route.fulfill({ json: { models: withFit, available: true, kokoro: { peakGb: 0.87, fit: fit(0.87) } } }));
  await page.route("**/api/voice/clone/models", (route) =>
    route.fulfill({ json: { available: true, allowedHere: true, models: models.filter((m) => m.capabilities.clone) } }),
  );
}

function libraryItems() {
  const at = (i: number) => Date.UTC(2026, 10, 8, 9, i) / 1000;
  const item = (i: number, kind: "speech" | "transcript", title: string, preview: string, extra: Record<string, unknown> = {}) => ({
    id: `v-${i}`, kind, title, preview, createdAt: at(i), updatedAt: at(i), audio: kind === "speech" ? ".wav" : ".m4a", speakers: {},
    speakerCount: kind === "transcript" ? 2 : 0, durationS: kind === "speech" ? 14 : 1860, projectId: null, projectName: null, ...extra,
  });
  return [
    item(1, "speech", "Good evening, everyone.", "Good evening, everyone. Tonight we look at the harbour.", { voice: "qwen3:Ryan", voiceName: "Ryan", voiceKind: "preset" }),
    item(2, "speech", "Chapter one", "Chapter one. The lighthouse keeper woke before dawn.", { voice: "clone:voice-1", voiceName: "Ada", voiceKind: "clone", clonedVoice: true }),
    item(3, "speech", "Welcome back", "Welcome back to the show.", { voice: "af_heart", voiceName: "Heart", voiceKind: "builtin" }),
    item(4, "transcript", "Team call 2026-11-08", "Speaker 1: Let's start with the release. Speaker 2: The build is green."),
    item(5, "speech", "A calm narrator", "Once upon a time, in a quiet valley.", { voice: "design", voiceName: "A calm woman in her forties, slow", voiceKind: "design" }),
  ];
}

async function openSpeak(page: Page) {
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: "Speak" }).click();
  await expect(page.getByLabel("Text to speak")).toBeVisible();
}

/** Speak's voices are chosen in the drawer behind Change voice. */
async function choose(page: Page, id: string) {
  await page.getByRole("button", { name: "Change voice" }).click();
  await pick(page, id).click();
  await expect(page.getByRole("dialog", { name: "Choose a voice" })).toHaveCount(0);
}

for (const phone of [false, true]) {
  const size = phone ? PHONE : DESKTOP;
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: size });

    test("pack not installed: Models, Studio gate, Library empty", async ({ page }) => {
      await goTo(page, "Voice", "Models");
      await shoot(page, "models-no-pack", phone);
      await openTab(page, "Voice", "Studio");
      await shoot(page, "studio-gate", phone);
      await openTab(page, "Voice", "Library");
      await shoot(page, "library-empty", phone);
    });

    test("Models with the pack and models installed", async ({ page }) => {
      await installedWorld(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-1.7b", "chatterbox-turbo"] });
      await goTo(page, "Voice", "Models");
      await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
      await shoot(page, "models-installed", phone);
    });

    test("Speak: first run, no natural voice", async ({ page }) => {
      await installedWorld(page, { installed: ["qwen3-tts-0.6b"], clones: [] });
      await openSpeak(page);
      await shoot(page, "speak-first-run", phone);
    });

    test("Speak with each kind of voice", async ({ page }) => {
      await installedWorld(page, { speakVoice: "qwen3:Ryan" });
      await openSpeak(page);
      await page.getByLabel("Text to speak").fill("Good evening, everyone. Tonight we look at the harbour, the boats and the people who work there.");
      await shoot(page, "speak-natural", phone);
      await page.getByRole("button", { name: "Change voice" }).click();
      await expect(page.getByRole("dialog", { name: "Choose a voice" })).toBeVisible();
      await page.screenshot({ path: path.join(OUT, `${phone ? "m" : "d"}-speak-drawer.png`), fullPage: false });
      await page.keyboard.press("Escape");
      await choose(page, "clone:voice-1");
      await shoot(page, "speak-clone", phone);
      await choose(page, "design");
      await shoot(page, "speak-design", phone);
      await choose(page, "chatterbox-turbo");
      await shoot(page, "speak-turbo", phone);
      await choose(page, "chatterbox");
      await shoot(page, "speak-chatterbox", phone);
      await choose(page, "orpheus:tara");
      await shoot(page, "speak-orpheus", phone);
      await choose(page, "af_heart");
      await shoot(page, "speak-quick", phone);
    });

    test("Voices, new clone form, Transcribe", async ({ page }) => {
      await installedWorld(page);
      await goTo(page, "Voice", "Studio");
      await page.getByRole("tab", { name: "Voices" }).click();
      await expect(page.getByRole("button", { name: "New cloned voice" })).toBeVisible();
      await shoot(page, "voices", phone);
      await page.getByRole("button", { name: "New cloned voice" }).click();
      await shoot(page, "voices-new-clone", phone);
      await page.getByRole("tab", { name: "Transcribe" }).click();
      await shoot(page, "transcribe", phone);
    });

    test("Library with speech and a transcript", async ({ page }) => {
      await installedWorld(page);
      await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items: libraryItems() } }));
      await goTo(page, "Voice", "Library");
      await expect(page.locator(".media-library-card").first()).toBeVisible();
      await shoot(page, "library-speech", phone);
    });

    test("Chat voice controls", async ({ page }) => {
      await installedWorld(page);
      await openApp(page);
      await page.getByRole("navigation", { name: "Primary" }).locator(".nav-group-header", { hasText: /^Chat/ }).click().catch(() => {});
      await page.getByRole("navigation", { name: "Primary" }).locator(".nav-child-button", { hasText: /^Chat$/ }).click().catch(() => {});
      await page.waitForTimeout(800);
      await shoot(page, "chat", phone);
      const voice = page.locator(".voice-popover__button").first();
      if (await voice.isVisible().catch(() => false)) {
        await voice.click();
        await expect(page.locator(".voice-popover__panel")).toBeVisible();
        await page.waitForTimeout(300);
        await page.screenshot({ path: path.join(OUT, `${phone ? "m" : "d"}-chat-voice-popover.png`), fullPage: false });
      }
    });
  });
}

test.describe("desktop only: the other areas, for comparison", () => {
  test.use({ viewport: DESKTOP });

  test("Music Models, Music Studio, Image Models, Settings", async ({ page }) => {
    await goTo(page, "Music and Audio", "Models");
    await shoot(page, "music-models");
    await openTab(page, "Music and Audio", "Studio");
    await shoot(page, "music-studio");
    await openTab(page, "Images", "Models");
    await shoot(page, "image-models");
    await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Settings/ }).click();
    await shoot(page, "settings");
  });
});
