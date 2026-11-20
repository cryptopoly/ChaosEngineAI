// Screenshot recipe for the v0.10.0 Voice review (docs/review/voice-v0100/README.md). Not a test of behaviour.
//
// It needs three local servers, all on throwaway HOME folders (never ~/.chaosengine):
//   1. the plain backend on E2E_API_PORT (voice runtime not installed: the real first-run state of this computer),
//   2. a second backend on REVIEW_FAKE_PORT whose voice runtime LOOKS installed (fake_installed_backend.py beside
//      this file: it patches the install checks and reads scenario.json; no venv, no model, nothing runs), so the
//      rows, fit numbers, add-ons and the dictation default are the backend's own answers for this GPU,
//   3. Vite on E2E_WEB_PORT pointed at the first backend.
// To re-run: copy this file to e2e/tests/99-voice-review.spec.ts, start the three servers, then from e2e/
//   E2E_WEB_PORT=8971 E2E_API_PORT=8972 REVIEW_FAKE_PORT=8973 REVIEW_SCENARIO=<path to scenario.json> \
//     npx playwright test 99-voice-review --workers=1   (with a config that has no webServer block)
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page, Route } from "@playwright/test";
import { API_BASE, TOKEN, goTo, openApp, openTab } from "./helpers";

const OUT = process.env.REVIEW_OUT ?? path.resolve(__dirname, "..", "..", "docs", "review", "voice-v0100", "shots");
const FAKE = `http://127.0.0.1:${process.env.REVIEW_FAKE_PORT ?? 8973}`;
const SCENARIO = process.env.REVIEW_SCENARIO ?? "";
const DESKTOP = { width: 1280, height: 900 };
const PHONE = { width: 375, height: 812 };
const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };

test.setTimeout(120_000);

type Scenario = { extras: string[]; speech: string[]; stt: string[] };
type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
type Patch = { runtime?: (r: Json) => void; models?: (m: Json) => void; voices?: (v: Json) => void };

const ALL_EXTRAS = ["kokoro", "speech-gpu", "stt-gpu"];
const SOME_SPEECH = ["qwen3-tts-custom-1.7b-cuda", "qwen3-tts-1.7b-cuda", "qwen3-tts-design-1.7b-cuda"];
const SOME_STT = ["Qwen/Qwen3-ASR-1.7B-hf", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"];

async function shoot(page: Page, name: string, phone = false, grow = true) {
  mkdirSync(OUT, { recursive: true });
  await page.waitForTimeout(500);
  const file = path.join(OUT, `${phone ? "m" : "d"}-${name}.png`);
  const size = page.viewportSize() ?? DESKTOP;
  if (!grow) {
    await page.screenshot({ path: file, fullPage: false });
    return;
  }
  // The app fills the viewport and scrolls inside its own boxes (zoom 0.9), so the capture grows the viewport to the
  // tallest scroll box instead of using Playwright's full-page mode.
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
  await page.screenshot({ path: file, fullPage: false });
  if (height !== size.height) await page.setViewportSize(size);
}

/** The voice API answered by the fake-installed backend for this scenario. Nothing but settings may be written. */
async function world(page: Page, scenario: Scenario, patch: Patch = {}) {
  writeFileSync(SCENARIO, JSON.stringify(scenario));
  await page.request.post(`${FAKE}/api/voice/settings`, {
    headers,
    data: { sttModel: "", language: "", speakVoice: "", ttsVoice: "af_heart", callVoice: "", fallbackVoice: "af_heart" },
  });
  await page.route("**/api/voice/**", async (route: Route) => {
    const request = route.request();
    const url = request.url().replace(API_BASE, FAKE);
    const pathname = new URL(url).pathname;
    if (request.method() !== "GET" && pathname !== "/api/voice/settings") {
      return route.fulfill({ status: 503, json: { detail: "Review run: nothing is installed, downloaded or spoken." } });
    }
    const response = await route.fetch({ url });
    if (!(response.headers()["content-type"] ?? "").includes("json")) return route.fulfill({ response });
    const body = (await response.json()) as Json;
    if (pathname === "/api/voice/runtime") patch.runtime?.(body);
    if (pathname === "/api/voice/models") patch.models?.(body);
    if (pathname === "/api/voice/voices") patch.voices?.(body);
    return route.fulfill({ response, json: body });
  });
}

function libraryItems() {
  const at = (i: number) => Date.UTC(2026, 10, 19, 9, i) / 1000;
  const item = (i: number, kind: "speech" | "transcript", title: string, preview: string, extra: Json = {}) => ({
    id: `v-${i}`, kind, title, preview, createdAt: at(i), updatedAt: at(i), audio: kind === "speech" ? ".wav" : ".m4a", speakers: {},
    speakerCount: kind === "transcript" ? 2 : 0, durationS: kind === "speech" ? 14 : 1860, projectId: null, projectName: null, ...extra,
  });
  return [
    item(1, "speech", "Good evening, everyone.", "Good evening, everyone. Tonight we look at the harbour.", { voice: "qwen3:Ryan", voiceName: "Ryan", voiceKind: "preset" }),
    item(2, "speech", "Chapter one", "Chapter one. The lighthouse keeper woke before dawn.", { voice: "clone:voice-1", voiceName: "Ada", voiceKind: "clone", clonedVoice: true }),
    item(3, "speech", "Welcome back", "Welcome back to the show.", { voice: "af_heart", voiceName: "Heart", voiceKind: "builtin" }),
    item(4, "transcript", "Team call 2026-11-19", "Speaker 1: Let's start with the release. Speaker 2: The build is green.", { model: "Qwen3-ASR 1.7B", language: "en" }),
    item(5, "speech", "A calm narrator", "Once upon a time, in a quiet valley.", { voice: "design", voiceName: "A calm woman in her forties, slow", voiceKind: "design" }),
  ];
}

async function openStudio(page: Page, tab: "Speak" | "Voices" | "Transcribe") {
  await goTo(page, "Voice", "Studio");
  await page.getByRole("tab", { name: tab, exact: true }).click();
  await page.waitForTimeout(600);
}

for (const phone of [false, true]) {
  const size = phone ? PHONE : DESKTOP;
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: size });

    test("01 first run: runtime not installed (this computer's own answer)", async ({ page }) => {
      await goTo(page, "Voice", "Models");
      await expect(page.getByRole("button", { name: "Install voice runtime" })).toBeVisible();
      await shoot(page, "01-models-not-installed", phone);
      await openTab(page, "Voice", "Studio");
      await shoot(page, "01-studio-gate", phone);
      await openTab(page, "Voice", "Library");
      await shoot(page, "01-library-empty", phone);
    });

    test("02 runtime installed, no GPU add-ons, nothing downloaded", async ({ page }) => {
      await world(page, { extras: ["kokoro"], speech: [], stt: [] });
      await goTo(page, "Voice", "Models");
      await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
      await shoot(page, "02-models-base-only", phone);
      await openStudio(page, "Speak");
      await shoot(page, "02-speak-base-only", phone);
    });

    test("03 all add-ons, nothing downloaded", async ({ page }) => {
      await world(page, { extras: ALL_EXTRAS, speech: [], stt: [] });
      await goTo(page, "Voice", "Models");
      await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
      await shoot(page, "03-models-addons-empty", phone);
      await openStudio(page, "Speak");
      await shoot(page, "03-speak-first-run", phone);
      await page.getByRole("tab", { name: "Voices", exact: true }).click();
      await shoot(page, "03-voices-first-run", phone);
      await page.getByRole("tab", { name: "Transcribe", exact: true }).click();
      await shoot(page, "03-transcribe-empty", phone);
    });

    test("04 Kokoro left out and nothing downloaded: no voice at all", async ({ page }) => {
      await world(page, { extras: ["speech-gpu", "stt-gpu"], speech: [], stt: [] });
      await openStudio(page, "Speak");
      await shoot(page, "04-speak-no-voice", phone);
      await page.getByRole("tab", { name: "Voices", exact: true }).click();
      await shoot(page, "04-voices-no-voice", phone);
    });

    test("05 models downloaded: Models, row detail, dictation default", async ({ page }) => {
      await world(page, { extras: ALL_EXTRAS, speech: SOME_SPEECH, stt: SOME_STT }, {
        runtime: (r) => Object.assign(r, { workerPid: 48213, inMemory: [{ lane: "stt", name: "Qwen3-ASR 1.7B", peakGb: 3.97 }] }),
      });
      await goTo(page, "Voice", "Models");
      await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
      await shoot(page, "05-models-downloaded", phone);
      const sizes = page.locator(".voice-models__sizes").first();
      if (await sizes.isVisible().catch(() => false)) await sizes.click();
      const row = page.locator('[data-model-id="qwen3-tts-custom-1.7b-cuda"]');
      await row.locator("button[aria-expanded]").first().click().catch(() => {});
      await shoot(page, "05-models-all-sizes-detail", phone);
      // P1 (D-102 A): the Mac-only models behind one closed line; opened here.
      const macOnly = page.locator(".voice-models__mac-only summary");
      if (await macOnly.isVisible().catch(() => false)) {
        await macOnly.click();
        await macOnly.scrollIntoViewIfNeeded();
        await shoot(page, "05-models-mac-only-open", phone);
      }
    });

    test("06 Speak, the voice drawer, Voices, the clone form, Transcribe", async ({ page }) => {
      await world(page, { extras: ALL_EXTRAS, speech: SOME_SPEECH, stt: SOME_STT });
      await openStudio(page, "Speak");
      await page.getByLabel("Text to speak").fill("Good evening, everyone. Tonight we look at the harbour, the boats and the people who work there.");
      await shoot(page, "06-speak", phone);
      await page.getByRole("button", { name: "Change voice" }).click();
      await expect(page.getByRole("dialog", { name: "Choose a voice" })).toBeVisible();
      await shoot(page, "06-speak-drawer", phone, false);
      await page.keyboard.press("Escape");
      await page.getByRole("tab", { name: "Voices", exact: true }).click();
      await shoot(page, "06-voices", phone);
      const newClone = page.getByRole("button", { name: "New cloned voice" });
      if (await newClone.isVisible().catch(() => false)) {
        await newClone.click();
        await shoot(page, "06-voices-new-clone", phone);
      }
      await page.getByRole("tab", { name: "Transcribe", exact: true }).click();
      await shoot(page, "06-transcribe", phone);
    });

    test("07 Library with speech and a transcript", async ({ page }) => {
      await world(page, { extras: ALL_EXTRAS, speech: SOME_SPEECH, stt: SOME_STT });
      await page.route(/\/api\/voice\/library(\?.*)?$/, (route) => route.fulfill({ json: { items: libraryItems() } }));
      await goTo(page, "Voice", "Library");
      await expect(page.locator(".media-library-card").first()).toBeVisible();
      await shoot(page, "07-library", phone);
    });

    test("08 install running and install failed", async ({ page }) => {
      const running = {
        phase: "packages", phaseLabel: "Installing the speech packages", phaseIndex: 2, phaseTotal: 5, percent: 38, message: "", error: null,
        log: ["Collecting faster-whisper>=1.2.1,<1.3", "Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-win_amd64.whl (679.6 MB)"],
        running: true, extras: ["kokoro", "stt-gpu"],
      };
      let job: Json = running;
      await page.route("**/api/voice/runtime", async (route) => {
        const response = await route.fetch();
        const body = (await response.json()) as Json;
        body.install = job;
        return route.fulfill({ response, json: body });
      });
      await page.route("**/api/voice/install/status", (route) => route.fulfill({ json: job }));
      await goTo(page, "Voice", "Models");
      await expect(page.locator(".progress-track")).toBeVisible();
      await shoot(page, "08-install-running", phone);
      job = {
        ...running, running: false, phase: "error", phaseLabel: "error",
        error: "pip could not install the speech-to-text GPU add-on: No matching distribution found for torch==2.9.1 (from https://download.pytorch.org/whl/cu128).",
      };
      await page.reload();
      await openApp(page);
      await openTab(page, "Voice", "Models");
      await expect(page.locator(".voice-error").first()).toBeVisible();
      await shoot(page, "08-install-failed", phone);
    });

    test("09 downloads: a speech model downloading, one failed, a dictation model failed", async ({ page }) => {
      await world(page, { extras: ALL_EXTRAS, speech: ["qwen3-tts-1.7b-cuda"], stt: [] }, {
        models: (body) => {
          for (const row of body.models as Json[]) {
            if (row.key === "qwen3-tts-custom-1.7b-cuda") row.download = { state: "downloading", error: null, phase: "weights", progress: 0.42, downloadedGb: 1.9, totalGb: 4.52 };
            if (row.key === "qwen3-tts-design-1.7b-cuda") row.download = { state: "failed", error: "The download stopped: connection reset by peer. Try again.", phase: null };
          }
        },
      });
      await page.route("**/api/models/download", (route) => {
        const repo = (route.request().postDataJSON() as { repo: string }).repo;
        const download = repo.includes("granite")
          ? { repo, state: "failed", progress: 0, downloadedGb: 0, totalGb: null, error: "Not enough free disk space: 4.64 GB needed, 2.1 GB free." }
          : { repo, state: "downloading", progress: 0.37, downloadedGb: 0.58, totalGb: 1.58, error: null };
        return route.fulfill({ json: { download } });
      });
      await goTo(page, "Voice", "Models");
      await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
      await page.locator('[data-stt-model="granite-speech-5.0-2b"]').getByRole("button", { name: /Download/ }).click();
      await page.locator('[data-stt-model="qwen3-asr-0.6b"]').getByRole("button", { name: /Download/ }).click();
      const sizes = page.locator(".voice-models__sizes").first();
      if (await sizes.isVisible().catch(() => false)) await sizes.click();
      await shoot(page, "09-models-download-states", phone);
    });
  });
}

test.describe("desktop only", () => {
  test.use({ viewport: DESKTOP });

  test("10 Chat: the Voice popover", async ({ page }) => {
    await world(page, { extras: ALL_EXTRAS, speech: SOME_SPEECH, stt: SOME_STT });
    await goTo(page, "Chat", "Chat");
    await page.waitForTimeout(800);
    const voice = page.locator(".voice-popover__button").first();
    if (await voice.isVisible().catch(() => false)) {
      await voice.click();
      await expect(page.locator(".voice-popover__panel")).toBeVisible();
    }
    await shoot(page, "10-chat-voice-popover", false, false);
  });

  test("11 German: Models and Speak", async ({ page, request }) => {
    await request.patch(`${API_BASE}/api/settings`, { headers, data: { locale: "de" } });
    try {
      await world(page, { extras: ALL_EXTRAS, speech: SOME_SPEECH, stt: SOME_STT });
      await page.goto("/");
      await expect(page.locator(".loading-state")).toHaveCount(0, { timeout: 60_000 });
      const nav = page.getByRole("navigation").first();
      const header = nav.locator(".nav-group-header", { hasText: "Sprache" });
      if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
      await header.locator("xpath=following-sibling::*").locator(".nav-child-button").first().click();
      await page.waitForTimeout(1200);
      await shoot(page, "11-de-models");
      await nav.locator(".nav-child-button").nth(1).click();
      await page.waitForTimeout(1200);
      await shoot(page, "11-de-studio");
    } finally {
      await request.patch(`${API_BASE}/api/settings`, { headers, data: { locale: "system" } });
    }
  });

  test("12 a light system theme changes nothing (the app has one theme)", async ({ page }) => {
    await page.emulateMedia({ colorScheme: "light" });
    await world(page, { extras: ALL_EXTRAS, speech: SOME_SPEECH, stt: SOME_STT });
    await goTo(page, "Voice", "Models");
    await expect(page.getByRole("heading", { name: "Speech models" })).toBeVisible();
    await shoot(page, "12-light-system-theme", false, false);
  });
});
