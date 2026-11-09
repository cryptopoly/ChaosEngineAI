import { mkdirSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { ADA, mockVoiceApi } from "./voiceMocks";

// D-75: Kokoro is an optional add-on of the voice runtime and a Fallback voice setting replaces the fixed Kokoro
// fallback. The install form has the Kokoro tick box, ticked by default; after install it is one of the add-ons
// (unticking removes it); Voices has a fourth Fallback card whose drawer also offers "No fallback voice". Desktop and
// 375 px. Every voice route is mocked: nothing is installed, downloaded or removed.
// E2E_KOKORO_SHOTS=1 also writes the desktop screenshots to docs/plans/voice-makeover/after/ (d-kokoro-*.png).

const SHOTS = process.env.E2E_KOKORO_SHOTS === "1" ? path.resolve(__dirname, "..", "..", "docs", "plans", "voice-makeover", "after") : "";
const IDLE_JOB = {
  phase: "idle", phaseLabel: "idle", phaseIndex: 0, phaseTotal: 5, percent: 0,
  message: "", error: null, log: [], startedAt: 0, finishedAt: 0, running: false, extras: [],
};
const PARTS = (kokoroGb: number) => [
  { id: "packages", sizeGb: 1.92 }, { id: "kokoro", sizeGb: kokoroGb }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 },
];

async function openRuntime(page: Page, patch: Record<string, unknown>) {
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, {
      lane: "mlx", installed: true, sttAvailable: true, ttsAvailable: true, ttsModelInstalled: true, vadAvailable: true,
      speakersAvailable: true, updateAvailable: false, installedAt: "2026-11-09T09:00:00", install: IDLE_JOB, workerPid: null,
      extras: [
        { id: "kokoro", label: "Kokoro quick voices", installed: true },
        { id: "chinese", label: "Mandarin Chinese voices", installed: false },
      ],
      parts: PARTS(0.33), inMemory: [],
      ...patch,
    });
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Voice", "Models");
  await expect(page.getByRole("heading", { name: "Voice runtime" })).toBeVisible();
}

async function fits(page: Page) {
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(sideways).toBeLessThanOrEqual(0);
}

async function shot(page: Page, selector: string, name: string, phone: boolean) {
  if (!SHOTS || phone) return;
  mkdirSync(SHOTS, { recursive: true });
  await page.locator(selector).first().screenshot({ path: path.join(SHOTS, name) });
}

for (const phone of [false, true]) {
  test.describe(phone ? "375 px" : "desktop", () => {
    test.use({ viewport: phone ? { width: 375, height: 812 } : { width: 1280, height: 900 } });

    test("install: Kokoro is a tick box, on by default; unticked, the runtime installs without it", async ({ page }) => {
      const bodies: unknown[] = [];
      await page.route("**/api/voice/install", (route) => {
        bodies.push(route.request().postDataJSON());
        return route.fulfill({ json: IDLE_JOB });
      });
      await openRuntime(page, {
        installed: false, sttAvailable: false, ttsAvailable: false, ttsModelInstalled: false, parts: [], installedAt: null,
        extras: [
          { id: "kokoro", label: "Kokoro quick voices", installed: false },
          { id: "chinese", label: "Mandarin Chinese voices", installed: false },
        ],
      });
      const kokoro = page.getByRole("checkbox", { name: "Kokoro quick voices (about 330 MB)" });
      await expect(kokoro).toBeChecked();
      await expect(page.getByRole("checkbox", { name: /Mandarin Chinese quick voices/ })).not.toBeChecked();
      await expect(page.getByText(/plus the Kokoro quick voices when ticked below/)).toBeVisible();
      await fits(page);
      await shot(page, ".voice-runtime", "d-kokoro-runtime-install.png", phone);
      await page.getByRole("button", { name: "Install voice runtime" }).click();
      await expect.poll(() => bodies).toEqual([{ extras: ["kokoro"] }]);
      await kokoro.uncheck();
      await page.getByRole("button", { name: "Install voice runtime" }).click();
      await expect.poll(() => bodies).toEqual([{ extras: ["kokoro"] }, { extras: [] }]);
    });

    test("installed: Kokoro is an add-on; unticking removes it; without it the runtime is healthy", async ({ page }) => {
      const removed: unknown[] = [];
      await page.route("**/api/voice/extras/remove", (route) => {
        removed.push(route.request().postDataJSON());
        return route.fulfill({ json: { removed: "kokoro", packages: [], files: ["prince-canuma/Kokoro-82M"], freedGb: 0.33 } });
      });
      await openRuntime(page, {});
      const kokoro = page.getByRole("checkbox", { name: "Kokoro quick voices (about 330 MB)" });
      await expect(kokoro).toBeChecked();
      await expect(page.getByText(/Kokoro quick voices 0\.3 GB/)).toBeVisible();
      await expect(page.locator('[data-model-id="kokoro"]')).toContainText("an add-on of the voice runtime");
      await fits(page);
      await shot(page, ".voice-runtime", "d-kokoro-runtime-installed.png", phone);
      await kokoro.click();
      await expect.poll(() => removed).toEqual([{ extra: "kokoro" }]);

      await page.unrouteAll({ behavior: "ignoreErrors" });
      await openRuntime(page, {
        ttsModelInstalled: false, parts: PARTS(0),
        extras: [{ id: "kokoro", label: "Kokoro quick voices", installed: false }, { id: "chinese", label: "Mandarin Chinese voices", installed: false }],
      });
      await expect(page.getByRole("checkbox", { name: "Kokoro quick voices (about 330 MB)" })).not.toBeChecked();
      await expect(page.getByText(/Installed, checked OK/)).toBeVisible();
      await expect(page.getByRole("button", { name: "Repair voice runtime" })).toHaveCount(0);
      await expect(page.getByText(/the Kokoro quick voices are not installed/)).toBeVisible();
      const row = page.locator('[data-model-id="kokoro"]');
      await expect(row).toContainText("Not installed");
      await expect(row).toContainText("Tick it under Voice runtime > Add-ons.");
      await fits(page);
      await shot(page, ".voice-runtime", "d-kokoro-runtime-without.png", phone);
    });

    test("Voices: the Fallback card picks any allowed voice or none", async ({ page }) => {
      const seen = await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b"], clones: [ADA], ttsVoice: "qwen3:Ryan" });
      await goTo(page, "Voice", "Studio");
      await page.getByRole("tab", { name: "Voices" }).click();
      const strip = page.locator(".voice-main");
      const fallback = strip.locator('[data-use="fallback"]');
      await expect(fallback).toContainText("Fallback");
      await expect(fallback).toContainText("Heart");
      await expect(strip).toContainText("the fallback voice Heart reads instead");
      await fits(page);
      await shot(page, ".voice-main", "d-kokoro-voices-strip.png", phone);

      await fallback.getByRole("button", { name: "Change the voice for Fallback" }).click();
      const drawer = page.getByRole("dialog", { name: "Choose the voice for Fallback" });
      await expect(drawer).toBeVisible();
      await expect(drawer.getByRole("radio", { name: /Design a voice/ })).toHaveCount(0);
      await expect(drawer.getByRole("radio", { name: /^Ada\b/ }).first()).toBeVisible();
      await drawer.getByRole("button", { name: "No fallback voice" }).click();
      await expect.poll(() => seen.fallbackVoices).toEqual([""]);
      await expect(drawer).toHaveCount(0);
      await expect(fallback).toContainText("None set");
      await expect(strip).toContainText("No fallback voice is set");
      await shot(page, ".voice-main", "d-kokoro-voices-strip-none.png", phone);

      await fallback.getByRole("button", { name: "Change the voice for Fallback" }).click();
      await page.getByRole("dialog", { name: "Choose the voice for Fallback" }).getByRole("radio", { name: /^Ryan/ }).first().click();
      await expect.poll(() => seen.fallbackVoices).toEqual(["", "qwen3:Ryan"]);
      await expect(fallback).toContainText("Ryan");
      await expect(strip).toContainText("the fallback voice Ryan reads instead");
      await fits(page);
    });

    test("Voices without Kokoro: no fallback in effect and the Quick group says how to add it", async ({ page }) => {
      await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b"], ttsVoice: "qwen3:Ryan", noKokoro: true });
      await goTo(page, "Voice", "Studio");
      await page.getByRole("tab", { name: "Voices" }).click();
      await expect(page.locator('[data-use="fallback"]')).toContainText("None set");
      await expect(page.locator(".voice-main")).toContainText("No fallback voice is set");
      await expect(page.locator("#voice-group-quick")).toBeVisible();
      await expect(page.getByText(/The Kokoro quick voices are not installed/)).toBeVisible();
      await expect(page.locator('[data-voice-row="af_heart"]')).toHaveCount(0);
      await fits(page);
      await shot(page, ".voice-voices", "d-kokoro-voices-without.png", phone);
    });

    test("Chat: the Voice popover has a Fallback line", async ({ page }) => {
      await mockVoiceApi(page, { installed: ["qwen3-tts-0.6b", "qwen3-tts-custom-0.6b"], ttsVoice: "qwen3:Ryan", fallbackVoice: "", ready: true });
      await goTo(page, "Chat", "Chat");
      await page.locator(".voice-popover__button").click();
      const popover = page.locator(".voice-popover__panel");
      await expect(popover).toBeVisible();
      const line = popover.locator(".voice-popover__line", { hasText: "Fallback" });
      await expect(line).toContainText("None set");
      await expect(popover).toContainText("No fallback voice is set");
      await fits(page);
    });
  });
}
