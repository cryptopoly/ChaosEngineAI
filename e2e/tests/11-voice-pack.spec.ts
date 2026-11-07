import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";

// Voice > Models, voice pack panel and the Mandarin add-on. The pack runtime route is mocked, so nothing
// is installed: a healthy pack leads with "Installed, checked OK" and only a small Reinstall link inside
// Troubleshooting; a failed or missing pack leads with the primary button; the Mandarin button sits in the
// Chinese voice group and starts the install with the chinese extra.

const IDLE_JOB = {
  phase: "idle", phaseLabel: "idle", phaseIndex: 0, phaseTotal: 5, percent: 0,
  message: "", error: null, log: [], startedAt: 0, finishedAt: 0, running: false, extras: [],
};

async function openModels(page: Page, patch: Record<string, unknown>, extraInstalled = false) {
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, {
      lane: "mlx",
      installed: true,
      sttAvailable: true,
      ttsAvailable: true,
      ttsModelInstalled: true,
      vadAvailable: true,
      speakersAvailable: true,
      updateAvailable: false,
      installedAt: "2026-11-07T20:00:00",
      extras: [{ id: "chinese", label: "Mandarin Chinese voices", installed: extraInstalled }],
      install: IDLE_JOB,
      ttsVoices: [
        { id: "af_heart", name: "Heart", gender: "female", language: "en-US", languageLabel: "American English", available: true },
        { id: "zf_xiaobei", name: "Xiaobei", gender: "female", language: "zh", languageLabel: "Mandarin Chinese", available: extraInstalled },
      ],
      ...patch,
    });
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Voice", "Models");
  // On a fresh backend the terms can appear after the page has settled; a full run has them accepted already.
  const agree = page.getByRole("dialog", { name: "Terms of use" }).getByRole("button", { name: "I agree" });
  if (await agree.waitFor({ state: "visible", timeout: 5000 }).then(() => true, () => false)) await agree.click();
  await expect(page.getByRole("heading", { name: "Voice pack" })).toBeVisible();
}

async function checkHealthy(page: Page) {
  await expect(page.getByText(/Installed, checked OK/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Repair voice pack" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Repair \/ reinstall/ })).toHaveCount(0);
  await expect(page.locator(".voice-pack .primary-button")).toHaveCount(0);
  // The reinstall link is tucked away inside a collapsed details element.
  const details = page.locator(".voice-pack__troubleshoot");
  await expect(details).not.toHaveAttribute("open", "");
  await expect(page.getByRole("button", { name: "Reinstall" })).toBeHidden();
  await details.locator("summary").click();
  await expect(page.getByRole("button", { name: "Reinstall" })).toBeVisible();
  // The Mandarin add-on is offered in the Chinese voices group.
  const group = page.locator(".voice-group", { hasText: "Mandarin Chinese" });
  await expect(group.getByText("Mandarin voices need a small extra add-on.")).toBeVisible();
  await expect(group.getByRole("button", { name: "Add Mandarin Chinese voices" })).toBeVisible();
  await expect(page.getByText("Tick Mandarin Chinese voices")).toHaveCount(0);
  await expect(page.getByRole("checkbox", { name: "Mandarin Chinese voices" })).toHaveCount(0);
}

test("healthy pack: no primary repair button, Mandarin add-on button present", async ({ page }) => {
  await openModels(page, {});
  await checkHealthy(page);
});

test("healthy pack at 375 px", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await openModels(page, {});
  await checkHealthy(page);
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(sideways).toBeLessThanOrEqual(0);
});

test("failed install leads with the repair button", async ({ page }) => {
  await openModels(page, { install: { ...IDLE_JOB, phase: "error", error: "verify failed", log: ["boom"] } });
  await expect(page.getByRole("button", { name: "Repair voice pack" })).toBeVisible();
  await expect(page.getByText(/Installed, checked OK/)).toHaveCount(0);
  await expect(page.getByText("verify failed")).toBeVisible();
});

test("missing voice model leads with the repair button", async ({ page }) => {
  await openModels(page, { ttsModelInstalled: false, ttsAvailable: false });
  await expect(page.getByRole("button", { name: "Repair voice pack" })).toBeVisible();
});

test("not installed leads with install", async ({ page }) => {
  await openModels(page, { installed: false, sttAvailable: false, ttsAvailable: false });
  await expect(page.getByRole("button", { name: "Install voice pack" })).toBeVisible();
});

test("the Mandarin button installs the chinese add-on and the voices refresh when it finishes", async ({ page }) => {
  let installBody: unknown = null;
  let finished = false;
  await page.route("**/api/voice/install", async (route) => {
    installBody = route.request().postDataJSON();
    await route.fulfill({ json: { ...IDLE_JOB, phase: "installing", phaseLabel: "Installing speech packages", phaseIndex: 3, percent: 40, running: true, extras: ["chinese"] } });
  });
  await page.route("**/api/voice/install/status", async (route) => {
    finished = true;
    await route.fulfill({ json: { ...IDLE_JOB, phase: "done", percent: 100, extras: ["chinese"] } });
  });
  await openModels(page, {});
  // Once the job is done the runtime reports the add-on as installed.
  await page.unroute("**/api/voice/runtime");
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, {
      lane: "mlx", installed: true, sttAvailable: true, ttsAvailable: true, ttsModelInstalled: true,
      vadAvailable: true, speakersAvailable: true, updateAvailable: false, installedAt: "2026-11-07T20:00:00",
      extras: [{ id: "chinese", label: "Mandarin Chinese voices", installed: finished }],
      install: finished ? { ...IDLE_JOB, phase: "done", percent: 100, extras: ["chinese"] } : IDLE_JOB,
      ttsVoices: [
        { id: "af_heart", name: "Heart", gender: "female", language: "en-US", languageLabel: "American English", available: true },
        { id: "zf_xiaobei", name: "Xiaobei", gender: "female", language: "zh", languageLabel: "Mandarin Chinese", available: finished },
      ],
    });
    await route.fulfill({ response, json: body });
  });
  const group = page.locator(".voice-group", { hasText: "Mandarin Chinese" });
  await group.getByRole("button", { name: "Add Mandarin Chinese voices" }).click();
  expect(installBody).toEqual({ extras: ["chinese"] });
  await expect(group.getByText("Installing speech packages (3/5)")).toBeVisible();
  // Polling sees the job finish, refetches the runtime, and the add-on prompt goes away.
  await expect(group.getByRole("button", { name: "Add Mandarin Chinese voices" })).toHaveCount(0, { timeout: 10_000 });
  await expect(group.getByRole("button", { name: /Xiaobei/ })).toBeEnabled();
});
