import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { mockVoiceApi } from "./voiceMocks";

// Voice > Models, the voice runtime panel (called the voice pack before D-73). The runtime route is mocked, so
// nothing is installed: a healthy runtime leads with "Installed, checked OK", its parts and sizes, what is in memory
// with Unload, and the add-ons as tick boxes (tick installs, untick removes); Troubleshooting holds Reinstall and
// Remove voice runtime; a failed or missing runtime leads with the primary button. Dictation models have Delete in a
// row menu. The Mandarin card on Voices starts the same install.

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
      workerPid: 4242,
      parts: [{ id: "packages", sizeGb: 1.92 }, { id: "kokoro", sizeGb: 0.33 }, { id: "turns", sizeGb: 0.002 }, { id: "speakers", sizeGb: 0.036 }],
      inMemory: [{ lane: "tts", name: "Kokoro", peakGb: 0.87 }],
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
  await expect(page.getByRole("heading", { name: "Voice runtime" })).toBeVisible();
}

async function checkHealthy(page: Page) {
  await expect(page.getByText(/Installed, checked OK/)).toBeVisible();
  await expect(page.getByText("Voice pack", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Repair voice runtime" })).toHaveCount(0);
  await expect(page.locator(".voice-pack .primary-button")).toHaveCount(0);
  // What the runtime is, as parts with sizes; memory with Unload; the add-on as a tick box.
  await expect(page.getByText(/2\.3 GB on disk: speech packages 1\.9 GB, Kokoro quick voices 0\.3 GB/)).toBeVisible();
  await expect(page.locator(".voice-runtime__memory")).toContainText("In memory now: Kokoro, 0.9 GB");
  await expect(page.getByRole("button", { name: "Unload", exact: true })).toBeVisible();
  await expect(page.getByRole("checkbox", { name: /Mandarin Chinese quick voices/ })).not.toBeChecked();
  // Reinstall and Remove are tucked away inside a collapsed details element.
  const details = page.locator(".voice-pack__troubleshoot");
  await expect(details).not.toHaveAttribute("open", "");
  await expect(page.getByRole("button", { name: "Reinstall" })).toBeHidden();
  await details.locator("summary").click();
  await expect(page.getByRole("button", { name: "Reinstall" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Remove voice runtime" })).toBeVisible();
  // The Quick voices and Dictation and memory panels are gone; Kokoro is the first speech-model row.
  await expect(page.getByRole("heading", { name: "Quick voices" })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Dictation and memory" })).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Dictation models" })).toBeVisible();
}

test("healthy runtime: no primary repair button, parts, memory, add-on tick box, Remove under Troubleshooting", async ({ page }) => {
  await openModels(page, {});
  await checkHealthy(page);
});

test("healthy runtime at 375 px", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await openModels(page, {});
  await checkHealthy(page);
  const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(sideways).toBeLessThanOrEqual(0);
});

test("Open Studio on Models opens Voice Studio on Speak, and the Studio header links back to Models and Library", async ({ page }) => {
  await openModels(page, {});
  await page.getByRole("button", { name: "Open Studio" }).click();
  await expect(page.getByRole("tab", { name: "Speak" })).toHaveAttribute("aria-selected", "true");
  const header = page.locator(".studio-shell__header");
  await header.getByRole("button", { name: "Library" }).click();
  await expect(page.locator("section").getByRole("heading", { name: "Voice Library" })).toBeVisible();
  await page.getByRole("button", { name: "Open Studio" }).click();
  await page.locator(".studio-shell__header").getByRole("button", { name: "Models" }).click();
  await expect(page.getByRole("heading", { name: "Voice runtime" })).toBeVisible();
});

test("failed install leads with the repair button", async ({ page }) => {
  await openModels(page, { install: { ...IDLE_JOB, phase: "error", error: "verify failed", log: ["boom"] } });
  await expect(page.getByRole("button", { name: "Repair voice runtime" })).toBeVisible();
  await expect(page.getByText(/Installed, checked OK/)).toHaveCount(0);
  await expect(page.getByText("verify failed")).toBeVisible();
});

test("a runtime without the Kokoro quick voices is healthy: Kokoro is an optional add-on (D-75)", async ({ page }) => {
  await openModels(page, { ttsModelInstalled: false, ttsAvailable: false });
  await expect(page.getByText(/Installed, checked OK/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Repair voice runtime" })).toHaveCount(0);
});

test("not installed leads with install", async ({ page }) => {
  await openModels(page, { installed: false, sttAvailable: false, ttsAvailable: false });
  await expect(page.getByRole("button", { name: "Install voice runtime" })).toBeVisible();
});

test("settings sit next to what they change: dictation language with the dictation models, idle minutes with the runtime", async ({ page }) => {
  const stops: string[] = [];
  await page.route("**/api/voice/worker/stop", (route) => {
    stops.push(route.request().method());
    return route.fulfill({ json: { stopped: true } });
  });
  await openModels(page, {});
  await expect(page.locator(".voice-dictation").getByRole("combobox")).toHaveValue("");
  await expect(page.getByLabel("Free voice memory after (minutes idle, 0 = never)")).toBeVisible();
  await expect(page.getByText("Read every chat reply aloud when it finishes")).toHaveCount(0);
  await page.getByRole("button", { name: "Unload", exact: true }).click();
  await expect.poll(() => stops).toEqual(["POST"]);
  await page.locator('[data-model-id="kokoro"]').getByRole("button", { name: "Browse voices" }).click();
  await expect(page.getByRole("tab", { name: "Voices" })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".voice-main")).toBeVisible();
});

test("ticking the Mandarin add-on installs it; unticking removes it", async ({ page }) => {
  let installBody: unknown = null;
  const removed: unknown[] = [];
  await page.route("**/api/voice/install", async (route) => {
    installBody = route.request().postDataJSON();
    await route.fulfill({ json: { ...IDLE_JOB, phase: "installing", phaseLabel: "Installing speech packages", phaseIndex: 3, percent: 40, running: true, extras: ["chinese"] } });
  });
  await page.route("**/api/voice/install/status", (route) =>
    route.fulfill({ json: { ...IDLE_JOB, phase: "installing", phaseLabel: "Installing speech packages", phaseIndex: 3, percent: 40, running: true, extras: ["chinese"] } }),
  );
  await openModels(page, {});
  // The box follows the backend: it shows ticked once the install job carries the add-on.
  await page.getByRole("checkbox", { name: /Mandarin Chinese quick voices/ }).click();
  await expect.poll(() => installBody).toEqual({ extras: ["chinese"] });
  await expect(page.getByRole("checkbox", { name: /Mandarin Chinese quick voices/ })).toBeChecked();
  await expect(page.getByText("Installing speech packages (3/5)")).toBeVisible();

  await page.unrouteAll({ behavior: "ignoreErrors" });
  await page.route("**/api/voice/extras/remove", (route) => {
    removed.push(route.request().postDataJSON());
    return route.fulfill({ json: { removed: "chinese", packages: ["jieba"] } });
  });
  await openModels(page, {}, true);
  await expect(page.getByRole("checkbox", { name: /Mandarin Chinese quick voices/ })).toBeChecked();
  await page.getByRole("checkbox", { name: /Mandarin Chinese quick voices/ }).click();
  await expect.poll(() => removed).toEqual([{ extra: "chinese" }]);
});

test("Remove voice runtime asks first, keeps the downloaded models unless asked, and says what stops working", async ({ page }) => {
  const bodies: unknown[] = [];
  await page.route("**/api/voice/uninstall", (route) => {
    bodies.push(route.request().postDataJSON());
    return route.fulfill({ json: { removed: true, freedGb: 2.3, removedModels: [] } });
  });
  await openModels(page, {});
  await page.locator(".voice-pack__troubleshoot summary").click();
  let asked = "";
  page.once("dialog", (dialog) => {
    asked = dialog.message();
    void dialog.dismiss();
  });
  await page.getByRole("button", { name: "Remove voice runtime" }).click();
  await expect.poll(() => asked).toContain("Dictation, read-aloud, voice calls and Voice Studio stop working");
  expect(asked).toContain("Downloaded dictation and speech models, cloned voices and the Library stay.");
  expect(bodies).toEqual([]);
  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Remove voice runtime" }).click();
  await expect.poll(() => bodies).toEqual([{ alsoRuntimeModels: false }]);
  await page.getByRole("checkbox", { name: /Also delete the Kokoro quick voices/ }).check();
  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Remove voice runtime" }).click();
  await expect.poll(() => bodies).toEqual([{ alsoRuntimeModels: false }, { alsoRuntimeModels: true }]);
});

test("a downloaded dictation model has Delete in its row menu, asked first", async ({ page }) => {
  const deleted: unknown[] = [];
  await page.route("**/api/voice/stt/delete", (route) => {
    deleted.push(route.request().postDataJSON());
    return route.fulfill({ json: { repo: "x", result: { repo: "x", state: "deleted" } } });
  });
  await openModels(page, {});
  const card = page.locator(".voice-dictation .voice-model-card").first();
  const name = (await card.locator(".voice-model-card__name").innerText()).trim();
  // The runtime route reports what is downloaded; force the first card to look downloaded for the menu.
  await page.unrouteAll({ behavior: "ignoreErrors" });
  await page.route("**/api/voice/stt/delete", (route) => {
    deleted.push(route.request().postDataJSON());
    return route.fulfill({ json: { repo: "x", result: { repo: "x", state: "deleted" } } });
  });
  await page.route("**/api/voice/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    const runtime = body.runtime ?? body;
    Object.assign(runtime, { installed: true, sttAvailable: true, ttsAvailable: true, ttsModelInstalled: true, vadAvailable: true, updateAvailable: false, install: IDLE_JOB });
    runtime.sttModels = runtime.sttModels.map((m: Record<string, unknown>, i: number) => ({ ...m, installed: i === 0 }));
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Voice", "Models");
  const first = page.locator(".voice-dictation .voice-model-card").first();
  await first.getByLabel(`More actions for ${name}`).click();
  page.once("dialog", (dialog) => void dialog.dismiss());
  await first.getByRole("menuitem", { name: /^Delete model/ }).click();
  expect(deleted).toEqual([]);
  await first.getByLabel(`More actions for ${name}`).click();
  page.once("dialog", (dialog) => void dialog.accept());
  await first.getByRole("menuitem", { name: /^Delete model/ }).click();
  await expect.poll(() => deleted.length).toBe(1);
  expect(Object.keys(deleted[0] as object)).toEqual(["key"]);
});

test("the Mandarin card on Voice Studio > Voices starts the same install", async ({ page }) => {
  let installBody: unknown = null;
  await mockVoiceApi(page);
  await page.route("**/api/voice/install", async (route) => {
    installBody = route.request().postDataJSON();
    await route.fulfill({ json: { ...IDLE_JOB, phase: "installing", phaseLabel: "Installing speech packages", phaseIndex: 3, percent: 40, running: true, extras: ["chinese"] } });
  });
  await page.route("**/api/voice/install/status", (route) =>
    route.fulfill({ json: { ...IDLE_JOB, phase: "installing", phaseLabel: "Installing speech packages", phaseIndex: 3, percent: 40, running: true, extras: ["chinese"] } }),
  );
  await openModels(page, {});
  await page.getByRole("button", { name: "Open Studio" }).click();
  await page.getByRole("tab", { name: "Voices" }).click();
  const cards = page.locator(".voice-voices");
  await expect(cards.getByText("Add Mandarin voices", { exact: true })).toBeVisible();
  await cards.getByRole("button", { name: "Add Mandarin Chinese voices" }).click();
  expect(installBody).toEqual({ extras: ["chinese"] });
  await expect(cards.getByText("Installing speech packages (3/5)")).toBeVisible();
});
