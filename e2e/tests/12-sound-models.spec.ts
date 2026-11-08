import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openTab } from "./helpers";

// Stable Audio Open 1.0 next to Small: its own licence gate on the Models page, and in the Sounds studio the
// length slider follows the selected model (11 s Small, 47 s 1.0) with a model picker once both are installed.
// The catalog and runtime responses are patched so both models count as installed; nothing is downloaded or run.

const SMALL = "stabilityai/stable-audio-open-small";
const FULL = "stabilityai/stable-audio-open-1.0";

async function bothInstalled(page: Page) {
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      if (family.kind !== "sound") continue;
      for (const variant of family.variants) {
        Object.assign(variant, { availableLocally: true, weightsSource: "checkpoints", packInstalled: true, textEncoderReady: true });
      }
    }
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.runtime.sound = { ...body.runtime.sound, installed: true, unavailableReason: null, lane: "mps" };
    await route.fulfill({ response, json: body });
  });
}

test("Music Models lists Stable Audio Open 1.0 with its own licence gate", async ({ page }) => {
  await goTo(page, "Music and Audio", "Models");
  const row = page.locator(`[data-model-id="${FULL}"]`);
  await expect(row).toBeVisible();
  await expect(row).toContainText("Stable Audio Open 1.0");
  await expect(row.getByText(/needs licence/i).first()).toBeVisible();

  await row.getByRole("button", { name: "Accept licence" }).click();
  const dialog = page.getByRole("dialog", { name: /Stable Audio Open 1\.0/ });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Stability AI Community Licence");
  await expect(dialog).toContainText("US$1 million a year");
  await expect(dialog).toContainText("Powered by Stability AI");
  await expect(dialog).toContainText("The weights are never bundled with ChaosEngineAI.");
  await expect(dialog.getByRole("button", { name: "Read the full licence" })).toBeVisible();
  await dialog.getByRole("button", { name: "Not now" }).click();
  await expect(dialog).toBeHidden();
  // Small's row is untouched by this.
  await expect(page.locator(`[data-model-id="${SMALL}"]`).getByRole("button", { name: "Accept licence" })).toBeVisible();
});

test("Sounds studio: the length slider max follows the selected sound model", async ({ page }) => {
  await bothInstalled(page);
  await goTo(page, "Music and Audio", "Studio");
  await page.getByRole("group", { name: "Music or sound effects" }).getByRole("button", { name: "Sounds", exact: true }).click();

  const picker = page.getByRole("combobox", { name: "Sound model" });
  const slider = page.getByRole("slider", { name: "Length in seconds" });
  await expect(picker).toBeVisible();
  await expect(picker.locator("option")).toHaveText(["Stable Audio Open Small", "Stable Audio Open 1.0"]);
  await expect(picker).toHaveValue(SMALL);
  await expect(slider).toHaveAttribute("max", "11");

  const estimate = page.waitForRequest((request) => {
    if (!request.url().endsWith("/api/music/estimate")) return false;
    const body = request.postDataJSON();
    return body.modelId === FULL && body.seconds === 40;
  });
  await picker.selectOption(FULL);
  await expect(slider).toHaveAttribute("max", "47");
  await slider.fill("40");
  await expect(slider).toHaveValue("40");
  await estimate;

  // Back to Small: the 40 s length is brought inside its 11 s range.
  await picker.selectOption(SMALL);
  await expect(slider).toHaveAttribute("max", "11");
  await expect(slider).toHaveValue("11");
});

test("Sounds studio: the full model alone gets no picker and its own limit", async ({ page }) => {
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      if (family.kind !== "sound") continue;
      for (const variant of family.variants) {
        const on = variant.id === FULL;
        Object.assign(variant, { availableLocally: on, weightsSource: on ? "checkpoints" : null, packInstalled: on, textEncoderReady: on });
      }
    }
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.runtime.sound = { ...body.runtime.sound, installed: true, unavailableReason: null, lane: "mps" };
    await route.fulfill({ response, json: body });
  });
  await goTo(page, "Music and Audio", "Studio");
  await page.getByRole("group", { name: "Music or sound effects" }).getByRole("button", { name: "Sounds", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Sound model" })).toHaveCount(0);
  await expect(page.getByText("Stable Audio Open 1.0").first()).toBeVisible();
  await expect(page.getByRole("slider", { name: "Length in seconds" })).toHaveAttribute("max", "47");
  await openTab(page, "Music and Audio", "Models");
  await expect(page.locator(`[data-model-id="${FULL}"]`)).toBeVisible();
});

test("Sounds studio: a run on the CPU shows a slow-run warning and Cancel stays available", async ({ page }) => {
  await bothInstalled(page);
  let release: () => void = () => {};
  const held = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/api/music/sound/generate", async (route) => {
    await held;
    await route.fulfill({ status: 499, json: { detail: "cancelled" } });
  });
  let warning: string | null = "cpu";
  await page.route("**/api/music/progress", (route) =>
    route.fulfill({
      json: {
        progress: {
          kind: "music", active: true, phase: "diffusing", message: "", step: 1, totalSteps: 8, startedAt: Date.now() / 1000,
          updatedAt: Date.now() / 1000, elapsedSeconds: 5, runLabel: null, warning,
        },
      },
    }),
  );
  await goTo(page, "Music and Audio", "Studio");
  await page.getByRole("group", { name: "Music or sound effects" }).getByRole("button", { name: "Sounds", exact: true }).click();
  await page.getByRole("textbox").first().fill("rain on a tin roof");
  await page.getByRole("button", { name: "Make sound" }).click();
  await expect(page.getByText("running on the CPU, not a GPU")).toBeVisible();
  await expect(page.getByRole("button", { name: "Cancel" })).toBeEnabled();
  warning = "cpu-fallback";
  await expect(page.getByText("restarted on the CPU")).toBeVisible();
  release();
});
