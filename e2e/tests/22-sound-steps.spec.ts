import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";
import { mkdirSync } from "node:fs";
import path from "node:path";

// Sounds > Options > Steps (D-74): presets and a free number per model, the run strip follows the steps and says
// when the time is scaled from the measured run, and the chosen steps travel with the request.
// The catalog and runtime are patched so both models count as installed; the estimate is the real backend's
// (a read-only call), the generate call is answered by the test, so no model is loaded and nothing is downloaded.

const FULL = "stabilityai/stable-audio-open-1.0";

async function bothInstalled(page: Page) {
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      if (family.kind !== "sound") continue;
      for (const variant of family.variants) Object.assign(variant, { availableLocally: true, weightsSource: "checkpoints", packInstalled: true, textEncoderReady: true });
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

async function openSoundsOptions(page: Page) {
  await bothInstalled(page);
  await goTo(page, "Music and Audio", "Studio");
  await page.getByRole("group", { name: "Music or sound effects" }).getByRole("button", { name: "Sounds", exact: true }).click();
  await page.getByRole("combobox", { name: "Sound model" }).selectOption(FULL);
  await page.locator("summary.disclosure__summary", { hasText: "Options" }).click();
  return page.getByRole("group", { name: "Steps", exact: true });
}

const strip = (page: Page) => page.locator(".run-estimate");

test("Sounds Options: Steps changes the estimate and is sent with the request", async ({ page }) => {
  const estimateBodies: Array<Record<string, unknown>> = [];
  const generateBodies: Array<Record<string, unknown>> = [];
  page.on("request", (request) => {
    if (request.method() !== "POST") return;
    if (request.url().endsWith("/api/music/estimate")) estimateBodies.push(request.postDataJSON());
  });
  await page.route("**/api/music/sound/generate", async (route) => {
    generateBodies.push(route.request().postDataJSON());
    await route.fulfill({ status: 500, json: { detail: "stopped by the test" } });
  });

  const steps = await openSoundsOptions(page);
  await page.getByRole("textbox").first().fill("a wooden door creaking");
  await expect(steps).toBeVisible();
  // Presets for the full model, with the model author's value as the default.
  for (const preset of ["25", "50", "100"]) await expect(steps.getByRole("button", { name: preset, exact: true })).toBeVisible();
  await expect(steps.getByRole("button", { name: "100", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(steps.getByRole("spinbutton", { name: "Number of steps" })).toHaveValue("100");
  await expect(steps).toContainText("Fewer steps are faster but can sound worse. 100 is the model author's setting.");
  await expect(strip(page)).toContainText("Measured");
  await expect(strip(page)).toContainText(/3 min|About 20\d s/);

  // Balanced: about half the time, labelled as scaled from the measured run.
  await steps.getByRole("button", { name: "50", exact: true }).click();
  await expect(strip(page)).toContainText("Estimated from the measured run");
  await expect(strip(page)).not.toContainText("Measured");
  await expect(strip(page)).toContainText(/1 min|About 1\d\d s/);
  expect(estimateBodies.at(-1)).toMatchObject({ modelId: FULL, steps: 50 });

  // A free number inside the range.
  const number = steps.getByRole("spinbutton", { name: "Number of steps" });
  await number.fill("30");
  await expect(steps.getByRole("button", { name: "25", exact: true })).toHaveAttribute("aria-pressed", "false");
  await expect.poll(() => estimateBodies.at(-1)?.steps).toBe(30);

  // Outside the range: the form says so and no request is made for it.
  const before = estimateBodies.length;
  await number.fill("9");
  await expect(page.getByText("Steps must be a whole number from 10 to 150.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Make sound" })).toBeDisabled();
  await page.waitForTimeout(500);
  expect(estimateBodies.length).toBe(before);

  // Back to a valid number, and the chosen steps go with the request.
  await number.fill("50");
  await expect(page.getByRole("button", { name: "Make sound" })).toBeEnabled();
  await page.getByRole("button", { name: "Make sound" }).click();
  await expect.poll(() => generateBodies.length).toBe(1);
  expect(generateBodies[0]).toMatchObject({ modelId: FULL, caption: "a wooden door creaking", steps: 50 });

  // The default is not sent: the run follows the model.
  await steps.getByRole("button", { name: "100", exact: true }).click();
  await expect(strip(page)).toContainText("Measured");
  await page.getByRole("button", { name: "Make sound" }).click();
  await expect.poll(() => generateBodies.length).toBe(2);
  expect(generateBodies[1]).not.toHaveProperty("steps");
});

test("Sounds Options: Steps fits on desktop and at 375 px, with a screenshot", async ({ page }) => {
  const shots = path.resolve(__dirname, "..", "..", "docs", "plans", "sound-steps");
  mkdirSync(shots, { recursive: true });
  await page.setViewportSize({ width: 1280, height: 900 });
  const steps = await openSoundsOptions(page);
  await steps.getByRole("button", { name: "50", exact: true }).click();
  await expect(strip(page)).toContainText("Estimated from the measured run");
  await steps.scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(shots, "sounds-options-steps-desktop.png") });

  await page.setViewportSize({ width: 375, height: 812 });
  await expect(steps).toBeVisible();
  await steps.scrollIntoViewIfNeeded();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow, "no sideways scroll at 375 px").toBeLessThanOrEqual(0);
  const box = await steps.boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(376);
  await expect(steps.getByRole("spinbutton", { name: "Number of steps" })).toBeInViewport();
  await expect(strip(page)).toContainText("Estimated from the measured run");
});
