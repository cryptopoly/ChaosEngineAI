import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";

// D-44: video rows that are slow on Apple Silicon carry a "Slow on this Mac" label (measured time with
// where and when, or "Expected to be slow" when untimed) and point at the faster LTX-2.5 row. The label
// comes from the catalog row's slowOn field and shows only on a Mac; the host is set by patching the
// workspace response, so the result does not depend on the machine running the test.

async function setHost(page: Page, platform: string, arch: string) {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.system = { ...body.system, platform, arch };
    await route.fulfill({ response, json: body });
  });
}

async function openVideoModels(page: Page) {
  await goTo(page, "Video", "Models");
  await expect(page.getByRole("heading", { name: "Video Models" }).first()).toBeVisible();
  // Every catalog row, installed or not.
  const all = page.getByRole("button", { name: /^All\b/ }).first();
  if (await all.isVisible().catch(() => false)) await all.click();
}

test("slow rows are labelled on an Apple Silicon Mac", async ({ page }) => {
  await setHost(page, "Darwin", "arm64");
  await openVideoModels(page);

  const measured = page.locator('[data-model-id="prince-canuma/LTX-2.3-distilled"]');
  await expect(measured.getByTestId("slow-row-chip")).toHaveText("Slow on this Mac");
  const note = measured.getByTestId("slow-row-note");
  await expect(note).toContainText("Measured on M4 Max, 64 GB, 2026-10-21");
  await expect(note).toContainText("18 min 50 s");
  await expect(note).toContainText("768x512");
  await expect(note).toContainText("33 frames");
  await expect(note).toContainText("Faster here: LTX-2.5 · distilled q4 (ltx-2-mlx)");

  const wan = page.locator('[data-model-id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]');
  await expect(wan.getByTestId("slow-row-chip")).toHaveText("Slow on this Mac");
  await expect(wan.getByTestId("slow-row-note")).toContainText("Expected to be slow on this Mac");
  await expect(wan.getByTestId("slow-row-note")).not.toContainText("Measured");

  // The fast MLX row is not labelled, and every row is still listed.
  await expect(page.locator('[data-model-id="dgrauet/ltx-2.5-mlx-q4"]').getByTestId("slow-row-chip")).toHaveCount(0);
  await expect(page.locator('[data-model-id="Wan-AI/Wan2.2-T2V-A14B-Diffusers"]')).toBeVisible();

  await measured.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, 160));
  await page.screenshot({ path: "../docs/plans/slow-row-labels/video-models-slow-labels.png" });
});

test("nothing new shows on a CUDA machine", async ({ page }) => {
  await setHost(page, "Windows", "x86_64");
  await openVideoModels(page);
  await expect(page.locator('[data-model-id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"]')).toBeVisible();
  await expect(page.getByTestId("slow-row-chip")).toHaveCount(0);
  await expect(page.getByTestId("slow-row-note")).toHaveCount(0);
});
