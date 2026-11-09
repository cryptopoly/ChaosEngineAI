import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo } from "./helpers";

// D-44: video rows that are slow on this kind of computer carry a "Slow on this computer" label with a
// reason and, where there is one, the measured time (place and date) and a faster row that runs here.
// The label comes from the catalog row's per-platform slowOn list; the host is set by patching the
// workspace response, so the result does not depend on the machine running the test.

async function setHost(page: Page, platform: string, arch: string, gpuVramTotalGb: number | null) {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.system = { ...body.system, platform, arch, gpuVramTotalGb };
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

const row = (page: Page, id: string) => page.locator(`[data-model-id="${id}"]`);

test("slow rows are labelled on an Apple Silicon Mac", async ({ page }) => {
  await setHost(page, "Darwin", "arm64", null);
  await openVideoModels(page);

  const measured = row(page, "prince-canuma/LTX-2.3-distilled");
  await expect(measured.getByTestId("slow-row-chip")).toHaveText("Slow on this computer");
  const note = measured.getByTestId("slow-row-note");
  await expect(note).toContainText("full-size bf16 weights");
  await expect(note).toContainText("Measured: about 18 min 50 s");
  await expect(note).toContainText("768x512 clip of 33 frames (M4 Max, 64 GB, 2026-10-21)");
  await expect(note).toContainText("Faster here: LTX-2.5 · distilled q4 (ltx-2-mlx)");

  const wan = row(page, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers");
  await expect(wan.getByTestId("slow-row-chip")).toHaveText("Slow on this computer");
  await expect(wan.getByTestId("slow-row-note")).toContainText("Runs through PyTorch on the Apple GPU.");
  await expect(wan.getByTestId("slow-row-note")).toContainText("Expected to be slow; not timed yet.");
  await expect(wan.getByTestId("slow-row-note")).not.toContainText("Measured");

  // The fast MLX row is not labelled, and every row is still listed.
  await expect(row(page, "dgrauet/ltx-2.5-mlx-q4").getByTestId("slow-row-chip")).toHaveCount(0);
  await expect(row(page, "Wan-AI/Wan2.2-T2V-A14B-Diffusers")).toBeVisible();

  await measured.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, 160));
  await page.screenshot({ path: "../docs/plans/slow-row-labels/video-models-slow-labels.png" });
});

test("a machine with no NVIDIA GPU: every diffusion row is expected to be slow, no number, no faster row", async ({ page }) => {
  await setHost(page, "Windows", "x86_64", null);
  await openVideoModels(page);

  for (const id of ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "THUDM/CogVideoX-5b", "Lightricks/LTX-Video"]) {
    await expect(row(page, id).getByTestId("slow-row-chip")).toHaveText("Slow on this computer");
    const note = row(page, id).getByTestId("slow-row-note");
    await expect(note).toContainText("No NVIDIA GPU found, so it runs on the CPU.");
    await expect(note).toContainText("Expected to be slow; not timed yet.");
    await expect(note).not.toContainText("Measured");
    await expect(note).not.toContainText("Faster here");
  }
});

test("nothing new shows on an NVIDIA CUDA machine", async ({ page }) => {
  await setHost(page, "Windows", "x86_64", 24);
  await openVideoModels(page);
  await expect(row(page, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")).toBeVisible();
  await expect(page.getByTestId("slow-row-chip")).toHaveCount(0);
  await expect(page.getByTestId("slow-row-note")).toHaveCount(0);
});
