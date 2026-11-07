import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openTab } from "./helpers";

// Busy-machine advice: /api/busy-advice is mocked for ok, slower and risky. In the Chat load dialog a
// risky answer gates the load behind "Load anyway"; slower only warns; ok shows nothing. In the Music
// Studio's estimate strip only a risky answer is shown. No model is loaded and nothing is run.

type Level = "ok" | "slower" | "risky";

const MESSAGES: Record<Level, string> = {
  ok: "",
  slower: "A video render is running, so this will go slower.",
  risky: "A video render is running; together they need 80 GB and 30 GB is free.",
};

async function mockAdvice(page: Page, level: Level, footprintGb = 4) {
  await page.route("**/api/busy-advice*", (route) =>
    route.fulfill({
      json: {
        advice: {
          level,
          message: MESSAGES[level],
          jobs: level === "ok" ? [] : [{ kind: "video", label: "Video render", source: "studio" }],
          combinedPeakGb: 80,
          totalGb: 64,
          availableGb: 30,
          footprintGb,
        },
      },
    }),
  );
}

/** One small model in the library (the workspace response is patched, nothing is on disk). */
async function mockLibrary(page: Page) {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.library = [
      {
        name: "Mock Model 4B",
        path: "/mock/model-4b",
        format: "mlx",
        sourceKind: "directory",
        quantization: "4bit",
        backend: "mlx",
        modelType: "text",
        sizeGb: 4,
        lastModified: "2026-11-01",
        actions: ["chat"],
        maxContext: 8192,
      },
    ];
    await route.fulfill({ response, json: body });
  });
}

async function openLoadDialog(page: Page) {
  await goTo(page, "Chat", "Models");
  const asked = page.waitForResponse(/\/api\/busy-advice/);
  await page.getByRole("button", { name: "Chat in a new thread" }).click();
  await expect(page.getByRole("button", { name: "Start Chat" })).toBeVisible();
  await asked; // so the "no notice" checks below do not pass before the answer arrives
  await page.waitForTimeout(250);
}

test("Chat load dialog, advice ok: no notice and the load is open", async ({ page }) => {
  await mockLibrary(page);
  await mockAdvice(page, "ok");
  await openLoadDialog(page);
  await expect(page.locator(".modal-busy-advice")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Start Chat" })).toBeEnabled();
});

test("Chat load dialog, advice slower: a notice, no confirmation needed", async ({ page }) => {
  await mockLibrary(page);
  await mockAdvice(page, "slower");
  await openLoadDialog(page);
  const notice = page.locator(".modal-busy-advice");
  await expect(notice.getByRole("status")).toContainText("Machine is busy");
  await expect(notice).toContainText(MESSAGES.slower);
  await expect(notice.getByText("Load anyway")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Start Chat" })).toBeEnabled();
});

test("Chat load dialog, advice risky: Load anyway gates Start Chat", async ({ page }) => {
  await mockLibrary(page);
  await mockAdvice(page, "risky");
  await openLoadDialog(page);
  const notice = page.locator(".modal-busy-advice");
  await expect(notice.getByRole("alert")).toContainText("Not enough memory for both");
  await expect(notice).toContainText(MESSAGES.risky);
  const start = page.getByRole("button", { name: "Start Chat" });
  const confirm = notice.getByRole("checkbox", { name: "Load anyway" });
  await expect(start).toBeDisabled();
  await confirm.check();
  await expect(start).toBeEnabled();
  await confirm.uncheck();
  await expect(start).toBeDisabled();
});

// The Music Studio only shows its estimate strip for an installed model, so the pack, the catalog and
// the estimate are patched to look installed.
async function mockInstalledMusic(page: Page) {
  await page.route("**/api/music/catalog", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    for (const family of body.families) {
      for (const variant of family.variants) Object.assign(variant, { availableLocally: true, packInstalled: true, ready: true, textEncoderReady: true });
    }
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/runtime", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    Object.assign(body.runtime ?? body, { installed: true, unavailableReason: null });
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/music/estimate", (route) =>
    route.fulfill({
      json: {
        estimate: { model: "ACE-Step/Ace-Step1.5", seconds: 60, coldStartSeconds: 20, warm: false, peakGb: 23, idleGb: 10, writesMb: 12, modelDiskGb: 10, basis: "estimated" },
        fit: null,
        totalGb: 64,
        queue: { running: null, waiting: [] },
      },
    }),
  );
}

for (const level of ["ok", "slower", "risky"] as const) {
  test(`Music Studio estimate strip, advice ${level}`, async ({ page }) => {
    await mockInstalledMusic(page);
    await mockAdvice(page, level, 23);
    const asked = page.waitForResponse(/\/api\/busy-advice/);
    await goTo(page, "Music", "Studio");
    await asked;
    await page.waitForTimeout(250);
    const strip = page.locator(".run-estimate");
    await expect(strip).toContainText("Peak 23 GB of 64 GB");
    await expect(strip.getByRole("button", { name: "Make track" })).toBeVisible();
    if (level === "risky") {
      await expect(strip.locator(".run-estimate__busy")).toContainText(MESSAGES.risky);
    } else {
      // A Studio job waits its turn behind other heavy jobs, so only risky is worth a note there.
      await expect(strip.locator(".run-estimate__busy")).toHaveCount(0);
    }
    // Leaving and coming back keeps working (the advice is read again, not cached as a block).
    await openTab(page, "Music", "Library");
    await openTab(page, "Music", "Studio");
    await expect(page.locator(".run-estimate")).toContainText("Peak 23 GB of 64 GB");
  });
}
