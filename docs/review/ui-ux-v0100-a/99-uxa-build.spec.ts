// D-105 build (B to E): the Dashboard's best-fit row and Start here, Chat Models' first-run list, no performance
// preview without a model. Screenshots in docs/review/ui-ux-v0100-a/build/. Copy beside uxaShared.ts into e2e/tests/
// and run with uxa.config.ts against a throwaway backend (nothing downloaded, so the first-run states show).
// Run with REVIEW_OUT=<repo>/docs/review/ui-ux-v0100-a/build so `shoot` writes there.
import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { openApp, openTab } from "./helpers";
import { DESKTOP, PHONE, shoot, top } from "./uxaShared";

async function shot(page: Page, name: string) {
  // `shoot` adds the d- / m- prefix and grows the viewport to the tallest scrolling box.
  await shoot(page, name.slice(2), name.startsWith("m-"));
}

for (const [label, viewport] of [["d", DESKTOP], ["m", PHONE]] as const) {
  test.describe(`${label} ${viewport.width}px`, () => {
    test.use({ viewport });

    test(`dashboard, chat models, benchmarks, select model (${label})`, async ({ page }) => {
      await openApp(page);
      await top(page, "Dashboard");
      await expect(page.getByText("Start here").first()).toBeVisible({ timeout: 60_000 });
      await expect(page.getByText("Best fit for this computer", { exact: true })).toBeVisible();
      await shot(page, `${label}-01-dashboard-start-here`);
      await page.getByRole("button", { name: "Other recommended models" }).click();
      await expect(page.getByTestId("chat-recommended-row").first()).toBeVisible({ timeout: 30_000 });
      await shot(page, `${label}-14-models-recommended`);
      await page.getByRole("button", { name: /Show all \d+ models and filters/ }).first().click();
      await page.waitForTimeout(800);
      await shot(page, `${label}-14-models-full-list`);
      await openTab(page, "Benchmarks", "Run");
      await expect(page.getByText("Pick a model to see its expected speed, memory and fit.").first()).toBeVisible();
      await expect(page.getByText(/tok\/s est\./)).toHaveCount(0);
      await shot(page, `${label}-07-benchmarks-run`);
      await openTab(page, "Chat", "Chat");
      await page.getByRole("button", { name: "Select Model" }).first().click();
      await expect(page.getByText("Pick a model to see its expected speed, memory and fit.").first()).toBeVisible();
      await shot(page, `${label}-12-select-model-empty`);
    });
  });
}
