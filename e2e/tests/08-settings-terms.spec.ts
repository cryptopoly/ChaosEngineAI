import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { openApp } from "./helpers";

// Terms of use after the first run: Settings shows when they were accepted and reopens them for reading
// only; a build whose terms version is newer than the accepted one asks again. The newer-version case is
// simulated by patching the settings in /api/workspace; accepting still goes to the real backend, which
// already holds an acceptance from 01-smoke, so later specs are unaffected.

async function openSettings(page: Page) {
  await page.getByRole("navigation", { name: "Primary" }).getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Terms of use" })).toBeVisible();
}

test("Settings shows the acceptance date and reopens the terms for reading only", async ({ page }) => {
  await openApp(page);
  await openSettings(page);
  await expect(page.getByText(/^Accepted on .+\.$/)).toBeVisible();
  await page.getByRole("button", { name: "Read the terms" }).click();

  const dialog = page.getByRole("dialog", { name: "Terms of use" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("listitem")).toHaveCount(5);
  // Reading mode: no agree or quit, just Close.
  await expect(dialog.getByRole("button", { name: "I agree" })).toHaveCount(0);
  await expect(dialog.getByRole("button", { name: "Quit" })).toHaveCount(0);
  await dialog.getByRole("button", { name: "Close" }).click();
  await expect(dialog).toBeHidden();
  // Reading again does not change the acceptance.
  await expect(page.getByText(/^Accepted on .+\.$/)).toBeVisible();
});

test("an older accepted version brings the terms back until they are accepted again", async ({ page }) => {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.settings = { ...body.settings, termsAcceptedVersion: 0, termsAcceptedAt: "" };
    await route.fulfill({ response, json: body });
  });
  await page.goto("/");
  await expect(page.locator("aside.sidebar")).toBeVisible();
  const dialog = page.getByRole("dialog", { name: "Terms of use" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "I agree" })).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Quit" })).toBeVisible();
  // It is a blocking first-run dialog, not a reading view: there is no Close.
  await expect(dialog.getByRole("button", { name: "Close" })).toHaveCount(0);

  await dialog.getByRole("button", { name: "I agree" }).click();
  await expect(dialog).toBeHidden();

  await page.getByRole("navigation", { name: "Primary" }).getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Terms of use" })).toBeVisible();
  await expect(page.getByText(/^Accepted on .+\.$/)).toBeVisible();
});
