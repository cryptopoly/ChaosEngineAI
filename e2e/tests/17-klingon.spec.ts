import { mkdirSync, statSync } from "node:fs";
import path from "node:path";
import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { openApp, SHOTS_DIR } from "./helpers";

// D-69: the Klingon locale (tlh). Picks it in Settings, checks the community note and that the
// shell re-renders in Klingon, then saves a screenshot per widest surface (Settings general,
// Accelerators which hosts the Setup install strings, Diagnostics, Dashboard, Chat, Models) at
// desktop and at 375 px, where the page must not scroll sideways. Finishes by switching back to
// English so a later spec sharing the backend is not affected.

async function sideways(page: Page): Promise<number> {
  return page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
}

async function look(page: Page, name: string) {
  mkdirSync(SHOTS_DIR, { recursive: true });
  const file = path.join(SHOTS_DIR, `klingon-${name}.png`);
  await page.waitForTimeout(400);
  await page.screenshot({ path: file, fullPage: true });
  expect(statSync(file).size, `${name} screenshot looks blank`).toBeGreaterThan(8_000);
  expect.soft(await sideways(page), `${name}: page scrolls sideways`).toBeLessThanOrEqual(0);
}

async function openSettings(page: Page) {
  // The Settings entry is the last standalone nav button; find it by its tab id class-free: it is
  // the only nav button whose label is "Settings" in English or "SeHlaw" in Klingon.
  const nav = page.getByRole("navigation", { name: /Primary|He Hal/ });
  await nav.locator(".nav-button", { hasText: /^(Settings|SeHlaw)/ }).click();
  await expect(page.locator(".settings-subtab-bar")).toBeVisible();
}

async function pickLocale(page: Page, code: string) {
  await page.locator("#settings-language-select").selectOption(code);
}

async function settingsSection(page: Page, index: number) {
  const tab = page.locator(".settings-subtab-bar [role=tab]").nth(index);
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

async function surfaces(page: Page, suffix: string) {
  await openSettings(page);
  await settingsSection(page, 0);
  await look(page, `settings-general${suffix}`);
  await settingsSection(page, 5); // Diagnostics
  await look(page, `settings-diagnostics${suffix}`);
  await settingsSection(page, 6); // Accelerators: hosts the Setup install strings
  await look(page, `settings-accelerators${suffix}`);
  const nav = page.getByRole("navigation", { name: /Primary|He Hal/ });
  await nav.locator(".nav-button").first().click(); // Dashboard
  await look(page, `dashboard${suffix}`);
  const groups = nav.locator(".nav-group-header");
  for (const [groupIndex, name] of [[0, "chat"], [1, "models"]] as const) {
    const header = groups.nth(groupIndex);
    if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
    await nav.locator(".nav-child-button").first().click();
    await look(page, `${name}${suffix}`);
  }
}

test("Klingon: picking tlhIngan Hol shows the community note and flips the shell", async ({ page }) => {
  await openApp(page);
  await openSettings(page);
  await expect(page.getByTestId("language-community-note")).toBeHidden();
  await pickLocale(page, "tlh");
  await expect(page.getByTestId("language-community-note")).toBeVisible();
  await expect(page.getByTestId("language-community-note")).toContainText("mughwI' ghom mugh");
  // The sidebar re-renders live: Settings becomes SeHlaw, Dashboard becomes wIy.
  const nav = page.getByRole("navigation", { name: /Primary|He Hal/ });
  await expect(nav.locator(".nav-button", { hasText: /^SeHlaw/ })).toBeVisible();
  await expect(nav.locator(".nav-button").first()).toContainText("wIy");
  await expect(page.locator("html")).toHaveAttribute("lang", "tlh");
  await surfaces(page, "");
});

test.describe("Klingon at 375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("the widest surfaces fit and do not scroll sideways", async ({ page }) => {
    await openApp(page);
    // The locale persisted through the previous test; check it survived the reload.
    await expect(page.locator("html")).toHaveAttribute("lang", "tlh");
    await surfaces(page, "-375");
  });
});

test("Klingon: switching back to English restores the shell", async ({ page }) => {
  await openApp(page);
  await openSettings(page);
  await pickLocale(page, "en");
  await expect(page.getByTestId("language-community-note")).toBeHidden();
  const nav = page.getByRole("navigation", { name: "Primary" });
  await expect(nav.locator(".nav-button", { hasText: /^Settings/ })).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
});
