import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { openApp, primaryNav } from "./helpers";

// Stronger borders on fields and buttons (Settings > General > Accessibility). On by default: a .text-input
// border is --border-control (#66727f); with the switch off it is the panel border (#27303a). The setting
// goes through the real backend and is put back on at the end of each test.

const STRONG = "rgb(102, 114, 127)";
const WEAK = "rgb(39, 48, 58)";

async function openSettings(page: Page) {
  await (await primaryNav(page)).getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Accessibility" })).toBeVisible();
}

const fieldBorder = (page: Page) => page.locator("#settings-language-select").evaluate((el) => getComputedStyle(el).borderTopColor);
const buttonBorder = (page: Page) =>
  page.locator("button.secondary-button").first().evaluate((el) => getComputedStyle(el).borderTopColor);

async function checkSwitch(page: Page) {
  await openApp(page);
  await openSettings(page);
  const toggle = page.getByRole("switch", { name: "Stronger borders on fields and buttons" });
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  expect(await fieldBorder(page)).toBe(STRONG);
  expect(await buttonBorder(page)).toBe(STRONG);

  const saved = (value: boolean) =>
    page.waitForResponse((r) => r.url().endsWith("/api/settings") && r.request().method() === "PATCH" && r.request().postData()?.includes(`"strongControlBorders":${value}`) === true);
  const off = saved(false);
  await toggle.click();
  await off;
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  await expect(page.locator("html")).toHaveAttribute("data-strong-borders", "off");
  expect(await fieldBorder(page)).toBe(WEAK);

  // The choice is saved and survives a reload.
  await page.reload();
  await openSettings(page);
  await expect(page.getByRole("switch", { name: "Stronger borders on fields and buttons" })).toHaveAttribute("aria-checked", "false");
  expect(await fieldBorder(page)).toBe(WEAK);

  const on = saved(true);
  await page.getByRole("switch", { name: "Stronger borders on fields and buttons" }).click();
  await on;
  await expect(page.locator("html")).not.toHaveAttribute("data-strong-borders", "off");
  expect(await fieldBorder(page)).toBe(STRONG);
}

test("field and button borders are strong by default, the switch turns them off and on", async ({ page }) => {
  await checkSwitch(page);
});

test.describe("at 375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("the same holds on a phone, with no sideways scroll", async ({ page }) => {
    await checkSwitch(page);
    expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
  });
});
