import { test, expect } from "./fixtures";

// The backend is started with a fresh HOME on every run (see playwright.config.ts),
// so every test starts as a first launch.

test("shell renders and the terms screen appears on a clean first run", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".app-shell")).toBeVisible();
  await expect(page.locator("aside.sidebar")).toBeVisible();
  const terms = page.getByRole("dialog", { name: "Terms of use" });
  await expect(terms).toBeVisible();
  await expect(terms.getByRole("button", { name: "I agree" })).toBeVisible();
  await expect(terms.getByRole("button", { name: "Quit" })).toBeVisible();
});

test("sidebar order: Dashboard, then Chat is the first group", async ({ page }) => {
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Primary" });
  await expect(nav).toBeVisible();
  // Top-level entries in DOM order: standalone tab buttons, group headers and section labels.
  const entries = nav.locator(".nav-button strong, .nav-group-label, .nav-section-label");
  await expect(entries.first()).toBeVisible();
  const names = (await entries.allTextContents()).map((s) => s.trim());
  expect(names[0]).toBe("Dashboard");
  expect(names[1]).toBe("Chat");
});

test("accepting the terms closes the dialog and it stays closed after reload", async ({ page }) => {
  await page.goto("/");
  const terms = page.getByRole("dialog", { name: "Terms of use" });
  await terms.getByRole("button", { name: "I agree" }).click();
  await expect(terms).toBeHidden();
  await page.reload();
  await expect(page.locator("aside.sidebar")).toBeVisible();
  await expect(terms).toBeHidden();
});
