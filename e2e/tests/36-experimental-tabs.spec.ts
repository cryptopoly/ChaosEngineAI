import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { openApp, primaryNav } from "./helpers";

// Settings > General > Appearance > "Show experimental tabs" (off by default, remembered with the sidebar
// preferences). Fine-Tuning and Plugins are experimental (overnight review, D-87): out of the sidebar until the
// switch is on. Prompts, Conversion and Benchmarks always show.

async function toolsChildren(page: Page): Promise<string[]> {
  const nav = await primaryNav(page);
  const header = nav.locator(".nav-group-header", { hasText: "Tools" });
  if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
  const group = nav.getByRole("group", { name: "Tools" });
  await expect(group.locator(".nav-child-button").first()).toBeVisible();
  return (await group.locator(".nav-child-label").allInnerTexts()).map((text) => text.trim());
}

async function setSwitch(page: Page, on: boolean) {
  await (await primaryNav(page)).getByRole("button", { name: "Settings", exact: true }).click();
  const box = page.getByTestId("show-experimental-tabs");
  await expect(box).toBeVisible();
  if ((await box.isChecked()) !== on) await box.click();
  await expect(box).toBeChecked({ checked: on });
}

test("Fine-Tuning and Plugins show only with the switch on, and the choice is remembered", async ({ page }) => {
  await openApp(page);
  let tools = await toolsChildren(page);
  expect(tools).toContain("Prompts");
  expect(tools).not.toContain("Fine-Tuning");
  expect(tools).not.toContain("Plugins");
  await expect((await primaryNav(page)).locator(".nav-group-header", { hasText: "Benchmarks" })).toHaveCount(1);

  await setSwitch(page, true);
  tools = await toolsChildren(page);
  expect(tools).toEqual(expect.arrayContaining(["Prompts", "Fine-Tuning", "Plugins"]));

  await page.reload();
  await expect(page.locator("aside.sidebar")).toBeVisible();
  tools = await toolsChildren(page);
  expect(tools).toContain("Plugins");

  // The page opens while the switch is on; switched off again, the entries go.
  await (await primaryNav(page)).locator(".nav-child-button", { hasText: /^Plugins$/ }).click();
  await expect(page.locator(".plugins-page")).toBeVisible();
  await setSwitch(page, false);
  tools = await toolsChildren(page);
  expect(tools).not.toContain("Fine-Tuning");
  expect(tools).not.toContain("Plugins");
});

test("a remembered hidden tab is not reopened: the Tools group opens on Conversion", async ({ page }) => {
  // Tabs-style sidebar remembering Plugins as the Tools group's last page, with the switch off.
  await page.addInitScript(() => {
    window.localStorage.setItem(
      "chaosengine.sidebar.prefs.v1",
      JSON.stringify({ mode: "tabs", lastChildByGroup: { tools: "plugins" }, showExperimentalTabs: false }),
    );
  });
  await openApp(page);
  await (await primaryNav(page)).locator(".nav-button", { hasText: "Tools" }).click();
  const bar = page.locator(".subtab-bar");
  await expect(bar.getByRole("tab", { name: "Conversion" })).toHaveAttribute("aria-selected", "true");
  await expect(bar.getByRole("tab", { name: "Plugins" })).toHaveCount(0);
  await expect(bar.getByRole("tab", { name: "Fine-Tuning" })).toHaveCount(0);
  await expect(page.locator(".plugins-page")).toHaveCount(0);
});
