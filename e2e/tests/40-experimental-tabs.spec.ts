import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { tabs } from "../../src/constants/tabs";
import { openApp, primaryNav } from "./helpers";

// Settings > General > Appearance > "Show experimental tabs" (off by default, remembered with the sidebar
// preferences). Tabs flagged `experimental` in src/constants/tabs.ts (Fine-Tuning since the overnight review, D-87;
// Plugins was removed and Prompts moved into Chat) stay out of the sidebar until the switch is on. The spec reads
// the flags, so it holds whichever tabs are flagged; with none flagged it only checks the switch is remembered.
// Off Apple Silicon no Tools page is listed with the switch off (Conversion is Mac-only), so the group is not shown.

const TOOLS = tabs.filter((tab) => tab.group === "tools");
const hiddenTools = TOOLS.filter((tab) => tab.experimental).map((tab) => tab.shortLabel ?? tab.label);
const listedTools = TOOLS.filter((tab) => !tab.experimental).map((tab) => tab.shortLabel ?? tab.label);
const onAppleSilicon = process.platform === "darwin" && process.arch === "arm64";
const listedHere = listedTools.filter((label) => label !== "Conversion" || onAppleSilicon);

async function toolsHeaderCount(page: Page): Promise<number> {
  return (await primaryNav(page)).locator(".nav-group-header", { hasText: "Tools" }).count();
}

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

test("experimental tabs show only with the switch on, and the choice is remembered", async ({ page }) => {
  await openApp(page);
  let tools: string[];
  if (listedHere.length === 0) {
    expect(await toolsHeaderCount(page)).toBe(0);
  } else {
    tools = await toolsChildren(page);
    expect(tools).toEqual(expect.arrayContaining(listedHere));
    for (const label of hiddenTools) expect(tools).not.toContain(label);
  }
  await expect((await primaryNav(page)).locator(".nav-group-header", { hasText: "Benchmarks" })).toHaveCount(1);

  await setSwitch(page, true);
  tools = await toolsChildren(page);
  expect(tools).toEqual(expect.arrayContaining(hiddenTools));

  await page.reload();
  await expect(page.locator("aside.sidebar")).toBeVisible();
  await (await primaryNav(page)).getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.getByTestId("show-experimental-tabs")).toBeChecked();
  tools = await toolsChildren(page);
  expect(tools).toEqual(expect.arrayContaining(hiddenTools));

  await setSwitch(page, false);
  if (listedHere.length === 0) {
    expect(await toolsHeaderCount(page)).toBe(0);
  } else {
    tools = await toolsChildren(page);
    for (const label of hiddenTools) expect(tools).not.toContain(label);
  }
});

test("a remembered hidden tab is not reopened: the Tools group opens on a listed page", async ({ page }) => {
  const hidden = TOOLS.find((tab) => tab.experimental);
  test.skip(!hidden, "no Tools tab is flagged experimental");
  test.skip(listedHere.length === 0, "no Tools page is listed on this platform with the switch off, so there is no group to open");
  await page.addInitScript((id) => {
    window.localStorage.setItem(
      "chaosengine.sidebar.prefs.v1",
      JSON.stringify({ mode: "tabs", lastChildByGroup: { tools: id }, showExperimentalTabs: false }),
    );
  }, hidden!.id);
  await openApp(page);
  await (await primaryNav(page)).locator(".nav-button", { hasText: "Tools" }).click();
  const bar = page.locator(".subtab-bar");
  // Conversion is listed on Apple Silicon only; one listed page left shows no tab bar, just that page.
  if (listedHere.length > 1) await expect(bar.getByRole("tab", { selected: true })).toHaveCount(1);
  else await expect(page.getByRole("heading", { name: listedHere[0], level: 2 })).toBeVisible();
  for (const label of hiddenTools) await expect(bar.getByRole("tab", { name: label })).toHaveCount(0);
});
