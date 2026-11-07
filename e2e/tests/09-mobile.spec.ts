import { mkdirSync, statSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openTab, SHOTS_DIR, seedProject } from "./helpers";

// 375 px wide (a phone). The page must not scroll sideways and nothing visible may stick out past the
// right edge. A screenshot of each screen is saved to e2e/screenshots/ (git-ignored) so the layout can be
// looked at, and the sanity check is that the file is a real picture of the page, not a blank one.

test.use({ viewport: { width: 375, height: 812 } });

async function overflow(page: Page) {
  return await page.evaluate(() => {
    const width = window.innerWidth;
    const wide = [...document.querySelectorAll("main *")]
      .filter((el) => {
        const box = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return box.width > 0 && box.right > width + 1 && style.visibility !== "hidden" && style.position !== "fixed";
      })
      .slice(0, 5)
      .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 40)} right=${Math.round(el.getBoundingClientRect().right)}`);
    return { page: document.documentElement.scrollWidth - width, wide };
  });
}

async function look(page: Page, name: string) {
  mkdirSync(SHOTS_DIR, { recursive: true });
  const file = path.join(SHOTS_DIR, `mobile-${name}.png`);
  await page.waitForTimeout(400);
  await page.screenshot({ path: file, fullPage: true });
  // A 375x812 page with content compresses well above 8 KB; an empty or all-black capture does not.
  expect(statSync(file).size, `${name} screenshot looks blank`).toBeGreaterThan(8_000);
  const { page: sideways, wide } = await overflow(page);
  expect.soft(sideways, `${name}: page scrolls sideways by ${sideways}px (${wide.join("; ")})`).toBeLessThanOrEqual(0);
}

test("375 px: the Image gallery fits", async ({ page }) => {
  await goTo(page, "Images", "Gallery");
  await expect(page.locator("aside.sidebar")).toBeVisible();
  await look(page, "images-gallery");
});

test("375 px: Storyboard projects, board and mixer", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Phone Film", shots: 2, takes: 2, clonedVoice: true });
  await goTo(page, "Storyboard", "Projects");
  await look(page, "storyboard-projects");
  await page.getByRole("article", { name: project.name }).getByRole("button", { name: "Open", exact: true }).click();
  await expect(page.getByRole("heading", { name: `${project.name} · Board` })).toBeVisible();
  await look(page, "storyboard-board");
  await page.getByRole("main").getByRole("button", { name: "Mixer", exact: true }).click();
  await expect(page.getByRole("heading", { name: `${project.name} · Mixer` })).toBeVisible();
  await look(page, "storyboard-mixer");
});

// Known layout bug F-1 (docs/research/e2e-findings.md): the Models table keeps its desktop width on a
// phone and the page scrolls sideways. Marked as an expected failure so this goes red the day it is fixed.
test("375 px: Music Models fits the width", async ({ page }) => {
  test.fail(true, "F-1: .models-table overflows below ~700 px");
  await goTo(page, "Music", "Models");
  await expect(page.getByText("Stable Audio Open Small").first()).toBeVisible();
  await look(page, "music-models");
});

test("375 px: Music Studio and Voice Library", async ({ page }) => {
  await goTo(page, "Music", "Studio");
  await expect(page.getByText("ACE-Step 1.5 turbo")).toBeVisible();
  await look(page, "music-studio");
  await openTab(page, "Voice", "Library");
  await expect(page.locator("section").getByRole("heading", { name: "Voice Library" })).toBeVisible();
  await look(page, "voice-library");
});

test("375 px: Settings with the terms panel", async ({ page }) => {
  await goTo(page, "Images", "Gallery");
  await page.getByRole("navigation", { name: "Primary" }).getByRole("button", { name: "Settings", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Terms of use" })).toBeVisible();
  await look(page, "settings");
});
