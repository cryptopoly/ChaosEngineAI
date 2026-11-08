import { mkdirSync, statSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, goToProjects, openProjectTab, openTab, SHOTS_DIR, seedProject } from "./helpers";

// 375 px wide (a phone). The page must not scroll sideways and nothing visible may stick out past the
// right edge. A screenshot of each screen is saved to e2e/screenshots/ (git-ignored) so the layout can be
// looked at, and the sanity check is that the file is a real picture of the page, not a blank one.

test.use({ viewport: { width: 375, height: 812 } });

/**
 * What sticks out at this width: elements whose right edge is past the viewport (unless a scroll box that
 * itself fits holds them, like the timeline or the tab row), and boxes that scroll sideways or have been
 * panned (the tab row and the timeline are meant to scroll, nothing else is; a clipped label with an ellipsis is fine).
 */
async function overflow(page: Page) {
  return await page.evaluate(() => {
    const width = window.innerWidth;
    const scrollersOk = (el: Element) => el.matches(".project-shell__tabs, .timeline__scroll, .sr-only, .preset-row, .picker-dialog__sources");
    const insideFittingScroller = (el: Element) => {
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        if (getComputedStyle(p).overflowX !== "visible") return p.getBoundingClientRect().right <= width + 1;
      }
      return false;
    };
    const wide = [...document.querySelectorAll("main *")]
      .filter((el) => {
        const box = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return box.width > 0 && box.right > width + 1 && style.visibility !== "hidden" && style.position !== "fixed" && !insideFittingScroller(el);
      })
      .slice(0, 5)
      .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 40)} right=${Math.round(el.getBoundingClientRect().right)}`);
    const panned = [document.querySelector("main"), ...document.querySelectorAll("main *")]
      .filter((el): el is Element => !!el && !scrollersOk(el) && (el.scrollLeft > 0 || (["auto", "scroll"].includes(getComputedStyle(el).overflowX) && el.scrollWidth > el.clientWidth + 1)))
      .slice(0, 5)
      .map((el) => `${String(el.className).slice(0, 40)} ${el.scrollWidth}/${el.clientWidth} left=${el.scrollLeft}`);
    return { page: document.documentElement.scrollWidth - width, wide, panned };
  });
}

async function look(page: Page, name: string) {
  mkdirSync(SHOTS_DIR, { recursive: true });
  const file = path.join(SHOTS_DIR, `mobile-${name}.png`);
  await page.waitForTimeout(400);
  await page.screenshot({ path: file, fullPage: true });
  // A 375x812 page with content compresses well above 8 KB; an empty or all-black capture does not.
  expect(statSync(file).size, `${name} screenshot looks blank`).toBeGreaterThan(8_000);
  const { page: sideways, wide, panned } = await overflow(page);
  expect.soft(sideways, `${name}: page scrolls sideways by ${sideways}px`).toBeLessThanOrEqual(0);
  // The element-level checks hold for the Projects screens (D-65); other screens keep the page-level one.
  if (name.startsWith("storyboard")) {
    expect(wide, `${name}: elements past the right edge`).toEqual([]);
    expect(panned, `${name}: boxes that scroll sideways`).toEqual([]);
  }
}

test("375 px: the Image gallery fits", async ({ page }) => {
  await goTo(page, "Images", "Gallery");
  await expect(page.locator("aside.sidebar")).toBeVisible();
  await look(page, "images-gallery");
});

test("375 px: the project list and every tab of a project", async ({ page, request }) => {
  const project = await seedProject(request, { name: "Phone Film", shots: 2, takes: 2, clonedVoice: true });
  await goToProjects(page);
  await look(page, "storyboard-projects");
  await page.getByRole("button", { name: `Open ${project.name}`, exact: true }).click();
  await expect(page.locator(".project-shell").getByRole("heading", { name: project.name })).toBeVisible();
  for (const [tab, shot] of [["Overview", "storyboard-overview"], ["Cast", "storyboard-cast"], ["Storyboard", "storyboard-shots"], ["Mixer", "storyboard-mixer"], ["Export", "storyboard-export"]] as const) {
    await openProjectTab(page, tab);
    // Moving along the tab row with the keyboard must not pan the shell either.
    await page.getByRole("tab", { name: new RegExp(`^${tab}`) }).focus();
    await look(page, shot);
  }
  // The picture picker opens as a sheet that fits the width.
  await openProjectTab(page, "Cast");
  await page.getByRole("button", { name: "Add character" }).click();
  await page.locator(".cast-card__add-picture").first().click();
  const picker = page.getByRole("dialog", { name: /^Add a picture of/ });
  await expect(picker).toBeVisible();
  const box = await picker.boundingBox();
  expect(box && box.x >= 0 && box.x + box.width <= 376).toBe(true);
  await look(page, "storyboard-picker");
  await picker.getByRole("button", { name: "Cancel" }).click();
});

// F-1 (docs/research/e2e-findings.md): the Models table used to keep its desktop width on a phone.
test("375 px: Music Models fits the width", async ({ page }) => {
  await goTo(page, "Music and Audio", "Models");
  await expect(page.getByText("Stable Audio Open Small").first()).toBeVisible();
  await look(page, "music-models");
});

test("375 px: Music Studio and Voice Library", async ({ page }) => {
  await goTo(page, "Music and Audio", "Studio");
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
