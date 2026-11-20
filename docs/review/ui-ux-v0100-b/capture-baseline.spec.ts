// Screenshot recipe for the v0.10.0 UI/UX review B (docs/review/ui-ux-v0100-b/README.md). Not a test of behaviour.
//
// Pass 1: every page in the review's area as the backend itself answers, at 1280 and 375 px, in two worlds:
//   A  throwaway HOME, nothing downloaded or installed (the real first run of this Windows 11 / RTX 4090 PC);
//   B  throwaway HOME, HF_HUB_CACHE pointed at the PC's model folder and HF_HUB_OFFLINE=1, so the rows that are on
//      disk show as downloaded. In world B every request that could change anything outside the throwaway HOME is
//      answered 503 here in the browser (see `readOnly`): nothing is deleted, downloaded, installed or generated.
// To re-run: copy this file to e2e/tests/98-uiux-b-baseline.spec.ts and playwright.review.config.ts to e2e/, start
// the servers, then from e2e/:
//   E2E_WEB_PORT=8981 E2E_API_PORT=8982 REVIEW_WORLD=A npx playwright test -c playwright.review.config.ts 98-uiux
//   E2E_WEB_PORT=8983 E2E_API_PORT=8984 REVIEW_WORLD=B npx playwright test -c playwright.review.config.ts 98-uiux
import { mkdirSync } from "node:fs";
import path from "node:path";
import { test } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, openTab, showExperimentalTabs } from "./helpers";

const WORLD = process.env.REVIEW_WORLD ?? "A";
const OUT = process.env.REVIEW_OUT ?? path.resolve(__dirname, "..", "..", "docs", "review", "ui-ux-v0100-b", "shots");
const DESKTOP = { width: 1280, height: 900 };
const PHONE = { width: 375, height: 812 };

test.setTimeout(900_000);

export async function shoot(page: Page, name: string, phone = false, grow = true) {
  mkdirSync(OUT, { recursive: true });
  await page.waitForTimeout(700);
  const file = path.join(OUT, `${phone ? "m" : "d"}-${WORLD}-${name}.png`);
  const size = page.viewportSize() ?? DESKTOP;
  if (!grow) {
    await page.screenshot({ path: file, fullPage: false });
    return;
  }
  // The app fills the viewport and scrolls inside its own boxes, so the capture grows the viewport to the tallest
  // scroll box instead of using Playwright's full-page mode.
  const tallest = await page.evaluate(() => {
    let max = 0;
    for (const el of document.querySelectorAll<HTMLElement>("main, main *")) {
      const style = getComputedStyle(el);
      if (["auto", "scroll"].includes(style.overflowY) && el.scrollHeight > el.clientHeight + 1) {
        max = Math.max(max, el.scrollHeight + el.getBoundingClientRect().top / 0.9 + 40);
      }
    }
    return max;
  });
  const height = Math.min(Math.max(size.height, Math.ceil(tallest * 0.9)), 9000);
  if (height !== size.height) {
    await page.setViewportSize({ width: size.width, height });
    await page.waitForTimeout(400);
  }
  await page.screenshot({ path: file, fullPage: false });
  if (height !== size.height) await page.setViewportSize(size);
}

/** World B: nothing but the throwaway HOME's own small state may be written. */
async function readOnly(page: Page) {
  if (WORLD !== "B") return;
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const url = new URL(request.url()).pathname;
    const harmless = /\/api\/(terms|settings|auth|busy-advice|music\/estimate|chat\/studio\/price|storyboard\/projects|images\/preview|video\/preview|images\/estimate|video\/estimate)/.test(url);
    if (request.method() === "GET" || request.method() === "OPTIONS" || harmless) return route.fallback();
    return route.fulfill({ status: 503, json: { detail: "Review run: read-only." } });
  });
}

const ONLY = (process.env.REVIEW_ONLY ?? "").split(",").filter(Boolean);
const ALL_PAGES: Array<[string, string, string]> = [
  ["Images", "Models", "image-models"],
  ["Images", "Studio", "image-studio"],
  ["Images", "Gallery", "image-gallery"],
  ["Video", "Models", "video-models"],
  ["Video", "Studio", "video-studio"],
  ["Video", "Gallery", "video-gallery"],
  ["Music and Audio", "Models", "music-models"],
  ["Music and Audio", "Studio", "music-studio"],
  ["Music and Audio", "Library", "music-library"],
  ["Benchmarks", "Run", "bench-run"],
  ["Benchmarks", "History", "bench-history"],
  // Tools > Conversion is listed on macOS only (sidebarModel.ts), so it is not in this Windows walk.
  ["Tools", "Fine-Tuning", "finetuning"],
];

const PAGES = ONLY.length ? ALL_PAGES.filter((p) => ONLY.includes(p[2])) : ALL_PAGES;

for (const phone of [false, true]) {
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: phone ? PHONE : DESKTOP });

    test(`baseline pages, world ${WORLD}`, async ({ page }) => {
      await showExperimentalTabs(page);
      await readOnly(page);
      const [g0, c0, n0] = PAGES[0];
      await goTo(page, g0, c0);
      await page.waitForTimeout(2500);
      await shoot(page, n0, phone);
      for (const [group, child, name] of PAGES.slice(1)) {
        await openTab(page, group, child);
        await page.waitForTimeout(1800);
        await shoot(page, name, phone);
      }
      // Music > Studio has a Music | Sounds switch.
      await openTab(page, "Music and Audio", "Studio");
      const sounds = page.getByRole("tab", { name: "Sounds", exact: true }).or(page.getByRole("button", { name: "Sounds", exact: true }));
      if (await sounds.first().isVisible().catch(() => false)) {
        await sounds.first().click();
        await page.waitForTimeout(1200);
        await shoot(page, "music-studio-sounds", phone);
      }
    });
  });
}
