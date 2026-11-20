// Shared helpers for the UI/UX review capture specs (docs/review/ui-ux-v0100-a). Not a test.
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { Page } from "@playwright/test";
import { primaryNav } from "./helpers";

export const OUT = process.env.REVIEW_OUT ?? path.resolve(__dirname, "..", "..", "docs", "review", "ui-ux-v0100-a", "shots");
export const DESKTOP = { width: 1280, height: 900 };
export const PHONE = { width: 375, height: 812 };
const texts: Record<string, string> = {};

export function dumpTexts(tag: string) {
  writeFileSync(path.join("E:/tmp/uxa", `texts-${tag}.json`), JSON.stringify(texts, null, 1));
}

/** Screenshot. The app scrolls inside its own boxes (zoom 0.9), so `grow` raises the viewport to the tallest one. */
export async function shoot(page: Page, name: string, phone: boolean, grow = true) {
  mkdirSync(OUT, { recursive: true });
  await page.waitForTimeout(600);
  const key = `${phone ? "m" : "d"}-${name}`;
  const file = path.join(OUT, `${key}.png`);
  const size = page.viewportSize() ?? DESKTOP;
  texts[key] = await page.evaluate(() => document.body.innerText);
  if (!grow) {
    await page.screenshot({ path: file });
    return;
  }
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
    await page.waitForTimeout(300);
  }
  await page.screenshot({ path: file });
  if (height !== size.height) await page.setViewportSize(size);
}

/** A sidebar entry that is not in a group (Dashboard, Server, Settings). */
export async function top(page: Page, label: string) {
  const nav = await primaryNav(page);
  await nav.locator(".nav-button", { hasText: new RegExp(`^${label}`) }).first().click();
}
