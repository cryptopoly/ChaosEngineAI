import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { openApp, openMenuIfCompact, openTab, primaryNav } from "./helpers";

// Review stream D: every tab outside the seven owned by other streams, at the smallest window the desktop app
// allows (1100 x 760, tauri.conf.json), at the default one (1400 x 920) and at a phone width. Three things must
// hold on each page: no panel cuts its own content with nothing to scroll (the Dashboard and Settings did), no
// two panels are drawn over each other (Image Studio did, at 1100 px and narrower), and the page does not crash.
// Then the cases the sweep found: Settings under the drawer's footer, the Storage folder rows, tick box rows.

test.setTimeout(180_000);

const TABS: Array<[string, string]> = [
  ["Chat", "Chat"], ["Chat", "Compare"], ["Chat", "HTML Challenge"], ["Chat", "Models"],
  ["Images", "Models"], ["Images", "Studio"], ["Images", "Gallery"],
  ["Video", "Models"], ["Video", "Studio"], ["Video", "Gallery"],
  ["Voice", "Models"], ["Voice", "Studio"], ["Voice", "Library"],
  ["Music and Audio", "Models"], ["Music and Audio", "Studio"], ["Music and Audio", "Library"],
];
const SETTINGS_SECTIONS = ["General", "Storage", "Providers", "Integrations", "Skills", "Diagnostics", "Accelerators", "Disk usage"];

/** Panels that hide part of their content, and sibling panels that overlap. */
async function layoutFaults(page: Page): Promise<{ cut: string[]; overlap: string[] }> {
  return page.evaluate(() => {
    const name = (el: Element) => `${el.tagName.toLowerCase()}.${String(el.className).trim().replace(/\s+/g, ".").slice(0, 60)}`;
    const cut = [...document.querySelectorAll("main .panel, main .content-grid, main .settings-storage-grid")]
      .filter((el) => {
        const style = getComputedStyle(el);
        return el.clientHeight > 0 && el.scrollHeight > el.clientHeight + 4 && (style.overflowY === "hidden" || style.overflowY === "clip");
      })
      .map((el) => `${name(el)} shows ${el.clientHeight} of ${el.scrollHeight}px`);
    const overlap: string[] = [];
    for (const parent of new Set([...document.querySelectorAll("main .panel")].map((panel) => panel.parentElement))) {
      if (!parent) continue;
      const kids = [...parent.children].filter((kid) => {
        const position = getComputedStyle(kid).position;
        return kid.getBoundingClientRect().height > 0 && position !== "absolute" && position !== "fixed";
      });
      for (let i = 0; i < kids.length; i++) {
        for (let j = i + 1; j < kids.length; j++) {
          const a = kids[i].getBoundingClientRect();
          const b = kids[j].getBoundingClientRect();
          const x = Math.min(a.right, b.right) - Math.max(a.left, b.left);
          const y = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
          if (x > 4 && y > 4) overlap.push(`${name(kids[i])} over ${name(kids[j])} by ${Math.round(y)}px`);
        }
      }
    }
    return { cut, overlap };
  });
}

async function check(page: Page, label: string): Promise<void> {
  await page.waitForTimeout(350);
  await expect(page.locator(".error-boundary"), `${label} crashed`).toHaveCount(0);
  const { cut, overlap } = await layoutFaults(page);
  expect.soft(cut, `${label}: panels that cut their content`).toEqual([]);
  expect.soft(overlap, `${label}: panels drawn over each other`).toEqual([]);
}

async function openSettings(page: Page): Promise<void> {
  await (await primaryNav(page)).locator(".nav-button", { hasText: /^Settings/ }).click();
  await expect(page.locator(".settings-subtab-bar")).toBeVisible();
}

async function settingsSection(page: Page, label: string): Promise<void> {
  const tab = page.locator(".settings-subtab-bar [role=tab]", { hasText: new RegExp(`^${label}$`) });
  await tab.click();
  await expect(tab).toHaveAttribute("aria-selected", "true");
}

for (const [width, height] of [[1100, 760], [1400, 920], [375, 812]] as const) {
  test.describe(`${width} x ${height}`, () => {
    test.use({ viewport: { width, height } });

    test("no page cuts or overlaps its panels", async ({ page }) => {
      await openApp(page);
      await (await primaryNav(page)).locator(".nav-button", { hasText: /^Dashboard/ }).click();
      await check(page, "Dashboard");
      await (await primaryNav(page)).locator(".nav-button", { hasText: /^Projects/ }).click();
      await check(page, "Projects");
      for (const [group, child] of TABS) {
        await openTab(page, group, child);
        await check(page, `${group} > ${child}`);
      }
      await openSettings(page);
      for (const section of SETTINGS_SECTIONS) {
        await settingsSection(page, section);
        await check(page, `Settings > ${section}`);
      }
    });
  });
}

test.describe("1100 x 760, the smallest window", () => {
  test.use({ viewport: { width: 1100, height: 760 } });

  test("Settings > General: every control can be brought into view", async ({ page }) => {
    await openApp(page);
    await openSettings(page);
    await settingsSection(page, "General");
    // These sat below the cut: the UI scale row, the clock format row and the translations link.
    for (const control of [
      page.getByRole("radiogroup", { name: "UI scale" }),
      page.getByRole("radiogroup", { name: "Clock format" }),
      page.locator("#settings-language-select"),
      page.getByRole("button", { name: "Read the terms" }),
    ]) {
      await control.scrollIntoViewIfNeeded();
      await expect(control).toBeInViewport({ ratio: 1 });
    }
    // The tick box sits on one line with its label, not centred above it.
    const row = page.locator("label.check-row", { hasText: "Keep all sidebar groups open" });
    const box = (await row.locator("input").boundingBox())!;
    const text = (await row.boundingBox())!;
    expect(box.height).toBeLessThanOrEqual(24);
    expect(text.height, "the tick box and its label are stacked").toBeLessThanOrEqual(30);
    // The Copy button beside the translations link is a styled button.
    await expect(page.locator(".panel", { hasText: "Help improve translations" }).getByRole("button", { name: "Copy" })).toHaveClass(/secondary-button/);
  });

  test("the drawer: Settings can be clicked with the Chat or Tools group open", async ({ page }) => {
    await openApp(page);
    for (const group of ["Chat", "Tools"]) {
      await openMenuIfCompact(page);
      const nav = page.getByRole("navigation", { name: "Primary" });
      const header = nav.locator(".nav-group-header", { hasText: new RegExp(`^${group}$`) });
      await header.scrollIntoViewIfNeeded();
      if ((await header.getAttribute("aria-expanded")) !== "true") await header.click();
      const settings = nav.locator(".nav-button", { hasText: /^Settings/ });
      await settings.scrollIntoViewIfNeeded();
      // Nothing may sit on top of it (the status footer did).
      const covered = await settings.evaluate((button) => {
        const box = button.getBoundingClientRect();
        const top = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
        return !top || !(button === top || button.contains(top));
      });
      expect(covered, `Settings is covered with ${group} open`).toBe(false);
      await settings.click({ timeout: 5000 });
      await expect(page.locator(".settings-subtab-bar")).toBeVisible();
    }
  });

  test("Settings > Storage: the folder path gets the room, not the Browse button", async ({ page }) => {
    await openApp(page);
    await openSettings(page);
    await settingsSection(page, "Storage");
    const row = page.locator(".panel", { hasText: "Data Directory" }).locator(".directory-add-row").first();
    const path = (await row.locator("input").boundingBox())!;
    const browse = (await row.getByRole("button", { name: "Browse..." }).boundingBox())!;
    expect(path.width).toBeGreaterThan(browse.width * 1.5);
    expect(path.width).toBeGreaterThanOrEqual(180);
  });
});

test.describe("375 px", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("Settings > Storage is one column and its buttons keep their text", async ({ page }) => {
    await openApp(page);
    await openSettings(page);
    await settingsSection(page, "Storage");
    const columns = page.locator(".settings-storage-col");
    const first = (await columns.nth(0).boundingBox())!;
    const second = (await columns.nth(1).boundingBox())!;
    expect(first.width).toBeGreaterThan(300);
    expect(second.y, "the two Storage columns are still side by side").toBeGreaterThan(first.y + 50);
    const squeezed = await page.evaluate(() =>
      [...document.querySelectorAll(".settings-storage-grid button")]
        .filter((button) => button.clientWidth > 0 && button.scrollWidth > button.clientWidth + 2)
        .map((button) => (button.textContent ?? "").trim()),
    );
    expect(squeezed, "buttons narrower than their label").toEqual([]);
  });
});
