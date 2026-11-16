import { mkdirSync } from "node:fs";
import path from "node:path";
import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE, TOKEN, goTo, openApp, openProject, openProjectTab, openTab, SHOTS_DIR, seedProject } from "./helpers";

// S-Q: below 1180 px the sidebar used to stack above the page and take about half a phone's screen. It is now a
// slim top bar with a menu button; the nav opens as a drawer (aria-expanded, focus kept inside, Escape closes,
// choosing a tab closes) and the page keeps the full width. At 1180 px and wider the sidebar column is unchanged.
// Screenshots go to e2e/screenshots/compact-*.png to be looked at.

async function shoot(page: Page, name: string): Promise<void> {
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.waitForTimeout(250);
  await page.screenshot({ path: path.join(SHOTS_DIR, `compact-${name}.png`) });
}

for (const width of [375, 768, 1024]) {
  test.describe(`${width} px`, () => {
    test.use({ viewport: { width, height: width === 375 ? 812 : 900 } });

    test("the sidebar is a top bar and the nav a drawer", async ({ page }) => {
      await openApp(page);
      const sidebar = page.locator("aside.sidebar");
      const menu = page.getByRole("button", { name: "Menu" });
      const nav = page.getByRole("navigation", { name: "Primary" });
      await expect(menu).toBeVisible();
      await expect(menu).toHaveAttribute("aria-expanded", "false");
      await expect(nav).toBeHidden();
      const bar = (await sidebar.boundingBox())!;
      expect(bar.height).toBeLessThanOrEqual(60);
      const main = (await page.locator("main.workspace").boundingBox())!;
      expect(main.width).toBeGreaterThanOrEqual(width - 1);
      expect(main.y).toBeLessThanOrEqual(bar.height + 1);
      const button = (await menu.boundingBox())!;
      expect(button.width).toBeGreaterThanOrEqual(44);
      expect(button.height).toBeGreaterThanOrEqual(44);
      await shoot(page, `${width}-closed`);

      await menu.click();
      await expect(menu).toHaveAttribute("aria-expanded", "true");
      await expect(nav).toBeVisible();
      // Focus moves into the drawer, onto the current tab.
      await expect(nav.locator(".nav-button", { hasText: /^Dashboard/ })).toBeFocused();
      await shoot(page, `${width}-open`);

      // Tab never leaves the menu button and the drawer.
      for (let i = 0; i < 40; i++) {
        await page.keyboard.press("Tab");
        const inside = await page.evaluate(() => {
          const el = document.activeElement;
          return !!el && (el.classList.contains("sidebar-menu-button") || !!el.closest("#sidebar-panel"));
        });
        expect(inside, `Tab ${i + 1} left the drawer`).toBe(true);
      }
      await menu.focus();
      await page.keyboard.press("Shift+Tab");
      expect(await page.evaluate(() => !!document.activeElement?.closest("#sidebar-panel"))).toBe(true);

      await page.keyboard.press("Escape");
      await expect(menu).toHaveAttribute("aria-expanded", "false");
      await expect(nav).toBeHidden();
      await expect(menu).toBeFocused();

      // The backdrop closes it too.
      await menu.click();
      await page.mouse.click(width - 10, 400);
      await expect(menu).toHaveAttribute("aria-expanded", "false");

      // Choosing a tab closes the drawer and shows that tab.
      await openTab(page, "Chat", "Chat");
      await expect(menu).toHaveAttribute("aria-expanded", "false");
      await expect(nav).toBeHidden();
      await expect(page.locator(".chat-layout-2col, .chat-layout").first()).toBeVisible();
      await shoot(page, `${width}-chat`);
    });

    test("Chat: the thread list no longer takes half the height; a project's tabs fit", async ({ page, request }) => {
      for (let i = 1; i <= 8; i++) {
        const made = await request.post(`${API_BASE}/api/chat/sessions`, {
          headers: { Authorization: `Bearer ${TOKEN}` },
          data: { title: `Thread ${i}` },
        });
        expect(made.ok()).toBeTruthy();
      }
      await goTo(page, "Chat", "Chat");
      const layout = (await page.locator(".chat-layout-2col").boundingBox())!;
      const thread = (await page.locator(".chat-thread").boundingBox())!;
      expect(thread.height / layout.height).toBeGreaterThan(0.6);
      await shoot(page, `${width}-chat-threads`);

      const project = await seedProject(request, { name: `Narrow ${width}`, shots: 3, takes: 1 });
      await openProject(page, project.name, "Storyboard");
      await shoot(page, `${width}-storyboard`);
      await page.locator(".shot-card").first().focus();
      await page.keyboard.press("Enter");
      await expect(page.locator(".shot-editor")).toBeVisible();
      await shoot(page, `${width}-shot-editor`);
      await page.keyboard.press("Escape");
      await openProjectTab(page, "Board");
      await shoot(page, `${width}-board`);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    });
  });
}

for (const width of [1180, 1440]) {
  test.describe(`${width} px`, () => {
    test.use({ viewport: { width, height: 900 } });

    test("the sidebar stays a column with no menu button", async ({ page }) => {
      await openApp(page);
      await expect(page.locator(".sidebar-menu-button")).toBeHidden();
      await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();
      const side = (await page.locator("aside.sidebar").boundingBox())!;
      const main = (await page.locator("main.workspace").boundingBox())!;
      expect(side.x).toBe(0);
      expect(side.width).toBeGreaterThanOrEqual(180); // 208 px at the default UI scale of 0.9 is about 187 on screen
      expect(side.width).toBeLessThanOrEqual(240);
      expect(side.height).toBeGreaterThanOrEqual(899);
      expect(main.x).toBeGreaterThanOrEqual(side.width - 1);
      await shoot(page, `${width}`);
    });
  });
}

test.describe("touch, 768 px", () => {
  test.use({ hasTouch: true, isMobile: true, viewport: { width: 768, height: 1024 } });

  test("the drawer's rows are 44 px", async ({ page }) => {
    await openApp(page);
    await page.getByRole("button", { name: "Menu" }).tap();
    const nav = page.getByRole("navigation", { name: "Primary" });
    await expect(nav).toBeVisible();
    const heights = await nav.locator(".nav-button, .nav-group-header, .nav-child-button").evaluateAll((els) =>
      els.filter((el) => (el as HTMLElement).offsetParent).map((el) => Math.round(el.getBoundingClientRect().height)),
    );
    expect(heights.length).toBeGreaterThan(5);
    expect(Math.min(...heights)).toBeGreaterThanOrEqual(44);
  });
});
