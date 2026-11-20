// UI/UX review, pass 1: every page in my areas as this PC's throwaway backend answers (first run, nothing installed).
import { test, expect } from "./fixtures";
import { openApp, openTab } from "./helpers";
import { DESKTOP, PHONE, dumpTexts, shoot, top } from "./uxaShared";

for (const phone of [false, true]) {
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: phone ? PHONE : DESKTOP });
    test.afterAll(() => dumpTexts(`p1-${phone ? "m" : "d"}`));

    test("00 first run", async ({ page }) => {
      await page.goto("/");
      await expect(page.locator("aside.sidebar")).toBeVisible();
      await expect(page.locator(".loading-state")).toHaveCount(0, { timeout: 60_000 });
      await shoot(page, "00-first-run", phone, false);
    });

    test("01 dashboard", async ({ page }) => {
      await openApp(page);
      await top(page, "Dashboard");
      await shoot(page, "01-dashboard", phone);
    });

    test("02 chat", async ({ page }) => {
      await openApp(page);
      await openTab(page, "Chat", "Chat");
      await shoot(page, "02-chat-empty", phone, false);
      for (const child of ["Compare", "HTML Challenge", "Models"]) {
        await openTab(page, "Chat", child);
        await shoot(page, `03-chat-${child.toLowerCase().replace(/ /g, "-")}`, phone);
      }
    });

    test("05 server", async ({ page }) => {
      await openApp(page);
      await top(page, "Server");
      await shoot(page, "05-server-overview", phone);
      await page.getByRole("tab", { name: "Activity" }).click();
      await shoot(page, "05-server-activity", phone);
    });

    test("06 settings", async ({ page }) => {
      await openApp(page);
      await top(page, "Settings");
      for (const s of ["General", "Storage", "Providers", "Integrations", "Skills", "Diagnostics", "Accelerators", "Disk usage"]) {
        await page.getByRole("tablist", { name: "Settings sections" }).getByRole("tab", { name: s, exact: true }).click();
        await page.waitForTimeout(s === "Diagnostics" || s === "Disk usage" ? 6000 : 800);
        await shoot(page, `06-settings-${s.toLowerCase().replace(/ /g, "-")}`, phone);
      }
    });

    test("07 benchmarks", async ({ page }) => {
      await openApp(page);
      await openTab(page, "Benchmarks", "Run");
      await shoot(page, "07-benchmarks-run", phone);
      await openTab(page, "Benchmarks", "History");
      await shoot(page, "07-benchmarks-history", phone);
    });
  });
}
