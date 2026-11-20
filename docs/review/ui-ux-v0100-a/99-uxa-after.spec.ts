// UI/UX review: the same screens after the fixes in this branch (docs/review/ui-ux-v0100-a/after).
import path from "node:path";
import { test, expect } from "./fixtures";
import { openApp, openTab } from "./helpers";
import { DESKTOP, top } from "./uxaShared";

const AFTER = path.resolve(__dirname, "..", "..", "docs", "review", "ui-ux-v0100-a", "after");
test.use({ viewport: DESKTOP });

test("after: dashboard, chat first run, model picker, models rows", async ({ page }) => {
  await openApp(page);
  await top(page, "Dashboard");
  await expect(page.getByText("Backend readiness")).toBeVisible({ timeout: 60_000 });
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(AFTER, "d-01-dashboard.png") });
  await openTab(page, "Chat", "Chat");
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(AFTER, "d-02-chat-empty.png") });
  await page.getByRole("button", { name: "Select Model" }).click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(AFTER, "d-12-select-model-dialog-empty.png") });
  await page.keyboard.press("Escape");
  await openTab(page, "Chat", "Models");
  await page.getByPlaceholder(/Search name/).fill("24 GB card");
  await page.waitForTimeout(2500);
  await page.locator(".models-card__toggle, [aria-expanded='false']").filter({ hasText: "Nemotron" }).first().click().catch(() => undefined);
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(AFTER, "d-14-models-24gb-rows.png") });
});
