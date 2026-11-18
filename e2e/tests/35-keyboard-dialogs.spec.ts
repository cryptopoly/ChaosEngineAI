import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { openApp, openTab } from "./helpers";

// Review stream D: keyboard paths in the main flows. The model dialogs (Select Model in Chat and Compare, the
// family details on Chat Models) were plain overlays: no dialog role, Escape did nothing, Tab walked out into the
// page behind and focus was lost on close. The first-run terms must be answered, so Escape leaves them open.

async function focusInside(page: Page, selector: string): Promise<boolean> {
  return page.evaluate((css) => !!document.activeElement?.closest(css), selector);
}

test("first-run terms: Escape does not close them, Tab stays on Quit and I agree, Enter accepts", async ({ page }) => {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.settings = { ...body.settings, termsAcceptedVersion: 0, termsAcceptedAt: "" };
    await route.fulfill({ response, json: body });
  });
  await page.goto("/");
  const dialog = page.getByRole("dialog", { name: "Terms of use" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeVisible();
  const order: string[] = [];
  for (let i = 0; i < 4; i++) {
    await page.keyboard.press("Tab");
    order.push(await page.evaluate(() => (document.activeElement?.textContent ?? "").trim()));
  }
  expect(order).toEqual(["Quit", "I agree", "Quit", "I agree"]);
  const saved = page.waitForRequest((r) => r.method() === "PATCH" && r.url().endsWith("/api/settings"));
  await page.keyboard.press("Enter");
  expect((await saved).postDataJSON()).toEqual({ termsAcceptedVersion: 1 });
  await expect(dialog).toBeHidden();
});

test("Chat: Select Model is a dialog; Tab stays inside, Escape closes it and focus returns to the button", async ({ page }) => {
  await openApp(page);
  await openTab(page, "Chat", "Chat");
  // The button reads "Select Model", or the thread's model when an earlier spec left a thread with one.
  const opener = page.locator(".thread-toolbar-actions > button.secondary-button").first();
  await opener.click();
  const dialog = page.getByRole("dialog", { name: "Select Model" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAttribute("aria-modal", "true");
  expect(await focusInside(page, "[role=dialog]")).toBe(true);
  for (let i = 0; i < 30; i++) {
    await page.keyboard.press("Tab");
    expect(await focusInside(page, "[role=dialog]"), `Tab ${i + 1} left the dialog`).toBe(true);
  }
  await page.keyboard.press("Shift+Tab");
  expect(await focusInside(page, "[role=dialog]")).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test("Compare: the slot's model dialog closes on Escape and gives focus back", async ({ page }) => {
  await openApp(page);
  await openTab(page, "Chat", "Compare");
  const opener = page.getByRole("button", { name: "Select", exact: true }).first();
  await opener.click();
  const dialog = page.getByRole("dialog", { name: /^Select Model/ });
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test("Settings: the read-only terms close on Escape and focus returns to Read the terms", async ({ page }) => {
  await openApp(page);
  await page.getByRole("navigation", { name: "Primary" }).locator(".nav-button", { hasText: /^Settings/ }).click();
  const opener = page.getByRole("button", { name: "Read the terms" });
  await opener.click();
  const dialog = page.getByRole("dialog", { name: "Terms of use" });
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(opener).toBeFocused();
});

test("Chat: the thread search, New thread and the composer are reachable by keyboard and Enter in an empty search does nothing", async ({ page }) => {
  await openApp(page);
  await openTab(page, "Chat", "Chat");
  await page.getByRole("button", { name: "New thread" }).click();
  await expect(page.locator(".error-boundary")).toHaveCount(0);
  const search = page.getByPlaceholder("Search threads...");
  await search.fill("no such thread");
  await page.keyboard.press("Enter");
  await expect(page.locator(".error-boundary")).toHaveCount(0);
  await search.fill("");
  // With no model loaded the composer explains itself and Enter sends nothing.
  const composer = page.locator("textarea").last();
  await expect(composer).toHaveAttribute("placeholder", /Load a model first/);
  let sent = false;
  page.on("request", (r) => { if (r.method() === "POST" && /\/api\/chat\/generate/.test(r.url())) sent = true; });
  await composer.fill("hello");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(800);
  expect(sent).toBe(false);
  await expect(page.locator(".error-boundary")).toHaveCount(0);
});
