import { mkdirSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { API_BASE, TOKEN, goTo, openTab, SHOTS_DIR } from "./helpers";

// Overnight review, stream C: the Prompts and Plugins pages (Tools group). Both run against the run's own
// backend (throwaway HOME), so the real prompt store and plugin registry answer; no model runs. Chat is faked
// the same way as 16-skills: a model looks loaded and the chat stream is a page-side fetch wrapper, so the
// test reads what the composer sends.
//
// Prompts: the five built-ins show, dates are dates (not epoch seconds), a save with no name says why, a
// template with a variable is created, found by its prompt text, filled in, and "Apply to chat" lands in Chat
// with the filled prompt sent as the system prompt; edit and delete (with a confirm); the page stacks at
// 375 px; a failed load says so on both pages.
// Plugins: the page lists what GET /api/plugins registers (it used to read a key the route never sent and
// said "No plugins found"), the last row can be reached, a switch flips and flips back, 375 px fits.

const AUTH = { Authorization: `Bearer ${TOKEN}` };
const NAME = "E2E review template";

async function removeTemplates(page: Page) {
  const response = await page.request.get(`${API_BASE}/api/prompts`, { headers: AUTH });
  const { templates } = (await response.json()) as { templates: Array<{ id: string; name: string }> };
  for (const template of templates.filter((t) => t.name.startsWith(NAME))) {
    await page.request.delete(`${API_BASE}/api/prompts/${template.id}`, { headers: AUTH });
  }
}

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      w.__promptPayload = JSON.parse(String(init?.body ?? "{}"));
      const body = new ReadableStream<Uint8Array>({ start: () => undefined });
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}

async function mockLoadedModel(page: Page) {
  let runtime: Record<string, unknown> = {};
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as { runtime: Record<string, unknown> };
    body.runtime = {
      ...body.runtime,
      state: "loaded",
      loadedModel: {
        ref: "mock/chat-8b", name: "Mock Chat 8B", canonicalRepo: "mock/chat-8b", backend: "mlx", source: "library",
        engine: "mlx", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
        contextTokens: 8192, loadedAt: "2026-11-18 08:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
      },
    };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
}

const layout = (page: Page) => page.locator(".prompt-library-layout");
const detail = (page: Page) => layout(page).locator(".panel").nth(1);

test.afterEach(async ({ page }) => {
  await removeTemplates(page);
});

test("Prompts: built-ins, dates, a refused save, variables, search, use in Chat, edit, delete", async ({ page }) => {
  await scriptChatStream(page);
  await mockLoadedModel(page);
  await goTo(page, "Tools", "Prompts");
  await removeTemplates(page);

  const list = layout(page).locator(".panel").first();
  await expect(list.locator(".session-button")).toHaveCount(5);
  await list.locator(".session-button", { hasText: "Translator" }).click();
  await expect(detail(page)).toContainText("You are a professional translator.");
  // Created / Updated are dates; the page used to print the backend's epoch seconds (1794987548.93451).
  const stamp = detail(page).getByText(/^Created: /);
  await expect(stamp).not.toHaveText(/\d{9,}/);
  await expect(stamp).toHaveText(/Created: .*20\d\d.*\| Updated: .*20\d\d/);

  // A save with no name is refused with a sentence, and the editor stays open.
  await list.getByRole("button", { name: "New", exact: true }).click();
  await detail(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(detail(page).getByRole("alert")).toHaveText("Give the template a name before saving.");

  // Create a template with one variable.
  // Every editor field has its visible label as its accessible name (none had one).
  await detail(page).getByRole("textbox", { name: "Name", exact: true }).fill(NAME);
  await detail(page).getByRole("textbox", { name: "Category", exact: true }).fill("Review");
  await detail(page).getByRole("textbox", { name: "Tags (comma-separated)" }).fill("e2e, review");
  await detail(page).getByRole("textbox", { name: "System Prompt" }).fill("Answer only in {{language}}. Keep quokkas in mind.");
  await detail(page).getByRole("textbox", { name: "Variables (JSON array)", exact: true })
    .fill('[{"name": "language", "type": "string", "default": "French"}]');
  await detail(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(list.locator(".session-button")).toHaveCount(6);
  await expect(detail(page)).toContainText("1 variable");

  // Search reads the prompt text, not only the name and tags.
  await list.getByRole("textbox", { name: "Search templates..." }).fill("quokka");
  await expect(list.locator(".session-button")).toHaveCount(1);
  await expect(list.locator(".session-button")).toContainText(NAME);
  await list.getByRole("textbox", { name: "Search templates..." }).fill("no-such-template-text");
  await expect(list.locator(".session-button")).toHaveCount(0);
  await expect(list).toContainText("No template matches the search.");
  await list.getByRole("textbox", { name: "Search templates..." }).fill("");

  // Use in Chat: fill the variable, see the resolved prompt, apply. Chat opens and the next message carries it.
  await detail(page).getByRole("button", { name: "Use in Chat...", exact: true }).click();
  await detail(page).getByRole("textbox", { name: "language", exact: true }).fill("Welsh");
  await expect(detail(page).locator("pre").last()).toHaveText("Answer only in Welsh. Keep quokkas in mind.");
  await detail(page).getByRole("button", { name: "Apply to chat", exact: true }).click();
  await page.locator(".composer textarea").waitFor();
  await page.locator(".composer textarea").fill("hello");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __promptPayload?: unknown }).__promptPayload));
  const payload = await page.evaluate(() => (window as unknown as { __promptPayload: Record<string, unknown> }).__promptPayload);
  expect(payload.systemPrompt).toBe("Answer only in Welsh. Keep quokkas in mind.");

  // Edit, then delete; the backend agrees.
  await openTab(page, "Tools", "Prompts");
  await list.locator(".session-button", { hasText: NAME }).click();
  await detail(page).getByRole("button", { name: "Edit", exact: true }).click();
  await detail(page).getByRole("textbox", { name: "Name", exact: true }).fill(`${NAME} v2`);
  await detail(page).getByRole("button", { name: "Save", exact: true }).click();
  await expect(list.locator(".session-button", { hasText: `${NAME} v2` })).toHaveCount(1);
  // Delete asks first: dismissing keeps the template, accepting removes it.
  const asked: string[] = [];
  page.once("dialog", (dialog) => { asked.push(dialog.message()); void dialog.dismiss(); });
  await detail(page).getByRole("button", { name: "Delete", exact: true }).click();
  await expect(list.locator(".session-button")).toHaveCount(6);
  expect(asked[0]).toBe(`Delete the template "${NAME} v2"? This cannot be undone.`);
  page.once("dialog", (dialog) => void dialog.accept());
  await detail(page).getByRole("button", { name: "Delete", exact: true }).click();
  await expect(list.locator(".session-button")).toHaveCount(5);
  const after = await (await page.request.get(`${API_BASE}/api/prompts`, { headers: AUTH })).json();
  expect((after.templates as Array<{ name: string }>).some((t) => t.name.startsWith(NAME))).toBe(false);
});

test("Prompts stacks at 375 px with no sideways scroll", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await goTo(page, "Tools", "Prompts");
  await layout(page).locator(".session-button", { hasText: "Summarizer" }).click();
  await expect(detail(page)).toContainText("concise summarization assistant");
  // The detail used to be 65 px wide beside a fixed 300 px list.
  const width = await detail(page).evaluate((el) => el.getBoundingClientRect().width);
  expect(width).toBeGreaterThan(300);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
  await detail(page).getByRole("button", { name: "Use in Chat", exact: true }).scrollIntoViewIfNeeded();
  await expect(detail(page).getByRole("button", { name: "Use in Chat", exact: true })).toBeInViewport();
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS_DIR, "29-prompts-375.png"), fullPage: true });
});

test("a failed load says so on both pages instead of looking empty", async ({ page }) => {
  await page.route("**/api/prompts", (route) => route.fulfill({ status: 500, json: { detail: "boom" } }));
  await page.route("**/api/plugins", (route) => route.fulfill({ status: 500, json: { detail: "boom" } }));
  await goTo(page, "Tools", "Prompts");
  await expect(layout(page).getByRole("alert")).toHaveText("The templates could not be loaded. Check that the backend is running.");
  await openTab(page, "Tools", "Plugins");
  const pluginPage = page.locator(".plugins-page");
  await expect(pluginPage.getByRole("alert")).toHaveText("The plugin list could not be loaded. Check that the backend is running.");
  await expect(pluginPage).not.toContainText("No plugins found.");
});

test("Plugins: lists what the backend registers, every row is reachable, a switch flips", async ({ page }) => {
  await goTo(page, "Tools", "Plugins");
  const registered = (await (await page.request.get(`${API_BASE}/api/plugins`, { headers: AUTH })).json()) as Record<
    string,
    Array<{ id: string; name: string }>
  >;
  const rows = Object.values(registered).flat();
  expect(rows.length).toBeGreaterThan(0);

  const pluginPage = page.locator(".plugins-page");
  await expect(pluginPage).not.toContainText("No plugins found.");
  await expect(pluginPage).toContainText(`${rows.length} plugins registered`);
  await expect(pluginPage.getByRole("button", { name: /^(ON|OFF)$/ })).toHaveCount(rows.length);

  // The last row (a tool) used to be cut off by a panel that could not scroll.
  const last = pluginPage.getByText(rows[rows.length - 1].name, { exact: true });
  await last.scrollIntoViewIfNeeded();
  await expect(last).toBeInViewport();

  // A switch flips and the backend agrees; flipped back at the end so later specs see the defaults.
  const first = pluginPage.getByRole("button", { name: /^(ON|OFF)$/ }).first();
  await expect(first).toHaveText("ON");
  await first.click();
  await expect(first).toHaveText("OFF");
  const off = (await (await page.request.get(`${API_BASE}/api/plugins`, { headers: AUTH })).json()) as Record<
    string,
    Array<{ id: string; enabled: boolean }>
  >;
  expect(Object.values(off).flat().find((row) => row.id === rows[0].id)?.enabled).toBe(false);
  await first.click();
  await expect(first).toHaveText("ON");
});

test("Plugins fits a 375 px window", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await goTo(page, "Tools", "Plugins");
  const pluginPage = page.locator(".plugins-page");
  await expect(pluginPage.getByRole("button", { name: /^(ON|OFF)$/ }).first()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(375);
  expect(await pluginPage.evaluate((el) => el.scrollWidth - el.clientWidth)).toBeLessThanOrEqual(1);
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.screenshot({ path: path.join(SHOTS_DIR, "29-plugins-375.png"), fullPage: true });
});
