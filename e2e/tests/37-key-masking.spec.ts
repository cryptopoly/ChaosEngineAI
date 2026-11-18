import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { API_BASE, openApp, primaryNav, TOKEN } from "./helpers";

// API keys on screen (overnight review, stream F): the Server page's key field, its Connect samples and Test
// commands, and Settings > Integrations show the key masked (dots and at most the last 4 characters) until
// Reveal; Copy always puts the real key on the clipboard.

const MASKED = `••••••••${TOKEN.slice(-4)}`;

async function stubClipboard(page: Page) {
  await page.addInitScript(() => {
    (window as unknown as { __copied: string[] }).__copied = [];
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          (window as unknown as { __copied: string[] }).__copied.push(text);
        },
      },
    });
  });
}

const lastCopied = (page: Page) => page.evaluate(() => (window as unknown as { __copied: string[] }).__copied.at(-1) ?? "");

async function openServer(page: Page) {
  await (await primaryNav(page)).locator(".nav-button", { hasText: /^Server/ }).click();
  await expect(page.locator(".server-api-token-row")).toBeVisible();
}

test("Server: key field, Connect samples and Test commands are masked; Copy copies the real key", async ({ page }) => {
  await stubClipboard(page);
  await openApp(page);
  await openServer(page);

  const keyRow = page.locator(".server-api-token-row");
  await expect(keyRow).toContainText(MASKED);
  await expect(keyRow).not.toContainText(TOKEN);

  const connect = page.getByTestId("server-connect");
  await connect.getByRole("button", { name: "Connect your app" }).click();
  await expect(connect.locator("pre").first()).toBeVisible();
  await expect(connect).not.toContainText(TOKEN);
  await expect(connect).toContainText(MASKED);
  await connect.locator(".server-command").first().getByRole("button", { name: "Copy" }).click();
  expect(await lastCopied(page)).toContain(`api_key="${TOKEN}"`);
  await connect.getByTestId("reveal-secret").click();
  await expect(connect).toContainText(TOKEN);
  await connect.getByTestId("reveal-secret").click();
  await expect(connect).not.toContainText(TOKEN);

  await page.getByRole("button", { name: "Test", exact: true }).click();
  const dialog = page.locator(".modal-content", { hasText: "API Test Commands" });
  await expect(dialog).toBeVisible();
  await expect(dialog).not.toContainText(TOKEN);
  await expect(dialog).toContainText(MASKED);
  await dialog.locator(".server-command", { hasText: "List models" }).getByRole("button", { name: "Copy" }).click();
  expect(await lastCopied(page)).toContain(`Bearer ${TOKEN}`);
  await dialog.getByTestId("reveal-secret").click();
  await expect(dialog).toContainText(`Bearer ${TOKEN}`);
});

test("Settings > Integrations: snippets are masked until Reveal; Copy copies the real key", async ({ page }) => {
  await stubClipboard(page);
  await openApp(page);
  await (await primaryNav(page)).getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("tab", { name: "Integrations" }).click();
  const panel = page.locator(".panel", { hasText: "Connect external tools" }).last();
  await expect(panel.locator("pre").first()).toBeVisible();
  await expect(panel).not.toContainText(TOKEN);
  await expect(panel).toContainText(MASKED);
  await panel.locator(".integration-card").first().getByRole("button", { name: "Copy" }).click();
  expect(await lastCopied(page)).toContain(TOKEN);
  await panel.getByTestId("reveal-secret").click();
  await expect(panel).toContainText(TOKEN);

  // Agent access: the bridge token in its snippets is masked the same way.
  const access = (await (await page.request.get(`${API_BASE}/api/agent/access`, { headers: { Authorization: `Bearer ${TOKEN}` } })).json()) as {
    token: string;
  };
  expect(access.token.length).toBeGreaterThan(8);
  const agent = page.locator(".panel", { hasText: "Agent access" }).last();
  await expect(agent.locator("pre").first()).toBeVisible();
  await expect(agent).not.toContainText(access.token);
  await expect(agent).toContainText(`••••••••${access.token.slice(-4)}`);
  await agent.locator(".integration-card").first().getByRole("button", { name: "Copy" }).click();
  expect(await lastCopied(page)).toContain(access.token);
  await agent.getByTestId("reveal-secret").click();
  await expect(agent).toContainText(access.token);
});
