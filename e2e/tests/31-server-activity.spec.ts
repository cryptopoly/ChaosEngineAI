import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { openApp, primaryNav, SHOTS_DIR } from "./helpers";

// Server page redesign (D-88): the connect block is one closed line that remembers being opened, the page has
// Overview and Activity views, Activity draws a timeline per model and a filterable event list with a detail row,
// and the Logs sidebar entry is gone (its lines show under Activity). The event stream is mocked with a real-shaped
// model life (load, answer, swap, a failed load); nothing is loaded.

const NOW = Date.now() / 1000;
let id = 0;
function event(kind: string, offset: number, extra: Record<string, unknown> = {}) {
  id += 1;
  const ts = NOW - 600 + offset;
  return {
    id,
    ts,
    time: new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19),
    kind,
    level: "info",
    model: "Qwen3 0.6B",
    modelRef: "mlx-community/Qwen3-0.6B-4bit",
    requestId: "a1b2c3d4e5",
    engine: "mlx",
    cacheStrategy: "native",
    speculative: null,
    message: kind,
    data: {},
    detail: {},
    ...extra,
  };
}

const EVENTS = [
  event("requested", 0, { data: { backend: "auto" } }),
  event("resolving", 0.2),
  event("loading", 0.5, { data: { bytes: 350_000_000 } }),
  event("loaded", 6.5, { data: { seconds: 6, memoryBytes: 420_000_000, warm: false }, detail: { command: "python -m backend_service.mlx_worker", env: ["CHAOSENGINE_PORT=8904"] } }),
  event("first_token", 40.3, { requestId: "f00dfeed01", data: { prefillSeconds: 0.3 } }),
  event("generation", 42, { requestId: "f00dfeed01", data: { seconds: 2, completionTokens: 96, tokS: 61.2, promptTokens: 12 } }),
  event("idle", 42, { requestId: "f00dfeed01" }),
  event("parked", 100),
  event("requested", 100, { model: "Broken GGUF", modelRef: "x/broken", requestId: "deadbeef00", engine: null }),
  event("load_failed", 103, {
    model: "Broken GGUF",
    modelRef: "x/broken",
    requestId: "deadbeef00",
    engine: "llama.cpp",
    level: "error",
    message: "llama-server exited: failed to load model",
    data: { seconds: 3, reason: "error" },
    detail: {
      command: "llama-server -m ~/Models/broken.gguf --port 8960 --api-key ***",
      stderr: ["gguf_init_from_file: invalid magic", "llama_model_load: error loading model"],
      env: ["CHAOSENGINE_PORT=8904"],
      error: "RuntimeError: llama-server exited: failed to load model",
    },
  }),
];

async function mockEvents(page: Page) {
  const body = EVENTS.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("") + ": keepalive\n\n";
  await page.route("**/api/server/events/stream", (route) =>
    route.fulfill({ status: 200, headers: { "content-type": "text/event-stream", "access-control-allow-origin": "*" }, body }),
  );
}

async function openServer(page: Page) {
  await (await primaryNav(page)).locator(".nav-button", { hasText: /^Server/ }).click();
  await expect(page.getByRole("tablist", { name: "Server views" })).toBeVisible();
}

test.describe("Server page and Activity", () => {
  test.beforeEach(async ({ page }) => {
    await mockEvents(page);
    await openApp(page);
    await page.evaluate(() => {
      window.localStorage.removeItem("chaosengine.server.connectOpen");
      window.localStorage.removeItem("chaosengine.server.view");
    });
  });

  test("the Logs entry is gone from the sidebar", async ({ page }) => {
    const nav = await primaryNav(page);
    await expect(nav.locator(".nav-button", { hasText: /^Server/ })).toBeVisible();
    await expect(nav.locator(".nav-button", { hasText: /^Logs$/ })).toHaveCount(0);
  });

  test("connect block is one closed line, and remembers being opened", async ({ page }) => {
    await openServer(page);
    const connect = page.getByTestId("server-connect");
    const toggle = connect.getByRole("button", { name: "Connect your app" });
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await expect(connect.locator(".server-connect-url")).toContainText("/v1");
    await expect(connect.getByRole("button", { name: "Copy" })).toBeVisible();
    await expect(connect.getByText("from openai import OpenAI")).toHaveCount(0);

    await toggle.click();
    await expect(connect.getByText(/from openai import OpenAI/)).toBeVisible();
    await page.reload();
    await openServer(page);
    await expect(page.getByTestId("server-connect").getByRole("button", { name: "Connect your app" })).toHaveAttribute("aria-expanded", "true");
    await page.getByTestId("server-connect").getByRole("button", { name: "Connect your app" }).click();
    await expect(page.getByTestId("server-connect").getByText(/from openai import OpenAI/)).toHaveCount(0);
  });

  test("Overview shows recent activity and opens Activity", async ({ page }) => {
    await openServer(page);
    const recent = page.getByRole("region", { name: "Recent activity" });
    await expect(recent.locator("li.level-error")).toContainText("Broken GGUF");
    await recent.getByRole("button", { name: "Open Activity" }).click();
    await expect(page.getByRole("tab", { name: "Activity" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByTestId("server-activity")).toBeVisible();
  });

  test("Activity: timeline, filters, detail, latest failure, selection, pause", async ({ page }) => {
    await openServer(page);
    await page.getByRole("tab", { name: "Activity" }).click();
    const activity = page.getByTestId("server-activity");

    // Timeline: one track per model, the failed load drawn as failed.
    await activity.getByRole("button", { name: "All", exact: true }).click();
    const timeline = activity.getByRole("region", { name: "Model timeline" });
    await expect(timeline.locator(".activity-timeline-row")).toHaveCount(2);
    await expect(timeline.locator(".activity-segment.phase-failed")).toHaveCount(1);
    await expect(timeline.locator(".activity-segment.phase-generating")).toHaveCount(1);
    await expect(timeline.locator(".activity-segment.phase-warm")).toHaveCount(1);

    // Event list, newest first.
    const rows = activity.locator(".activity-event");
    await expect(rows).toHaveCount(EVENTS.length);
    await expect(rows.first()).toHaveAttribute("data-kind", "load_failed");

    // Filters: level, model, request id, text.
    await activity.getByLabel("Level").selectOption("error");
    await expect(rows).toHaveCount(1);
    await activity.getByLabel("Level").selectOption("");
    await activity.getByLabel("Model", { exact: true }).selectOption("Qwen3 0.6B");
    await expect(rows).toHaveCount(EVENTS.length - 2);
    await activity.getByRole("searchbox", { name: "Request id" }).fill("f00d");
    await expect(rows).toHaveCount(3);
    await activity.getByRole("searchbox", { name: "Request id" }).fill("");
    await activity.getByLabel("Model", { exact: true }).selectOption("");

    // Latest failure opens its detail: command line, engine output, environment.
    await activity.getByRole("button", { name: "Latest failure" }).click();
    const detail = activity.getByTestId("activity-detail");
    await expect(detail).toBeVisible();
    await expect(detail).toContainText("llama-server -m ~/Models/broken.gguf");
    await expect(detail).toContainText("invalid magic");
    await expect(detail).toContainText("CHAOSENGINE_PORT=8904");

    // Selection drives Copy and Export.
    await expect(activity.getByRole("button", { name: "Copy selected (0)" })).toBeDisabled();
    await rows.first().getByRole("checkbox").check();
    await expect(activity.getByRole("button", { name: "Copy selected (1)" })).toBeEnabled();
    const download = page.waitForEvent("download");
    await activity.getByRole("button", { name: "Export selected (1)" }).click();
    expect((await download).suggestedFilename()).toMatch(/^model-events-\d{8}-\d{4}\.json$/);

    // Pause and resume the live tail.
    await activity.getByRole("button", { name: "Pause" }).click();
    await expect(activity.getByRole("status").filter({ hasText: "Paused" })).toBeVisible();
    await activity.getByRole("button", { name: "Resume" }).click();

    // The old log lines are still here.
    await expect(activity.getByRole("region", { name: "Log lines" })).toBeVisible();

    await page.screenshot({ path: `${SHOTS_DIR}/31-server-activity.png`, fullPage: true });
  });

  test("the chosen view is remembered, and the view tabs answer the arrow keys", async ({ page }) => {
    await openServer(page);
    await page.getByRole("tab", { name: "Overview" }).focus();
    await page.keyboard.press("ArrowRight");
    await expect(page.getByRole("tab", { name: "Activity" })).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("tab", { name: "Activity" })).toBeFocused();
    await page.reload();
    await openServer(page);
    await expect(page.getByRole("tab", { name: "Activity" })).toHaveAttribute("aria-selected", "true");
  });

  test("Activity fits a narrow window", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 800 });
    await openServer(page);
    await page.getByRole("tab", { name: "Activity" }).click();
    await expect(page.getByTestId("server-activity").locator(".activity-event").first()).toBeVisible();
    const overflow = await page.evaluate(() => document.scrollingElement!.scrollWidth - document.scrollingElement!.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
    await page.screenshot({ path: `${SHOTS_DIR}/31-server-activity-375.png`, fullPage: true });
  });
});
