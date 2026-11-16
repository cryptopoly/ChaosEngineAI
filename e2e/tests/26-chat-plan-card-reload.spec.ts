import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { goTo, openTab, SHOTS_DIR } from "./helpers";

// S-J (v0.10.0 item 1): Chat's plan card. No model runs: the workspace says a tool-capable model is loaded,
// the chat stream is a scripted stream the test feeds, and the turn, decision and workspace answers are mocked.
// Covers: a running thread is named by its first message; a refused first call shows its reason in a notice
// card; a plan card that waits survives a window reload, is answered there, and the reply then shows.

const PLAN = {
  planId: "plan-reload",
  sessionId: "session-reload",
  items: [{ callId: "c1", tool: "generate_image", kind: "image", prompt: "Rose shop poster", model: "FLUX.1-schnell", size: "1024x1536", estimatedSeconds: 40, peakGb: 12 }],
  memory: { chatModel: "Mock Chat 8B", freeGb: 30, totalGb: 64, level: "ok", advice: null },
  status: "pending",
};
const REASON = "Invalid input for storyboard_generate: 'what' must be one of still, clip, both, final, redo-clip, redo-still.";

const LOADED = {
  state: "loaded",
  loadedModel: {
    ref: "mock/chat-8b", name: "Mock Chat 8B", canonicalRepo: "mock/chat-8b", backend: "mlx", source: "library",
    engine: "mlx", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
    contextTokens: 8192, loadedAt: "2026-11-16 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
    speculativeDecoding: false, treeBudget: 0,
    capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
  },
};

type Workspace = { runtime: Record<string, unknown>; chatSessions: Record<string, unknown>[] };

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
      const body = new ReadableStream<Uint8Array>({ start: (c) => { controller = c; } });
      w.__studioPush = (event: unknown) => controller?.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}

async function push(page: Page, event: unknown) {
  await page.evaluate((e) => (window as unknown as { __studioPush: (x: unknown) => void }).__studioPush(e), event);
}

/** The workspace says a model is loaded; `extra` puts sessions in front of the backend's own. */
async function mockWorkspace(page: Page, extra: () => Record<string, unknown>[] = () => []) {
  let runtime: Record<string, unknown> = {};
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as Workspace;
    body.runtime = { ...body.runtime, ...LOADED };
    body.chatSessions = [...extra(), ...body.chatSessions];
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
}

test("a running thread has its first message as title; a refused call shows its reason", async ({ page }) => {
  await scriptChatStream(page);
  await mockWorkspace(page);
  await goTo(page, "Chat", "Chat");
  await page.getByRole("switch", { name: "Studio" }).click();
  await page.locator(".composer textarea").fill("Make the stills for my lighthouse film");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __studioPush?: unknown }).__studioPush));
  // Still running: the thread list already names it.
  await expect(page.locator(".session-button.active")).toContainText("Make the stills for my lighthouse");
  await expect(page.locator(".session-button.active")).not.toContainText("New chat");
  await push(page, { phase: "generating", ttftSeconds: 0.5 });
  await push(page, { toolCallStart: { id: "c1", name: "storyboard_generate", arguments: { project_id: "p1", what: "stills" } } });
  await push(page, {
    toolCallResult: {
      id: "c1", name: "storyboard_generate", arguments: { project_id: "p1", what: "stills" }, elapsed: 0, renderAs: "studio",
      result: `Error: ${REASON}`, data: { tool: "storyboard_generate", error: REASON },
    },
  });
  const notice = page.locator(".studio-job-card--refused");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("Not started");
  await expect(notice).toContainText("'what' must be one of still");
  await expect(notice).toContainText("The model was told why and can fix the call.");
  mkdirSync(SHOTS_DIR, { recursive: true });
  await notice.screenshot({ path: path.join(SHOTS_DIR, "chat-refused-call.png") });
});

test("a waiting plan card survives a reload and is answered there", async ({ page }) => {
  const state = { decided: false, finished: false };
  const decisions: unknown[] = [];
  const session = () => ({
    id: "session-reload", title: "Paint a rose poster for the shop", updatedAt: "2026-11-16 10:00:00", pinned: false,
    model: "Mock Chat 8B", modelRef: "mock/chat-8b", thinkingMode: "off", cacheLabel: "Native f16",
    messages: [
      { role: "user", text: "Paint a rose poster for the shop", metrics: null, createdAt: Date.now() - 60_000 },
      ...(state.finished ? [{ role: "assistant", text: "The rose poster is queued.", metrics: null }] : []),
    ],
  });
  await mockWorkspace(page, () => [session()]);
  await page.route("**/api/chat/studio/sessions/session-reload/turn", (route) => {
    // After the click the turn runs on for one poll, then the reply is saved.
    const running = !state.finished;
    if (state.decided) state.finished = true;
    return route.fulfill({ json: { sessionId: "session-reload", running, plans: state.decided ? [] : [PLAN] } });
  });
  await page.route("**/api/chat/studio/plans/*/decision", async (route) => {
    decisions.push(route.request().postDataJSON());
    state.decided = true;
    await route.fulfill({ json: { planId: PLAN.planId } });
  });

  await goTo(page, "Chat", "Chat");
  const detached = page.getByTestId("detached-turn");
  await expect(detached).toContainText("waiting for your answer");
  const card = detached.getByRole("region", { name: "Plan to approve" });
  await expect(card).toContainText("Rose shop poster");
  await expect(page.locator(".session-button.active")).toContainText("Paint a rose poster for the shop");
  // A second reload: the card is still there.
  await page.reload();
  await page.locator("aside.sidebar").waitFor();
  await openTab(page, "Chat", "Chat");
  await expect(page.getByTestId("detached-turn").getByRole("region", { name: "Plan to approve" })).toBeVisible();
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.getByTestId("detached-turn").screenshot({ path: path.join(SHOTS_DIR, "chat-plan-after-reload.png") });
  await page.getByTestId("detached-turn").getByRole("button", { name: "Approve", exact: true }).click();
  await expect.poll(() => decisions).toEqual([{ decisions: { c1: "approve" }, edit: false, alwaysAllow: false }]);
  await expect(page.getByText("The rose poster is queued.")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId("detached-turn")).toHaveCount(0);
});
