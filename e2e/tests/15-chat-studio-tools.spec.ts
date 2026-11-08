import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { goTo, openTab, SHOTS_DIR } from "./helpers";

// D-60: studio tools in Chat. No model runs: the workspace says a tool-capable model is loaded, the chat
// stream is a scripted stream the test feeds event by event (a page-side fetch wrapper), and the plan
// decision and job routes are mocked. Covers: the Studio switch (off by default, remembered), the plan
// card for a 2-image plan, Approve all (decision body, job cards with the finished image) and Deny
// (declined results, nothing made).

const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

const PLAN = {
  planId: "plan-e2e",
  items: [
    { callId: "c1", tool: "generate_image", kind: "image", prompt: "Rose shop poster", model: "FLUX.1-schnell", size: "1024x1536", estimatedSeconds: 40, peakGb: 12 },
    { callId: "c2", tool: "generate_image", kind: "image", prompt: "Tulip shop poster", model: "FLUX.1-schnell", size: "1024x1536", estimatedSeconds: 40, peakGb: 12 },
  ],
  memory: { chatModel: "Mock Chat 8B", freeGb: 30, totalGb: 64, level: "ok", advice: null },
  status: "pending",
};

type Workspace = { runtime: Record<string, unknown> };

/** The page's fetch for the chat stream returns a stream this test writes to. */
async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      w.__studioPayload = JSON.parse(String(init?.body ?? "{}"));
      let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
      const body = new ReadableStream<Uint8Array>({ start: (c) => { controller = c; } });
      w.__studioPush = (event: unknown) => controller?.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      w.__studioClose = () => controller?.close();
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}

async function push(page: Page, event: unknown) {
  await page.evaluate((e) => (window as unknown as { __studioPush: (x: unknown) => void }).__studioPush(e), event);
}

/** A tool-capable chat model looks loaded (the workspace answer is patched; nothing is loaded). */
async function mockLoadedModel(page: Page): Promise<{ runtime: () => Record<string, unknown> }> {
  let runtime: Record<string, unknown> = {};
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as Workspace;
    body.runtime = {
      ...body.runtime,
      state: "loaded",
      loadedModel: {
        ref: "mock/chat-8b", name: "Mock Chat 8B", canonicalRepo: "mock/chat-8b", backend: "mlx", source: "library",
        engine: "mlx", cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true,
        contextTokens: 8192, loadedAt: "2026-11-08 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
      },
    };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  // A new thread asks for the default chat model first; answer as if it were the loaded one.
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
  return { runtime: () => runtime };
}

async function mockJobs(page: Page) {
  const decisions: unknown[] = [];
  await page.route("**/api/chat/studio/plans/*/decision", async (route) => {
    decisions.push(route.request().postDataJSON());
    await route.fulfill({ json: { planId: PLAN.planId } });
  });
  await page.route(/\/api\/chat\/studio\/jobs\/job-\d+$/, (route) => {
    const id = new URL(route.request().url()).pathname.split("/").pop();
    return route.fulfill({ json: { job: { jobId: id, kind: "image", label: "FLUX.1-schnell", state: "done", output: { path: `/mock/${id}.png`, width: 1024, height: 1536 } } } });
  });
  await page.route(/\/api\/chat\/studio\/jobs\/job-\d+\/media/, (route) => route.fulfill({ body: PNG_1X1, contentType: "image/png" }));
  return decisions;
}

async function startStudioTurn(page: Page) {
  await scriptChatStream(page);
  const workspace = await mockLoadedModel(page);
  const decisions = await mockJobs(page);
  await goTo(page, "Chat", "Chat");
  const studio = page.getByRole("switch", { name: "Studio" });
  await expect(studio).toHaveAttribute("aria-checked", "false");
  await studio.click();
  await expect(page.getByRole("switch", { name: "Studio ON" })).toHaveAttribute("aria-checked", "true");
  await page.locator(".composer textarea").fill("Make two posters for a flower shop");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __studioPush?: unknown }).__studioPush));
  const payload = await page.evaluate(() => (window as unknown as { __studioPayload: Record<string, unknown> }).__studioPayload);
  expect(payload.studioTools).toBe(true);
  expect(payload.studioAutoApprove).toBeUndefined();
  await push(page, { phase: "prompt_eval" });
  await push(page, { phase: "generating", ttftSeconds: 1.2 });
  await push(page, { studioPlan: PLAN });
  const card = page.getByRole("region", { name: "Plan to approve" });
  await expect(card).toBeVisible();
  await expect(card).toContainText("Rose shop poster");
  await expect(card).toContainText("Tulip shop poster");
  await expect(card).toContainText("about 40 s");
  await expect(card).toContainText("Mock Chat 8B stays loaded: about 30 GB free.");
  await expect(card.getByRole("checkbox", { name: "Always allow renders in this chat" })).not.toBeChecked();
  return { card, decisions, payload, workspace };
}

function doneEvent(sessionId: string, runtime: Record<string, unknown>, assistant: Record<string, unknown>) {
  return {
    done: true,
    session: {
      id: sessionId, title: "Flower posters", updatedAt: "2026-11-08 10:01:00", model: "Mock Chat 8B", cacheLabel: "Native f16",
      messages: [{ role: "user", text: "Make two posters for a flower shop" }, { role: "assistant", ...assistant }],
    },
    assistant: { role: "assistant", ...assistant },
    runtime,
  };
}

test("Studio switch: approve a 2-image plan, the jobs run and show their images", async ({ page }) => {
  const { card, decisions, payload, workspace } = await startStudioTurn(page);
  mkdirSync(SHOTS_DIR, { recursive: true });
  await card.screenshot({ path: path.join(SHOTS_DIR, "chat-studio-plan.png") });
  await expect(card.getByRole("button", { name: "Approve only this Image: Tulip shop poster" })).toBeVisible();
  await card.getByRole("checkbox", { name: "Always allow renders in this chat" }).check();
  await card.getByRole("button", { name: "Approve all" }).click();
  await expect(card.getByRole("status")).toContainText("Sent");
  expect(decisions).toEqual([{ decisions: { c1: "approve", c2: "approve" }, edit: false, alwaysAllow: true }]);
  // Always allow is visible and can be turned off.
  const chip = page.locator(".composer-studio-auto");
  await expect(chip).toContainText("Renders auto-approved");
  await chip.getByRole("button", { name: "Turn off" }).click();
  await expect(chip).toHaveCount(0);

  const resolved = { ...PLAN, status: "decided", decisions: { c1: "approve", c2: "approve" }, alwaysAllow: false };
  const results = ["c1", "c2"].map((id, i) => ({
    id, name: "generate_image", arguments: { prompt: PLAN.items[i].prompt }, elapsed: 0.1, renderAs: "studio",
    result: JSON.stringify({ jobId: `job-${i + 1}`, state: "queued" }), data: { tool: "generate_image", jobId: `job-${i + 1}`, label: "FLUX.1-schnell" },
  }));
  await push(page, { studioPlanResolved: resolved });
  for (const r of results) await push(page, { toolCallStart: { id: r.id, name: r.name } }), await push(page, { toolCallResult: r });
  await expect(card).toContainText("Approved 2 of 2.");
  const jobs = page.locator(".studio-job-card");
  await expect(jobs).toHaveCount(2);
  await expect(jobs.first().getByRole("status")).toHaveText("Done");
  await expect(jobs.first().locator("img")).toBeVisible();
  await expect(jobs.first().getByRole("button", { name: "Open in Library" })).toBeVisible();
  await page.locator(".chat-thread, .message-list, main").first().screenshot({ path: path.join(SHOTS_DIR, "chat-studio-jobs.png") });

  await push(page, doneEvent(String(payload.sessionId), workspace.runtime(), {
    text: "Both posters are ready.", toolCalls: results, studioPlans: [resolved],
  }));
  await expect(page.getByText("Both posters are ready.")).toBeVisible();
  await expect(page.locator(".studio-job-card")).toHaveCount(2);

  // The switch is remembered.
  await page.reload();
  await page.locator("aside.sidebar").waitFor();
  await openTab(page, "Chat", "Chat");
  await expect(page.getByRole("switch", { name: "Studio ON" })).toHaveAttribute("aria-checked", "true");
});

test("Studio switch: deny the plan, the model gets the declined result and nothing is made", async ({ page }) => {
  const { card, decisions, payload, workspace } = await startStudioTurn(page);
  await card.getByRole("button", { name: "Deny" }).click();
  expect(decisions).toEqual([{ decisions: { c1: "deny", c2: "deny" }, edit: false, alwaysAllow: false }]);
  const declined = "Declined: the user did not approve this call, so nothing was made. Do not call it again; ask the user what to change.";
  const resolved = { ...PLAN, status: "decided", decisions: { c1: "deny", c2: "deny" }, alwaysAllow: false };
  const results = ["c1", "c2"].map((id, i) => ({
    id, name: "generate_image", arguments: { prompt: PLAN.items[i].prompt }, elapsed: 0, renderAs: "studio", result: declined,
    data: { tool: "generate_image", declined: true, message: declined },
  }));
  await push(page, { studioPlanResolved: resolved });
  for (const r of results) await push(page, { toolCallResult: r });
  await expect(card).toContainText("Approved 0 of 2.");
  await expect(page.locator(".studio-job-card--declined")).toHaveCount(2);
  await expect(page.locator(".studio-job-card--declined").first()).toContainText("Declined. Nothing was made.");
  await push(page, doneEvent(String(payload.sessionId), workspace.runtime(), {
    text: "Understood, nothing was made. What should I change?", toolCalls: results, studioPlans: [resolved],
  }));
  await expect(page.getByText("What should I change?")).toBeVisible();
  await expect(page.locator(".studio-job-card img")).toHaveCount(0);
});

test("plan card fits a 375 px window", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const { card } = await startStudioTurn(page);
  // The Chat thread pane is very short at phone width (a layout issue of the Chat tab, not of the card), so
  // its scroll boxes are opened up for the picture; the width checks below do not depend on that.
  await card.evaluate((el) => {
    for (let node = el.parentElement; node && node !== document.body; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (/(auto|scroll|hidden)/.test(style.overflowY)) Object.assign(node.style, { height: "auto", maxHeight: "none", overflow: "visible" });
    }
  });
  await card.scrollIntoViewIfNeeded();
  mkdirSync(SHOTS_DIR, { recursive: true });
  await card.screenshot({ path: path.join(SHOTS_DIR, "chat-studio-plan-375.png") });
  const overflow = await card.evaluate((el) => {
    const box = el.getBoundingClientRect();
    const wide = [...el.querySelectorAll("*")].filter((c) => c.getBoundingClientRect().right > box.right + 1).length;
    return { wide, cardRight: box.right, viewport: window.innerWidth, scroll: el.scrollWidth - el.clientWidth };
  });
  expect(overflow.wide).toBe(0);
  expect(overflow.scroll).toBeLessThanOrEqual(0);
  expect(overflow.cardRight).toBeLessThanOrEqual(overflow.viewport);
});
