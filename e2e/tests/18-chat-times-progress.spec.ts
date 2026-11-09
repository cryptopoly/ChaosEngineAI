import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { goTo, SHOTS_DIR } from "./helpers";

// Times and long-job progress in Chat. No model runs: the workspace says a tool-capable model is loaded
// and the chat stream is a scripted stream the test feeds event by event (as in spec 15). Covers: a running
// wait_for_job card with its start time and a ticking elapsed time, the progress line ("clip 2 of 4", time
// left from the clips made so far), 24 polls of one job shown as ONE card, the send time on messages (full
// date on hover, date when not today) and an old message without a time showing none.

type Workspace = { runtime: Record<string, unknown> };

async function scriptChatStream(page: Page) {
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    const encoder = new TextEncoder();
    const w = window as unknown as Record<string, unknown>;
    window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (!url.includes("/api/chat/generate/stream")) return original(input, init);
      w.__payload = JSON.parse(String(init?.body ?? "{}"));
      let controller: ReadableStreamDefaultController<Uint8Array> | null = null;
      const body = new ReadableStream<Uint8Array>({ start: (c) => { controller = c; } });
      w.__push = (event: unknown) => controller?.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } }));
    };
  });
}

async function push(page: Page, event: unknown) {
  await page.evaluate((e) => (window as unknown as { __push: (x: unknown) => void }).__push(e), event);
}

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
        contextTokens: 8192, loadedAt: "2026-11-09 10:00:00", path: "/mock/chat-8b", runtimeTarget: "/mock/chat-8b",
        speculativeDecoding: false, treeBudget: 0,
        capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: false },
      },
    };
    runtime = body.runtime;
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", (route) => route.fulfill({ json: { runtime } }));
  return { runtime: () => runtime };
}

test("a long wait shows start time, ticking elapsed time and the clip progress; 24 polls are one card; messages carry times", async ({ page }) => {
  await scriptChatStream(page);
  const workspace = await mockLoadedModel(page);
  await goTo(page, "Chat", "Chat");
  await page.getByRole("switch", { name: "Studio" }).click();
  await page.locator(".composer textarea").fill("Make a 4 shot film");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await page.waitForFunction(() => Boolean((window as unknown as { __push?: unknown }).__push));
  const payload = await page.evaluate(() => (window as unknown as { __payload: Record<string, unknown> }).__payload);
  await push(page, { phase: "prompt_eval" });
  await push(page, { phase: "generating", ttftSeconds: 1.2 });

  // The user's message got its send time at once.
  const userTime = page.locator(".message-bubble.user time.message-time");
  await expect(userTime).toHaveCount(1);
  await expect(userTime).toHaveText(/\d{1,2}:\d{2}/);
  expect(await userTime.getAttribute("title")).toMatch(/20\d\d/);

  // A running wait_for_job: started 4 min 12 s ago, clip 2 of 4, time left from clip 1 (369 s).
  const startedAt = Date.now() - 252_000;
  await push(page, { toolCallStart: { id: "w0", name: "wait_for_job", arguments: { job_id: "job-11" }, startedAt } });
  await push(page, {
    toolCallProgress: {
      id: "w0", name: "wait_for_job",
      data: {
        tool: "wait_for_job", waiting: true, jobId: "job-11", state: "running",
        progress: { kind: "clip", kindIndex: 2, kindTotal: 4, step: 5, totalSteps: 8, elapsedSeconds: 432, etaSeconds: 738, etaBasis: "clips" },
      },
    },
  });
  const timing = page.getByTestId("tool-call-timing").first();
  await expect(timing).toContainText(/started \d{1,2}:\d{2}/);
  await expect(timing).toContainText("4 min");
  const first = await timing.innerText();
  await expect.poll(async () => timing.innerText(), { timeout: 5000 }).not.toBe(first); // it ticks while running
  const line = page.getByTestId("job-progress-line").first();
  await expect(line).toContainText("clip 2 of 4");
  await expect(line).toContainText("elapsed");
  await expect(line).toContainText("left (from the clips made so far)");
  mkdirSync(SHOTS_DIR, { recursive: true });
  await page.locator(".message-bubble.assistant").last().screenshot({ path: path.join(SHOTS_DIR, "chat-wait-progress.png") });

  // 24 polls of the same job: one card.
  for (let i = 0; i < 24; i++) {
    const id = i === 0 ? "w0" : `w${i}`;
    if (i > 0) await push(page, { toolCallStart: { id, name: "wait_for_job", arguments: { job_id: "job-11" }, startedAt: startedAt + i * 60_000 } });
    await push(page, {
      toolCallResult: {
        id, name: "wait_for_job", arguments: { job_id: "job-11" }, elapsed: 60, startedAt: startedAt + i * 60_000, renderAs: "studio",
        result: "{}", data: { tool: "wait_for_job", jobId: "job-11", state: i === 23 ? "done" : "running" },
      },
    });
  }
  await expect(page.getByTestId("tool-call-timing")).toHaveCount(1);
  await expect(page.getByTestId("tool-call-timing")).toContainText("24 checks");

  // The finished turn: assistant message with its time; an older message without one shows none.
  const yesterday = Date.now() - 26 * 3600_000;
  const assistant = {
    text: "The film is done.", createdAt: Date.now(),
    toolCalls: [{ id: "w0", name: "wait_for_job", arguments: { job_id: "job-11" }, elapsed: 1440, startedAt, result: "{}", renderAs: "studio", data: { tool: "wait_for_job", jobId: "job-11", state: "done" } }],
  };
  await push(page, {
    done: true,
    session: {
      id: String(payload.sessionId), title: "Film", updatedAt: "2026-11-09 10:01:00", model: "Mock Chat 8B", cacheLabel: "Native f16",
      messages: [
        { role: "user", text: "An old question", createdAt: yesterday },
        { role: "assistant", text: "An old answer from before times existed" },
        { role: "user", text: "Make a 4 shot film", createdAt: Date.now() - 1500_000 },
        { role: "assistant", ...assistant },
      ],
    },
    assistant: { role: "assistant", ...assistant },
    runtime: workspace.runtime(),
  });
  await expect(page.getByText("The film is done.")).toBeVisible();
  const times = page.locator("time.message-time");
  await expect(times).toHaveCount(3); // the old answer has none
  const oldTime = await times.first().innerText();
  expect(oldTime).toMatch(/[A-Za-z]/); // not today: the date is in front of the clock
  await expect(page.locator(".message-bubble").filter({ hasText: "An old answer" }).locator("time.message-time")).toHaveCount(0);
  await expect(page.getByTestId("tool-call-timing")).toContainText("started");
  await expect(page.getByTestId("tool-call-timing")).toContainText("24 min");
});
