import type { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { openApp, openTab } from "./helpers";

// Review stream D: a local model that is not itself a catalogue row. Two faults met on a real library:
// 1. The app joined the model to another catalogue row by shared words ("qwen", "4bit", MLX, 4-bit) and sent
//    that row's repo to the backend as the model's own. For mlx-community/Qwen3-0.6B-4bit the row was Qwen3.8
//    Flash Next, and the backend refused the load: "runs only on TensorFold".
// 2. Chat in a new thread loaded the model but opened no thread, so the first message went into the last thread
//    and that thread's own model was loaded back in.
// The library item and an older thread are added to the real workspace answer; the load itself is answered here.

const HUB = "/Users/someone/.cache/huggingface/hub";
const LOCAL = {
  name: "mlx-community/Qwen3-0.6B-4bit",
  path: `${HUB}/models--mlx-community--Qwen3-0.6B-4bit`,
  format: "MLX",
  sourceKind: "HF cache",
  quantization: "4-bit",
  backend: "mlx",
  modelType: "text",
  sizeGb: 0.33,
  loadSizeGb: null,
  lastModified: "2026-11-15 21:36",
  actions: ["Run Chat", "Run Server", "Cache Preview", "Delete"],
  directoryId: "hf-cache",
  directoryLabel: "Hugging Face cache",
  directoryPath: HUB,
  maxContext: 40960,
  mtpHeads: false,
  broken: false,
  brokenReason: null,
};
const OLDER = {
  id: "older-thread",
  title: "An older thread",
  updatedAt: "2026-11-17 13:53:08",
  pinned: false,
  thinkingMode: "off",
  model: "mlx-community/Some-Other-27B-4bit",
  modelRef: "mlx-community/Some-Other-27B-4bit",
  canonicalRepo: "mlx-community/Some-Other-27B-4bit",
  modelSource: "library",
  modelPath: `${HUB}/models--mlx-community--Some-Other-27B-4bit`,
  modelBackend: "mlx",
  cacheLabel: "Native f16",
  cacheStrategy: "native",
  cacheBits: 0,
  fp16Layers: 0,
  fusedAttention: false,
  fitModelInMemory: true,
  contextTokens: 8192,
  speculativeDecoding: false,
  dflashDraftModel: null,
  treeBudget: 0,
  messages: [
    { role: "user", text: "An older question", metrics: null },
    { role: "assistant", text: "An older answer", metrics: null },
  ],
};

async function withLocalModel(page: Page): Promise<{ loads: Array<Record<string, unknown>> }> {
  const loads: Array<Record<string, unknown>> = [];
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.library = [LOCAL];
    body.libraryStatus = "ready";
    body.chatSessions = [OLDER];
    await route.fulfill({ response, json: body });
  });
  await page.route("**/api/models/load", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    loads.push(route.request().postDataJSON());
    // The answer does not matter here; the request and the thread do.
    await route.fulfill({ status: 500, json: { detail: "Stopped by the test before a real load." } });
  });
  return { loads };
}

test("Chat in a new thread on a local model: its own repo is sent, and a new thread opens", async ({ page }) => {
  const { loads } = await withLocalModel(page);
  await openApp(page);
  await openTab(page, "Chat", "Models");
  await page.locator(".models-availability__option", { hasText: "Installed" }).click();
  await page.getByRole("button", { name: "Chat in a new thread" }).first().click();

  const dialog = page.getByRole("dialog", { name: "Select Model" });
  await expect(dialog).toBeVisible();
  await expect(dialog.locator(".model-selected-card")).toContainText(LOCAL.name);
  // The dialog describes this model, not a 256K-context catalogue row it was mistaken for.
  await expect(dialog.locator(".model-selected-card")).not.toContainText("256K");
  await dialog.getByRole("button", { name: "Start Chat" }).click();

  await expect.poll(() => loads.length).toBe(1);
  expect(loads[0]).toMatchObject({
    modelRef: LOCAL.name,
    canonicalRepo: LOCAL.name,
    path: LOCAL.path,
    backend: "mlx",
    source: "library",
  });

  // A new, empty thread bound to the model is the active one; the older thread is still in the list.
  await expect(page.getByRole("heading", { name: "Chat", exact: true })).toBeVisible();
  await expect(page.locator(".thread-toolbar-actions > button.secondary-button").first()).toHaveText(LOCAL.name);
  await expect(page.locator(".message-bubble")).toHaveCount(0);
  await expect(page.getByText("An older answer")).toHaveCount(0);
  await expect(page.getByText("An older thread")).toBeVisible();
});
