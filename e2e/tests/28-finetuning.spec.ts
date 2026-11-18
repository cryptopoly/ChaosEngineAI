import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { goTo, RUN_HOME } from "./helpers";

// Tools > Fine-Tuning. The app has no training backend, so the page must say so: Start is off, one sentence
// explains it, and nothing is sent to /api/finetuning/start. The adapter list is real: a folder with an
// adapter_config.json under the backend's ~/Models (the run's throwaway HOME) shows as a card. That state used
// to crash the whole tab ("Fine-Tuning crashed: Cannot read properties of undefined (reading 'toFixed')").

const ADAPTER_DIR = path.join(RUN_HOME, "Models", "e2e-lora-adapter");

function seedAdapter() {
  mkdirSync(ADAPTER_DIR, { recursive: true });
  // The file mlx-lm writes next to the weights: its training arguments, with the base model under "model".
  writeFileSync(
    path.join(ADAPTER_DIR, "adapter_config.json"),
    JSON.stringify({ model: "mlx-community/Example-0.5B-4bit", fine_tune_type: "lora", lora_parameters: { rank: 8 } }),
  );
  writeFileSync(path.join(ADAPTER_DIR, "adapters.safetensors"), Buffer.alloc(3 * 1048576));
}

async function openFineTuning(page: Page) {
  await goTo(page, "Tools", "Fine-Tuning");
  await expect(page.getByRole("heading", { name: "LoRA Adapters" })).toBeVisible();
}

test.afterEach(() => rmSync(ADAPTER_DIR, { recursive: true, force: true }));

test("empty state: no adapters, and the page says training is not built", async ({ page }) => {
  const starts: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/api/finetuning/start")) starts.push(request.method());
  });
  await openFineTuning(page);
  await expect(page.getByText("0 adapters found")).toBeVisible();
  await expect(page.getByText("No LoRA adapters found.")).toBeVisible();
  await expect(page.getByText("Not available in this version")).toBeVisible();
  await expect(page.getByText(/Training is not built yet/)).toBeVisible();
  // The raw status word of the backend ("idle") used to be the panel's subtitle.
  await expect(page.locator("main").getByText("idle", { exact: true })).toHaveCount(0);
  const start = page.getByRole("button", { name: "Start Fine-Tuning" });
  await expect(start).toBeDisabled();
  await start.click({ force: true });
  expect(starts).toEqual([]);
});

test("loaded state: an adapter folder on disk is a card, not a crash", async ({ page }) => {
  seedAdapter();
  await openFineTuning(page);
  await expect(page.getByText("1 adapter found")).toBeVisible();
  await expect(page.getByText("e2e-lora-adapter")).toBeVisible();
  await expect(page.getByText("Base: mlx-community/Example-0.5B-4bit")).toBeVisible();
  await expect(page.getByText("Size: 3.0 MB")).toBeVisible();
  await expect(page.getByText(/^Created: \d{4}-\d{2}-\d{2} \d{2}:\d{2}$/)).toBeVisible();
  await expect(page.getByText("Fine-Tuning crashed")).toHaveCount(0);
});

test("a row with fields missing still renders", async ({ page }) => {
  await page.route("**/api/adapters", (route) =>
    route.fulfill({ json: { adapters: [{ name: "odd-row", path: "/mock/odd-row", base_model: "unknown", config: {} }], count: 1 } }),
  );
  await openFineTuning(page);
  await expect(page.getByText("odd-row")).toBeVisible();
  await expect(page.getByText("Size: 0.0 MB")).toBeVisible();
  await expect(page.getByText("Fine-Tuning crashed")).toHaveCount(0);
});

test("error state: a failing adapter list leaves the page standing", async ({ page }) => {
  await page.route("**/api/adapters", (route) => route.fulfill({ status: 500, json: { detail: "boom" } }));
  await openFineTuning(page);
  // It used to read "No LoRA adapters found.", as if the list were empty.
  await expect(page.getByRole("alert").filter({ hasText: "Could not load the adapter list" })).toBeVisible();
  await expect(page.getByText("No LoRA adapters found.")).toHaveCount(0);
  await expect(page.getByText("Loading adapters...")).toHaveCount(0);
  await expect(page.getByText("Fine-Tuning crashed")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Start Fine-Tuning" })).toBeDisabled();
});

test.describe("375 px wide", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("no sideways scroll, with an adapter card", async ({ page }) => {
    seedAdapter();
    await openFineTuning(page);
    await expect(page.getByText("e2e-lora-adapter")).toBeVisible();
    const sideways = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(sideways).toBeLessThanOrEqual(0);
    const stickingOut = await page.evaluate(() =>
      [...document.querySelectorAll("main *")].filter((el) => {
        const box = el.getBoundingClientRect();
        return box.width > 0 && box.right > window.innerWidth + 1;
      }).length,
    );
    expect(stickingOut).toBe(0);
  });
});
