import { test, expect } from "./fixtures";
import type { Page, Route } from "@playwright/test";
import path from "node:path";
import { goTo, openTab, SHOTS_DIR } from "./helpers";

// Benchmarks (Run, History) and Conversion to MLX (overnight review, stream A). The library, the saved
// runs and the two long requests (POST /api/benchmarks/run, POST /api/models/convert) are mocked: no
// model is loaded and nothing is converted here. The real runs are in scripts/e2e_phases/
// phase_21_benchmarks_conversion.py. The app has one theme (dark), so there is no light pass.

const MLX_ITEM = {
  name: "mock-org/Mock-Model-1B-4bit",
  path: "/mock/hub/models--mock-org--Mock-Model-1B-4bit",
  format: "MLX",
  sourceKind: "HF cache",
  quantization: "4-bit",
  backend: "mlx",
  modelType: "text",
  sizeGb: 0.6,
  loadSizeGb: 0.6,
  lastModified: "2026-11-01 10:00",
  actions: ["Run Chat"],
  maxContext: 8192,
  broken: false,
  brokenReason: null,
};
const TRANSFORMERS_ITEM = {
  ...MLX_ITEM,
  name: "mock-org/Mock-Model-4B",
  path: "/mock/hub/models--mock-org--Mock-Model-4B",
  format: "Transformers",
  quantization: "BF16",
  sizeGb: 8,
  loadSizeGb: 8,
};
const GGUF_ITEM = {
  ...MLX_ITEM,
  name: "mock-org/Mock-Model-1B-GGUF",
  path: "/mock/hub/models--mock-org--Mock-Model-1B-GGUF",
  format: "GGUF",
  quantization: "Q4_K_M",
  backend: "llama.cpp",
  sizeGb: 0.5,
  loadSizeGb: 0.5,
};
const GGUF_FILE = `${GGUF_ITEM.path}/snapshots/abc/mock-model-1b-Q4_K_M.gguf`;

function run(overrides: Record<string, unknown>) {
  return {
    id: "bench-speed-1",
    mode: "throughput",
    label: "Mock-Model-1B-4bit / Native f16 / 8K ctx",
    model: "Mock-Model-1B-4bit",
    modelRef: MLX_ITEM.name,
    backend: "mlx",
    engineLabel: "MLX",
    source: "library",
    measuredAt: "2026-11-18 08:00:00",
    bits: 16,
    fp16Layers: 0,
    cacheStrategy: "native",
    cacheLabel: "Native f16",
    cacheGb: 0.3,
    baselineCacheGb: 0.3,
    compression: 1,
    contextTokens: 8192,
    maxTokens: 512,
    loadSeconds: 1.2,
    tokS: 120.5,
    quality: 100,
    responseSeconds: 2.1,
    totalSeconds: 3.3,
    promptTokens: 40,
    completionTokens: 250,
    totalTokens: 290,
    notes: "Mock run.",
    ...overrides,
  };
}

const SPEED_OLD = run({ id: "bench-speed-old", tokS: 100.5, measuredAt: "2026-11-18 07:00:00", label: "Older speed run" });
const SPEED_GGUF = run({
  id: "bench-speed-gguf", tokS: 327.5, model: "Mock-Model-1B-GGUF", modelRef: GGUF_ITEM.name, backend: "llama.cpp",
  engineLabel: "llama.cpp + GGUF", label: "GGUF speed run", measuredAt: "2026-11-18 07:30:00",
});
const SPEED_NEW = run({ id: "bench-speed-new", tokS: 120.5, label: "Newer speed run" });
const ACCURACY = run({
  id: "bench-mmlu", mode: "task_accuracy", label: "Mock-Model-1B-4bit / MMLU / 4/10", tokS: 0, taskName: "mmlu",
  taskAccuracy: 0.4, taskCorrect: 4, taskTotal: 10, taskNumShots: 0, evalSeconds: 3.2, responseSeconds: 3.2,
  completionTokens: 0, promptTokens: 0, totalTokens: 0, measuredAt: "2026-11-18 08:10:00",
});

type Workspace = { library: unknown[]; benchmarks: unknown[]; runtime: unknown };
let lastWorkspace: Workspace | null = null;

/** Patch the real workspace answer with a small library and the given saved runs. */
async function mockWorkspace(page: Page, benchmarks: unknown[], library: unknown[] = [MLX_ITEM, TRANSFORMERS_ITEM, GGUF_ITEM]) {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = await response.json();
    body.library = library;
    body.benchmarks = benchmarks;
    lastWorkspace = body;
    await route.fulfill({ response, json: body });
  });
}

async function noSidewaysScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(1);
}

async function shot(page: Page, name: string) {
  // At 375 px the nav drawer slides shut after a tab is chosen; wait for it so the page is what is captured.
  const menu = page.locator(".sidebar-menu-button");
  if (await menu.isVisible()) {
    await expect(menu).toHaveAttribute("aria-expanded", "false");
    await page.waitForTimeout(400);
  }
  await page.screenshot({ path: path.join(SHOTS_DIR, `${name}.png`), fullPage: true });
}

// ---- Benchmarks -------------------------------------------------------------------------------

test("Benchmarks, empty: no latest-run card, History says there are no runs", async ({ page }) => {
  await mockWorkspace(page, []);
  await goTo(page, "Benchmarks", "Run");
  await expect(page.getByRole("heading", { name: "Run Benchmark" })).toBeVisible();
  await expect(page.getByText("No benchmark runs yet.")).toBeVisible();
  await expect(page.locator(".benchmark-last-run-card")).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Run benchmark/ })).toBeEnabled();

  await openTab(page, "Benchmarks", "History");
  await expect(page.getByRole("heading", { name: "Benchmark History" })).toBeVisible();
  await expect(page.getByText("0 runs")).toBeVisible();
  await expect(page.locator(".benchmark-history-list button")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Swap" })).toBeDisabled();
  await shot(page, "27-benchmark-history-empty");
});

test("Benchmarks, loaded: speed is compared only between speed runs", async ({ page }) => {
  // Latest is a speed run: it is compared with the previous speed run of the same model and the fastest run.
  await mockWorkspace(page, [SPEED_NEW, SPEED_GGUF, SPEED_OLD]);
  await goTo(page, "Benchmarks", "Run");
  const card = page.locator(".benchmark-last-run-card");
  await expect(card).toContainText("120.5");
  await expect(card.locator(".benchmark-delta-note")).toContainText("+20.0 tok/s vs your previous Native f16 run");
  await expect(card.locator(".benchmark-delta-note")).toContainText("-207.0 tok/s vs fastest");
});

test("Benchmarks: no invented Quality figure; quality is what a run measured", async ({ page }) => {
  // Every run used to carry "Quality 100%" (a constant) on the card, in the list, the compare table and the chart.
  const perplexity = run({
    id: "bench-ppl-q", mode: "perplexity", label: "Perplexity run", tokS: 5200, perplexity: 21.4, perplexityStdError: 0.6,
    perplexityDataset: "wikitext-2", perplexityNumSamples: 16, evalTokensPerSecond: 5200, evalSeconds: 1.5,
    measuredAt: "2026-11-18 07:10:00",
  });
  await mockWorkspace(page, [SPEED_NEW, SPEED_OLD, perplexity]);
  await goTo(page, "Benchmarks", "Run");
  const card = page.locator(".benchmark-last-run-card");
  await expect(card).toContainText("Tokens generated");
  await expect(card).toContainText("250");
  await expect(card).not.toContainText("Quality");
  await openTab(page, "Benchmarks", "History");
  const history = page.locator(".benchmark-history-page");
  await expect(history).toContainText("Lowest perplexity");
  await expect(history.locator(".stat-card", { hasText: "Lowest perplexity" })).toContainText("21.40");
  await expect(history).not.toContainText("Best quality");
  await expect(history).not.toContainText("Dot size = quality");
  await expect(history.locator(".bm-comparison-table")).not.toContainText("Quality");
  await expect(history).not.toContainText("100%");
});

test("Benchmarks, loaded: an accuracy run shows its score and no tok/s comparison", async ({ page }) => {
  // The accuracy run stores 0 tok/s; the card used to say "-327.5 tok/s vs your previous Native f16 run".
  await mockWorkspace(page, [ACCURACY, SPEED_GGUF, SPEED_OLD]);
  await goTo(page, "Benchmarks", "Run");
  const card = page.locator(".benchmark-last-run-card");
  await expect(card).toContainText("4/10 correct");
  await expect(card).toContainText("MMLU");
  await expect(card.locator(".benchmark-delta-note")).toHaveCount(0);
  await expect(card).not.toContainText("tok/s vs");
});

test("Benchmark History: list, filter by model, select, compare and delete", async ({ page }) => {
  const saved = [ACCURACY, SPEED_NEW, SPEED_GGUF, SPEED_OLD];
  await mockWorkspace(page, saved);
  await goTo(page, "Benchmarks", "History");
  await expect(page.getByText("4 runs")).toBeVisible();
  // Latest is the accuracy run: its card shows the score, not "0.0 tok/s"; Fastest is a speed run.
  const stat = (label: string) => page.locator(".benchmark-summary-row .stat-card", { hasText: label });
  await expect(stat("Latest")).toContainText("40.0%");
  await expect(stat("Latest")).not.toContainText("tok/s");
  await expect(stat("Fastest")).toContainText("327.5 tok/s");
  const rows = page.locator(".benchmark-history-list button");
  await expect(rows).toHaveCount(4);
  await expect(rows.nth(0)).toContainText("MMLU");
  await expect(rows.nth(0)).toContainText("40.0%");
  await expect(rows.nth(2)).toContainText("327.5 tok/s");

  // Select the GGUF run, make the older run the baseline (right-click): the delta row follows.
  await rows.nth(2).click();
  await rows.nth(3).click({ button: "right" });
  const speedRow = page.locator(".bm-comparison-row, tr", { hasText: /^Speed/ }).first();
  await expect(speedRow).toContainText("327.5 tok/s");
  await expect(speedRow).toContainText("100.5 tok/s");

  // A double click makes a row the baseline and keeps the selection (it used to select the row
  // as well, so the baseline fell back to some other run).
  await rows.nth(1).dblclick();
  await expect(speedRow).toContainText("327.5 tok/s");
  await expect(speedRow).toContainText("120.5 tok/s");
  await expect(rows.nth(2)).toHaveClass(/active/);
  await expect(rows.nth(1)).toHaveClass(/compare/);

  // A baseline that measured something else gives no value and no delta, not "0.0 tok/s".
  await rows.nth(0).click({ button: "right" });
  await expect(speedRow).toContainText("327.5 tok/s");
  await expect(speedRow).not.toContainText("0.0 tok/s");
  await expect(speedRow).not.toContainText("%");
  await rows.nth(3).click({ button: "right" });

  // Filter to one model.
  await page.locator("select.bm-model-filter").selectOption("Mock-Model-1B-GGUF");
  await expect(rows).toHaveCount(1);
  await expect(page.getByText("1 run for Mock-Model-1B-GGUF")).toBeVisible();
  await page.locator("select.bm-model-filter").selectOption("");
  await expect(rows).toHaveCount(4);

  // Table, Chart and Both views.
  await page.getByRole("button", { name: "Chart", exact: true }).click();
  await expect(page.locator(".benchmark-history-list")).toHaveCount(0);
  await page.getByRole("button", { name: "Table", exact: true }).click();
  await expect(rows).toHaveCount(4);
  await shot(page, "27-benchmark-history-loaded");

  // The toolbar keeps the three views and both buttons on one line at the default window width.
  await expect(page.getByRole("button", { name: "Both", exact: true })).toBeVisible();
  const toolbar = await page.locator(".bm-toolbar").boundingBox();
  for (const name of ["Both", "Copy CSV", "Swap"]) {
    const box = await page.getByRole("button", { name, exact: true }).boundingBox();
    expect(box!.height).toBeLessThan(40);
    expect(box!.x + box!.width).toBeLessThanOrEqual(toolbar!.x + toolbar!.width + 1);
  }

  // Copy CSV: the runs on screen go to the clipboard, a header line and one line per run.
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.getByRole("button", { name: "Copy CSV" }).click();
  await expect(page.getByRole("button", { name: "Copied" })).toBeVisible();
  const csv = await page.evaluate(() => navigator.clipboard.readText());
  const csvLines = csv.trimEnd().split("\n");
  expect(csvLines).toHaveLength(5);
  expect(csvLines[0].startsWith("measuredAt,label,model,")).toBe(true);
  expect(csvLines[1]).toContain("task_accuracy");
  expect(csvLines[3]).toContain("327.5");

  // Delete the selected run: asks first, then the run leaves the list (the backend answer is mocked).
  let deleted = "";
  await page.route("**/api/benchmarks/*", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    deleted = decodeURIComponent(new URL(route.request().url()).pathname.split("/").pop() ?? "");
    saved.splice(saved.indexOf(SPEED_GGUF), 1); // the workspace poll answers without it from now on
    await route.fulfill({ json: { deleted, benchmarks: saved } });
  });
  await rows.nth(2).click();
  page.once("dialog", (dialog) => void dialog.dismiss());
  await page.getByRole("button", { name: "Delete run" }).click();
  await expect(rows).toHaveCount(4);
  expect(deleted).toBe("");
  page.once("dialog", (dialog) => {
    expect(dialog.message()).toContain("GGUF speed run");
    void dialog.accept();
  });
  await page.getByRole("button", { name: "Delete run" }).click();
  await expect(rows).toHaveCount(3);
  expect(deleted).toBe("bench-speed-gguf");
  await expect(page.locator(".benchmark-history-list")).not.toContainText("GGUF speed run");
});

test("Benchmark run: the chosen mode is sent, the result card and the history follow", async ({ page }) => {
  await mockWorkspace(page, [SPEED_OLD]);
  let sent: Record<string, unknown> | null = null;
  const perplexity = run({
    id: "bench-ppl", mode: "perplexity", label: "Mock-Model-1B-4bit / Perplexity / wikitext-2", tokS: 5200,
    perplexity: 21.4, perplexityStdError: 0.6, perplexityDataset: "wikitext-2", perplexityNumSamples: 16,
    evalTokensPerSecond: 5200, evalSeconds: 1.5, measuredAt: "2026-11-18 08:20:00",
  });
  await page.route("**/api/benchmarks/run", async (route: Route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({ json: { result: perplexity, benchmarks: [perplexity, SPEED_OLD], runtime: lastWorkspace?.runtime } });
  });
  await goTo(page, "Benchmarks", "Run");
  await page.locator(".benchmark-run-mode-stack select").first().selectOption("perplexity");
  await expect(page.getByLabel("Dataset")).toHaveValue("wikitext-2");
  await page.getByLabel("Samples").fill("16");
  await page.getByRole("button", { name: /Run benchmark/ }).click();

  const modal = page.locator(".benchmark-result-modal");
  await expect(modal.getByRole("heading", { name: "Benchmark complete" })).toBeVisible();
  await expect(modal).toContainText("21.4");
  expect(sent).toMatchObject({ mode: "perplexity", perplexityNumSamples: 16, modelRef: MLX_ITEM.name, path: MLX_ITEM.path });
  await modal.getByRole("button", { name: "OK" }).click();
  await expect(modal).toHaveCount(0);
  await expect(page.locator(".benchmark-last-run-card")).toContainText("21.4");
  await expect(page.locator(".benchmark-last-run-card .benchmark-delta-note")).toHaveCount(0);

  await openTab(page, "Benchmarks", "History");
  await expect(page.locator(".benchmark-history-list button")).toHaveCount(2);
  await expect(page.locator(".benchmark-history-list button").first()).toContainText("PPL");
});

test("Benchmark run: perplexity and task accuracy are MLX only, and the page says so for a GGUF model", async ({ page }) => {
  // With a GGUF model the run used to load the model and then fail with a 400.
  await mockWorkspace(page, [], [GGUF_ITEM]);
  let asked = false;
  await page.route("**/api/benchmarks/run", (route) => {
    asked = true;
    return route.fulfill({ status: 400, json: { detail: "Perplexity evaluation is not supported by the llama.cpp backend." } });
  });
  await goTo(page, "Benchmarks", "Run");
  await expect(page.locator(".benchmark-run-header-grid .model-selected-card")).toContainText(GGUF_ITEM.name);
  const mode = page.locator(".benchmark-run-mode-stack select").first();
  const runButton = page.getByRole("button", { name: /Run benchmark/ });
  await expect(runButton).toBeEnabled();
  for (const value of ["perplexity", "task_accuracy"]) {
    await mode.selectOption(value);
    await expect(page.getByText("Perplexity and task accuracy run on MLX models only.")).toBeVisible();
    await expect(runButton).toBeDisabled();
  }
  await mode.selectOption("throughput");
  await expect(page.getByText("Perplexity and task accuracy run on MLX models only.")).toHaveCount(0);
  await expect(runButton).toBeEnabled();
  expect(asked).toBe(false);
});

test("Benchmark run: a backend error is shown in the dialog, not lost", async ({ page }) => {
  await mockWorkspace(page, []);
  await page.route("**/api/benchmarks/run", (route) =>
    route.fulfill({ status: 400, json: { detail: { message: "Cannot benchmark 'Mock': incomplete or corrupt.", localized: "Cannot benchmark 'Mock': incomplete or corrupt." } } }),
  );
  await goTo(page, "Benchmarks", "Run");
  await page.getByRole("button", { name: /Run benchmark/ }).click();
  const modal = page.locator(".benchmark-result-modal");
  await expect(modal.getByRole("heading", { name: "Benchmark failed" }).first()).toBeVisible();
  await expect(modal).toContainText("incomplete or corrupt");
  await modal.getByRole("button", { name: /Close|OK/ }).click();
  await expect(page.getByText("No benchmark runs yet.")).toBeVisible();
});

/** A progress snapshot as GET /api/benchmarks/progress and /api/models/convert/progress answer it. */
function progress(kind: string, overrides: Record<string, unknown>) {
  return {
    progress: {
      kind, active: true, phase: "loading", message: "", step: 0, totalSteps: 0, startedAt: 1, updatedAt: 1,
      elapsedSeconds: 1, runLabel: null, cancelRequested: false, thumbnail: null, warning: null, detail: {},
      ...overrides,
    },
  };
}

test("Benchmark History: accuracy runs scored by different methods show both values and no delta", async ({ page }) => {
  const scoredNew = { ...ACCURACY, id: "bench-mmlu-new", label: "MMLU next-token", taskAccuracy: 0.3, taskCorrect: 30, taskTotal: 100, taskScoringMethod: "next_token", measuredAt: "2026-11-18 09:00:00" };
  const scoredOld = { ...ACCURACY, id: "bench-mmlu-old", label: "MMLU generated", taskAccuracy: 0.09, taskCorrect: 9, taskTotal: 100, measuredAt: "2026-11-18 07:00:00" };
  await mockWorkspace(page, [scoredNew, scoredOld]);
  await goTo(page, "Benchmarks", "Run");
  // The latest-run card names the method next to the shot count.
  await expect(page.locator(".benchmark-last-run-card")).toContainText("next-token choice");
  await page.locator(".benchmark-run-mode-stack select").first().selectOption("task_accuracy");
  await expect(page.getByText("The answer is the choice letter the model scores highest as its next token.")).toBeVisible();
  await openTab(page, "Benchmarks", "History");
  const history = page.locator(".benchmark-history-page");
  await expect(history).toContainText("Scoring");
  await expect(history).toContainText("next-token choice");
  await expect(history).toContainText("generated text (older method)");
  // 30.0% against 9.0% would be "+233%": with two scoring methods no percentage is shown for it.
  await expect(history).not.toContainText("233");
});

test("Benchmark run: Cancel stops the run in the backend, nothing is saved, the dialog says what happened", async ({ page }) => {
  // Cancel used to stop only the wait: the backend finished the run, saved it and kept the model loaded.
  await mockWorkspace(page, [SPEED_OLD]);
  let release: () => void = () => {};
  const cancelled = new Promise<void>((resolve) => { release = resolve; });
  let cancelCalls = 0;
  await page.route("**/api/benchmarks/run", async (route) => {
    await cancelled;
    await route.fulfill({
      json: {
        cancelled: true, modelUnloaded: true, result: null, benchmarks: [SPEED_OLD], runtime: lastWorkspace?.runtime,
        message: "Benchmark cancelled. Nothing was saved. The model was unloaded.",
      },
    });
  });
  await page.route("**/api/benchmarks/cancel", async (route) => {
    cancelCalls += 1;
    await route.fulfill({ json: { cancelled: true } });
    release();
  });
  await goTo(page, "Benchmarks", "Run");
  await page.getByRole("button", { name: /Run benchmark/ }).click();
  const modal = page.locator(".benchmark-result-modal");
  await expect(modal.getByRole("heading", { name: "Running benchmark" })).toBeVisible();
  await modal.getByRole("button", { name: "Cancel benchmark" }).click();
  await expect(modal.getByRole("heading", { name: "Benchmark cancelled" })).toBeVisible();
  await expect(modal.getByTestId("benchmark-cancelled")).toContainText("Nothing was saved to the history.");
  await expect(modal.getByTestId("benchmark-cancelled")).toContainText("The model was unloaded.");
  await expect(modal.getByRole("heading", { name: "Benchmark failed" })).toHaveCount(0);
  expect(cancelCalls).toBe(1);
  await shot(page, "27-benchmark-cancelled");
  await modal.getByRole("button", { name: "Close" }).click();
  // The run on the card is still the old one.
  await expect(page.locator(".benchmark-last-run-card")).toContainText("100.5");
  await openTab(page, "Benchmarks", "History");
  await expect(page.locator(".benchmark-history-list button")).toHaveCount(1);
});

test("Benchmark run: the dialog shows the run's own progress, a percentage only for counted items", async ({ page }) => {
  await mockWorkspace(page, []);
  let release: () => void = () => {};
  const finished = new Promise<void>((resolve) => { release = resolve; });
  let snapshot = progress("benchmark", { phase: "loading", message: "Loading weights into MLX" });
  await page.route("**/api/benchmarks/progress", (route) => route.fulfill({ json: snapshot }));
  await page.route("**/api/benchmarks/run", async (route) => {
    await finished;
    await route.fulfill({ json: { result: ACCURACY, benchmarks: [ACCURACY], runtime: lastWorkspace?.runtime } });
  });
  await goTo(page, "Benchmarks", "Run");
  await page.locator(".benchmark-run-mode-stack select").first().selectOption("task_accuracy");
  await page.getByRole("button", { name: /Run benchmark/ }).click();
  const modal = page.locator(".benchmark-result-modal");
  const bar = modal.getByRole("progressbar");
  // A model load cannot be counted: the loader's own text, no percentage.
  await expect(modal.locator(".live-progress__phase--active")).toContainText("Loading weights into MLX");
  await expect(bar).not.toHaveAttribute("aria-valuenow", /.*/);
  // An accuracy run answers questions; it never "generates N tokens".
  await expect(modal.locator(".live-progress__phases")).not.toContainText("Generating");
  await expect(modal.locator(".live-progress__phases")).toContainText("Answering questions");

  snapshot = progress("benchmark", { phase: "evaluating", step: 12, totalSteps: 100, detail: { correct: 5 } });
  await expect(modal.locator(".live-progress__phase--active")).toContainText("12 of 100 questions, 5 correct");
  await expect(bar).toHaveAttribute("aria-valuenow", "12");
  await shot(page, "27-benchmark-progress");

  snapshot = progress("benchmark", { phase: "saving" });
  await expect(modal.locator(".live-progress__phase--active")).toContainText("Saving the result");
  await expect(bar).not.toHaveAttribute("aria-valuenow", /.*/);
  release();
  await expect(modal.getByRole("heading", { name: "Benchmark complete" })).toBeVisible();
});

test("Benchmark run: a throughput run says it cannot count tokens while it generates", async ({ page }) => {
  await mockWorkspace(page, []);
  let release: () => void = () => {};
  const finished = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/benchmarks/progress", (route) =>
    route.fulfill({ json: progress("benchmark", { phase: "generating", message: "Generating up to 4096 tokens" }) }),
  );
  await page.route("**/api/benchmarks/run", async (route) => {
    await finished;
    await route.fulfill({ json: { result: SPEED_NEW, benchmarks: [SPEED_NEW], runtime: lastWorkspace?.runtime } });
  });
  await goTo(page, "Benchmarks", "Run");
  await page.getByRole("button", { name: /Run benchmark/ }).click();
  const modal = page.locator(".benchmark-result-modal");
  await expect(modal.locator(".live-progress__phase--active")).toContainText("Generating up to 4096 tokens");
  await expect(modal.locator(".live-progress__phase--active")).toContainText("the engine reports the token count when it finishes");
  await expect(modal.getByRole("progressbar")).not.toHaveAttribute("aria-valuenow", /.*/);
  release();
  await expect(modal.getByRole("heading", { name: "Benchmark complete" })).toBeVisible();
});

test("Benchmarks at 375 px: Run and History have no sideways scroll", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await mockWorkspace(page, [ACCURACY, SPEED_NEW, SPEED_GGUF, SPEED_OLD]);
  await goTo(page, "Benchmarks", "Run");
  await expect(page.getByRole("heading", { name: "Run Benchmark" })).toBeVisible();
  await noSidewaysScroll(page);
  // The model card sits above the mode fields at this width (it was squeezed into a 30 px column beside them).
  const card = await page.locator(".benchmark-run-header-grid .model-selected-card").boundingBox();
  const mode = await page.locator(".benchmark-run-mode-stack").boundingBox();
  expect(card!.width).toBeGreaterThan(250);
  expect(card!.y + card!.height).toBeLessThanOrEqual(mode!.y + 1);
  const runButton = page.getByRole("button", { name: /Run benchmark/ });
  await runButton.scrollIntoViewIfNeeded();
  await expect(runButton).toBeVisible();
  await shot(page, "27-benchmark-run-375");
  await openTab(page, "Benchmarks", "History");
  await expect(page.getByRole("heading", { name: "Benchmark History" })).toBeVisible();
  await noSidewaysScroll(page);
  await shot(page, "27-benchmark-history-375");
});

// ---- Conversion -------------------------------------------------------------------------------

test("Conversion, empty: nothing to convert until a source is picked", async ({ page }) => {
  await mockWorkspace(page, []);
  await goTo(page, "Tools", "Conversion");
  await expect(page.getByRole("heading", { name: "MLX Conversion" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Select a model to convert..." })).toBeVisible();
  await expect(page.getByRole("button", { name: "Convert to MLX" })).toBeDisabled();
  // With no source there is nothing to project: no tok/s figure, no KV-cache panel, no cache size.
  const estCard = page.locator(".conversion-visuals .stat-card", { hasText: "Est. tok/s" });
  await expect(estCard).toContainText("\u2014");
  await expect(estCard).toContainText("Pick a source first");
  await expect(estCard).not.toContainText(/\d tok\/s/);
  await expect(page.locator(".conversion-visuals")).not.toContainText("Performance preview");
  await expect(page.locator(".conversion-visuals")).not.toContainText("Cache footprint");
  // Quality is measured after converting; no percentage is shown before.
  await expect(page.getByTestId("conversion-quality-row")).toContainText("Measured after converting");
  await expect(page.getByTestId("conversion-quality-row")).not.toContainText("%");
  await shot(page, "27-conversion-empty");
});

test("Conversion: only non-MLX models are offered; a Transformers source converts at 8-bit", async ({ page }) => {
  await mockWorkspace(page, []);
  let sent: Record<string, unknown> | null = null;
  await page.route("**/api/models/convert", async (route: Route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({
      json: {
        conversion: {
          sourceRef: TRANSFORMERS_ITEM.name, sourcePath: TRANSFORMERS_ITEM.path, sourceLabel: "Mock-Model-4B",
          hfRepo: TRANSFORMERS_ITEM.name, outputPath: "/mock/Models/Mock-Model-4B-mlx", quantize: true, qBits: 8,
          qGroupSize: 64, dtype: "float16", sourceSizeGb: 8, outputSizeGb: 4.2, ggufMetadata: null,
          log: "[INFO] Quantized model with 8.502 bits per weight.", sourceFormat: "Transformers", paramsB: 4,
          contextWindow: "262K", architecture: null, estimatedTokS: 95, baselineCacheGb: 4, optimizedCacheGb: 4,
          diskRatio: 1.9, bitsPerWeight: 8.502,
          quality: {
            status: "measured", reason: null, dataset: "wikitext-2", samples: 8, seqLength: 256,
            sourcePerplexity: 12.5, convertedPerplexity: 12.56, changePercent: 0.48,
          },
        },
        library: lastWorkspace?.library,
        runtime: lastWorkspace?.runtime,
      },
    });
  });
  await goTo(page, "Tools", "Conversion");
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  const picker = page.locator(".modal-content", { hasText: "Select Source Model" });
  await expect(picker.locator(".model-select-item")).toHaveCount(2);
  await expect(picker).not.toContainText(MLX_ITEM.name);
  await picker.locator(".model-select-item", { hasText: TRANSFORMERS_ITEM.name }).click();

  await expect(page.locator(".model-selected-card")).toContainText(TRANSFORMERS_ITEM.name);
  const convert = page.getByRole("button", { name: "Convert to MLX" });
  await expect(convert).toBeEnabled();
  await page.locator('input[type="range"]').first().fill("8");
  await convert.click();

  const modal = page.locator(".conversion-result-modal");
  await expect(modal.getByRole("heading", { name: "Conversion complete" })).toBeVisible();
  await expect(modal).toContainText("/mock/Models/Mock-Model-4B-mlx");
  expect(sent).toMatchObject({
    modelRef: TRANSFORMERS_ITEM.name, path: TRANSFORMERS_ITEM.path, quantize: true, qBits: 8, dtype: "float16",
    measureQuality: true,
  });
  // The dialog shows what was measured: the size ratio, the converter's bits per weight and the
  // perplexity of both models. It used to show "Quality 100.0%" and a KV-cache "cache reduction".
  await expect(modal).toContainText("1.9x smaller on disk");
  await expect(modal).toContainText("8.502 (measured by the converter)");
  const quality = modal.getByTestId("conversion-quality-result");
  await expect(quality).toContainText("+0.5% perplexity vs the source");
  await expect(quality).toContainText("Perplexity on 8 samples of wikitext-2: source 12.50, converted 12.56. Lower is better.");
  await expect(modal).not.toContainText("cache reduction");
  await expect(modal).not.toContainText("100.0%");
  await shot(page, "27-conversion-complete");
  await modal.getByRole("button", { name: "OK" }).click();
  await expect(modal).toHaveCount(0);
});

test("Conversion: quality that could not be measured says why, and switching it off is sent", async ({ page }) => {
  await mockWorkspace(page, []);
  let sent: Record<string, unknown> | null = null;
  const reason = "The comparison loads the source once (8.0 GB) and 4.0 GB of memory is free. Unload a model and run a Perplexity benchmark on both to compare.";
  await page.route("**/api/models/convert", async (route: Route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({
      json: {
        conversion: {
          sourceRef: TRANSFORMERS_ITEM.name, sourcePath: TRANSFORMERS_ITEM.path, sourceLabel: "Mock-Model-4B",
          hfRepo: TRANSFORMERS_ITEM.name, outputPath: "/mock/Models/Mock-Model-4B-mlx", quantize: false, qBits: 4,
          qGroupSize: 64, dtype: "float16", sourceSizeGb: 8, outputSizeGb: 8, ggufMetadata: null, log: "[INFO] Loading",
          sourceFormat: "Transformers", paramsB: 4, contextWindow: "262K", architecture: null, estimatedTokS: 40,
          baselineCacheGb: 4, optimizedCacheGb: 4, diskRatio: 1, bitsPerWeight: null,
          quality: sent?.measureQuality ? { status: "skipped", reason } : null,
        },
        library: lastWorkspace?.library,
        runtime: lastWorkspace?.runtime,
      },
    });
  });
  await goTo(page, "Tools", "Conversion");
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  await page.locator(".model-select-item", { hasText: TRANSFORMERS_ITEM.name }).click();
  const measure = page.getByLabel("Measure quality after converting");
  await expect(measure).toBeChecked();
  await page.getByRole("button", { name: "Convert to MLX" }).click();
  const modal = page.locator(".conversion-result-modal");
  const quality = modal.getByTestId("conversion-quality-result");
  await expect(quality).toContainText("Not measured");
  await expect(quality).toContainText(reason);
  await expect(quality).not.toContainText("%");
  await expect(modal).not.toContainText("measured by the converter"); // no quantization, no bits per weight
  await modal.getByRole("button", { name: "OK" }).click();

  await page.getByRole("button", { name: "Clear" }).click();
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  await page.locator(".model-select-item", { hasText: TRANSFORMERS_ITEM.name }).click();
  await page.getByLabel("Measure quality after converting").uncheck();
  await expect(page.getByTestId("conversion-quality-row")).toContainText("Not measured (switched off)");
  await page.getByRole("button", { name: "Convert to MLX" }).click();
  await expect(modal.getByTestId("conversion-quality-result")).toContainText("Not measured");
  expect(sent).toMatchObject({ measureQuality: false });
});

test("Conversion: a GGUF picked as one weights file still shows as the source", async ({ page }) => {
  // The picker hands over the .gguf file inside the item's folder. The source card stayed on
  // "Select a model to convert..." with Convert enabled, because the lookup wanted equal paths.
  await mockWorkspace(page, []);
  await page.route("**/api/models/list-weights*", (route) =>
    route.fulfill({
      json: { path: GGUF_ITEM.path, format: "GGUF", broken: false, brokenReason: null,
        files: [{ name: "mock-model-1b-Q4_K_M.gguf", path: GGUF_FILE, sizeGb: 0.5, role: "main" }] },
    }),
  );
  await goTo(page, "Tools", "Conversion");
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  await page.locator(".model-select-item", { hasText: GGUF_ITEM.name }).click();
  await expect(page.locator(".model-selected-card")).toContainText(GGUF_ITEM.name);
  await expect(page.getByRole("button", { name: "Select a model to convert..." })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Convert to MLX" })).toBeEnabled();
  // The page says what really happens with a GGUF: the original model is downloaded and converted.
  // It used to warn about "re-quantizing an already quantized source", which is not what runs.
  const note = page.locator(".callout.warning", { hasText: "A GGUF source is downloaded again in full size" });
  await expect(note).toContainText("downloaded from Hugging Face");
  await expect(page.getByText("Re-quantizing an already quantized source")).toHaveCount(0);
});

test("Conversion: a backend error is shown in the dialog", async ({ page }) => {
  await mockWorkspace(page, []);
  const message = "Not enough free disk space for the converted model: about 2.45 GB needed, 0.01 GB free for /mock/out. Free some space or choose another Output path.";
  await page.route("**/api/models/convert", (route) =>
    route.fulfill({ status: 400, json: { detail: { message, localized: message } } }),
  );
  await goTo(page, "Tools", "Conversion");
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  await page.locator(".model-select-item", { hasText: TRANSFORMERS_ITEM.name }).click();
  await page.getByRole("button", { name: "Convert to MLX" }).click();
  const modal = page.locator(".conversion-result-modal");
  await expect(modal.getByRole("heading", { name: "Conversion failed" }).first()).toBeVisible();
  await expect(modal).toContainText("Not enough free disk space");
  await modal.getByRole("button", { name: /Close|OK/ }).click();
  await expect(page.getByRole("button", { name: "Convert to MLX" })).toBeEnabled();
});

test("Conversion: Cancel stops the run and the dialog says it was cancelled", async ({ page }) => {
  await mockWorkspace(page, []);
  const message = "Conversion cancelled. Nothing was kept at /mock/Models/Mock-Model-4B-mlx.";
  let release: () => void = () => {};
  const cancelled = new Promise<void>((resolve) => { release = resolve; });
  let cancelCalls = 0;
  // The convert request stays open until Cancel is pressed, as the real one does.
  await page.route("**/api/models/convert", async (route) => {
    await cancelled;
    await route.fulfill({ status: 400, json: { detail: { message, localized: message } } });
  });
  await page.route("**/api/models/convert/cancel", async (route) => {
    cancelCalls += 1;
    await route.fulfill({ json: { cancelled: true } });
    release();
  });
  await goTo(page, "Tools", "Conversion");
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  await page.locator(".model-select-item", { hasText: TRANSFORMERS_ITEM.name }).click();
  await page.getByRole("button", { name: "Convert to MLX" }).click();
  const modal = page.locator(".conversion-result-modal");
  await expect(modal.getByRole("heading", { name: "Converting model" }).first()).toBeVisible();
  await modal.getByRole("button", { name: "Cancel conversion" }).click();
  await expect(modal).toContainText("Conversion cancelled. Nothing was kept");
  await expect(modal.getByRole("button", { name: "Cancel conversion" })).toHaveCount(0);
  expect(cancelCalls).toBe(1);
  await modal.getByRole("button", { name: /Close|OK/ }).click();
  await expect(page.getByRole("button", { name: "Convert to MLX" })).toBeEnabled();
});

test("Conversion: the dialog shows the converter's own progress, bytes written and no made-up percentage", async ({ page }) => {
  await mockWorkspace(page, []);
  let release: () => void = () => {};
  const cancelled = new Promise<void>((resolve) => { release = resolve; });
  let snapshot = progress("conversion", { phase: "preparing", message: "Checking the source" });
  await page.route("**/api/models/convert/progress", (route) => route.fulfill({ json: snapshot }));
  await page.route("**/api/models/convert", async (route) => {
    await cancelled;
    await route.fulfill({ status: 400, json: { detail: "Conversion cancelled. Nothing was kept at /mock/Models/Mock-Model-4B-mlx." } });
  });
  await page.route("**/api/models/convert/cancel", async (route) => {
    await route.fulfill({ json: { cancelled: true } });
    release();
  });
  await goTo(page, "Tools", "Conversion");
  await page.getByRole("button", { name: "Select a model to convert..." }).click();
  await page.locator(".model-select-item", { hasText: TRANSFORMERS_ITEM.name }).click();
  await page.getByRole("button", { name: "Convert to MLX" }).click();
  const modal = page.locator(".conversion-result-modal");
  const bar = modal.getByRole("progressbar");
  const active = modal.locator(".live-progress__phase--active");
  await expect(active).toContainText("Resolving source");
  await expect(bar).not.toHaveAttribute("aria-valuenow", /.*/);
  // The old dialog always listed "Fetching weights" and "Verifying output", whatever the run did.
  await expect(modal.locator(".live-progress__phases")).not.toContainText("Verifying output");

  // A Hub source: files are counted, so the bar has a value.
  snapshot = progress("conversion", { phase: "downloading", step: 5, totalSteps: 11 });
  await expect(active).toContainText("5 of 11 files");
  await expect(bar).toHaveAttribute("aria-valuenow", "45");

  // Writing: bytes on disk are measured, the final size is an estimate and is called one.
  snapshot = progress("conversion", { phase: "saving", detail: { bytesWritten: 1024 ** 3, expectedBytes: 2 * 1024 ** 3, bitsPerWeight: 4.5 } });
  await expect(active).toContainText("1.00 GB written, about 2.0 GB expected (estimate)");
  await expect(bar).not.toHaveAttribute("aria-valuenow", /.*/);
  await shot(page, "27-conversion-progress");
  await modal.getByRole("button", { name: "Cancel conversion" }).click();
  await expect(modal).toContainText("Conversion cancelled");
});

test("Conversion at 375 px: no sideways scroll", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 800 });
  await mockWorkspace(page, []);
  await goTo(page, "Tools", "Conversion");
  await expect(page.getByRole("heading", { name: "MLX Conversion" })).toBeVisible();
  await noSidewaysScroll(page);
  await shot(page, "27-conversion-375");
});
