// UI/UX review, pass 3: Server with a loaded model, LAN access and events; Settings in their used states; the
// backend offline; German; the tabs sidebar; Dashboard with a model and processes. All mocked, nothing runs.
import { test, expect } from "./fixtures";
import type { Page } from "@playwright/test";
import { API_BASE, TOKEN, openApp, openTab } from "./helpers";
import { DESKTOP, PHONE, dumpTexts, shoot, top } from "./uxaShared";

test.setTimeout(240_000);

const NOW = Date.now() / 1000;
let id = 0;
function event(kind: string, offset: number, extra: Record<string, unknown> = {}) {
  id += 1;
  const ts = NOW - 600 + offset;
  return {
    id, ts, time: new Date(ts * 1000).toISOString().replace("T", " ").slice(0, 19), kind, level: "info",
    model: "Nemotron 3.5 Lightning GGUF Q4_0 (24 GB card)", modelRef: "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
    requestId: "a1b2c3d4e5", engine: "llama.cpp", cacheStrategy: "native", speculative: null, message: kind, data: {}, detail: {}, ...extra,
  };
}
const EVENTS = [
  event("requested", 0, { data: { backend: "auto" } }),
  event("loading", 0.5, { data: { bytes: 19_100_000_000 } }),
  event("loaded", 18.5, { data: { seconds: 18, memoryBytes: 18_100_000_000, warm: false }, detail: { command: "llama-server.exe -m E:\\AI Models\\nemotron.gguf --port 8960 --api-key ***", env: ["CHAOSENGINE_PORT=8961"] } }),
  event("first_token", 40.3, { requestId: "f00dfeed01", data: { prefillSeconds: 0.3 } }),
  event("generation", 42, { requestId: "f00dfeed01", data: { seconds: 2, completionTokens: 96, tokS: 233.1, promptTokens: 9349 } }),
  event("idle", 42, { requestId: "f00dfeed01" }),
  event("requested", 100, { model: "Broken GGUF", modelRef: "x/broken", requestId: "deadbeef00", engine: null }),
  event("load_failed", 103, {
    model: "Broken GGUF", modelRef: "x/broken", requestId: "deadbeef00", engine: "llama.cpp", level: "error",
    message: "llama-server exited: failed to load model", data: { seconds: 3, reason: "error" },
    detail: { command: "llama-server.exe -m E:\\Models\\broken.gguf --port 8960 --api-key ***", stderr: ["gguf_init_from_file: invalid magic", "llama_model_load: error loading model"], env: ["CHAOSENGINE_PORT=8961"], error: "RuntimeError: llama-server exited: failed to load model" },
  }),
];

const LOADED_MODEL = {
  ref: "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", name: "Nemotron 3.5 Lightning GGUF Q4_0 (24 GB card)",
  canonicalRepo: "bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", backend: "llama.cpp", source: "library", engine: "llama.cpp",
  cacheStrategy: "native", cacheBits: 0, fp16Layers: 0, fusedAttention: false, fitModelInMemory: true, contextTokens: 16384,
  loadedAt: "2026-11-20 10:00:00", path: "E:\\AI Models\\nemotron.gguf", runtimeTarget: "E:\\AI Models\\nemotron.gguf", speculativeDecoding: false, treeBudget: 0,
  capabilities: { supportsTools: true, supportsVision: false, supportsReasoning: true },
};

type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

async function mockEvents(page: Page) {
  const body = EVENTS.map((e) => `data: ${JSON.stringify(e)}\n\n`).join("") + ": keepalive\n\n";
  await page.route("**/api/server/events/stream", (route) =>
    route.fulfill({ status: 200, headers: { "content-type": "text/event-stream", "access-control-allow-origin": "*" }, body }));
}

async function patchWorkspace(page: Page, patch: (w: Json) => void) {
  await page.route("**/api/workspace", async (route) => {
    const response = await route.fetch();
    const body = (await response.json()) as Json;
    patch(body);
    await route.fulfill({ response, json: body });
  });
}

function loadedServer(w: Json) {
  w.runtime = {
    ...w.runtime, state: "loaded", loadedModel: LOADED_MODEL,
    warmModels: [
      { ref: LOADED_MODEL.ref, name: LOADED_MODEL.name, engine: "llama.cpp", warm: true, active: true },
      { ref: "google/gemma-4-12B-it-qat-q4_0-gguf", name: "Gemma 4 12B Official QAT GGUF", engine: "llama.cpp", warm: true, active: false },
    ],
  };
  w.server = {
    ...w.server, status: "running", remoteAccessActive: true, bindHost: "0.0.0.0", lanUrls: ["http://192.168.1.42:8961/v1"],
    loadedModelName: LOADED_MODEL.name, requestsServed: 128, activeConnections: 2,
  };
  w.settings = { ...w.settings, allowRemoteConnections: true };
}

for (const phone of [false, true]) {
  test.describe(phone ? "phone" : "desktop", () => {
    test.use({ viewport: phone ? PHONE : DESKTOP });
    test.afterAll(() => dumpTexts(`p3-${phone ? "m" : "d"}`));

    test("20 server: loaded, LAN on, connect open, activity with a failure, test modal", async ({ page }) => {
      await mockEvents(page);
      await patchWorkspace(page, loadedServer);
      await openApp(page);
      await top(page, "Server");
      await page.getByText("Connect your app", { exact: true }).click();
      await shoot(page, "20-server-loaded-lan-connect", phone);
      await page.getByRole("button", { name: "Test", exact: true }).first().click();
      await shoot(page, "20-server-test-modal", phone, false);
      await page.getByRole("button", { name: "Close", exact: true }).click();
      await page.getByRole("tab", { name: "Activity" }).click();
      await page.waitForTimeout(800);
      await shoot(page, "20-server-activity-events", phone);
      await page.getByRole("button", { name: "Latest failure" }).click();
      await shoot(page, "20-server-activity-failure", phone);
    });

    test("21a settings: experimental tabs on, tabs sidebar", async ({ page }) => {
      await openApp(page);
      await top(page, "Settings");
      await page.getByLabel("Show experimental tabs").check();
      await page.locator(".settings-content").getByText("Tabs", { exact: true }).first().click();
      await shoot(page, "21-settings-general-tabs-experimental", phone);
      await page.locator(".settings-content").getByText("Collapsible", { exact: true }).first().click();
      await page.getByLabel("Show experimental tabs").uncheck();
    });

    test("21b settings: providers form, agent permissions, accelerators settled, disk scanned", async ({ page }) => {
      await openApp(page);
      await top(page, "Settings");
      const sections = page.getByRole("tablist", { name: "Settings sections" });
      await sections.getByRole("tab", { name: "Providers", exact: true }).click();
      await page.getByRole("button", { name: "+ Add Provider" }).click();
      await shoot(page, "21-settings-providers-add", phone);
      await sections.getByRole("tab", { name: "Integrations", exact: true }).click();
      await page.locator("summary", { hasText: "Permissions" }).click();
      await shoot(page, "21-settings-integrations-permissions", phone);
      await sections.getByRole("tab", { name: "Accelerators", exact: true }).click();
      await page.waitForTimeout(8000);
      await shoot(page, "21-settings-accelerators-settled", phone);
      await sections.getByRole("tab", { name: "Disk usage", exact: true }).click();
      await page.getByRole("button", { name: "Scan disk usage" }).click();
      await page.waitForTimeout(3000);
      await shoot(page, "21-settings-disk-scanned", phone);
    });

    test("22 accelerators: install error under its row", async ({ page }) => {
      await page.route("**/api/setup/install-package", (route) => route.fulfill({
        status: 400,
        json: { detail: "Nunchaku (SVDQuant) is not installable from PyPI. Download the wheel that matches your torch, CUDA and Python versions from the Nunchaku GitHub releases page." },
      }));
      await openApp(page);
      await top(page, "Settings");
      await page.getByRole("tab", { name: "Accelerators" }).click();
      await page.locator('tr[data-accelerator-id="nunchaku"]').getByRole("button", { name: /Install/ }).click();
      await page.waitForTimeout(800);
      await shoot(page, "22-accelerators-install-error", phone);
    });

    test("23 backend offline", async ({ page }) => {
      await openApp(page);
      await top(page, "Dashboard");
      await page.route(`${API_BASE}/**`, (route) => route.abort("connectionrefused"));
      await expect(page.locator(".backend-offline-banner")).toBeVisible({ timeout: 40_000 });
      await shoot(page, "23-offline-dashboard", phone, false);
      await openTab(page, "Chat", "Chat");
      await shoot(page, "23-offline-chat", phone, false);
      await openTab(page, "Chat", "Models");
      await shoot(page, "23-offline-models", phone, false);
      await top(page, "Server");
      await shoot(page, "23-offline-server", phone, false);
      await top(page, "Settings");
      for (const s of ["Diagnostics", "Accelerators", "Disk usage", "Skills"]) {
        await page.getByRole("tablist", { name: "Settings sections" }).getByRole("tab", { name: s, exact: true }).click();
        await page.waitForTimeout(1500);
        await shoot(page, `23-offline-settings-${s.toLowerCase().replace(/ /g, "-")}`, phone, false);
      }
    });

    test("24 dashboard with a model and processes", async ({ page }) => {
      await patchWorkspace(page, (w) => {
        loadedServer(w);
        w.system.runningLlmProcesses = [
          { pid: 4242, name: "llama-server.exe", owner: "ChaosEngineAI", modelName: LOADED_MODEL.name, modelStatus: "active", kind: "llama_server", memoryGb: 18.4, cpuPercent: 12.5 },
          { pid: 5151, name: "python.exe", owner: "Other app", modelName: null, modelStatus: null, kind: "other", memoryGb: 6.2, cpuPercent: 0.4 },
        ];
      });
      await openApp(page);
      await top(page, "Dashboard");
      await shoot(page, "24-dashboard-loaded", phone);
    });

    test("25 German: dashboard, chat models, server, settings", async ({ page }) => {
      const headers = { Authorization: `Bearer ${TOKEN}`, "Content-Type": "application/json" };
      await page.request.patch(`${API_BASE}/api/settings`, { headers, data: { locale: "de" } });
      try {
        await page.goto("/");
        await expect(page.locator(".loading-state")).toHaveCount(0, { timeout: 60_000 });
        await page.waitForTimeout(1000);
        await shoot(page, "25-de-dashboard", phone);
        const menu = page.locator(".sidebar-menu-button");
        const nav = page.getByRole("navigation").first();
        for (const [index, name] of [[1, "chat-models"], [-2, "server"], [-1, "settings"]] as const) {
          if ((await menu.isVisible()) && (await menu.getAttribute("aria-expanded")) !== "true") await menu.click();
          const buttons = nav.locator(".nav-button, .nav-group-header");
          const count = await buttons.count();
          await buttons.nth(index < 0 ? count + index : index).click();
          if (name === "chat-models") await nav.locator(".nav-child-button").nth(3).click();
          await page.waitForTimeout(1500);
          await shoot(page, `25-de-${name}`, phone, false);
        }
      } finally {
        await page.request.patch(`${API_BASE}/api/settings`, { headers, data: { locale: "en" } });
      }
    });
  });
}
