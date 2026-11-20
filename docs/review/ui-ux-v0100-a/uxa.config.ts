import { defineConfig } from "@playwright/test";
// Review-only config: no webServer block; the servers are started by hand on throwaway HOME folders.
export default defineConfig({
  testDir: "./tests",
  outputDir: "E:/tmp/uxa/test-results",
  reporter: [["list"]],
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180_000,
  use: { baseURL: `http://127.0.0.1:${process.env.E2E_WEB_PORT ?? 8960}`, headless: true, chromiumSandbox: true, actionTimeout: 15_000, trace: "off", screenshot: "off" },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
