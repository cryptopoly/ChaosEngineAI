import { defineConfig } from "@playwright/test";

// Config for the review captures: no webServer block, the servers are started by hand on throwaway HOME folders
// (see README.md, "How the screenshots were made"). Copy to e2e/ beside playwright.config.ts to run.
const WEB_PORT = Number(process.env.E2E_WEB_PORT ?? 8981);

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results-review",
  reporter: [["list"]],
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 180_000,
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    headless: true,
    chromiumSandbox: true,
    trace: "off",
    actionTimeout: 8_000,
    screenshot: "off",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
