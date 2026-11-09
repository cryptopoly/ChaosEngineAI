import { existsSync } from "node:fs";
import path from "node:path";
import { defineConfig } from "@playwright/test";

// Local-only E2E: Vite dev server on 8909, backend on 8910, throwaway HOME so
// the run never touches ~/.chaosengine, dummy token, headless sandboxed Chromium.
// See docs/research/playwright-security.md for the conditions this follows.
// E2E_WEB_PORT / E2E_API_PORT let a second run (another worktree) use other ports.
const WEB_PORT = Number(process.env.E2E_WEB_PORT ?? 8909);
const API_PORT = Number(process.env.E2E_API_PORT ?? 8910);
const TOKEN = "e2e-dummy-token";
const repoRoot = path.resolve(__dirname, "..");
const home = path.join(__dirname, ".run", "home");


function pythonBin(): string {
  if (process.env.E2E_PYTHON) return process.env.E2E_PYTHON;
  // Walk up so a worktree under .claude/worktrees/ finds the main checkout's .venv.
  const rel = path.join(".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  for (let dir = repoRoot; ; dir = path.dirname(dir)) {
    if (existsSync(path.join(dir, rel))) return path.join(dir, rel);
    if (dir === path.dirname(dir)) return "python3";
  }
}

export default defineConfig({
  testDir: "./tests",
  outputDir: "./test-results",
  reporter: [["list"], ["html", { outputFolder: "./playwright-report", open: "never" }]],
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    headless: true,
    chromiumSandbox: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
  webServer: [
    {
      // Fresh data dir on every run so the terms screen appears on a clean first run.
      command:
        `rm -rf "${home}" && mkdir -p "${home}" && ` +
        `HOME="${home}" USERPROFILE="${home}" CHAOSENGINE_PORT=${API_PORT} CHAOSENGINE_HOST=127.0.0.1 ` +
        `CHAOSENGINE_API_TOKEN=${TOKEN} CHAOSENGINE_ALLOWED_ORIGINS=http://127.0.0.1:${WEB_PORT} ` +
        `HF_TOKEN= HUGGING_FACE_HUB_TOKEN= "${pythonBin()}" -m backend_service.app`,
      cwd: repoRoot,
      url: `http://127.0.0.1:${API_PORT}/api/health`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `npx vite --host 127.0.0.1 --port ${WEB_PORT} --strictPort`,
      cwd: repoRoot,
      env: {
        VITE_CHAOSENGINE_API_BASE: `http://127.0.0.1:${API_PORT}`,
        VITE_CHAOSENGINE_API_TOKEN: TOKEN,
        // Music and Storyboard stay hidden in a normal build; the run turns them on.
        VITE_CHAOSENGINE_HOME_STUDIO: "1",
        // Own dependency cache, so a test run never rewrites the one `npm run tauri:dev` is serving.
        CHAOSENGINE_VITE_CACHE_DIR: path.join(__dirname, ".run", "vite-cache"),
      },
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
