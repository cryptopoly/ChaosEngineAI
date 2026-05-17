import { defineConfig } from "vitest/config";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const configDir = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    setupFiles: ["./src/test-setup.ts"],
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    exclude: [
      "node_modules/**",
      "dist/**",
      "src-tauri/**",
      "vendor/**",
      ".venv/**",
      ".claude/worktrees/**",
    ],
  },
  resolve: {
    alias: {
      "@": resolve(configDir, "src"),
    },
  },
});
