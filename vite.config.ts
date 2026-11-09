import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  // One dependency cache per checkout and per server. The default (node_modules/.vite) is shared by
  // every worktree that links node_modules, and two dev servers writing it at once leave the app blank.
  cacheDir: process.env.CHAOSENGINE_VITE_CACHE_DIR || path.resolve(__dirname, ".vite-cache"),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
  },
});
