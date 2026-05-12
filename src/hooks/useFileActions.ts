import { openHtmlChallengeFile, revealModelPath } from "../api";


function fileUrlFromPath(path: string): string {
  if (/^(https?|file):\/\//i.test(path)) return path;
  const normalized = path.replace(/\\/g, "/");
  const encoded = normalized.split("/").map((part) => encodeURIComponent(part)).join("/");
  return `${normalized.startsWith("/") ? "file://" : "file:///"}${encoded}`;
}


/**
 * Cross-platform file / URL action handlers used across the My Models /
 * library / chat / images / video tabs. Each handler tries the backend
 * first when it's online (so reveal-in-finder uses the same persisted
 * path resolution as the rest of the app) and falls back to Tauri's
 * opener plugin when the backend is unreachable.
 *
 * Extracted from ``src/App.tsx`` as part of the v0.8.0 refactor.
 */
export function useFileActions(
  backendOnline: boolean,
  setError: (msg: string | null) => void,
) {
  async function handleRevealPath(path: string) {
    try {
      if (backendOnline) {
        await revealModelPath(path);
        return;
      }
    } catch { /* fallback below */ }
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      const parentDir = path.replace(/\/[^/]+$/, "");
      await tauriInvoke("plugin:opener|open_path", { path: parentDir });
    } catch {
      setError("Could not open file location. Try navigating manually to: " + path);
    }
  }

  async function handleOpenFilePath(path: string) {
    if (backendOnline) {
      try {
        await openHtmlChallengeFile(path);
        return;
      } catch { /* fallback below */ }
    }
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      await tauriInvoke("plugin:opener|open_url", { url: fileUrlFromPath(path) });
      return;
    } catch { /* fall through */ }
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      await tauriInvoke("plugin:opener|open_path", { path });
    } catch {
      setError("Could not open file. Try opening this path manually: " + path);
    }
  }

  async function handleOpenExternalUrl(url: string) {
    if (/^(\/|[A-Za-z]:[\\/])/.test(url)) {
      await handleOpenFilePath(url);
      return;
    }
    try {
      const { invoke: tauriInvoke } = await import("@tauri-apps/api/core");
      await tauriInvoke("plugin:opener|open_url", { url });
      return;
    } catch { /* fall through */ }
    try {
      const opened = window.open(url, "_blank", "noopener,noreferrer");
      if (opened) return;
    } catch { /* fall through */ }
    setError(`Could not open link. Try opening this URL manually: ${url}`);
  }

  return {
    handleRevealPath,
    handleOpenFilePath,
    handleOpenExternalUrl,
  };
}
