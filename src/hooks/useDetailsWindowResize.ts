import { useRef, useState } from "react";


/**
 * Tauri-only: grow / shrink the OS window when expandable detail panels
 * open inside the app. The first details-open captures the current
 * window size, scales it up (1.15× width capped at 1800, 1.10× height
 * capped at 1100), then restores the captured size when the last open
 * details panel closes.
 *
 * Tracks an open-count so multiple simultaneous details panels stack
 * cleanly — the first to open does the resize, intermediate opens are
 * no-ops, the last to close does the restore.
 *
 * No-op outside Tauri (the dynamic ``isTauri()`` check returns false in
 * a plain browser preview, so the hook quietly does nothing).
 *
 * Extracted from ``src/App.tsx`` as part of the v0.8.0 refactor.
 */
export function useDetailsWindowResize() {
  const originalWindowSizeRef = useRef<{ width: number; height: number } | null>(null);
  const [, setOpenDetailsCount] = useState(0);

  async function handleDetailsToggle(opened: boolean) {
    try {
      const { isTauri } = await import("@tauri-apps/api/core");
      if (!isTauri()) return;
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      setOpenDetailsCount((prev) => {
        const next = opened ? prev + 1 : Math.max(0, prev - 1);
        void (async () => {
          if (next > 0 && prev === 0) {
            const size = await win.innerSize();
            originalWindowSizeRef.current = { width: size.width, height: size.height };
            await win.setSize(new (await import("@tauri-apps/api/window")).LogicalSize(
              Math.min(1800, Math.round(size.width * 1.15)),
              Math.min(1100, Math.round(size.height * 1.1)),
            ));
          } else if (next === 0 && prev > 0 && originalWindowSizeRef.current) {
            const { LogicalSize } = await import("@tauri-apps/api/window");
            await win.setSize(new LogicalSize(originalWindowSizeRef.current.width, originalWindowSizeRef.current.height));
            originalWindowSizeRef.current = null;
          }
        })();
        return next;
      });
    } catch { /* Not running in Tauri */ }
  }

  return { handleDetailsToggle };
}
