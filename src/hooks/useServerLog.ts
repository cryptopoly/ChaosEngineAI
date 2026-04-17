import { useEffect, useState } from "react";
import { apiFetch } from "../api";
import type { TabId } from "../types";

export function useServerLog(
  activeTab: TabId,
  backendOnline: boolean,
) {
  const [serverLogEntries, setServerLogEntries] = useState<Array<{ ts: string; level: string; message: string }>>([]);

  // SSE connection for server logs
  useEffect(() => {
    if (activeTab !== "server" || !backendOnline) return;

    let cancelled = false;
    let currentAbort: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    async function connect() {
      if (cancelled) return;
      reconnectTimer = null;
      const controller = new AbortController();
      currentAbort = controller;
      try {
        const response = await apiFetch("/api/server/logs/stream", {
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error(`Server log stream failed: HTTP ${response.status}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() ?? "";
          for (const event of events) {
            const dataLine = event
              .split("\n")
              .find((line) => line.startsWith("data: "));
            if (!dataLine) {
              continue;
            }
            const payload = dataLine.slice(6);
            try {
              const entry = JSON.parse(payload);
              if (entry.level === "debug") continue;
              setServerLogEntries((prev) => {
                const next = [...prev, entry];
                return next.length > 100 ? next.slice(-100) : next;
              });
            } catch {
              // ignore malformed data
            }
          }
        }
      } catch {
        if (!cancelled) {
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            void connect();
          }, 3000);
        }
      } finally {
        if (!cancelled && !reconnectTimer) {
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            void connect();
          }, 3000);
        }
        currentAbort = null;
      }
    }

    void connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      currentAbort?.abort();
    };
  }, [activeTab, backendOnline]);

  return {
    serverLogEntries,
  };
}
