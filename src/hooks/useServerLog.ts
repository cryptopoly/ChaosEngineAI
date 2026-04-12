import { useEffect, useState } from "react";
import { resolveApiBase } from "../api";
import type { TabId } from "../types";

export function useServerLog(
  activeTab: TabId,
  backendOnline: boolean,
) {
  const [serverLogEntries, setServerLogEntries] = useState<Array<{ ts: string; level: string; message: string }>>([]);

  // SSE connection for server logs
  useEffect(() => {
    if (activeTab !== "server" || !backendOnline) return;

    let eventSource: EventSource | null = null;
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    function connect(base: string) {
      if (cancelled) return;
      eventSource = new EventSource(`${base}/api/server/logs/stream`);
      eventSource.onmessage = (event) => {
        try {
          const entry = JSON.parse(event.data);
          if (entry.level === "debug") return;
          setServerLogEntries((prev) => {
            const next = [...prev, entry];
            return next.length > 100 ? next.slice(-100) : next;
          });
        } catch {
          // ignore malformed data
        }
      };
      eventSource.onerror = () => {
        eventSource?.close();
        if (!cancelled) {
          reconnectTimer = setTimeout(() => connect(base), 3000);
        }
      };
    }

    void (async () => {
      const base = await resolveApiBase();
      connect(base);
    })();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      eventSource?.close();
    };
  }, [activeTab, backendOnline]);

  return {
    serverLogEntries,
  };
}
