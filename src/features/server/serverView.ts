/**
 * Which view the Server page shows (Overview or Activity), remembered per
 * user. The old Logs tab id redirects here with "activity"
 * (``constants/tabRedirects.ts``), so the write fires an event the open page
 * listens to.
 */
import { useEffect, useState } from "react";

export type ServerViewId = "overview" | "activity";

export const SERVER_VIEW_KEY = "chaosengine.server.view";
const SERVER_VIEW_EVENT = "chaosengine:server-view";

export function isServerView(value: unknown): value is ServerViewId {
  return value === "overview" || value === "activity";
}

export function readServerView(): ServerViewId {
  try {
    const stored = window.localStorage.getItem(SERVER_VIEW_KEY);
    return isServerView(stored) ? stored : "overview";
  } catch {
    return "overview";
  }
}

export function writeServerView(view: ServerViewId): void {
  try {
    window.localStorage.setItem(SERVER_VIEW_KEY, view);
  } catch {
    // not remembered; the event below still switches an open page
  }
  try {
    window.dispatchEvent(new CustomEvent(SERVER_VIEW_EVENT, { detail: view }));
  } catch {
    // no window (tests without a DOM)
  }
}

export function useServerView(): [ServerViewId, (view: ServerViewId) => void] {
  const [view, setView] = useState<ServerViewId>(() => readServerView());
  useEffect(() => {
    const sync = (event: Event) => {
      const next = (event as CustomEvent<unknown>).detail;
      setView(isServerView(next) ? next : readServerView());
    };
    window.addEventListener(SERVER_VIEW_EVENT, sync);
    return () => window.removeEventListener(SERVER_VIEW_EVENT, sync);
  }, []);
  return [view, writeServerView];
}
