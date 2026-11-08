/**
 * Where the Projects entry opens (D-65): the project list, or one project on one of its five tabs. The route
 * lives in local storage so the sidebar entry reopens where the user left off, and each project remembers
 * its own last tab. Anything that sends the user into a project (a gallery's "Use in storyboard", an old tab
 * id) writes the route first and then moves to the "storyboard" tab; an already open view hears the change
 * through a window event. The old single "open project" key is read once and migrated.
 */
import { useCallback, useEffect, useState } from "react";

export const PROJECT_TABS = ["overview", "cast", "storyboard", "mixer", "export"] as const;
export type ProjectTabId = (typeof PROJECT_TABS)[number];

export interface ProjectRoute {
  /** The open project, or null for the list. */
  projectId: string | null;
  /** The tab of the open project (its remembered tab when not given). */
  tab: ProjectTabId;
  /** Last tab per project id. */
  lastTabs: Record<string, ProjectTabId>;
}

export const ROUTE_KEY = "chaos.storyboard.route";
/** Before D-65: the id of the project the Board and Mixer worked on. */
export const LEGACY_OPEN_KEY = "chaos.storyboard.openProject";
export const ROUTE_EVENT = "chaos:storyboard-route";
/** At most this many projects keep a remembered tab. */
const LAST_TABS_KEEP = 50;

export function isProjectTab(value: unknown): value is ProjectTabId {
  return typeof value === "string" && (PROJECT_TABS as readonly string[]).includes(value);
}

const EMPTY: ProjectRoute = { projectId: null, tab: "overview", lastTabs: {} };

/** A stored value made safe: unknown tabs, ids and shapes fall back to the list. */
export function parseRoute(raw: string | null, legacyOpenId: string | null = null): ProjectRoute {
  let parsed: unknown = null;
  try {
    parsed = raw ? JSON.parse(raw) : null;
  } catch {
    parsed = null;
  }
  if (!parsed || typeof parsed !== "object") {
    // The old key held only the project the Board showed: open it on the Storyboard tab.
    return legacyOpenId ? { projectId: legacyOpenId, tab: "storyboard", lastTabs: { [legacyOpenId]: "storyboard" } } : { ...EMPTY };
  }
  const value = parsed as Record<string, unknown>;
  const lastTabs: Record<string, ProjectTabId> = {};
  if (value.lastTabs && typeof value.lastTabs === "object") {
    for (const [id, tab] of Object.entries(value.lastTabs as Record<string, unknown>)) {
      if (id && isProjectTab(tab)) lastTabs[id] = tab;
    }
  }
  const projectId = typeof value.projectId === "string" && value.projectId ? value.projectId : null;
  const tab = isProjectTab(value.tab) ? value.tab : (projectId && lastTabs[projectId]) || "overview";
  return { projectId, tab, lastTabs };
}

function storage(): Storage | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

export function readRoute(): ProjectRoute {
  const store = storage();
  if (!store) return { ...EMPTY };
  try {
    const raw = store.getItem(ROUTE_KEY);
    const legacy = raw ? null : store.getItem(LEGACY_OPEN_KEY);
    const route = parseRoute(raw, legacy || null);
    if (!raw && legacy) {
      store.setItem(ROUTE_KEY, JSON.stringify(route));
      store.removeItem(LEGACY_OPEN_KEY);
    }
    return route;
  } catch {
    return { ...EMPTY };
  }
}

export function writeRoute(route: ProjectRoute): void {
  const store = storage();
  try {
    store?.setItem(ROUTE_KEY, JSON.stringify(route));
  } catch {
    // storage unavailable: the route is simply not remembered
  }
  if (typeof window !== "undefined") window.dispatchEvent(new Event(ROUTE_EVENT));
}

/** The route after opening `projectId` (null: the list) on `tab`, or on the project's remembered tab. */
export function routeWith(current: ProjectRoute, projectId: string | null, tab?: ProjectTabId): ProjectRoute {
  if (!projectId) return { ...current, projectId: null };
  const nextTab = tab ?? current.lastTabs[projectId] ?? "overview";
  const lastTabs = { ...current.lastTabs };
  delete lastTabs[projectId];
  lastTabs[projectId] = nextTab; // most recent last
  const ids = Object.keys(lastTabs);
  for (const id of ids.slice(0, Math.max(0, ids.length - LAST_TABS_KEEP))) delete lastTabs[id];
  return { projectId, tab: nextTab, lastTabs };
}

/**
 * Open a project on a tab (or the list with `projectId` null) the next time the Projects view shows, or now
 * when it already shows. The caller then moves the app to the "storyboard" tab.
 */
export function openProjectAt(projectId: string | null, tab?: ProjectTabId): void {
  writeRoute(routeWith(readRoute(), projectId, tab));
}

/** A request that keeps the open project but picks the tab, or the list when no project was open. */
export function openLastProjectAt(tab: ProjectTabId): void {
  const current = readRoute();
  writeRoute(current.projectId ? routeWith(current, current.projectId, tab) : current);
}

/** A navigation's route: an explicit project (null: the list), or a tab of the last open project. */
export function applyStoryboardNavigation(request: { projectId?: string | null; tab?: ProjectTabId }): void {
  if (request.projectId !== undefined) openProjectAt(request.projectId, request.tab);
  else if (request.tab) openLastProjectAt(request.tab);
}

/** Kept for one release for callers that still name only a project: opens it on its remembered tab. */
export function readOpenProjectId(): string | null {
  return readRoute().projectId;
}

export function writeOpenProjectId(id: string | null): void {
  openProjectAt(id);
}

// ---- the open project's name, for the page header ----
const names = new Map<string, string>();

/** The shell reports the name once the project loads, so the page header can show it. */
export function setProjectName(projectId: string, name: string): void {
  if (names.get(projectId) === name) return;
  names.set(projectId, name);
  if (typeof window !== "undefined") window.dispatchEvent(new Event(ROUTE_EVENT));
}

export function projectName(projectId: string | null): string | null {
  return projectId ? (names.get(projectId) ?? null) : null;
}

// ---- reload lands back in Projects ----
const LAST_TAB_KEY = "chaos.app.lastTab";

/** Whether the app was on the Projects entry when it was last closed or reloaded. */
export function readReopenProjects(): boolean {
  try {
    return window.localStorage.getItem(LAST_TAB_KEY) === "storyboard";
  } catch {
    return false;
  }
}

export function writeLastAppTab(tab: string): void {
  try {
    window.localStorage.setItem(LAST_TAB_KEY, tab);
  } catch {
    // not remembered
  }
}

/** The stored route, kept in step with writes from anywhere in the app. */
export function useProjectRoute(): [ProjectRoute, (projectId: string | null, tab?: ProjectTabId) => void] {
  const [route, setRoute] = useState<ProjectRoute>(() => readRoute());
  useEffect(() => {
    const sync = () => setRoute(readRoute());
    window.addEventListener(ROUTE_EVENT, sync);
    return () => window.removeEventListener(ROUTE_EVENT, sync);
  }, []);
  const go = useCallback((projectId: string | null, tab?: ProjectTabId) => openProjectAt(projectId, tab), []);
  return [route, go];
}
