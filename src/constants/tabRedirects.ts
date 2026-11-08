/**
 * One redirect map for tab ids that no longer exist.
 *
 * Discover and My Models became one Models page per area. The old ids
 * are no longer tabs; anything that still asks for one (a stale caller,
 * a persisted sidebar memory) is sent to the area's Models page, and a
 * navigation carries a preset for the page's availability filter so the
 * old buttons keep their meaning: an old Discover id opens the page on
 * "Available", an old My Models id on "Installed".
 *
 * The Storyboard's Projects, Board and Mixer became one Projects entry
 * (D-65). Their ids open that entry with a route: the list, or the last
 * open project on its Storyboard or Mixer tab.
 *
 * App applies ``resolveTabRequest`` in the single wrapper around its tab
 * setter, and ``useSidebarPrefs`` applies ``normalizeStoredTabId`` when it
 * reads ``lastChildByGroup`` from localStorage.
 */

import type { TabId } from "../types";
import type { ModelsArea } from "../components/models/modelEntry";
import type { ProjectTabId } from "../features/storyboard/projectRoute";

/** Tab ids that no longer exist. */
export type LegacyTabId =
  | "online-models"
  | "my-models"
  | "image-discover"
  | "video-discover"
  | "story-projects"
  | "story-board"
  | "story-mixer";

/** Availability filter a navigation can preset on a Models page. */
export type ModelsPreset = "installed" | "available";

/** Where the Projects entry opens: a project (null: the list) and one of its tabs (D-65). */
export interface StoryboardNavigation {
  /** Omitted: the last open project, or the list when none. Null: the list. */
  projectId?: string | null;
  tab?: ProjectTabId;
}

export interface TabNavigation {
  tab: TabId;
  /** Applied to the target Models page's availability filter. */
  modelsPreset?: ModelsPreset;
  /** Applied to the Projects view's route before it shows. */
  storyboard?: StoryboardNavigation;
}

/** What a tab setter accepts: a tab id, a legacy id, or an explicit
 * navigation with a preset. */
export type TabRequest = TabId | LegacyTabId | TabNavigation;

export const MODELS_TAB_BY_AREA: Record<ModelsArea, TabId> = {
  chat: "chat-models",
  image: "image-models",
  video: "video-models",
  music: "music-models",
};

export const LEGACY_TAB_REDIRECTS: Record<LegacyTabId, TabNavigation> = {
  "online-models": { tab: "chat-models", modelsPreset: "available" },
  "my-models": { tab: "chat-models", modelsPreset: "installed" },
  "image-discover": { tab: "image-models", modelsPreset: "available" },
  "video-discover": { tab: "video-models", modelsPreset: "available" },
  "story-projects": { tab: "storyboard", storyboard: { projectId: null } },
  "story-board": { tab: "storyboard", storyboard: { tab: "storyboard" } },
  "story-mixer": { tab: "storyboard", storyboard: { tab: "mixer" } },
};

export function isLegacyTabId(value: unknown): value is LegacyTabId {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(LEGACY_TAB_REDIRECTS, value);
}

/** The Models page of ``area``, optionally with a preset. */
export function modelsPageRequest(area: ModelsArea, preset?: ModelsPreset): TabNavigation {
  return preset ? { tab: MODELS_TAB_BY_AREA[area], modelsPreset: preset } : { tab: MODELS_TAB_BY_AREA[area] };
}

/** The Models area a tab id belongs to, or ``null``. */
export function modelsAreaForTab(tab: TabId): ModelsArea | null {
  if (tab === "chat-models") return "chat";
  if (tab === "image-models") return "image";
  if (tab === "video-models") return "video";
  if (tab === "music-models") return "music";
  return null;
}

function navigation(tab: TabId, modelsPreset?: ModelsPreset, storyboard?: StoryboardNavigation): TabNavigation {
  const result: TabNavigation = { tab };
  if (modelsPreset) result.modelsPreset = modelsPreset;
  if (storyboard) result.storyboard = { ...storyboard };
  return result;
}

/** Resolve whatever a caller passed into a real tab plus an optional
 * preset or route. Legacy ids redirect; ``null`` / empty input resolves to
 * ``null``. Unknown strings pass through as tab ids (the caller's type
 * already says they are). */
export function resolveTabRequest(request: TabRequest | null | undefined): TabNavigation | null {
  if (!request) return null;
  if (typeof request === "object") {
    // The type says ``tab`` is a real tab, but a stale caller can still
    // hold an old id at runtime.
    const tab: string = request.tab;
    if (isLegacyTabId(tab)) {
      const redirect = LEGACY_TAB_REDIRECTS[tab];
      return navigation(redirect.tab, request.modelsPreset ?? redirect.modelsPreset, request.storyboard ?? redirect.storyboard);
    }
    return navigation(request.tab, request.modelsPreset, request.storyboard);
  }
  if (isLegacyTabId(request)) {
    const redirect = LEGACY_TAB_REDIRECTS[request];
    return navigation(redirect.tab, redirect.modelsPreset, redirect.storyboard);
  }
  return { tab: request };
}

/** A tab id read from storage, with legacy ids mapped to their page. The
 * preset is dropped: a remembered sidebar child is not an intent, and the
 * page keeps its own last filter. */
export function normalizeStoredTabId(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0) return null;
  if (isLegacyTabId(value)) return LEGACY_TAB_REDIRECTS[value].tab;
  return value;
}

/** Groups that no longer exist; what they remembered is dropped. */
const RETIRED_GROUPS: ReadonlySet<string> = new Set(["storyboard"]);

/** ``lastChildByGroup`` with every legacy id mapped and retired groups dropped. */
export function migrateLastChildByGroup<K extends string>(
  stored: Partial<Record<K, unknown>> | null | undefined,
): Partial<Record<K, string>> {
  const result: Partial<Record<K, string>> = {};
  if (!stored || typeof stored !== "object") return result;
  for (const [group, value] of Object.entries(stored) as Array<[K, unknown]>) {
    if (RETIRED_GROUPS.has(group)) continue;
    const normalized = normalizeStoredTabId(value);
    if (normalized) result[group] = normalized;
  }
  return result;
}
