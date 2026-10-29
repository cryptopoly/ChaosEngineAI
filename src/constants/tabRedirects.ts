/**
 * One redirect map for the tab ids that the Models pages replaced.
 *
 * Discover and My Models became one Models page per area. The old ids
 * are no longer tabs; anything that still asks for one (a stale caller,
 * a persisted sidebar memory) is sent to the area's Models page, and a
 * navigation carries a preset for the page's availability filter so the
 * old buttons keep their meaning: an old Discover id opens the page on
 * "Available", an old My Models id on "Installed".
 *
 * App applies ``resolveTabRequest`` in the single wrapper around its tab
 * setter, and ``useSidebarPrefs`` applies ``normalizeStoredTabId`` when it
 * reads ``lastChildByGroup`` from localStorage.
 */

import type { TabId } from "../types";
import type { ModelsArea } from "../components/models/modelEntry";

/** Tab ids that no longer exist. */
export type LegacyTabId = "online-models" | "my-models" | "image-discover" | "video-discover";

/** Availability filter a navigation can preset on a Models page. */
export type ModelsPreset = "installed" | "available";

export interface TabNavigation {
  tab: TabId;
  /** Applied to the target Models page's availability filter. */
  modelsPreset?: ModelsPreset;
}

/** What a tab setter accepts: a tab id, a legacy id, or an explicit
 * navigation with a preset. */
export type TabRequest = TabId | LegacyTabId | TabNavigation;

export const MODELS_TAB_BY_AREA: Record<ModelsArea, TabId> = {
  chat: "chat-models",
  image: "image-models",
  video: "video-models",
};

export const LEGACY_TAB_REDIRECTS: Record<LegacyTabId, TabNavigation> = {
  "online-models": { tab: "chat-models", modelsPreset: "available" },
  "my-models": { tab: "chat-models", modelsPreset: "installed" },
  "image-discover": { tab: "image-models", modelsPreset: "available" },
  "video-discover": { tab: "video-models", modelsPreset: "available" },
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
  return null;
}

/** Resolve whatever a caller passed into a real tab plus an optional
 * preset. Legacy ids redirect; ``null`` / empty input resolves to
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
      return { tab: redirect.tab, modelsPreset: request.modelsPreset ?? redirect.modelsPreset };
    }
    return request.modelsPreset ? { tab: request.tab, modelsPreset: request.modelsPreset } : { tab: request.tab };
  }
  if (isLegacyTabId(request)) return { ...LEGACY_TAB_REDIRECTS[request] };
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

/** ``lastChildByGroup`` with every legacy id mapped. */
export function migrateLastChildByGroup<K extends string>(
  stored: Partial<Record<K, unknown>> | null | undefined,
): Partial<Record<K, string>> {
  const result: Partial<Record<K, string>> = {};
  if (!stored || typeof stored !== "object") return result;
  for (const [group, value] of Object.entries(stored) as Array<[K, unknown]>) {
    const normalized = normalizeStoredTabId(value);
    if (normalized) result[group] = normalized;
  }
  return result;
}
