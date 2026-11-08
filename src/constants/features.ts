import type { SidebarGroupId, TabId } from "../types";

/**
 * The Music and Storyboard areas ship from v0.10.0 and are always on. The flag is kept so the tab
 * and sidebar code can still be tested with the areas switched off (tests pass the value in).
 */
export const HOME_STUDIO_ENABLED = true;

export const HOME_STUDIO_GROUPS: readonly SidebarGroupId[] = ["music"];

export const HOME_STUDIO_TABS: readonly TabId[] = [
  "music-models",
  "music-studio",
  "music-library",
  "storyboard",
];

export function isHomeStudioTab(tab: TabId): boolean {
  return HOME_STUDIO_TABS.includes(tab);
}
