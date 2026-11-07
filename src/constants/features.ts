import type { SidebarGroupId, TabId } from "../types";

/**
 * Master switch for the Music and Storyboard areas. Their tabs, groups and
 * placeholder views are registered, but stay out of the sidebar and the
 * router until the backends land and this flips to true. The browser E2E run
 * turns it on with VITE_CHAOSENGINE_HOME_STUDIO=1; a normal build leaves it off.
 */
export const HOME_STUDIO_ENABLED = import.meta.env.VITE_CHAOSENGINE_HOME_STUDIO === "1";

export const HOME_STUDIO_GROUPS: readonly SidebarGroupId[] = ["music", "storyboard"];

export const HOME_STUDIO_TABS: readonly TabId[] = [
  "music-models",
  "music-studio",
  "music-library",
  "story-projects",
  "story-board",
  "story-mixer",
];

export function isHomeStudioTab(tab: TabId): boolean {
  return HOME_STUDIO_TABS.includes(tab);
}
