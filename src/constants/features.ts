import type { SidebarGroupId, TabId } from "../types";

/**
 * Master switch for the Music and Storyboard areas. Their tabs, groups and
 * placeholder views are registered, but stay out of the sidebar and the
 * router until the backends land and this flips to true.
 */
export const HOME_STUDIO_ENABLED = false;

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
