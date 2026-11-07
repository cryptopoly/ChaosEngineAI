/** Gallery provenance: the Storyboard project a still, clip, track or voice item was made for (set by the backend, absent otherwise). */

export interface ProjectTagged {
  projectId?: string | null;
  projectName?: string | null;
  /** ISO text (stills, clips) or unix seconds (music, voice); only used to order projects by latest output. */
  createdAt?: string | number | null;
}

export interface ProjectOption {
  id: string;
  name: string;
  count: number;
  /** Milliseconds of the newest output in the project (0 when unknown). */
  latest: number;
}

export const ALL_PROJECTS = "all";
export const NO_PROJECT = "__none__";
/** From this many projects on, the dropdown gets a search box. */
export const SEARCH_FROM = 8;

function millis(value: string | number | null | undefined): number {
  if (typeof value === "number") return value < 1e11 ? value * 1000 : value;
  const parsed = value ? Date.parse(value) : NaN;
  return Number.isNaN(parsed) ? 0 : parsed;
}

/** One option per project that has outputs, the one with the newest output first, then by name. */
export function projectOptions(items: ProjectTagged[]): ProjectOption[] {
  const byId = new Map<string, ProjectOption>();
  for (const item of items) {
    if (!item.projectId) continue;
    const at = millis(item.createdAt);
    const entry = byId.get(item.projectId);
    if (entry) {
      entry.count += 1;
      entry.latest = Math.max(entry.latest, at);
    } else byId.set(item.projectId, { id: item.projectId, name: item.projectName || item.projectId, count: 1, latest: at });
  }
  return [...byId.values()].sort((a, b) => b.latest - a.latest || a.name.localeCompare(b.name));
}

/** How many outputs belong to no project. */
export function untaggedCount(items: ProjectTagged[]): number {
  return items.filter((item) => !item.projectId).length;
}

export function filterByProject<T extends ProjectTagged>(items: T[], projectId: string): T[] {
  if (projectId === ALL_PROJECTS) return items;
  if (projectId === NO_PROJECT) return items.filter((item) => !item.projectId);
  return items.filter((item) => item.projectId === projectId);
}

/** The options whose name contains the search text (case-insensitive); all of them for an empty search. */
export function searchProjects(options: ProjectOption[], query: string): ProjectOption[] {
  const needle = query.trim().toLowerCase();
  return needle ? options.filter((option) => option.name.toLowerCase().includes(needle)) : options;
}
