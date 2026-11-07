/** Gallery provenance: the Storyboard project a still or clip was made for (set by the backend, absent otherwise). */

export interface ProjectTagged {
  projectId?: string | null;
  projectName?: string | null;
}

export interface ProjectOption {
  id: string;
  name: string;
  count: number;
}

export const ALL_PROJECTS = "all";

/** One option per project that has outputs, most outputs first, then by name. */
export function projectOptions(items: ProjectTagged[]): ProjectOption[] {
  const byId = new Map<string, ProjectOption>();
  for (const item of items) {
    if (!item.projectId) continue;
    const entry = byId.get(item.projectId);
    if (entry) entry.count += 1;
    else byId.set(item.projectId, { id: item.projectId, name: item.projectName || item.projectId, count: 1 });
  }
  return [...byId.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

export function filterByProject<T extends ProjectTagged>(items: T[], projectId: string): T[] {
  return projectId === ALL_PROJECTS ? items : items.filter((item) => item.projectId === projectId);
}
