/**
 * Pure sidebar model: which tabs are visible, how they group, in which order
 * the sections list them, and where a group header navigates. The Sidebar
 * component only renders what this returns.
 */
import type { SidebarGroupId, TabId } from "../types";
import type { TabConfig } from "../constants";
import { HOME_STUDIO_ENABLED, isHomeStudioTab, sidebarGroups } from "../constants";

export type SidebarSection = "make" | "run";

export type SidebarItem =
  | { kind: "tab"; tab: TabConfig }
  | { kind: "group"; id: SidebarGroupId; label: string; children: TabConfig[]; section?: SidebarSection };

export type SidebarRow = SidebarItem | { kind: "section"; section: SidebarSection };

export function visibleSidebarTabs(
  allTabs: TabConfig[],
  platform: string | undefined,
  homeStudioEnabled: boolean = HOME_STUDIO_ENABLED,
): TabConfig[] {
  return allTabs.filter((tab) => {
    if (tab.id === "conversion" && platform && platform !== "Darwin") return false;
    if (!homeStudioEnabled && isHomeStudioTab(tab.id)) return false;
    return true;
  });
}

export function buildItems(visibleTabs: TabConfig[]): SidebarItem[] {
  const items: SidebarItem[] = [];
  const groupsSeen = new Set<SidebarGroupId>();

  for (const tab of visibleTabs) {
    if (!tab.group) {
      items.push({ kind: "tab", tab });
      continue;
    }
    if (groupsSeen.has(tab.group)) continue;
    groupsSeen.add(tab.group);

    const groupDef = sidebarGroups.find((g) => g.id === tab.group);
    if (!groupDef) {
      items.push({ kind: "tab", tab });
      continue;
    }
    const children = visibleTabs.filter((t) => t.group === tab.group);
    items.push({ kind: "group", id: groupDef.id, label: groupDef.label, children, section: groupDef.section });
  }

  return items;
}

/**
 * Dashboard first, then Chat (a group with no section label), then the Make
 * groups, then the Run groups (each under a thin section label), then the
 * remaining single tabs (Server, Logs, Settings). Group order inside a section follows `sidebarGroups`.
 */
export function arrangeRows(items: SidebarItem[]): SidebarRow[] {
  const groups = items.filter((i): i is Extract<SidebarItem, { kind: "group" }> => i.kind === "group");
  const singles = items.filter((i) => i.kind === "tab");
  const order = (a: { id: SidebarGroupId }, b: { id: SidebarGroupId }) =>
    sidebarGroups.findIndex((g) => g.id === a.id) - sidebarGroups.findIndex((g) => g.id === b.id);
  const rows: SidebarRow[] = [];
  const lead = singles.filter((i) => i.kind === "tab" && i.tab.id === "dashboard");
  rows.push(...lead);
  rows.push(...groups.filter((g) => !g.section).sort(order));
  const sectioned = (section: SidebarSection) => singles.filter((i) => i.kind === "tab" && i.tab.section === section);
  for (const section of ["make", "run"] as const) {
    // A standalone tab with a section (Projects) follows that section's groups.
    const inSection: SidebarItem[] = [...groups.filter((g) => g.section === section).sort(order), ...sectioned(section)];
    if (inSection.length === 0) continue;
    rows.push({ kind: "section", section }, ...inSection);
  }
  rows.push(...singles.filter((i) => !lead.includes(i) && !(i.kind === "tab" && i.tab.section)));
  return rows;
}

export function resolveGroupTarget(
  groupId: SidebarGroupId,
  children: TabConfig[],
  lastChildByGroup: Partial<Record<SidebarGroupId, string>>,
  readyGroups?: ReadonlySet<SidebarGroupId>,
): TabId | null {
  const remembered = lastChildByGroup[groupId];
  if (remembered) {
    const match = children.find((c) => c.id === remembered);
    if (match) return match.id;
  }
  const groupDef = sidebarGroups.find((g) => g.id === groupId);
  if (groupDef) {
    // A group with a "ready" child (Music with a model installed) opens there
    // instead of on its default.
    if (groupDef.readyChild && readyGroups?.has(groupId)) {
      const ready = children.find((c) => c.id === groupDef.readyChild);
      if (ready) return ready.id;
    }
    const defaultChild = children.find((c) => c.id === groupDef.defaultChild);
    if (defaultChild) return defaultChild.id;
  }
  return children[0]?.id ?? null;
}
